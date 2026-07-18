"""Aggregate run artifacts into experiment summary CSV files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from paper05.analysis.emissions_recast import add_absolute_emission_fields, fixed_baseline_reference_ton
from paper05.analysis.metrics import marginal_abatement_cost
from paper05.analysis.metrics import dual_based_mac
from paper05.analysis.node_typology import assign_node_typology, typology_summary
from paper05.analysis.sensitivity import LHS_PARAMETERS, partial_rank_correlation, rank_correlation
from paper05.data.loaders import read_yaml
from paper05.model.validity import evaluate_result_validity, split_invalid_reasons


SUMMARY_MAP = {
    "E0_legacy_audit": ("E0", ["main_capacity", "main_cost", "main_emissions", "qa_checks"]),
    "E1_reference_floor": ("E1", ["reference_emissions", "floor_shift_by_technology", "floor_shift_by_grid_carbon"]),
    "E2_8760_validation": ("E2", ["repday_vs_8760", "full8760_benchmark"]),
    "E3_repday_sensitivity": ("E3", ["repday_sensitivity_frontier", "repday_sensitivity_capacity", "repday_sensitivity_storage"]),
    "E4_storage_lifecycle": ("E4", ["storage_lifecycle_frontier", "storage_lifecycle_capacity", "storage_cycles", "storage_costs"]),
    "E5_flexibility_portfolio": ("E5", ["flexibility_frontier", "flexibility_floor_shift", "flexibility_storage_reduction", "flexibility_curtailment_reduction"]),
    "E6_network_spatial_robustness": ("E6", ["network_frontier", "network_storage", "network_curtailment", "network_binding_hours", "network_proxy_edges"]),
    "E7_spatial_aggregation": ("E7", ["spatial_frontier", "spatial_capacity", "spatial_minimum_emissions", "spatial_8760_validation"]),
    "E8_uncertainty_lhs": ("E8", ["lhs_samples", "lhs_outputs", "rank_correlation", "rank_correlation_by_target", "prcc_by_target", "uncertainty_top5_by_target", "uncertainty_band"]),
    "E9_dual_bottleneck": ("E9", ["duals_by_constraint", "binding_constraints_by_cap", "node_binding_score"]),
    "E10_node_typology": ("E10", ["node_metrics", "node_typology", "node_typology_summary"]),
}

SUMMARY_INPUT_COLUMN_RENAMES = {
    "BESS_degradation_cost": "BESS_degradation_cost_input",
    "DR_fraction": "DR_fraction_input",
}

MAC_BASE_GROUP_COLUMNS = ["experiment_id", "technology_set", "grid_carbon_scenario"]
MAC_EXTRA_GROUP_COLUMNS = {
    "E2_8760_validation": ["temporal_mode"],
    "E3_repday_sensitivity": ["n_rep_days"],
    "E4_storage_lifecycle": ["storage_model"],
    "E6_network_spatial_robustness": ["network_mode"],
    "E7_spatial_aggregation": ["spatial_resolution", "objective_name"],
    "E8_uncertainty_lhs": ["lhs_sample_id"],
}

ABSOLUTE_EMISSION_COLUMNS = [
    "co2_cap_mt",
    "actual_emissions_mt",
    "scenario_relative_cap_ratio",
    "fixed_baseline_reference_ton",
    "fixed_baseline_cap_ratio",
]

E8_TARGETS = ["total_system_cost", "actual_emissions_ton", "bess_energy_mwh", "grid_import_mwh"]
E8_REPRESENTATIVE_CAPS = [0.25, 0.14]

VALIDITY_AUDIT_MIN_COLUMNS = [
    "experiment",
    "run_id",
    "config",
    "status",
    "valid",
    "invalid_reason",
    "failed_reason",
    "mode",
    "technology_set",
    "co2_cap_ratio",
    "co2_cap_ton",
    "actual_emissions_ton",
    "cap_violation_ton",
    "load_shedding_mwh",
    "objective_value",
    "total_system_cost",
    "objective_cost_gap",
    "objective_cost_rel_gap",
    "has_size_guard",
    "is_full8760",
    "can_use_for_paper",
    "paper_exclusion_reason",
    "recommended_action",
]


def summarize_results(results_root: Path = Path("results")) -> None:
    summary_dir = results_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    all_runs = []
    status_paths = sorted((results_root / "runs").glob("*/*/run_status.json"))
    for status_path in status_paths:
        run_dir = status_path.parent
        status = _read_json_safe(status_path)
        row = dict(status)
        row["recorded_is_valid_result"] = row.get("is_valid_result")
        row["recorded_invalid_reason"] = row.get("invalid_reason")
        row.update(_config_scalars(run_dir / "config_resolved.yaml"))
        row.update(_first_row(run_dir / "capacities.csv"))
        row.update(_first_row(run_dir / "emissions.csv"))
        row.update(_first_row(run_dir / "costs.csv"))
        row.update(_first_row(run_dir / "curtailment.csv"))
        row.update(_first_row(run_dir / "storage_cycles.csv"))
        row.update(_first_row(run_dir / "grid_import.csv"))
        row.update(_first_row(run_dir / "flexibility.csv"))
        row.update(_read_json_safe(run_dir / "qa_checks.json"))
        row.update(_apply_current_validity(row))
        row["validity_changed"] = bool(row.get("recorded_is_valid_result") != row.get("is_valid_result") or str(row.get("recorded_invalid_reason") or "") != str(row.get("invalid_reason") or ""))
        all_runs.append(row)
    if all_runs:
        all_df = _sanitize_summary_columns(pd.DataFrame(all_runs))
    else:
        existing = summary_dir / "all_run_summary.csv"
        all_df = pd.read_csv(existing) if existing.exists() else pd.DataFrame()
        all_df = _sanitize_summary_columns(all_df)
    for col in ["co2_cap_ratio", "lhs_sample_id", "n_rep_days"]:
        if col in all_df.columns:
            try:
                all_df[col] = pd.to_numeric(all_df[col])
            except Exception:
                pass
    if "co2_cap_ratio" in all_df.columns:
        all_df["co2_cap_ratio"] = all_df["co2_cap_ratio"].round(6)
    all_df = _add_absolute_fields_if_possible(all_df)
    all_df.to_csv(summary_dir / "all_run_summary.csv", index=False)
    if all_df.empty:
        return

    for experiment_id, (label, names) in SUMMARY_MAP.items():
        exp_df = all_df[all_df["experiment_id"].astype(str) == experiment_id].copy()
        if exp_df.empty:
            continue
        _write_experiment_run_status(summary_dir, experiment_id, exp_df)
        for name in names:
            _write_named_summary(summary_dir, label, name, exp_df, results_root)

    _write_mac(summary_dir, all_df)
    _write_mac_dual_comparison(summary_dir, all_df, results_root)
    _write_absolute_emissions_recast(summary_dir, all_df)
    _write_validity_audit(summary_dir, all_df)


def _write_experiment_run_status(summary_dir: Path, experiment_id: str, df: pd.DataFrame) -> None:
    cols = _existing(
        df,
        [
            "experiment_id",
            "run_id",
            "dataset_id",
            "spatial_resolution",
            "temporal_mode",
            "technology_set",
            "grid_carbon_scenario",
            "co2_cap_ratio",
            "objective_name",
            "solver_name",
            "solver_status",
            "objective_value",
            "solver_objective_value",
            "is_valid_result",
            "invalid_reason",
            "runtime_seconds",
            "timestamp",
            "git_commit",
            "input_hash",
            "reference_case_error",
            "recorded_is_valid_result",
            "recorded_invalid_reason",
            "validity_changed",
        ],
    )
    df[cols].to_csv(summary_dir / f"{experiment_id}_run_status.csv", index=False)


def _write_named_summary(summary_dir: Path, exp: str, name: str, df: pd.DataFrame, results_root: Path) -> None:
    out = summary_dir / f"{exp}_{name}.csv"
    if name in {"main_capacity", "storage_lifecycle_capacity", "repday_sensitivity_capacity", "spatial_capacity"}:
        if name == "spatial_capacity":
            df = df[(df["temporal_mode"] == "repday_opt") & (df["objective_name"] == "min_cost")].copy()
        cols = _existing(df, ["experiment_id", "run_id", "technology_set", "spatial_resolution", "temporal_mode", "grid_carbon_scenario", "co2_cap_ratio", "pv_mw", "wind_mw", "bess_power_mw", "bess_energy_mwh", "ldes_power_mw", "ldes_energy_mwh", "grid_expansion_mw"])
        df[cols].to_csv(out, index=False)
    elif name in {"main_cost", "storage_costs"}:
        cost_cols = [c for c in df.columns if c.endswith("_cost") or c in {"total_system_cost", "load_shedding_penalty"}]
        if "total_system_cost_billion_cny" in df.columns:
            cost_cols.append("total_system_cost_billion_cny")
        df[_existing(df, ["experiment_id", "run_id", "technology_set", "grid_carbon_scenario", "co2_cap_ratio"]) + cost_cols].to_csv(out, index=False)
    elif name == "main_emissions":
        cols = _existing(df, _with_absolute_columns(["experiment_id", "run_id", "technology_set", "grid_carbon_scenario", "co2_cap_ratio", "E_ref_grid_only_ton", "E_lc_unconstrained_ton", "E_min_techset_ton", "co2_cap_ton", "actual_emissions_ton", "reduction_vs_grid_only", "reduction_vs_unconstrained_lc", "cap_violation_ton"]))
        df[cols].to_csv(out, index=False)
    elif name == "qa_checks":
        cols = _existing(df, ["experiment_id", "run_id", "objective_name", "solver_status", "is_valid_result", "invalid_reason", "total_load_shedding_mwh", "cap_violation_ton", "cost_component_sum_error", "objective_cost_consistency_error", "ess_zero_capacity_zero_cost"])
        df[cols].to_csv(out, index=False)
    elif name == "reference_emissions":
        _reference_emissions_summary(df).to_csv(out, index=False)
    elif name == "floor_shift_by_technology":
        _floor_shift_by_technology(df).to_csv(out, index=False)
    elif name == "floor_shift_by_grid_carbon":
        _floor_shift_by_grid_carbon(df).to_csv(out, index=False)
    elif name == "repday_vs_8760":
        _repday_vs_8760(df).to_csv(out, index=False)
    elif name == "full8760_benchmark":
        bench = _valid_only(df)
        bench = bench[
            (bench["temporal_mode"] == "full8760_opt")
            & (bench["grid_carbon_scenario"] == "baseline")
            & (bench["co2_cap_ratio"].isin([0.40, 0.25, 0.14]))
        ].copy()
        bench.to_csv(out, index=False)
    elif name == "repday_sensitivity_frontier":
        _select(df, _with_absolute_columns(["experiment_id", "run_id", "n_rep_days", "co2_cap_ratio", "grid_carbon_scenario", "solver_status", "is_valid_result", "invalid_reason", "actual_emissions_ton", "total_system_cost", "total_system_cost_billion_cny"])).to_csv(out, index=False)
    elif name == "repday_sensitivity_storage":
        _select(df, ["experiment_id", "run_id", "n_rep_days", "co2_cap_ratio", "bess_power_mw", "bess_energy_mwh", "bess_throughput_mwh", "bess_equivalent_full_cycles", "ldes_power_mw", "ldes_energy_mwh"]).to_csv(out, index=False)
    elif name == "storage_lifecycle_frontier":
        _select(df, _with_absolute_columns(["experiment_id", "run_id", "storage_model", "grid_carbon_scenario", "co2_cap_ratio", "solver_status", "is_valid_result", "invalid_reason", "actual_emissions_ton", "total_system_cost", "total_system_cost_billion_cny"])).to_csv(out, index=False)
    elif name == "storage_cycles":
        _select(df, ["experiment_id", "run_id", "storage_model", "grid_carbon_scenario", "co2_cap_ratio", "bess_charge_mwh", "bess_discharge_mwh", "bess_throughput_mwh", "bess_equivalent_full_cycles", "bess_average_duration_h", "bess_max_duration_h", "ldes_charge_mwh", "ldes_discharge_mwh", "ldes_equivalent_full_cycles"]).to_csv(out, index=False)
    elif name == "flexibility_frontier":
        _select(df, _with_absolute_columns(["experiment_id", "run_id", "technology_set", "co2_cap_ratio", "solver_status", "is_valid_result", "invalid_reason", "actual_emissions_ton", "total_system_cost", "total_system_cost_billion_cny", "total_load_shedding_mwh"])).to_csv(out, index=False)
    elif name == "flexibility_floor_shift":
        _flexibility_floor_shift(results_root).to_csv(out, index=False)
    elif name == "flexibility_storage_reduction":
        _delta_against_base(df, ["bess_power_mw", "bess_energy_mwh", "ldes_power_mw", "ldes_energy_mwh"]).to_csv(out, index=False)
    elif name == "flexibility_curtailment_reduction":
        _delta_against_base(df, ["pv_curtailment_mwh", "wind_curtailment_mwh"]).to_csv(out, index=False)
    elif name == "network_frontier":
        _select(df, _with_absolute_columns(["experiment_id", "run_id", "network_mode", "technology_set", "co2_cap_ratio", "solver_status", "is_valid_result", "invalid_reason", "actual_emissions_ton", "total_system_cost", "total_system_cost_billion_cny", "total_load_shedding_mwh"])).to_csv(out, index=False)
    elif name == "network_storage":
        _select(df, ["experiment_id", "run_id", "network_mode", "technology_set", "co2_cap_ratio", "bess_power_mw", "bess_energy_mwh", "ldes_power_mw", "ldes_energy_mwh"]).to_csv(out, index=False)
    elif name == "network_curtailment":
        _select(df, ["experiment_id", "run_id", "network_mode", "technology_set", "co2_cap_ratio", "pv_curtailment_mwh", "wind_curtailment_mwh"]).to_csv(out, index=False)
    elif name == "network_binding_hours":
        _network_binding_hours(_valid_only(df), results_root).to_csv(out, index=False)
    elif name == "spatial_frontier":
        frontier = df[(df["temporal_mode"] == "repday_opt") & (df["objective_name"] == "min_cost")].copy()
        _select(frontier, _with_absolute_columns(["experiment_id", "run_id", "spatial_resolution", "technology_set", "temporal_mode", "co2_cap_ratio", "solver_status", "is_valid_result", "invalid_reason", "actual_emissions_ton", "total_system_cost", "total_system_cost_billion_cny", "total_load_shedding_mwh"])).to_csv(out, index=False)
    elif name == "spatial_minimum_emissions":
        _select(df[df["objective_name"] == "min_emissions"], ["experiment_id", "run_id", "spatial_resolution", "technology_set", "actual_emissions_ton", "E_min_techset_ton", "solver_status", "is_valid_result", "invalid_reason"]).to_csv(out, index=False)
    elif name == "spatial_8760_validation":
        valid_8760 = _valid_only(df[df["temporal_mode"] == "full8760_dispatch_validation"].copy())
        _select(valid_8760, ["experiment_id", "run_id", "spatial_resolution", "technology_set", "co2_cap_ratio", "solver_status", "is_valid_result", "invalid_reason", "actual_emissions_ton", "total_system_cost"]).to_csv(out, index=False)
    elif name == "lhs_samples":
        lhs_df = _with_lhs_parameter_aliases(df)
        _select(lhs_df, ["experiment_id", "run_id", "lhs_sample_id", "co2_cap_ratio"] + LHS_PARAMETERS).to_csv(out, index=False)
    elif name == "lhs_outputs":
        _select(df, _with_absolute_columns(["experiment_id", "run_id", "lhs_sample_id", "co2_cap_ratio", "solver_status", "is_valid_result", "invalid_reason", "total_system_cost", "total_system_cost_billion_cny", "actual_emissions_ton", "pv_mw", "wind_mw", "bess_power_mw", "bess_energy_mwh", "grid_import_mwh", "pv_curtailment_mwh", "wind_curtailment_mwh"])).to_csv(out, index=False)
    elif name == "rank_correlation":
        valid_df = _valid_only(df)
        _annotate_valid_only_scope(_lhs_rank_correlation(valid_df), source_df=df, valid_df=valid_df).to_csv(out, index=False)
    elif name == "rank_correlation_by_target":
        valid_df = _valid_only(df)
        _annotate_valid_only_scope(_lhs_rank_correlation_by_target(valid_df), source_df=df, valid_df=valid_df).to_csv(out, index=False)
    elif name == "prcc_by_target":
        valid_df = _valid_only(df)
        _annotate_valid_only_scope(_lhs_prcc_by_target(valid_df), source_df=df, valid_df=valid_df).to_csv(out, index=False)
    elif name == "uncertainty_top5_by_target":
        valid_df = _valid_only(df)
        _annotate_valid_only_scope(_lhs_uncertainty_top5_by_target(valid_df), source_df=df, valid_df=valid_df).to_csv(out, index=False)
    elif name == "uncertainty_band":
        valid_df = _valid_only(df)
        _annotate_valid_only_scope(_lhs_uncertainty_band(valid_df), source_df=df, valid_df=valid_df).to_csv(out, index=False)
    elif name.startswith("node_") or name in {"duals_by_constraint", "binding_constraints_by_cap"}:
        valid_df = _valid_only(df)
        _write_node_or_dual_summary(out, name, df, valid_df, results_root)
    elif name == "network_proxy_edges":
        valid_df = _valid_only(df)
        _write_network_proxy_edges(out, df, valid_df, results_root)
    else:
        df.to_csv(out, index=False)


def _write_node_or_dual_summary(out: Path, name: str, source_df: pd.DataFrame, valid_df: pd.DataFrame, results_root: Path) -> None:
    frames = []
    dual_names = {"duals_by_constraint", "binding_constraints_by_cap"}
    for _, row in valid_df.iterrows():
        run_dir = results_root / "runs" / row["experiment_id"] / row["run_id"]
        fname = "duals.csv" if name in dual_names else "node_metrics.csv"
        path = run_dir / fname
        if path.exists():
            part = _read_csv_safe(path)
            if part.empty:
                continue
            part.insert(0, "run_id", row["run_id"])
            part.insert(0, "experiment_id", row["experiment_id"])
            frames.append(part)
    out_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out_df.empty and not (results_root / "runs").exists() and out.exists():
        return
    if name == "node_typology" and not out_df.empty:
        out_df = assign_node_typology(out_df)
    if name == "node_typology_summary" and not out_df.empty:
        out_df = typology_summary(assign_node_typology(out_df))
    if name == "node_binding_score" and not out_df.empty:
        meta = valid_df[["experiment_id", "run_id", "co2_cap_ratio", "technology_set", "solver_status", "is_valid_result", "invalid_reason"]].drop_duplicates()
        out_df = out_df.merge(meta, on=["experiment_id", "run_id"], how="left")
        out_df = out_df[
            _existing(
                out_df,
                [
                    "experiment_id",
                    "run_id",
                    "technology_set",
                    "co2_cap_ratio",
                    "node_id",
                    "node_binding_score",
                    "annual_load_mwh",
                    "peak_load_mw",
                    "RSI",
                    "GDI",
                    "SDI",
                    "solver_status",
                    "is_valid_result",
                    "invalid_reason",
                ],
            )
        ].sort_values(["co2_cap_ratio", "node_binding_score"], ascending=[True, False])
    if name == "binding_constraints_by_cap" and not out_df.empty:
        out_df = _binding_constraints_by_cap(valid_df, out_df)
    _annotate_valid_only_scope(out_df, source_df=source_df, valid_df=valid_df).to_csv(out, index=False)


def _write_network_proxy_edges(out: Path, source_df: pd.DataFrame, valid_df: pd.DataFrame, results_root: Path) -> None:
    frames = []
    for _, row in valid_df.iterrows():
        run_dir = results_root / "runs" / row["experiment_id"] / row["run_id"]
        path = run_dir / "network_edges.csv"
        if not path.exists():
            continue
        part = _read_csv_safe(path)
        if part.empty:
            continue
        part.insert(0, "network_mode", row.get("network_mode"))
        part.insert(0, "co2_cap_ratio", row.get("co2_cap_ratio"))
        part.insert(0, "run_id", row["run_id"])
        part.insert(0, "experiment_id", row["experiment_id"])
        frames.append(part)
    out_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out_df.empty and not (results_root / "runs").exists() and out.exists():
        return
    _annotate_valid_only_scope(out_df, source_df=source_df, valid_df=valid_df).to_csv(out, index=False)


def _write_mac(summary_dir: Path, df: pd.DataFrame) -> None:
    valid = _valid_only(df).dropna(subset=["actual_emissions_ton", "total_system_cost"])
    evidence_frames = []
    diagnostic_frames = []
    for experiment_id, exp_valid in valid.groupby("experiment_id", dropna=False):
        group_cols = _mac_group_columns(str(experiment_id), exp_valid)
        if not group_cols:
            continue
        for _, group in exp_valid.groupby(group_cols, dropna=False):
            mac = _mac_rows_for_group(group, group_cols)
            if mac.empty:
                continue
            diagnostic_frames.append(mac)
            evidence_frames.append(mac[mac["is_valid_mac"].fillna(False)].copy())
    out = pd.concat(evidence_frames, ignore_index=True) if evidence_frames else pd.DataFrame()
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True) if diagnostic_frames else pd.DataFrame()
    out.to_csv(summary_dir / "all_mac.csv", index=False)
    diagnostics.to_csv(summary_dir / "all_mac_diagnostics.csv", index=False)


def _write_mac_dual_comparison(summary_dir: Path, df: pd.DataFrame, results_root: Path) -> None:
    valid = _valid_only(df)
    finite = _read_csv_safe(summary_dir / "all_mac.csv")
    rows = _dual_rows_from_run_artifacts(valid, finite, results_root)
    if not rows:
        rows = _dual_rows_from_summary_files(summary_dir, valid, finite)
    pd.DataFrame(rows).to_csv(summary_dir / "all_mac_dual_comparison.csv", index=False)


def _dual_rows_from_run_artifacts(valid: pd.DataFrame, finite: pd.DataFrame, results_root: Path) -> list[dict[str, Any]]:
    rows = []
    for _, run in valid.iterrows():
        dual_path = results_root / "runs" / str(run["experiment_id"]) / str(run["run_id"]) / "duals.csv"
        if not dual_path.exists():
            continue
        duals = _read_csv_safe(dual_path)
        if duals.empty:
            continue
        co2_dual = duals[duals["constraint"].astype(str).eq("co2_cap")].copy()
        if co2_dual.empty:
            continue
        rows.append(_mac_dual_row(run, co2_dual, finite))
    return rows


def _dual_rows_from_summary_files(summary_dir: Path, valid: pd.DataFrame, finite: pd.DataFrame) -> list[dict[str, Any]]:
    binding = _read_csv_safe(summary_dir / "E9_binding_constraints_by_cap.csv")
    if binding.empty:
        return []
    co2 = binding[binding["constraint_family"].astype(str).eq("co2_cap")].copy()
    if co2.empty:
        return []
    meta_cols = _existing(valid, ["experiment_id", "run_id", "technology_set", "grid_carbon_scenario", "co2_cap_ratio", "co2_cap_mt", "actual_emissions_mt", "is_valid_result", "invalid_reason"])
    meta = valid[meta_cols].drop_duplicates("run_id") if "run_id" in meta_cols else pd.DataFrame()
    rows = []
    for _, item in co2.iterrows():
        if not meta.empty and item.get("run_id") in set(meta["run_id"]):
            run = meta[meta["run_id"].astype(str).eq(str(item.get("run_id")))].iloc[0]
        else:
            run = item
        dual_df = pd.DataFrame(
            [
                {
                    "constraint": "co2_cap",
                    "shadow_price": item.get("max_shadow_price"),
                    "is_binding": float(item.get("binding_count", 0) or 0) > 0,
                }
            ]
        )
        rows.append(_mac_dual_row(run, dual_df, finite))
    return rows


def _mac_dual_row(run: pd.Series, co2_dual: pd.DataFrame, finite: pd.DataFrame) -> dict[str, Any]:
    shadow = pd.to_numeric(co2_dual["shadow_price"], errors="coerce").dropna()
    raw_shadow = float(shadow.iloc[0]) if not shadow.empty else None
    binding = bool(co2_dual["is_binding"].astype(str).str.lower().isin({"true", "1", "yes"}).any())
    dual_mac = dual_based_mac(co2_dual)
    finite_mac = _finite_mac_for_run(finite, str(run.get("run_id", "")))
    abs_diff = abs(finite_mac - dual_mac) if finite_mac is not None and dual_mac is not None else None
    rel_diff = abs_diff / abs(finite_mac) if abs_diff is not None and finite_mac not in (None, 0) else None
    return {
        "experiment_id": run.get("experiment_id"),
        "run_id": run.get("run_id"),
        "technology_set": run.get("technology_set"),
        "grid_carbon_scenario": run.get("grid_carbon_scenario"),
        "co2_cap_ratio": run.get("co2_cap_ratio"),
        "co2_cap_mt": run.get("co2_cap_mt"),
        "actual_emissions_mt": run.get("actual_emissions_mt"),
        "finite_difference_MAC_CNY_per_tCO2": finite_mac,
        "dual_MAC_CNY_per_tCO2": dual_mac,
        "dual_shadow_price_raw": raw_shadow,
        "co2_cap_is_binding": binding,
        "mac_abs_difference": abs_diff,
        "mac_relative_difference": rel_diff,
        "is_valid_result": run.get("is_valid_result"),
        "invalid_reason": run.get("invalid_reason"),
    }


def _finite_mac_for_run(finite: pd.DataFrame, run_id: str) -> float | None:
    if finite.empty or not run_id:
        return None
    for col in ["to_run_id", "from_run_id"]:
        if col not in finite.columns:
            continue
        match = finite[finite[col].astype(str).eq(run_id)]
        if not match.empty and "MAC_CNY_per_tCO2" in match.columns:
            value = pd.to_numeric(match["MAC_CNY_per_tCO2"], errors="coerce").dropna()
            if not value.empty:
                return float(value.iloc[0])
    return None


def _write_absolute_emissions_recast(summary_dir: Path, df: pd.DataFrame) -> None:
    cols = _existing(
        df,
        _with_absolute_columns(
            [
                "experiment_id",
                "run_id",
                "technology_set",
                "grid_carbon_scenario",
                "spatial_resolution",
                "temporal_mode",
                "network_mode",
                "co2_cap_ratio",
                "co2_cap_ton",
                "actual_emissions_ton",
                "total_system_cost",
                "total_system_cost_billion_cny",
                "is_valid_result",
                "invalid_reason",
                "solver_status",
            ]
        ),
    )
    out = df[cols].copy()
    out.to_csv(summary_dir / "absolute_emissions_recast_all_runs.csv", index=False)
    _valid_only(out).to_csv(summary_dir / "absolute_emissions_recast_valid_runs.csv", index=False)


def _mac_group_columns(experiment_id: str, df: pd.DataFrame) -> list[str]:
    cols = MAC_BASE_GROUP_COLUMNS + MAC_EXTRA_GROUP_COLUMNS.get(experiment_id, [])
    return _existing(df, cols)


def _mac_rows_for_group(group: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if len(group) < 2:
        return pd.DataFrame()
    frontier = group.sort_values("actual_emissions_ton", ascending=False).reset_index(drop=True)
    mac = marginal_abatement_cost(frontier, "total_system_cost", "actual_emissions_ton")
    if mac.empty:
        return mac
    annotated = mac.copy()
    for col in group_cols:
        annotated[col] = frontier.iloc[0][col]
    for source_col in [
        "run_id",
        "co2_cap_ratio",
        "co2_cap_mt",
        "fixed_baseline_cap_ratio",
        "scenario_relative_cap_ratio",
        "objective_name",
        "actual_emissions_ton",
        "actual_emissions_mt",
        "total_system_cost",
        "total_system_cost_billion_cny",
    ]:
        if source_col not in frontier.columns:
            continue
        annotated[f"from_{source_col}"] = [frontier.iloc[int(idx)][source_col] for idx in annotated["from_index"]]
        annotated[f"to_{source_col}"] = [frontier.iloc[int(idx)][source_col] for idx in annotated["to_index"]]
    annotated["source_valid_runs_only"] = True
    annotated["source_validity_scope"] = "valid_runs_only"
    annotated["source_valid_run_count"] = int(len(frontier))
    return annotated


def _first_row(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {}
    return df.iloc[0].to_dict() if not df.empty else {}


def _existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _with_absolute_columns(cols: list[str]) -> list[str]:
    out = list(cols)
    for col in ABSOLUTE_EMISSION_COLUMNS:
        if col not in out:
            out.append(col)
    return out


def _add_absolute_fields_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "total_system_cost" in out.columns:
        out["total_system_cost_billion_cny"] = pd.to_numeric(out["total_system_cost"], errors="coerce") / 1_000_000_000.0
    try:
        fixed_ref = fixed_baseline_reference_ton(out)
    except ValueError:
        return out
    return add_absolute_emission_fields(out, fixed_reference_ton=fixed_ref)


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json_safe(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _apply_current_validity(row: dict[str, Any]) -> dict[str, Any]:
    recorded_reason = str(row.get("recorded_invalid_reason") or row.get("invalid_reason") or "").strip()
    solver_success = _coerce_bool(row.get("solver_success"), default=str(row.get("solver_status", "")) == "optimal")
    if not solver_success:
        return {
            "solver_success": False,
            "is_valid_result": False,
            "invalid_reason": recorded_reason or "solver_failed",
        }

    total_cost = _maybe_float(row.get("total_system_cost"))
    if total_cost is None:
        total_cost = _maybe_float(row.get("objective_value"))
    validity = evaluate_result_validity(
        solver_success=True,
        objective_name=str(row.get("objective_name", "min_cost")),
        total_load_shedding_mwh=_maybe_float(row.get("total_load_shedding_mwh")),
        cap_violation_ton=_maybe_float(row.get("cap_violation_ton")),
        total_cost=total_cost,
        cost_component_sum_error=_cost_component_error(row, total_cost),
        solver_objective_value=_maybe_float(row.get("solver_objective_value")),
        ess_zero_capacity_zero_cost=_coerce_bool(row.get("ess_zero_capacity_zero_cost"), default=True),
        load_shedding_tolerance_mwh=_coerce_float(row.get("load_shedding_tolerance_mwh"), default=1e-6),
        co2_tolerance_ton=_coerce_float(row.get("co2_tolerance_ton"), default=1e-4),
        cost_tolerance_yuan=_coerce_float(row.get("cost_tolerance_yuan"), default=1e-4),
        cost_relative_tolerance=_coerce_float(row.get("cost_relative_tolerance"), default=1e-10),
    )
    return {"solver_success": True, **validity}


def _cost_component_error(row: dict[str, Any], total_cost: float | None) -> float | None:
    existing = _maybe_float(row.get("cost_component_sum_error"))
    if existing is not None:
        return existing
    if total_cost is None:
        return None
    excluded_cost_like_inputs = set(SUMMARY_INPUT_COLUMN_RENAMES)
    excluded_cost_like_inputs.update(SUMMARY_INPUT_COLUMN_RENAMES.values())
    cost_cols = [
        key
        for key in row
        if (key.endswith("_cost") or key == "load_shedding_penalty")
        and key != "total_system_cost"
        and key not in excluded_cost_like_inputs
    ]
    if not cost_cols:
        return None
    component_sum = 0.0
    for key in cost_cols:
        value = _maybe_float(row.get(key))
        if value is None:
            continue
        component_sum += value
    return abs(total_cost - component_sum)


def _write_validity_audit(summary_dir: Path, all_df: pd.DataFrame) -> None:
    audit_df = all_df[
        (all_df["solver_status"].astype(str) != "optimal")
        | (~all_df["is_valid_result"].fillna(False))
    ].copy()
    audit_cols = [
        *VALIDITY_AUDIT_MIN_COLUMNS,
        "experiment_id",
        "solver_status",
        "is_valid_result",
        "recorded_is_valid_result",
        "recorded_invalid_reason",
        "validity_changed",
        "primary_issue",
        "audit_category",
        "paper_disposition",
        "temporal_mode",
        "grid_carbon_scenario",
        "total_load_shedding_mwh",
        "objective_cost_consistency_error",
        "objective_cost_consistency_rel_error",
        "objective_cost_tolerance",
        "cost_component_sum_error",
        "reference_case_error",
    ]
    if not audit_df.empty:
        for col in audit_cols:
            if col not in audit_df.columns:
                audit_df[col] = pd.NA
        audit_df["primary_issue"] = audit_df.apply(_primary_issue, axis=1)
        audit_df["audit_category"] = audit_df.apply(_audit_category, axis=1)
        audit_df["recommended_action"] = audit_df["audit_category"].map(_recommended_action)
        audit_df["paper_disposition"] = audit_df["audit_category"].map(_paper_disposition)
        audit_df["experiment"] = audit_df["experiment_id"]
        audit_df["config"] = audit_df.apply(_config_descriptor, axis=1)
        audit_df["status"] = audit_df.apply(_audit_status, axis=1)
        audit_df["valid"] = audit_df["is_valid_result"].fillna(False)
        audit_df["failed_reason"] = audit_df.apply(_failed_reason_text, axis=1)
        audit_df["invalid_reason"] = audit_df.apply(_invalid_reason_text, axis=1)
        audit_df["mode"] = audit_df["temporal_mode"]
        audit_df["load_shedding_mwh"] = audit_df["total_load_shedding_mwh"]
        audit_df["objective_cost_gap"] = audit_df["objective_cost_consistency_error"]
        audit_df["objective_cost_rel_gap"] = audit_df["objective_cost_consistency_rel_error"]
        audit_df["has_size_guard"] = audit_df.apply(_has_size_guard, axis=1)
        audit_df["is_full8760"] = audit_df["temporal_mode"].astype(str).str.startswith("full8760")
        audit_df["can_use_for_paper"] = False
        audit_df["paper_exclusion_reason"] = audit_df["audit_category"]
        audit_out = audit_df[_existing(audit_df, audit_cols)]
    else:
        audit_out = pd.DataFrame(columns=audit_cols)
    audit_out.to_csv(summary_dir / "validity_audit.csv", index=False)
    _invalid_reason_breakdown(audit_out).to_csv(summary_dir / "invalid_reason_breakdown.csv", index=False)
    (summary_dir / "validity_audit.md").write_text(_validity_audit_markdown(all_df, audit_out), encoding="utf-8")


def _invalid_reason_breakdown(audit_df: pd.DataFrame) -> pd.DataFrame:
    if audit_df.empty:
        return pd.DataFrame(columns=["experiment", "experiment_id", "status", "paper_exclusion_reason", "audit_category", "reason_field", "reason_token", "count"])
    rows: list[dict[str, Any]] = []
    for _, row in audit_df.iterrows():
        tokens = _audit_reason_tokens(row)
        if not tokens:
            tokens = [str(row.get("primary_issue") or row.get("audit_category") or "unknown")]
        reason_field = "failed_reason" if str(row.get("status")) == "failed" else "invalid_reason"
        for token in tokens:
            rows.append(
                {
                    "experiment": row.get("experiment"),
                    "experiment_id": row.get("experiment_id"),
                    "status": row.get("status"),
                    "paper_exclusion_reason": row.get("paper_exclusion_reason"),
                    "audit_category": row.get("audit_category"),
                    "reason_field": reason_field,
                    "reason_token": token,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["experiment", "experiment_id", "status", "paper_exclusion_reason", "audit_category", "reason_field", "reason_token", "count"])
    return (
        out.groupby(["experiment", "experiment_id", "status", "paper_exclusion_reason", "audit_category", "reason_field", "reason_token"], dropna=False, as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "experiment", "reason_token"], ascending=[False, True, True])
    )


def _validity_audit_markdown(all_df: pd.DataFrame, audit_df: pd.DataFrame) -> str:
    valid_count = int(all_df["is_valid_result"].fillna(False).sum()) if "is_valid_result" in all_df.columns else 0
    failed_count = int((all_df["solver_status"].astype(str) != "optimal").sum()) if "solver_status" in all_df.columns else 0
    invalid_count = int(len(all_df) - valid_count - failed_count)
    reclassified = int(all_df.get("validity_changed", pd.Series(dtype=bool)).sum()) if "validity_changed" in all_df.columns else 0
    lines = [
        "# Validity Audit",
        "",
        f"- Total runs: {len(all_df)}",
        f"- Valid runs: {valid_count}",
        f"- Invalid runs: {invalid_count}",
        f"- Failed runs: {failed_count}",
        f"- Reclassified after current validity logic: {reclassified}",
        "",
        "## Family Disposition",
        "| experiment_id | valid | invalid | failed | disposition | notes |",
        "|---|---:|---:|---:|---|---|",
    ]
    for experiment_id, group in all_df.groupby("experiment_id", dropna=False):
        valid = int(group["is_valid_result"].fillna(False).sum())
        failed = int((group["solver_status"].astype(str) != "optimal").sum())
        invalid = int(len(group) - valid - failed)
        experiment_audit = audit_df[audit_df["experiment_id"] == experiment_id].copy()
        notes = _family_notes(group, experiment_audit)
        lines.append(f"| {experiment_id} | {valid} | {invalid} | {failed} | {_family_disposition(group, experiment_audit)} | {notes} |")
    lines.extend(
        [
            "",
            "## Current Invalid/Failed Breakdown",
            "| audit_category | count | can_use_for_paper | recommended_action |",
            "|---|---:|---|---|",
        ]
    )
    if audit_df.empty:
        lines.append("| none | 0 | n/a | n/a |")
    else:
        grouped = (
            audit_df.groupby(["audit_category", "can_use_for_paper", "recommended_action"], dropna=False, as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values(["count", "audit_category"], ascending=[False, True])
        )
        for row in grouped.to_dict("records"):
            lines.append(f"| {row['audit_category']} | {int(row['count'])} | {row['can_use_for_paper']} | {row['recommended_action']} |")
    return "\n".join(lines) + "\n"


def _family_disposition(group: pd.DataFrame, audit_df: pd.DataFrame) -> str:
    valid = int(group["is_valid_result"].fillna(False).sum())
    failed = int((group["solver_status"].astype(str) != "optimal").sum())
    invalid = int(len(group) - valid - failed)
    categories = set(audit_df.get("audit_category", pd.Series(dtype=str)).dropna().astype(str))
    if failed == 0 and invalid == 0:
        return "reported_summary"
    if valid > 0 and categories <= {"unavailable_due_to_size_guard"}:
        return "reported_summary_except_size_guard_rows"
    if valid > 0 and "unavailable_due_to_v2g_model_gap" in categories:
        return "filter_valid_rows_exclude_v2g"
    if valid > 0:
        return "filter_invalid_rows"
    if categories and categories <= {"unavailable_due_to_size_guard", "unavailable_due_to_v2g_model_gap"}:
        return "unavailable"
    return "exclude"


def _family_notes(group: pd.DataFrame, audit_df: pd.DataFrame) -> str:
    notes = []
    if "validity_changed" in group.columns:
        changed = int(group["validity_changed"].sum())
        if changed:
            notes.append(f"reclassified={changed}")
    categories = set(audit_df.get("audit_category", pd.Series(dtype=str)).dropna().astype(str))
    if "invalid_load_shedding" in categories:
        notes.append("load_shedding_present")
    if "unavailable_due_to_size_guard" in categories:
        notes.append("size_guard_rows")
    if "unavailable_due_to_v2g_model_gap" in categories:
        notes.append("v2g_model_gap")
    return ", ".join(notes) if notes else "none"


def _primary_issue(row: pd.Series) -> str:
    reason = str(row.get("invalid_reason") or "").strip()
    if str(row.get("solver_status", "")) != "optimal":
        if ":" in reason:
            return reason.split(":", 1)[0]
        return reason or "solver_failed"
    for token in ["load_shedding", "co2_cap_violation", "cost_inconsistency", "objective_cost_mismatch", "ess_zero_capacity_nonzero_cost"]:
        if token in split_invalid_reasons(reason):
            return token
    return reason or "unknown"


def _audit_category(row: pd.Series) -> str:
    reason = str(row.get("invalid_reason") or "")
    if str(row.get("solver_status", "")) != "optimal":
        if any(token in reason for token in ["failed_full8760_opt", "failed_full8760_dispatch_validation", "failed_spatial_transfer_proxy"]):
            return "unavailable_due_to_size_guard"
        if "V2G" in str(row.get("technology_set", "")) and "infeasible" in reason.lower():
            return "unavailable_due_to_v2g_model_gap"
        if reason.startswith("failed_reference_"):
            return "failed_reference_context"
        return "solver_failed"
    reasons = split_invalid_reasons(reason)
    if "load_shedding" in reasons:
        return "invalid_load_shedding"
    if "co2_cap_violation" in reasons:
        return "invalid_co2_cap_violation"
    if "cost_inconsistency" in reasons:
        return "invalid_cost_inconsistency"
    if "objective_cost_mismatch" in reasons:
        return "invalid_objective_cost_mismatch"
    if "ess_zero_capacity_nonzero_cost" in reasons:
        return "invalid_storage_cost_accounting"
    return "invalid_other"


def _recommended_action(category: str) -> str:
    mapping = {
        "unavailable_due_to_size_guard": "keep_failed_row_and_mark_unavailable",
        "unavailable_due_to_v2g_model_gap": "exclude_v2g_until_ev_trip_energy_model_exists",
        "failed_reference_context": "rerun_after_reference_context_fix",
        "solver_failed": "exclude_failed_run",
        "invalid_load_shedding": "exclude_invalid_run_from_evidence",
        "invalid_co2_cap_violation": "exclude_invalid_run_from_evidence",
        "invalid_cost_inconsistency": "rerun_after_cost_accounting_fix",
        "invalid_objective_cost_mismatch": "rerun_or_recompute_validity_after_tolerance_fix",
        "invalid_storage_cost_accounting": "rerun_after_storage_cost_fix",
        "invalid_other": "inspect_run_manually",
    }
    return mapping.get(category, "inspect_run_manually")


def _paper_disposition(category: str) -> str:
    if category == "unavailable_due_to_size_guard":
        return "unavailable"
    if category == "unavailable_due_to_v2g_model_gap":
        return "must_exclude"
    if category in {"solver_failed", "invalid_load_shedding", "invalid_co2_cap_violation", "invalid_cost_inconsistency", "invalid_storage_cost_accounting", "invalid_other"}:
        return "must_exclude"
    return "filterable"


def _audit_status(row: pd.Series) -> str:
    if str(row.get("solver_status", "")) != "optimal":
        return "failed"
    if _coerce_bool(row.get("is_valid_result"), default=False):
        return "valid"
    return "invalid"


def _failed_reason_text(row: pd.Series) -> str:
    if _audit_status(row) != "failed":
        return ""
    return str(row.get("invalid_reason") or row.get("recorded_invalid_reason") or "solver_failed").strip()


def _invalid_reason_text(row: pd.Series) -> str:
    if _audit_status(row) != "invalid":
        return ""
    return str(row.get("invalid_reason") or row.get("primary_issue") or "").strip()


def _has_size_guard(row: pd.Series) -> bool:
    if str(row.get("audit_category") or "") == "unavailable_due_to_size_guard":
        return True
    text = " ".join(
        [
            _safe_text(row.get("invalid_reason")),
            _safe_text(row.get("failed_reason")),
            _safe_text(row.get("recorded_invalid_reason")),
        ]
    ).lower()
    return "size guard" in text or "failed_full8760" in text or "failed_spatial_transfer_proxy" in text


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _audit_reason_tokens(row: pd.Series) -> list[str]:
    if str(row.get("status")) == "failed":
        primary_issue = str(row.get("primary_issue") or "").strip()
        if primary_issue.startswith("failed_"):
            return [primary_issue]
        category = str(row.get("audit_category") or row.get("paper_exclusion_reason") or "").strip()
        if category:
            return [category]
        reason = str(row.get("failed_reason") or "").strip()
        return [reason] if reason else []
    return split_invalid_reasons(row.get("invalid_reason"))


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any, *, default: float) -> float:
    numeric = _maybe_float(value)
    return default if numeric is None else numeric


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def _select(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[_existing(df, cols)].copy()


def _valid_only(df: pd.DataFrame) -> pd.DataFrame:
    if "is_valid_result" not in df.columns:
        return df.copy()
    return df[df["is_valid_result"].fillna(False)].copy()


def _annotate_valid_only_scope(out_df: pd.DataFrame, *, source_df: pd.DataFrame, valid_df: pd.DataFrame) -> pd.DataFrame:
    annotated = out_df.copy()
    valid_runs = int(valid_df["run_id"].nunique()) if "run_id" in valid_df.columns else int(len(valid_df))
    total_runs = int(source_df["run_id"].nunique()) if "run_id" in source_df.columns else int(len(source_df))
    annotated["source_valid_runs_only"] = True
    annotated["source_valid_run_count"] = valid_runs
    annotated["source_validity_scope"] = "valid_runs_only"
    annotated["source_total_runs"] = total_runs
    annotated["source_invalid_or_failed_runs_excluded"] = total_runs - valid_runs
    scope_cols = [
        "source_valid_runs_only",
        "source_valid_run_count",
        "source_validity_scope",
        "source_total_runs",
        "source_invalid_or_failed_runs_excluded",
    ]
    return annotated[scope_cols + [col for col in annotated.columns if col not in scope_cols]]


def _config_scalars(path: Path) -> dict:
    if not path.exists():
        return {}
    data = read_yaml(path)
    out = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out


def _sanitize_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {old: new for old, new in SUMMARY_INPUT_COLUMN_RENAMES.items() if old in df.columns}
    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def _with_lhs_parameter_aliases(df: pd.DataFrame) -> pd.DataFrame:
    aliased = df.copy()
    for original, renamed in SUMMARY_INPUT_COLUMN_RENAMES.items():
        if renamed in aliased.columns and original not in aliased.columns:
            aliased[original] = aliased[renamed]
    return aliased


def _reference_emissions_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _select(
        df,
        [
            "experiment_id",
            "run_id",
            "objective_name",
            "technology_set",
            "grid_carbon_scenario",
            "E_ref_grid_only_ton",
            "E_lc_unconstrained_ton",
            "E_min_techset_ton",
            "actual_emissions_ton",
            "solver_status",
            "is_valid_result",
            "invalid_reason",
        ],
    )


def _floor_shift_by_technology(df: pd.DataFrame) -> pd.DataFrame:
    floor = df[df["objective_name"] == "min_emissions"].copy()
    return _select(
        floor.sort_values(["technology_set", "grid_carbon_scenario"]),
        [
            "technology_set",
            "grid_carbon_scenario",
            "E_ref_grid_only_ton",
            "E_min_techset_ton",
            "actual_emissions_ton",
            "reduction_vs_grid_only",
            "solver_status",
            "is_valid_result",
            "invalid_reason",
        ],
    )


def _floor_shift_by_grid_carbon(df: pd.DataFrame) -> pd.DataFrame:
    floor = df[df["objective_name"] == "min_emissions"].copy()
    return _select(
        floor.sort_values(["grid_carbon_scenario", "technology_set"]),
        [
            "grid_carbon_scenario",
            "technology_set",
            "E_ref_grid_only_ton",
            "E_min_techset_ton",
            "actual_emissions_ton",
            "reduction_vs_grid_only",
            "solver_status",
            "is_valid_result",
            "invalid_reason",
        ],
    )


def _repday_vs_8760(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["spatial_resolution", "technology_set", "grid_carbon_scenario", "co2_cap_ratio"]
    metrics = [
        "actual_emissions_ton",
        "total_system_cost",
        "pv_curtailment_mwh",
        "wind_curtailment_mwh",
        "grid_import_mwh",
        "bess_throughput_mwh",
        "bess_equivalent_full_cycles",
        "peak_grid_import_mw",
    ]
    base_cols = key_cols + ["temporal_mode", "run_id", "solver_status", "is_valid_result", "invalid_reason", "comparison_run_id"]
    metric_cols = []
    for metric in metrics:
        metric_cols.extend(
            [
                f"{metric}_repday",
                f"{metric}_target",
                f"{metric}_deviation",
                f"{metric}_deviation_ratio",
            ]
        )
    empty = pd.DataFrame(columns=base_cols + metric_cols)
    valid = _valid_only(df)
    rep = valid[valid["temporal_mode"] == "repday_opt"].copy()
    if rep.empty:
        return empty
    rep = rep.set_index(key_cols)
    rows = []
    for _, row in valid[valid["temporal_mode"] != "repday_opt"].iterrows():
        key = tuple(row[col] for col in key_cols)
        if key not in rep.index:
            continue
        base = rep.loc[key]
        if isinstance(base, pd.DataFrame):
            base = base.iloc[0]
        item = {col: row[col] for col in _existing(df, key_cols + ["temporal_mode", "run_id", "solver_status", "is_valid_result", "invalid_reason"])}
        item["comparison_run_id"] = base.get("run_id")
        for metric in metrics:
            item[f"{metric}_repday"] = base.get(metric)
            item[f"{metric}_target"] = row.get(metric)
            if pd.notna(base.get(metric)) and pd.notna(row.get(metric)):
                item[f"{metric}_deviation"] = row.get(metric) - base.get(metric)
                item[f"{metric}_deviation_ratio"] = (row.get(metric) - base.get(metric)) / abs(base.get(metric)) if base.get(metric) not in (0, None) else None
            else:
                item[f"{metric}_deviation"] = None
                item[f"{metric}_deviation_ratio"] = None
        rows.append(item)
    if not rows:
        return empty
    return pd.DataFrame(rows, columns=base_cols + metric_cols)


def _flexibility_floor_shift(results_root: Path) -> pd.DataFrame:
    path = results_root / "summaries" / "E1_floor_shift_by_technology.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    techs = {
        "base_with_storage_degradation",
        "base_with_DR05",
        "base_with_DR10",
        "base_with_DR15",
        "base_with_EV_low_smart",
        "base_with_EV_medium_smart",
        "base_with_EV_high_smart",
        "base_with_EV_medium_smart_V2G",
        "base_with_thermal05",
        "base_with_thermal10",
        "base_with_thermal15",
        "base_with_DR10_EV_medium_thermal10",
        "base_with_DR10_EV_medium_thermal10_LDES",
    }
    return df[df["technology_set"].isin(techs)].copy()


def _delta_against_base(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    key_cols = ["grid_carbon_scenario", "co2_cap_ratio"]
    base = df[df["technology_set"] == "base_with_storage_degradation"].set_index(key_cols)
    rows = []
    for _, row in df.iterrows():
        key = tuple(row[col] for col in key_cols)
        if key not in base.index:
            continue
        ref = base.loc[key]
        if isinstance(ref, pd.DataFrame):
            ref = ref.iloc[0]
        item = {col: row[col] for col in _existing(df, ["experiment_id", "run_id", "technology_set"] + key_cols + ["solver_status", "is_valid_result", "invalid_reason"])}
        for metric in metrics:
            item[f"{metric}_baseline"] = ref.get(metric)
            item[metric] = row.get(metric)
            item[f"{metric}_delta"] = row.get(metric) - ref.get(metric) if pd.notna(row.get(metric)) and pd.notna(ref.get(metric)) else None
        rows.append(item)
    return pd.DataFrame(rows)


def _network_binding_hours(df: pd.DataFrame, results_root: Path) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        dual_path = results_root / "runs" / row["experiment_id"] / row["run_id"] / "duals.csv"
        if not dual_path.exists():
            continue
        duals = _read_csv_safe(dual_path)
        if duals.empty:
            continue
        flow = duals[duals["constraint"].astype(str).str.startswith("grid_interface") | duals["constraint"].astype(str).str.startswith("flow")]
        rows.append(
            {
                "experiment_id": row["experiment_id"],
                "run_id": row["run_id"],
                "network_mode": row.get("network_mode"),
                "technology_set": row.get("technology_set"),
                "co2_cap_ratio": row.get("co2_cap_ratio"),
                "binding_constraint_count": int(flow.get("is_binding", pd.Series(dtype=bool)).sum()),
                "max_shadow_price": float(flow["shadow_price"].max()) if not flow.empty else None,
                "solver_status": row.get("solver_status"),
                "is_valid_result": row.get("is_valid_result"),
                "invalid_reason": row.get("invalid_reason"),
            }
        )
    return pd.DataFrame(rows)


def _lhs_rank_correlation(df: pd.DataFrame) -> pd.DataFrame:
    df = _with_lhs_parameter_aliases(df)
    frames = []
    for cap, group in df.groupby("co2_cap_ratio", dropna=False):
        sample_cols = [col for col in LHS_PARAMETERS if col in group.columns]
        if not sample_cols:
            continue
        ranked = rank_correlation(group[sample_cols], group, target="total_system_cost")
        ranked["co2_cap_ratio"] = cap
        ranked["source_valid_runs_only"] = True
        ranked["source_valid_run_count"] = int(len(group))
        frames.append(ranked)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _lhs_rank_correlation_by_target(df: pd.DataFrame) -> pd.DataFrame:
    df = _with_lhs_parameter_aliases(df)
    frames = []
    for cap, group in _representative_e8_groups(df):
        sample_cols = [col for col in LHS_PARAMETERS if col in group.columns]
        if not sample_cols:
            continue
        for target in E8_TARGETS:
            if target not in group.columns:
                continue
            ranked = rank_correlation(group[sample_cols], group, target=target)
            ranked["target"] = target
            ranked["co2_cap_ratio"] = cap
            ranked["source_valid_runs_only"] = True
            ranked["source_valid_run_count"] = int(len(group))
            frames.append(ranked)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _lhs_prcc_by_target(df: pd.DataFrame) -> pd.DataFrame:
    df = _with_lhs_parameter_aliases(df)
    frames = []
    for cap, group in _representative_e8_groups(df):
        sample_cols = [col for col in LHS_PARAMETERS if col in group.columns]
        if not sample_cols:
            continue
        for target in E8_TARGETS:
            if target not in group.columns:
                continue
            prcc = partial_rank_correlation(group[sample_cols], group, target=target)
            prcc["co2_cap_ratio"] = cap
            prcc["source_valid_runs_only"] = True
            prcc["source_valid_run_count"] = int(len(group))
            frames.append(prcc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _lhs_uncertainty_top5_by_target(df: pd.DataFrame) -> pd.DataFrame:
    prcc = _lhs_prcc_by_target(df)
    if prcc.empty:
        return prcc
    out = prcc.copy()
    out["abs_prcc"] = pd.to_numeric(out["prcc"], errors="coerce").abs()
    frames = []
    for _, group in out.groupby(["co2_cap_ratio", "target"], dropna=False):
        frames.append(group.sort_values("abs_prcc", ascending=False).head(5).copy())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _representative_e8_groups(df: pd.DataFrame):
    for cap, group in df.groupby("co2_cap_ratio", dropna=False):
        numeric_cap = _maybe_float(cap)
        if numeric_cap is None or not any(abs(numeric_cap - target_cap) <= 1e-9 for target_cap in E8_REPRESENTATIVE_CAPS):
            continue
        yield cap, group


def _lhs_uncertainty_band(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["total_system_cost", "actual_emissions_ton", "pv_mw", "wind_mw", "bess_power_mw", "bess_energy_mwh", "grid_import_mwh"]
    rows = []
    for cap, group in df.groupby("co2_cap_ratio", dropna=False):
        for metric in metrics:
            if metric not in group.columns:
                continue
            s = pd.to_numeric(group[metric], errors="coerce").dropna()
            if s.empty:
                continue
            rows.append(
                {
                    "co2_cap_ratio": cap,
                    "metric": metric,
                    "p05": float(s.quantile(0.05)),
                    "p50": float(s.quantile(0.50)),
                    "p95": float(s.quantile(0.95)),
                    "mean": float(s.mean()),
                    "count": int(s.count()),
                    "source_valid_runs_only": True,
                    "source_valid_run_count": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def _binding_constraints_by_cap(summary_df: pd.DataFrame, dual_df: pd.DataFrame) -> pd.DataFrame:
    if dual_df.empty:
        return dual_df
    meta = summary_df[["experiment_id", "run_id", "co2_cap_ratio", "technology_set", "solver_status", "is_valid_result", "invalid_reason"]].drop_duplicates()
    out = dual_df.copy()
    out["constraint_family"] = out["constraint"].astype(str).str.split(":").str[0]
    out = out.merge(meta, on=["experiment_id", "run_id"], how="left")
    grouped = (
        out.groupby(["experiment_id", "run_id", "co2_cap_ratio", "technology_set", "constraint_family", "solver_status", "is_valid_result", "invalid_reason"], dropna=False, as_index=False)
        .agg(binding_count=("is_binding", "sum"), max_shadow_price=("shadow_price", "max"))
        .sort_values(["co2_cap_ratio", "technology_set", "binding_count"], ascending=[True, True, False])
    )
    grouped["source_valid_runs_only"] = True
    grouped["source_valid_run_count"] = int(summary_df["run_id"].nunique()) if "run_id" in summary_df.columns else 0
    return grouped


def _config_descriptor(row: pd.Series) -> str:
    parts = [
        str(row.get("dataset_id") or ""),
        str(row.get("spatial_resolution") or ""),
        str(row.get("temporal_mode") or ""),
        str(row.get("technology_set") or ""),
        str(row.get("grid_carbon_scenario") or ""),
    ]
    if pd.notna(row.get("co2_cap_ratio")):
        parts.append(f"cap={row.get('co2_cap_ratio')}")
    if pd.notna(row.get("lhs_sample_id")):
        parts.append(f"lhs={int(row.get('lhs_sample_id'))}")
    if pd.notna(row.get("storage_model")):
        parts.append(f"storage={row.get('storage_model')}")
    if pd.notna(row.get("network_mode")):
        parts.append(f"network={row.get('network_mode')}")
    return " | ".join(part for part in parts if part and part != "nan")
