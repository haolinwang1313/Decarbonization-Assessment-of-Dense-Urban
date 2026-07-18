from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from paper05.experiments.summarize import summarize_results


def test_summarize_recomputes_validity_and_gates_valid_only_outputs(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_run(
        results_root,
        experiment_id="E10_node_typology",
        run_id="valid_reclassified",
        status={
            "experiment_id": "E10_node_typology",
            "run_id": "valid_reclassified",
            "solver_status": "optimal",
            "objective_name": "min_cost",
            "objective_value": 10_000_000_000.0002,
            "solver_objective_value": 10_000_000_000.0,
            "is_valid_result": False,
            "invalid_reason": "objective_cost_mismatch",
        },
        config={
            "experiment_id": "E10_node_typology",
            "run_id": "valid_reclassified",
            "technology_set": "base_with_DR10_EV_medium_thermal10",
            "grid_carbon_scenario": "baseline",
            "co2_cap_ratio": 0.25,
            "temporal_mode": "repday_opt",
            "objective": "min_cost",
        },
        qa_checks={
            "solver_success": True,
            "objective_name": "min_cost",
            "solver_objective_value": 10_000_000_000.0,
            "total_load_shedding_mwh": 0.0,
            "cap_violation_ton": 0.0,
            "load_shedding_tolerance_mwh": 1e-6,
            "co2_tolerance_ton": 1e-4,
            "cost_tolerance_yuan": 1e-4,
            "objective_cost_consistency_error": 0.0002,
            "is_valid_result": False,
            "invalid_reason": "objective_cost_mismatch",
        },
        costs={"grid_purchase_cost": 4_000_000_000.0, "bess_power_cost": 6_000_000_000.0002, "total_system_cost": 10_000_000_000.0002},
        node_metrics=[{"node_id": "n1", "node_binding_score": 8, "annual_load_mwh": 10.0, "peak_load_mw": 2.0, "RSI": 0.7, "GDI": 0.3, "SDI": 1.1}],
    )
    _write_run(
        results_root,
        experiment_id="E10_node_typology",
        run_id="invalid_load_shedding",
        status={
            "experiment_id": "E10_node_typology",
            "run_id": "invalid_load_shedding",
            "solver_status": "optimal",
            "objective_name": "min_cost",
            "objective_value": 12_000_000_000.0,
            "solver_objective_value": 12_000_000_000.0,
            "is_valid_result": False,
            "invalid_reason": "load_shedding;objective_cost_mismatch",
        },
        config={
            "experiment_id": "E10_node_typology",
            "run_id": "invalid_load_shedding",
            "technology_set": "base_with_DR10_EV_medium_thermal10",
            "grid_carbon_scenario": "baseline",
            "co2_cap_ratio": 0.14,
            "temporal_mode": "repday_opt",
            "objective": "min_cost",
        },
        qa_checks={
            "solver_success": True,
            "objective_name": "min_cost",
            "solver_objective_value": 12_000_000_000.0,
            "total_load_shedding_mwh": 5.0,
            "cap_violation_ton": 0.0,
            "load_shedding_tolerance_mwh": 1e-6,
            "co2_tolerance_ton": 1e-4,
            "cost_tolerance_yuan": 1e-4,
            "objective_cost_consistency_error": 0.0,
            "is_valid_result": False,
            "invalid_reason": "load_shedding",
        },
        costs={"grid_purchase_cost": 4_000_000_000.0, "bess_power_cost": 8_000_000_000.0, "total_system_cost": 12_000_000_000.0},
        node_metrics=[{"node_id": "n2", "node_binding_score": 99, "annual_load_mwh": 20.0, "peak_load_mw": 3.0, "RSI": 0.2, "GDI": 0.9, "SDI": 0.5}],
    )
    _write_run(
        results_root,
        experiment_id="E8_uncertainty_lhs",
        run_id="e8_valid_sample",
        status={
            "experiment_id": "E8_uncertainty_lhs",
            "run_id": "e8_valid_sample",
            "solver_status": "optimal",
            "objective_name": "min_cost",
            "objective_value": 6_000_000_000.0,
            "solver_objective_value": 6_000_000_000.0,
            "is_valid_result": True,
            "invalid_reason": "",
        },
        config={
            "experiment_id": "E8_uncertainty_lhs",
            "run_id": "e8_valid_sample",
            "technology_set": "base_with_storage_degradation",
            "grid_carbon_scenario": "baseline",
            "co2_cap_ratio": 0.25,
            "temporal_mode": "repday_opt",
            "objective": "min_cost",
            "lhs_sample_id": 7,
            "demand_scale": 1.05,
            "PV_capex": 3200.0,
            "BESS_degradation_cost": 0.08,
            "DR_fraction": 0.11,
        },
        qa_checks={
            "solver_success": True,
            "objective_name": "min_cost",
            "solver_objective_value": 6_000_000_000.0,
            "total_load_shedding_mwh": 0.0,
            "cap_violation_ton": 0.0,
            "load_shedding_tolerance_mwh": 1e-6,
            "co2_tolerance_ton": 1e-4,
            "cost_tolerance_yuan": 1e-4,
            "objective_cost_consistency_error": 0.0,
            "is_valid_result": True,
            "invalid_reason": "",
        },
        costs={"grid_purchase_cost": 2_000_000_000.0, "bess_power_cost": 4_000_000_000.0, "total_system_cost": 6_000_000_000.0},
        node_metrics=[{"node_id": "n4", "node_binding_score": 5, "annual_load_mwh": 12.0, "peak_load_mw": 2.2, "RSI": 0.8, "GDI": 0.2, "SDI": 1.0}],
    )
    _write_run(
        results_root,
        experiment_id="E8_uncertainty_lhs",
        run_id="e8_invalid_sample",
        status={
            "experiment_id": "E8_uncertainty_lhs",
            "run_id": "e8_invalid_sample",
            "solver_status": "optimal",
            "objective_name": "min_cost",
            "objective_value": 9_000_000_000.0,
            "solver_objective_value": 9_000_000_000.0,
            "is_valid_result": False,
            "invalid_reason": "load_shedding",
        },
        config={
            "experiment_id": "E8_uncertainty_lhs",
            "run_id": "e8_invalid_sample",
            "technology_set": "base_with_storage_degradation",
            "grid_carbon_scenario": "baseline",
            "co2_cap_ratio": 0.25,
            "temporal_mode": "repday_opt",
            "objective": "min_cost",
            "lhs_sample_id": 8,
            "demand_scale": 0.95,
            "PV_capex": 4100.0,
            "BESS_degradation_cost": 0.02,
            "DR_fraction": 0.03,
        },
        qa_checks={
            "solver_success": True,
            "objective_name": "min_cost",
            "solver_objective_value": 9_000_000_000.0,
            "total_load_shedding_mwh": 2.5,
            "cap_violation_ton": 0.0,
            "load_shedding_tolerance_mwh": 1e-6,
            "co2_tolerance_ton": 1e-4,
            "cost_tolerance_yuan": 1e-4,
            "objective_cost_consistency_error": 0.0,
            "is_valid_result": False,
            "invalid_reason": "load_shedding",
        },
        costs={"grid_purchase_cost": 3_000_000_000.0, "bess_power_cost": 6_000_000_000.0, "total_system_cost": 9_000_000_000.0},
        node_metrics=[{"node_id": "n5", "node_binding_score": 3, "annual_load_mwh": 14.0, "peak_load_mw": 2.8, "RSI": 0.4, "GDI": 0.5, "SDI": 0.6}],
    )
    _write_run(
        results_root,
        experiment_id="E9_dual_bottleneck",
        run_id="e9_valid",
        status={
            "experiment_id": "E9_dual_bottleneck",
            "run_id": "e9_valid",
            "solver_status": "optimal",
            "objective_name": "min_cost",
            "objective_value": 7_000_000_000.0,
            "solver_objective_value": 7_000_000_000.0,
            "is_valid_result": True,
            "invalid_reason": "",
        },
        config={
            "experiment_id": "E9_dual_bottleneck",
            "run_id": "e9_valid",
            "technology_set": "base_with_DR10_EV_medium_thermal10",
            "grid_carbon_scenario": "baseline",
            "co2_cap_ratio": 0.25,
            "temporal_mode": "repday_opt",
            "objective": "min_cost",
        },
        qa_checks={
            "solver_success": True,
            "objective_name": "min_cost",
            "solver_objective_value": 7_000_000_000.0,
            "total_load_shedding_mwh": 0.0,
            "cap_violation_ton": 0.0,
            "load_shedding_tolerance_mwh": 1e-6,
            "co2_tolerance_ton": 1e-4,
            "cost_tolerance_yuan": 1e-4,
            "objective_cost_consistency_error": 0.0,
            "is_valid_result": True,
            "invalid_reason": "",
        },
        costs={"grid_purchase_cost": 3_000_000_000.0, "bess_power_cost": 4_000_000_000.0, "total_system_cost": 7_000_000_000.0},
        node_metrics=[{"node_id": "n3", "node_binding_score": 7, "annual_load_mwh": 30.0, "peak_load_mw": 4.0, "RSI": 0.5, "GDI": 0.4, "SDI": 0.8}],
        duals=[{"constraint": "grid_interface:0:n3", "shadow_price": 1.2, "slack": 0.0, "is_binding": True}],
    )
    _write_run(
        results_root,
        experiment_id="E5_flexibility_portfolio",
        run_id="failed_v2g_semicolon_message",
        status={
            "experiment_id": "E5_flexibility_portfolio",
            "run_id": "failed_v2g_semicolon_message",
            "solver_status": "failed",
            "objective_name": "min_cost",
            "objective_value": None,
            "solver_objective_value": None,
            "is_valid_result": False,
            "invalid_reason": "solver_failed: The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is At lower/fixed bound)",
        },
        config={
            "experiment_id": "E5_flexibility_portfolio",
            "run_id": "failed_v2g_semicolon_message",
            "technology_set": "base_with_EV_medium_smart_V2G",
            "grid_carbon_scenario": "baseline",
            "co2_cap_ratio": 0.14,
            "temporal_mode": "repday_opt",
            "objective": "min_cost",
        },
        qa_checks={
            "solver_success": False,
            "objective_name": "min_cost",
            "solver_objective_value": None,
            "total_load_shedding_mwh": None,
            "cap_violation_ton": None,
            "load_shedding_tolerance_mwh": 1e-6,
            "co2_tolerance_ton": 1e-4,
            "cost_tolerance_yuan": 1e-4,
            "objective_cost_consistency_error": None,
            "is_valid_result": False,
            "invalid_reason": "solver_failed: The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is At lower/fixed bound)",
        },
        costs={"total_system_cost": 0.0},
        node_metrics=[],
    )

    summarize_results(results_root)

    with (results_root / "summaries" / "all_run_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    lowered = [name.lower() for name in header]
    assert len(lowered) == len(set(lowered))
    assert "BESS_degradation_cost_input" in header
    assert "DR_fraction_input" in header

    all_runs = pd.read_csv(results_root / "summaries" / "all_run_summary.csv")
    reclassified = all_runs.loc[all_runs["run_id"] == "valid_reclassified"].iloc[0]
    assert bool(reclassified["is_valid_result"]) is True
    assert str(reclassified["invalid_reason"]) in {"", "nan"}
    assert bool(reclassified["validity_changed"]) is True

    audit = pd.read_csv(results_root / "summaries" / "validity_audit.csv")
    assert set(audit["run_id"]) == {"invalid_load_shedding", "e8_invalid_sample", "failed_v2g_semicolon_message"}
    assert {
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
    } <= set(audit.columns)
    invalid_row = audit.loc[audit["run_id"] == "invalid_load_shedding"].iloc[0]
    assert invalid_row["status"] == "invalid"
    assert invalid_row["audit_category"] == "invalid_load_shedding"
    assert invalid_row["invalid_reason"] == "load_shedding"
    assert str(invalid_row["failed_reason"]) in {"", "nan"}
    assert bool(invalid_row["can_use_for_paper"]) is False
    assert invalid_row["paper_exclusion_reason"] == "invalid_load_shedding"
    failed_row = audit.loc[audit["run_id"] == "failed_v2g_semicolon_message"].iloc[0]
    assert failed_row["status"] == "failed"
    assert failed_row["audit_category"] == "unavailable_due_to_v2g_model_gap"
    assert str(failed_row["invalid_reason"]) in {"", "nan"}
    assert "primal_status is At lower/fixed bound" in str(failed_row["failed_reason"])
    assert failed_row["paper_exclusion_reason"] == "unavailable_due_to_v2g_model_gap"

    breakdown = pd.read_csv(results_root / "summaries" / "invalid_reason_breakdown.csv")
    assert "load_shedding" in set(breakdown["reason_token"])
    assert "primal_status is At lower/fixed bound)" not in set(breakdown["reason_token"])
    assert "unavailable_due_to_v2g_model_gap" in set(breakdown["reason_token"])

    e8_rank = pd.read_csv(results_root / "summaries" / "E8_rank_correlation.csv")
    assert bool(e8_rank.loc[0, "source_valid_runs_only"]) is True
    assert int(e8_rank.loc[0, "source_valid_run_count"]) == 1
    assert e8_rank.loc[0, "source_validity_scope"] == "valid_runs_only"
    assert int(e8_rank.loc[0, "source_total_runs"]) == 2
    assert int(e8_rank.loc[0, "source_invalid_or_failed_runs_excluded"]) == 1

    e8_band = pd.read_csv(results_root / "summaries" / "E8_uncertainty_band.csv")
    assert bool(e8_band.loc[0, "source_valid_runs_only"]) is True
    assert int(e8_band.loc[0, "source_valid_run_count"]) == 1
    assert e8_band.loc[0, "source_validity_scope"] == "valid_runs_only"
    assert int(e8_band.loc[0, "source_total_runs"]) == 2
    assert int(e8_band.loc[0, "source_invalid_or_failed_runs_excluded"]) == 1

    e8_samples = pd.read_csv(results_root / "summaries" / "E8_lhs_samples.csv")
    assert "BESS_degradation_cost" in e8_samples.columns
    assert "DR_fraction" in e8_samples.columns
    assert "BESS_degradation_cost_input" not in e8_samples.columns
    assert "DR_fraction_input" not in e8_samples.columns

    e10_node_metrics = pd.read_csv(results_root / "summaries" / "E10_node_metrics.csv")
    assert list(e10_node_metrics["run_id"].unique()) == ["valid_reclassified"]
    assert bool(e10_node_metrics.loc[0, "source_valid_runs_only"]) is True
    assert int(e10_node_metrics.loc[0, "source_valid_run_count"]) == 1
    assert e10_node_metrics.loc[0, "source_validity_scope"] == "valid_runs_only"
    assert int(e10_node_metrics.loc[0, "source_total_runs"]) == 2
    assert int(e10_node_metrics.loc[0, "source_invalid_or_failed_runs_excluded"]) == 1

    e9_binding_score = pd.read_csv(results_root / "summaries" / "E9_node_binding_score.csv")
    assert "node_id" in e9_binding_score.columns
    assert "constraint" not in e9_binding_score.columns
    assert int(e9_binding_score.loc[0, "node_binding_score"]) == 7
    assert bool(e9_binding_score.loc[0, "source_valid_runs_only"]) is True
    assert e9_binding_score.loc[0, "source_validity_scope"] == "valid_runs_only"

    e9_binding_constraints = pd.read_csv(results_root / "summaries" / "E9_binding_constraints_by_cap.csv")
    assert bool(e9_binding_constraints.loc[0, "source_valid_runs_only"]) is True
    assert int(e9_binding_constraints.loc[0, "source_valid_run_count"]) == 1
    assert e9_binding_constraints.loc[0, "source_validity_scope"] == "valid_runs_only"


def test_summarize_mac_respects_frontier_slices_and_filters_invalid_segments(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    runs = [
        ("lhs001_cap025", 1, 0.25, 100.0, 10_000_000_000.0),
        ("lhs001_cap014", 1, 0.14, 80.0, 12_000_000_000.0),
        ("lhs002_cap025", 2, 0.25, 110.0, 11_000_000_000.0),
        ("lhs002_cap014", 2, 0.14, 90.0, 9_000_000_000.0),
    ]
    for run_id, lhs_sample_id, cap_ratio, emissions_ton, total_cost in runs:
        _write_run(
            results_root,
            experiment_id="E8_uncertainty_lhs",
            run_id=run_id,
            status={
                "experiment_id": "E8_uncertainty_lhs",
                "run_id": run_id,
                "solver_status": "optimal",
                "objective_name": "min_cost",
                "objective_value": total_cost,
                "solver_objective_value": total_cost,
                "is_valid_result": True,
                "invalid_reason": "",
            },
            config={
                "experiment_id": "E8_uncertainty_lhs",
                "run_id": run_id,
                "technology_set": "base_with_storage_degradation",
                "grid_carbon_scenario": "baseline",
                "co2_cap_ratio": cap_ratio,
                "temporal_mode": "repday_opt",
                "objective": "min_cost",
                "lhs_sample_id": lhs_sample_id,
            },
            qa_checks={
                "solver_success": True,
                "objective_name": "min_cost",
                "solver_objective_value": total_cost,
                "total_load_shedding_mwh": 0.0,
                "cap_violation_ton": 0.0,
                "load_shedding_tolerance_mwh": 1e-6,
                "co2_tolerance_ton": 1e-4,
                "cost_tolerance_yuan": 1e-4,
                "objective_cost_consistency_error": 0.0,
                "is_valid_result": True,
                "invalid_reason": "",
            },
            costs={"grid_purchase_cost": total_cost, "total_system_cost": total_cost},
            node_metrics=[{"node_id": "n1", "node_binding_score": 0, "annual_load_mwh": 1.0, "peak_load_mw": 1.0, "RSI": 1.0, "GDI": 0.0, "SDI": 0.0}],
            actual_emissions_ton=emissions_ton,
        )

    summarize_results(results_root)

    mac = pd.read_csv(results_root / "summaries" / "all_mac.csv")
    diagnostics = pd.read_csv(results_root / "summaries" / "all_mac_diagnostics.csv")

    assert set(mac["lhs_sample_id"]) == {1}
    assert list(mac["from_run_id"]) == ["lhs001_cap025"]
    assert list(mac["to_run_id"]) == ["lhs001_cap014"]
    assert bool(mac.loc[0, "source_valid_runs_only"]) is True
    assert int(mac.loc[0, "source_valid_run_count"]) == 2
    assert mac.loc[0, "source_validity_scope"] == "valid_runs_only"
    assert set(diagnostics["lhs_sample_id"]) == {1, 2}
    invalid_mac = ~diagnostics["is_valid_mac"].fillna(False)
    assert int(invalid_mac.sum()) == 1
    assert set(diagnostics.loc[invalid_mac, "lhs_sample_id"]) == {2}


def test_summarize_excludes_size_guard_failures_from_e2_benchmark_evidence(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_run(
        results_root,
        experiment_id="E2_8760_validation",
        run_id="repday_valid",
        status={
            "experiment_id": "E2_8760_validation",
            "run_id": "repday_valid",
            "solver_status": "optimal",
            "objective_name": "min_cost",
            "objective_value": 7_000_000_000.0,
            "solver_objective_value": 7_000_000_000.0,
            "is_valid_result": True,
            "invalid_reason": "",
        },
        config={
            "experiment_id": "E2_8760_validation",
            "run_id": "repday_valid",
            "technology_set": "base_with_storage_degradation",
            "grid_carbon_scenario": "baseline",
            "co2_cap_ratio": 0.4,
            "temporal_mode": "repday_opt",
            "objective": "min_cost",
            "spatial_resolution": "k100",
        },
        qa_checks={
            "solver_success": True,
            "objective_name": "min_cost",
            "solver_objective_value": 7_000_000_000.0,
            "total_load_shedding_mwh": 0.0,
            "cap_violation_ton": 0.0,
            "load_shedding_tolerance_mwh": 1e-6,
            "co2_tolerance_ton": 1e-4,
            "cost_tolerance_yuan": 1e-4,
            "objective_cost_consistency_error": 0.0,
            "is_valid_result": True,
            "invalid_reason": "",
        },
        costs={"grid_purchase_cost": 3_000_000_000.0, "bess_power_cost": 4_000_000_000.0, "total_system_cost": 7_000_000_000.0},
        node_metrics=[{"node_id": "n1", "node_binding_score": 0, "annual_load_mwh": 10.0, "peak_load_mw": 2.0, "RSI": 0.7, "GDI": 0.3, "SDI": 1.1}],
    )
    _write_run(
        results_root,
        experiment_id="E2_8760_validation",
        run_id="full8760_failed_size_guard",
        status={
            "experiment_id": "E2_8760_validation",
            "run_id": "full8760_failed_size_guard",
            "solver_status": "failed",
            "objective_name": "min_cost",
            "objective_value": None,
            "solver_objective_value": None,
            "is_valid_result": False,
            "invalid_reason": "failed_full8760_opt: disabled by local size guard for SciPy/HiGHS sparse LP path",
        },
        config={
            "experiment_id": "E2_8760_validation",
            "run_id": "full8760_failed_size_guard",
            "technology_set": "base_with_storage_degradation",
            "grid_carbon_scenario": "baseline",
            "co2_cap_ratio": 0.4,
            "temporal_mode": "full8760_opt",
            "objective": "min_cost",
            "spatial_resolution": "k100",
        },
        qa_checks={
            "solver_success": False,
            "objective_name": "min_cost",
            "solver_objective_value": None,
            "total_load_shedding_mwh": None,
            "cap_violation_ton": None,
            "load_shedding_tolerance_mwh": 1e-6,
            "co2_tolerance_ton": 1e-4,
            "cost_tolerance_yuan": 1e-4,
            "objective_cost_consistency_error": None,
            "is_valid_result": False,
            "invalid_reason": "failed_full8760_opt: disabled by local size guard for SciPy/HiGHS sparse LP path",
        },
        costs={"total_system_cost": None},
        node_metrics=[],
    )
    _write_run(
        results_root,
        experiment_id="E2_8760_validation",
        run_id="dispatch_validation_failed_size_guard",
        status={
            "experiment_id": "E2_8760_validation",
            "run_id": "dispatch_validation_failed_size_guard",
            "solver_status": "failed",
            "objective_name": "min_cost",
            "objective_value": None,
            "solver_objective_value": None,
            "is_valid_result": False,
            "invalid_reason": "failed_full8760_dispatch_validation: disabled by local size guard for SciPy/HiGHS sparse LP path",
        },
        config={
            "experiment_id": "E2_8760_validation",
            "run_id": "dispatch_validation_failed_size_guard",
            "technology_set": "base_with_storage_degradation",
            "grid_carbon_scenario": "baseline",
            "co2_cap_ratio": 0.4,
            "temporal_mode": "full8760_dispatch_validation",
            "objective": "min_cost",
            "spatial_resolution": "k100",
        },
        qa_checks={
            "solver_success": False,
            "objective_name": "min_cost",
            "solver_objective_value": None,
            "total_load_shedding_mwh": None,
            "cap_violation_ton": None,
            "load_shedding_tolerance_mwh": 1e-6,
            "co2_tolerance_ton": 1e-4,
            "cost_tolerance_yuan": 1e-4,
            "objective_cost_consistency_error": None,
            "is_valid_result": False,
            "invalid_reason": "failed_full8760_dispatch_validation: disabled by local size guard for SciPy/HiGHS sparse LP path",
        },
        costs={"total_system_cost": None},
        node_metrics=[],
    )

    summarize_results(results_root)

    benchmark = pd.read_csv(results_root / "summaries" / "E2_full8760_benchmark.csv")
    repday_vs_8760 = pd.read_csv(results_root / "summaries" / "E2_repday_vs_8760.csv")
    audit = pd.read_csv(results_root / "summaries" / "validity_audit.csv")

    assert benchmark.empty
    assert repday_vs_8760.empty
    assert set(audit["run_id"]) == {"full8760_failed_size_guard", "dispatch_validation_failed_size_guard"}
    assert set(audit["status"]) == {"failed"}
    assert set(audit["paper_exclusion_reason"]) == {"unavailable_due_to_size_guard"}
    assert set(audit["has_size_guard"]) == {True}
    assert set(audit["is_full8760"]) == {True}
    assert set(audit["invalid_reason"].fillna("")) == {""}
    assert all("failed_full8760" in reason for reason in audit["failed_reason"].tolist())


def test_summarize_excludes_size_guard_failures_from_e7_spatial_validation_evidence(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_run(
        results_root,
        experiment_id="E7_spatial_aggregation",
        run_id="repday_valid",
        status={
            "experiment_id": "E7_spatial_aggregation",
            "run_id": "repday_valid",
            "solver_status": "optimal",
            "objective_name": "min_cost",
            "objective_value": 6_000_000_000.0,
            "solver_objective_value": 6_000_000_000.0,
            "is_valid_result": True,
            "invalid_reason": "",
        },
        config={
            "experiment_id": "E7_spatial_aggregation",
            "run_id": "repday_valid",
            "technology_set": "base_with_storage_degradation",
            "grid_carbon_scenario": "baseline",
            "co2_cap_ratio": 0.14,
            "temporal_mode": "repday_opt",
            "objective": "min_cost",
            "spatial_resolution": "k100",
        },
        qa_checks={
            "solver_success": True,
            "objective_name": "min_cost",
            "solver_objective_value": 6_000_000_000.0,
            "total_load_shedding_mwh": 0.0,
            "cap_violation_ton": 0.0,
            "load_shedding_tolerance_mwh": 1e-6,
            "co2_tolerance_ton": 1e-4,
            "cost_tolerance_yuan": 1e-4,
            "objective_cost_consistency_error": 0.0,
            "is_valid_result": True,
            "invalid_reason": "",
        },
        costs={"grid_purchase_cost": 2_000_000_000.0, "bess_power_cost": 4_000_000_000.0, "total_system_cost": 6_000_000_000.0},
        node_metrics=[{"node_id": "n1", "node_binding_score": 1, "annual_load_mwh": 10.0, "peak_load_mw": 2.0, "RSI": 0.7, "GDI": 0.3, "SDI": 1.1}],
    )
    _write_run(
        results_root,
        experiment_id="E7_spatial_aggregation",
        run_id="dispatch_validation_failed_size_guard",
        status={
            "experiment_id": "E7_spatial_aggregation",
            "run_id": "dispatch_validation_failed_size_guard",
            "solver_status": "failed",
            "objective_name": "min_cost",
            "objective_value": None,
            "solver_objective_value": None,
            "is_valid_result": False,
            "invalid_reason": "failed_full8760_dispatch_validation: disabled by local size guard for SciPy/HiGHS sparse LP path",
        },
        config={
            "experiment_id": "E7_spatial_aggregation",
            "run_id": "dispatch_validation_failed_size_guard",
            "technology_set": "base_with_storage_degradation",
            "grid_carbon_scenario": "baseline",
            "co2_cap_ratio": 0.14,
            "temporal_mode": "full8760_dispatch_validation",
            "objective": "min_cost",
            "spatial_resolution": "k100",
        },
        qa_checks={
            "solver_success": False,
            "objective_name": "min_cost",
            "solver_objective_value": None,
            "total_load_shedding_mwh": None,
            "cap_violation_ton": None,
            "load_shedding_tolerance_mwh": 1e-6,
            "co2_tolerance_ton": 1e-4,
            "cost_tolerance_yuan": 1e-4,
            "objective_cost_consistency_error": None,
            "is_valid_result": False,
            "invalid_reason": "failed_full8760_dispatch_validation: disabled by local size guard for SciPy/HiGHS sparse LP path",
        },
        costs={"total_system_cost": None},
        node_metrics=[],
    )

    summarize_results(results_root)

    spatial_validation = pd.read_csv(results_root / "summaries" / "E7_spatial_8760_validation.csv")
    audit = pd.read_csv(results_root / "summaries" / "validity_audit.csv")

    assert spatial_validation.empty
    assert set(audit["run_id"]) == {"dispatch_validation_failed_size_guard"}
    assert set(audit["status"]) == {"failed"}
    assert set(audit["paper_exclusion_reason"]) == {"unavailable_due_to_size_guard"}
    assert set(audit["is_full8760"]) == {True}


def _write_run(
    results_root: Path,
    *,
    experiment_id: str,
    run_id: str,
    status: dict,
    config: dict,
    qa_checks: dict,
    costs: dict[str, float | None],
    node_metrics: list[dict],
    duals: list[dict] | None = None,
    actual_emissions_ton: float = 70.0,
) -> None:
    run_dir = results_root / "runs" / experiment_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (run_dir / "config_resolved.yaml").write_text("\n".join(f"{key}: {value}" for key, value in config.items()) + "\n", encoding="utf-8")
    (run_dir / "qa_checks.json").write_text(json.dumps(qa_checks, indent=2), encoding="utf-8")

    pd.DataFrame([{"pv_mw": 1.0, "wind_mw": 2.0, "bess_power_mw": 3.0, "bess_energy_mwh": 12.0, "ldes_power_mw": 0.0, "ldes_energy_mwh": 0.0, "grid_expansion_mw": 0.0}]).to_csv(run_dir / "capacities.csv", index=False)
    pd.DataFrame([{"E_ref_grid_only_ton": 100.0, "E_lc_unconstrained_ton": 80.0, "E_min_techset_ton": 60.0, "co2_cap_ton": 75.0, "actual_emissions_ton": actual_emissions_ton, "reduction_vs_grid_only": 1 - actual_emissions_ton / 100.0, "reduction_vs_unconstrained_lc": 1 - actual_emissions_ton / 80.0, "cap_violation_ton": qa_checks.get("cap_violation_ton", 0.0)}]).to_csv(run_dir / "emissions.csv", index=False)
    pd.DataFrame([costs]).to_csv(run_dir / "costs.csv", index=False)
    pd.DataFrame([{"pv_curtailment_mwh": 1.0, "wind_curtailment_mwh": 2.0}]).to_csv(run_dir / "curtailment.csv", index=False)
    pd.DataFrame([{"bess_charge_mwh": 1.0, "bess_discharge_mwh": 1.0, "bess_throughput_mwh": 2.0, "bess_equivalent_full_cycles": 0.2, "bess_average_duration_h": 4.0, "bess_max_duration_h": 4.0, "ldes_charge_mwh": 0.0, "ldes_discharge_mwh": 0.0, "ldes_equivalent_full_cycles": 0.0}]).to_csv(run_dir / "storage_cycles.csv", index=False)
    pd.DataFrame([{"grid_import_mwh": 10.0, "peak_grid_import_mw": 5.0}]).to_csv(run_dir / "grid_import.csv", index=False)
    pd.DataFrame([{"DR_shifted_mwh": 0.0, "DR_peak_reduction_mw": 0.0, "thermal_shifted_mwh": 0.0, "thermal_peak_reduction_mw": 0.0, "EV_charge_mwh": 0.0, "EV_discharge_mwh": 0.0}]).to_csv(run_dir / "flexibility.csv", index=False)
    node_metric_defaults = {
        "node_id": "n0",
        "node_binding_score": 0.0,
        "annual_load_mwh": 0.0,
        "peak_load_mw": 0.0,
        "PV_potential_mw": 0.0,
        "wind_potential_mw": 0.0,
        "installed_PV_mw": 0.0,
        "installed_wind_mw": 0.0,
        "RSI": 0.0,
        "GDI": 0.0,
        "PSI": 0.0,
        "WSI": 0.0,
        "SDI": 0.0,
        "curtailment_rate": 0.0,
    }
    pd.DataFrame([{**node_metric_defaults, **item} for item in node_metrics]).to_csv(run_dir / "node_metrics.csv", index=False)
    pd.DataFrame(duals or []).to_csv(run_dir / "duals.csv", index=False)
