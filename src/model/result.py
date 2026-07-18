"""Convert LP solver vectors into run output tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

from paper05.analysis.metrics import reference_metrics
from paper05.model.objective import annualized_capex
from paper05.model.solve import LPBuild
from paper05.model.validity import evaluate_result_validity


@dataclass
class RunTables:
    capacities: pd.DataFrame
    dispatch: pd.DataFrame
    dispatch_artifact: str
    emissions: pd.DataFrame
    costs: pd.DataFrame
    curtailment: pd.DataFrame
    storage_cycles: pd.DataFrame
    grid_import: pd.DataFrame
    flexibility: pd.DataFrame
    duals: pd.DataFrame
    node_metrics: pd.DataFrame
    network_edges: pd.DataFrame
    qa_checks: dict[str, Any]


def extract_tables(
    build: LPBuild,
    res: OptimizeResult,
    inputs,
    runtime_seconds: float,
    reference_case: Mapping[str, float | None] | None = None,
) -> RunTables:
    reg = build.registry
    x = np.asarray(res.x, dtype=float)
    nodes = build.nodes
    N = len(nodes)
    idx = build.demand.index
    w = build.weights
    cfg = build.config

    def arr(name: str) -> np.ndarray:
        if not reg.has(name):
            return np.zeros((len(idx), N))
        return reg.values(x, name)

    cap = {
        "pv_mw": _sum_var(reg, x, "pv_cap") / 1000,
        "wind_mw": _sum_var(reg, x, "wind_cap") / 1000,
        "bess_power_mw": _sum_var(reg, x, "bess_power") / 1000,
        "bess_energy_mwh": _sum_var(reg, x, "bess_energy") / 1000,
        "ldes_power_mw": _sum_var(reg, x, "ldes_power") / 1000,
        "ldes_energy_mwh": _sum_var(reg, x, "ldes_energy") / 1000,
        "grid_expansion_mw": _sum_var(reg, x, "grid_expansion") / 1000,
        "total_grid_interface_mw": (_sum_var(reg, x, "grid_expansion") + _existing_grid_total(build, inputs)) / 1000,
    }
    capacities = pd.DataFrame([cap])

    reference_case = reference_case or {}
    dispatch = _dispatch_table(build, reg, x)
    grid = arr("grid_import")
    pv_gen = arr("pv_gen")
    wind_gen = arr("wind_gen")
    pv_curtail = arr("pv_curtail")
    wind_curtail = arr("wind_curtail")
    ls = arr("load_shedding")
    bess_ch = arr("bess_charge")
    bess_dis = arr("bess_discharge")
    ldes_dis = arr("ldes_discharge")

    annual_grid_kwh = _weighted_sum(grid, w)
    actual_emissions = float((pd.DataFrame(grid, index=idx).mul(build.co2, axis=0).mul(w, axis=0) / 1000).sum().sum())
    cap_ton = build.co2_cap_ton
    e_ref = build.e_ref_grid_only_ton
    e_lc = reference_case.get("E_lc_unconstrained_ton")
    e_min = reference_case.get("E_min_techset_ton")
    cap_violation = max(0.0, actual_emissions - cap_ton) if cap_ton is not None else 0.0
    emissions = pd.DataFrame(
        [
            reference_metrics(
                actual_emissions,
                e_ref,
                float(e_lc) if e_lc is not None else None,
                cap_ton,
                e_min_techset_ton=float(e_min) if e_min is not None else None,
                co2_cap_ratio=(cap_ton / e_ref) if cap_ton is not None and e_ref > 0 else None,
            )
        ]
    )

    costs = pd.DataFrame([_costs(build, inputs, reg, x, annual_grid_kwh, ls, bess_ch, bess_dis, ldes_dis)])
    total_ls_mwh = _weighted_sum(ls, w) / 1000
    total_cost = float(costs["total_system_cost"].iloc[0])
    component_sum = float(costs.drop(columns=["total_system_cost"]).iloc[0].sum())
    objective_name = str(cfg.get("objective", "min_cost"))
    solver_objective = float(getattr(res, "fun", np.nan)) if getattr(res, "fun", None) is not None else np.nan
    qa_checks: dict[str, Any] = evaluate_result_validity(
        solver_success=bool(getattr(res, "success", False)),
        objective_name=objective_name,
        total_load_shedding_mwh=total_ls_mwh,
        cap_violation_ton=float(cap_violation),
        total_cost=total_cost,
        cost_component_sum_error=abs(total_cost - component_sum),
        solver_objective_value=None if not np.isfinite(solver_objective) else solver_objective,
        ess_zero_capacity_zero_cost=_ess_zero_cost_check(capacities, costs),
        load_shedding_tolerance_mwh=float(cfg.get("load_shedding_tolerance_mwh", 1e-6)),
        co2_tolerance_ton=float(cfg.get("co2_tolerance_ton", 1e-4)),
        cost_tolerance_yuan=float(cfg.get("cost_tolerance_yuan", 1e-4)),
        cost_relative_tolerance=float(cfg.get("cost_relative_tolerance", 1e-10)),
    )
    qa_checks.update(
        {
        "solver_success": bool(getattr(res, "success", False)),
        "solver_message": str(getattr(res, "message", "")),
        "objective_name": objective_name,
        "solver_objective_value": None if not np.isfinite(solver_objective) else solver_objective,
        "cap_violation_ratio": float(emissions["cap_violation_ratio"].iloc[0]),
        }
    )

    curtailment = pd.DataFrame(
        [
            {
                "pv_curtailment_mwh": _weighted_sum(pv_curtail, w) / 1000,
                "wind_curtailment_mwh": _weighted_sum(wind_curtail, w) / 1000,
                "pv_generation_mwh": _weighted_sum(pv_gen, w) / 1000,
                "wind_generation_mwh": _weighted_sum(wind_gen, w) / 1000,
            }
        ]
    )
    storage_cycles = storage_cycle_table(build, reg, x)
    grid_import = pd.DataFrame(
        [
            {
                "grid_import_mwh": annual_grid_kwh / 1000,
                "peak_grid_import_mw": float(grid.sum(axis=1).max()) / 1000,
            }
        ]
    )
    flexibility = _flexibility_table(build, reg, x)
    duals = _dual_table(build, res)
    node_metrics = _node_metrics(build, inputs, reg, x, duals)
    network_edges = network_edges_table(build, inputs)
    dispatch_artifact = "dispatch_8760" if str(build.config.get("temporal_mode", "repday_opt")).startswith("full8760") else "dispatch_repday"
    return RunTables(capacities, dispatch, dispatch_artifact, emissions, costs, curtailment, storage_cycles, grid_import, flexibility, duals, node_metrics, network_edges, qa_checks)


def storage_cycle_table(build: LPBuild, reg, x: np.ndarray) -> pd.DataFrame:
    w = build.weights
    bess_ch = reg.values(x, "bess_charge") if reg.has("bess_charge") else np.zeros((len(w), len(build.nodes)))
    bess_dis = reg.values(x, "bess_discharge") if reg.has("bess_discharge") else np.zeros((len(w), len(build.nodes)))
    bess_e = reg.values(x, "bess_energy") if reg.has("bess_energy") else np.zeros(len(build.nodes))
    ldes_ch = reg.values(x, "ldes_charge") if reg.has("ldes_charge") else np.zeros_like(bess_ch)
    ldes_dis = reg.values(x, "ldes_discharge") if reg.has("ldes_discharge") else np.zeros_like(bess_dis)
    ldes_e = reg.values(x, "ldes_energy") if reg.has("ldes_energy") else np.zeros(len(build.nodes))

    bess_dis_kwh = _weighted_sum(bess_dis, w)
    bess_ch_kwh = _weighted_sum(bess_ch, w)
    ldes_dis_kwh = _weighted_sum(ldes_dis, w)
    ldes_ch_kwh = _weighted_sum(ldes_ch, w)
    bess_e_kwh = float(np.sum(bess_e))
    ldes_e_kwh = float(np.sum(ldes_e))
    dod = float(build.config.get("bess_dod_max", 0.8))
    return pd.DataFrame(
        [
            {
                "bess_charge_mwh": bess_ch_kwh / 1000,
                "bess_discharge_mwh": bess_dis_kwh / 1000,
                "bess_throughput_mwh": (bess_ch_kwh + bess_dis_kwh) / 1000,
                "bess_equivalent_full_cycles": safe_efc(bess_dis_kwh, bess_e_kwh, dod),
                "bess_cycling_degradation_cost_yuan": (bess_ch_kwh + bess_dis_kwh) * _bess_deg_unit(build.tech_config),
                "bess_calendar_ageing_cost_yuan": bess_e_kwh * float(build.tech_config.get("bess_calendar_ageing_cost_yuan_per_kwh_year", 0.0)),
                "bess_average_duration_h": bess_e_kwh / float(np.sum(reg.values(x, "bess_power"))) if reg.has("bess_power") and np.sum(reg.values(x, "bess_power")) > 0 else 0.0,
                "bess_max_duration_h": float(np.max(np.divide(bess_e, reg.values(x, "bess_power"), out=np.zeros_like(bess_e), where=reg.values(x, "bess_power") > 1e-9))) if reg.has("bess_power") else 0.0,
                "ldes_charge_mwh": ldes_ch_kwh / 1000,
                "ldes_discharge_mwh": ldes_dis_kwh / 1000,
                "ldes_equivalent_full_cycles": safe_efc(ldes_dis_kwh, ldes_e_kwh, 1.0),
            }
        ]
    )


def safe_efc(discharge_kwh: float, energy_kwh: float, dod: float) -> float:
    denom = energy_kwh * dod
    if denom <= 1e-12:
        return 0.0
    return float(discharge_kwh / denom)


def write_tables(run_dir: Path, tables: RunTables) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ["capacities", "emissions", "costs", "curtailment", "storage_cycles", "grid_import", "flexibility", "duals", "node_metrics", "network_edges"]:
        getattr(tables, name).to_csv(run_dir / f"{name}.csv", index=False)
    if tables.dispatch_artifact == "dispatch_8760" and len(tables.dispatch) > 300_000:
        tables.dispatch.to_parquet(run_dir / "dispatch_8760.parquet", index=False)
    else:
        tables.dispatch.to_csv(run_dir / f"{tables.dispatch_artifact}.csv", index=False)
    (run_dir / "qa_checks.json").write_text(json.dumps(tables.qa_checks, indent=2), encoding="utf-8")


def _costs(build: LPBuild, inputs, reg, x: np.ndarray, grid_kwh: float, ls: np.ndarray, bess_ch: np.ndarray, bess_dis: np.ndarray, ldes_dis: np.ndarray) -> dict[str, float]:
    d = inputs.discount_rate
    pv_cap = _sum_var(reg, x, "pv_cap")
    wind_cap = _sum_var(reg, x, "wind_cap")
    bess_p = _sum_var(reg, x, "bess_power")
    bess_e = _sum_var(reg, x, "bess_energy")
    ldes_p = _sum_var(reg, x, "ldes_power")
    ldes_e = _sum_var(reg, x, "ldes_energy")
    grid_exp_vec = reg.values(x, "grid_expansion") if reg.has("grid_expansion") else np.zeros(len(build.nodes))
    if build.config.get("aggregate_nodes", False) or build.config.get("network_mode") == "copperplate":
        grid_cost_vec = np.asarray([float(inputs.grid_capacity["grid_connection_cost_yuan_per_kW"].mean())], dtype=float)
    else:
        grid_cost_vec = (
            inputs.grid_capacity.reindex(build.nodes)["grid_connection_cost_yuan_per_kW"].fillna(800.0).to_numpy(dtype=float)
        )
    cfg = build.tech_config
    bess_cycling_degradation = (_weighted_sum(bess_ch, build.weights) + _weighted_sum(bess_dis, build.weights)) * _bess_deg_unit(cfg)
    bess_calendar_ageing = bess_e * float(cfg.get("bess_calendar_ageing_cost_yuan_per_kwh_year", 0.0))
    costs = {
        "pv_cost": pv_cap * inputs.pv_cost["capex_yuan_per_kW"] * (annualized_capex(1.0, d, inputs.pv_cost["lifetime_years"]) + inputs.pv_cost.get("om_fix_fraction", 0.0)),
        "wind_cost": wind_cap * (inputs.wind_cost["capex_yuan_per_kW"] * annualized_capex(1.0, d, inputs.wind_cost["lifetime_years"]) + inputs.wind_cost.get("om_fix_yuan_per_kW_per_year", 0.0)),
        "bess_power_cost": bess_p * inputs.bess_cost["capex_power_yuan_per_kW"] * (annualized_capex(1.0, d, inputs.bess_cost["lifetime_years"]) + inputs.bess_cost.get("om_fix_fraction", 0.0)),
        "bess_energy_cost": bess_e * inputs.bess_cost["capex_energy_yuan_per_kWh"] * (annualized_capex(1.0, d, inputs.bess_cost["lifetime_years"]) + inputs.bess_cost.get("om_fix_fraction", 0.0)),
        "bess_degradation_cost": bess_cycling_degradation + bess_calendar_ageing,
        "bess_replacement_proxy_cost": 0.0,
        "ldes_power_cost": ldes_p * float(cfg.get("ldes_power_capex", 1600.0)) * annualized_capex(1.0, d, float(cfg.get("ldes_lifetime", 25.0))),
        "ldes_energy_cost": ldes_e * float(cfg.get("ldes_energy_capex", 240.0)) * annualized_capex(1.0, d, float(cfg.get("ldes_lifetime", 25.0))),
        "grid_purchase_cost": float((pd.DataFrame(reg.values(x, "grid_import"), index=build.demand.index).mul(build.price, axis=0).mul(build.weights, axis=0)).sum().sum()),
        "grid_expansion_cost": float(np.dot(grid_exp_vec, grid_cost_vec)) * annualized_capex(1.0, d, inputs.planning_horizon_years),
        "demand_response_cost": _shift_cost(build, reg, x, "dr", float(cfg.get("demand_response_cost", 0.02))),
        "ev_flexibility_cost": _ev_cost(build, reg, x, float(cfg.get("ev_flexibility_cost", 0.01))),
        "thermal_flexibility_cost": _shift_cost(build, reg, x, "thermal", float(cfg.get("thermal_flexibility_cost", 0.015))),
        "load_shedding_penalty": _weighted_sum(ls, build.weights) * float(cfg.get("ls_penalty", 1e5)),
    }
    costs["total_system_cost"] = float(sum(costs.values()))
    return {k: float(v) for k, v in costs.items()}


def _shift_cost(build: LPBuild, reg, x: np.ndarray, prefix: str, unit: float) -> float:
    if not reg.has(f"{prefix}_pos"):
        return 0.0
    return (_weighted_sum(reg.values(x, f"{prefix}_pos"), build.weights) + _weighted_sum(reg.values(x, f"{prefix}_neg"), build.weights)) * unit


def _ev_cost(build: LPBuild, reg, x: np.ndarray, unit: float) -> float:
    if not reg.has("ev_charge"):
        return 0.0
    return (_weighted_sum(reg.values(x, "ev_charge"), build.weights) + _weighted_sum(reg.values(x, "ev_discharge"), build.weights)) * unit


def _flexibility_table(build: LPBuild, reg, x: np.ndarray) -> pd.DataFrame:
    def shifted(prefix: str) -> float:
        if not reg.has(f"{prefix}_pos"):
            return 0.0
        return min(_weighted_sum(reg.values(x, f"{prefix}_pos"), build.weights), _weighted_sum(reg.values(x, f"{prefix}_neg"), build.weights)) / 1000

    return pd.DataFrame(
        [
            {
                "DR_shifted_mwh": shifted("dr"),
                "DR_peak_reduction_mw": _peak_reduction(build, reg, x, "dr") / 1000,
                "thermal_shifted_mwh": shifted("thermal"),
                "thermal_peak_reduction_mw": _peak_reduction(build, reg, x, "thermal") / 1000,
                "EV_charge_mwh": _weighted_sum(reg.values(x, "ev_charge"), build.weights) / 1000 if reg.has("ev_charge") else 0.0,
                "EV_discharge_mwh": _weighted_sum(reg.values(x, "ev_discharge"), build.weights) / 1000 if reg.has("ev_discharge") else 0.0,
            }
        ]
    )


def _peak_reduction(build: LPBuild, reg, x: np.ndarray, prefix: str) -> float:
    if not reg.has(f"{prefix}_pos"):
        return 0.0
    net = reg.values(x, f"{prefix}_pos") - reg.values(x, f"{prefix}_neg")
    return float(np.maximum(0.0, -net.sum(axis=1)).max())


def _dual_table(build: LPBuild, res: object) -> pd.DataFrame:
    rows = []
    ineqlin = getattr(res, "ineqlin", None)
    marg = getattr(ineqlin, "marginals", None) if ineqlin is not None else None
    residual = getattr(ineqlin, "residual", None) if ineqlin is not None else None
    if marg is None:
        return pd.DataFrame(columns=["constraint", "shadow_price", "slack", "is_binding"])
    for i, name in enumerate(build.ub_labels):
        slack = float(residual[i]) if residual is not None else np.nan
        rows.append(
            {
                "constraint": name,
                "shadow_price": float(marg[i]),
                "slack": slack,
                "is_binding": bool(abs(slack) <= 1e-6) if np.isfinite(slack) else False,
            }
        )
    return pd.DataFrame(rows)


def network_edges_table(build: LPBuild, inputs) -> pd.DataFrame:
    """Return the spatial-transfer proxy edge list used by the LP build."""

    columns = [
        "edge_id",
        "src_node",
        "dst_node",
        "src_centroid_x",
        "src_centroid_y",
        "dst_centroid_x",
        "dst_centroid_y",
        "distance",
        "network_k_nearest",
        "transfer_cap_mw",
        "edge_construction_rule",
    ]
    if not build.edges:
        return pd.DataFrame(columns=columns)

    coords = _node_coordinates(build, inputs)
    rows = []
    for edge_id, (src, dst) in enumerate(build.edges):
        src_node = build.nodes[src]
        dst_node = build.nodes[dst]
        src_xy = coords[src_node]
        dst_xy = coords[dst_node]
        rows.append(
            {
                "edge_id": edge_id,
                "src_node": src_node,
                "dst_node": dst_node,
                "src_centroid_x": src_xy[0],
                "src_centroid_y": src_xy[1],
                "dst_centroid_x": dst_xy[0],
                "dst_centroid_y": dst_xy[1],
                "distance": float(np.linalg.norm(np.asarray(src_xy) - np.asarray(dst_xy))),
                "network_k_nearest": int(build.config.get("network_k_nearest", 3)),
                "transfer_cap_mw": float(build.tech_config.get("transfer_cap_kw", 0.0)) / 1000.0,
                "edge_construction_rule": "undirected k-nearest-neighbour links from node centroids with uniform transfer-capacity bound",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _node_coordinates(build: LPBuild, inputs) -> dict[str, tuple[float, float]]:
    fallback = {node: (float(i), 0.0) for i, node in enumerate(build.nodes)}
    summary = getattr(inputs, "node_summary", None)
    if summary is None or summary.empty or "node_id" not in summary.columns:
        return fallback
    df = summary.set_index("node_id").reindex(build.nodes)
    if {"centroid_x", "centroid_y"}.issubset(df.columns):
        return {
            node: (
                float(pd.to_numeric(df.loc[node, "centroid_x"], errors="coerce")) if pd.notna(df.loc[node, "centroid_x"]) else fallback[node][0],
                float(pd.to_numeric(df.loc[node, "centroid_y"], errors="coerce")) if pd.notna(df.loc[node, "centroid_y"]) else fallback[node][1],
            )
            for node in build.nodes
        }
    return fallback


def _node_metrics(build: LPBuild, inputs, reg, x: np.ndarray, duals: pd.DataFrame) -> pd.DataFrame:
    nodes = build.nodes
    demand = build.demand
    w = build.weights
    pv_cap = reg.values(x, "pv_cap")
    wind_cap = reg.values(x, "wind_cap")
    bess_power = reg.values(x, "bess_power") if reg.has("bess_power") else np.zeros(len(nodes))
    bess_e = reg.values(x, "bess_energy") if reg.has("bess_energy") else np.zeros(len(nodes))
    ldes_power = reg.values(x, "ldes_power") if reg.has("ldes_power") else np.zeros(len(nodes))
    ldes_e = reg.values(x, "ldes_energy") if reg.has("ldes_energy") else np.zeros(len(nodes))
    grid_expansion = reg.values(x, "grid_expansion") if reg.has("grid_expansion") else np.zeros(len(nodes))
    existing_grid = inputs.grid_capacity.reindex(nodes).get("P_grid_existing_kW", pd.Series(0.0, index=nodes)).fillna(0.0).to_numpy(dtype=float)
    grid = reg.values(x, "grid_import")
    pv_gen = reg.values(x, "pv_gen")
    wind_gen = reg.values(x, "wind_gen")
    curtail = reg.values(x, "pv_curtail") + reg.values(x, "wind_curtail")
    rows = []
    for i, node in enumerate(nodes):
        annual_load = float((demand[node] * w).sum()) / 1000
        available = float(((pv_gen[:, i] + wind_gen[:, i] + curtail[:, i]) * w.to_numpy()).sum()) / 1000
        import_mwh = float((grid[:, i] * w.to_numpy()).sum()) / 1000
        node_duals = duals[duals["constraint"].astype(str).str.endswith(f":{node}")]
        rows.append(
            {
                "node_id": node,
                "annual_load_mwh": annual_load,
                "peak_load_mw": float(demand[node].max()) / 1000,
                "PV_potential_mw": float(inputs.pv_cap_max_kw.reindex([node]).fillna(0).iloc[0]) / 1000,
                "wind_potential_mw": float(inputs.wind_cap_max_kw.reindex([node]).fillna(0).iloc[0]) / 1000,
                "installed_PV_mw": float(pv_cap[i]) / 1000,
                "installed_wind_mw": float(wind_cap[i]) / 1000,
                "installed_bess_power_mw": float(bess_power[i]) / 1000,
                "installed_bess_energy_mwh": float(bess_e[i]) / 1000,
                "installed_ldes_power_mw": float(ldes_power[i]) / 1000,
                "installed_ldes_energy_mwh": float(ldes_e[i]) / 1000,
                "installed_grid_expansion_mw": float(grid_expansion[i]) / 1000,
                "existing_grid_interface_mw": float(existing_grid[i]) / 1000,
                "total_grid_interface_mw": float(existing_grid[i] + grid_expansion[i]) / 1000,
                "RSI": available / annual_load if annual_load > 0 else 0.0,
                "GDI": import_mwh / annual_load if annual_load > 0 else 0.0,
                "PSI": float(pv_cap[i]) / float(inputs.pv_cap_max_kw.reindex([node]).fillna(0).iloc[0]) if float(inputs.pv_cap_max_kw.reindex([node]).fillna(0).iloc[0]) > 0 else 0.0,
                "WSI": float(wind_cap[i]) / float(inputs.wind_cap_max_kw.reindex([node]).fillna(0).iloc[0]) if float(inputs.wind_cap_max_kw.reindex([node]).fillna(0).iloc[0]) > 0 else 0.0,
                "SDI": float(bess_e[i]) / float(demand[node].max()) if float(demand[node].max()) > 0 else 0.0,
                "curtailment_rate": float((curtail[:, i] * w.to_numpy()).sum()) / (available * 1000) if available > 0 else 0.0,
                "node_binding_score": int(node_duals["is_binding"].sum()) if not node_duals.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _dispatch_table(build: LPBuild, reg, x: np.ndarray) -> pd.DataFrame:
    index = pd.MultiIndex.from_product([build.demand.index, build.nodes], names=["timestamp", "node_id"])
    out = build.demand.stack().rename("load_kwh").to_frame().reindex(index).fillna(0.0)
    out["weight"] = index.get_level_values("timestamp").map(build.weights).to_numpy(dtype=float)
    out["co2_kg_per_kwh"] = index.get_level_values("timestamp").map(build.co2).to_numpy(dtype=float)

    for column, var_name in [
        ("pv_kwh", "pv_gen"),
        ("wind_kwh", "wind_gen"),
        ("grid_import_kwh", "grid_import"),
        ("ess_charge_kwh", "bess_charge"),
        ("ess_discharge_kwh", "bess_discharge"),
        ("ldes_charge_kwh", "ldes_charge"),
        ("ldes_discharge_kwh", "ldes_discharge"),
        ("load_shedding_kwh", "load_shedding"),
        ("pv_curtailment_kwh", "pv_curtail"),
        ("wind_curtailment_kwh", "wind_curtail"),
        ("dr_pos_kwh", "dr_pos"),
        ("dr_neg_kwh", "dr_neg"),
        ("thermal_pos_kwh", "thermal_pos"),
        ("thermal_neg_kwh", "thermal_neg"),
        ("ev_charge_kwh", "ev_charge"),
        ("ev_discharge_kwh", "ev_discharge"),
    ]:
        out[column] = _dispatch_series(build, reg, x, var_name).reindex(index).fillna(0.0).to_numpy(dtype=float)

    transfer_in, transfer_out = _transfer_tables(build, reg, x)
    out["transfer_in_kwh"] = transfer_in.reindex(index).fillna(0.0).to_numpy(dtype=float)
    out["transfer_out_kwh"] = transfer_out.reindex(index).fillna(0.0).to_numpy(dtype=float)
    out["net_transfer_kwh"] = out["transfer_in_kwh"] - out["transfer_out_kwh"]
    out["curtailment_kwh"] = out["pv_curtailment_kwh"] + out["wind_curtailment_kwh"]

    aliases = {
        "pv_gen": "pv_kwh",
        "wind_gen": "wind_kwh",
        "grid_import": "grid_import_kwh",
        "bess_charge": "ess_charge_kwh",
        "bess_discharge": "ess_discharge_kwh",
        "ldes_charge": "ldes_charge_kwh",
        "ldes_discharge": "ldes_discharge_kwh",
        "load_shedding": "load_shedding_kwh",
        "pv_curtail": "pv_curtailment_kwh",
        "wind_curtail": "wind_curtailment_kwh",
        "ev_charge": "ev_charge_kwh",
        "ev_discharge": "ev_discharge_kwh",
    }
    for alias, source in aliases.items():
        out[alias] = out[source]
    return out.reset_index()


def _weighted_sum(arr: np.ndarray, weights: pd.Series) -> float:
    arr = np.asarray(arr, dtype=float)
    weight_arr = weights.to_numpy(dtype=float)
    if arr.ndim == 1:
        return float((arr * weight_arr).sum())
    return float((arr * weight_arr[:, None]).sum())


def _sum_var(reg, x: np.ndarray, name: str) -> float:
    return float(np.sum(reg.values(x, name))) if reg.has(name) else 0.0


def _existing_grid_total(build: LPBuild, inputs) -> float:
    if build.config.get("aggregate_nodes", False) or build.config.get("network_mode") == "copperplate":
        return float(inputs.grid_capacity["P_grid_existing_kW"].sum())
    return float(inputs.grid_capacity.reindex(build.nodes)["P_grid_existing_kW"].fillna(0.0).sum())


def _bess_deg_unit(config: dict) -> float:
    return float(config.get("bess_deg_cost", config.get("bess_degradation_cost_yuan_per_kwh", 0.0)))


def _ess_zero_cost_check(capacities: pd.DataFrame, costs: pd.DataFrame) -> bool:
    if capacities["bess_power_mw"].iloc[0] <= 1e-9 and capacities["bess_energy_mwh"].iloc[0] <= 1e-9:
        return float(costs[["bess_power_cost", "bess_energy_cost", "bess_degradation_cost", "bess_replacement_proxy_cost"]].iloc[0].sum()) <= 1e-8
    return True
def _dispatch_series(build: LPBuild, reg, x: np.ndarray, name: str) -> pd.Series:
    index = pd.MultiIndex.from_product([build.demand.index, build.nodes], names=["timestamp", "node_id"])
    if not reg.has(name):
        return pd.Series(0.0, index=index, dtype=float)
    data = pd.DataFrame(reg.values(x, name), index=build.demand.index, columns=build.nodes).stack()
    return data.reindex(index).fillna(0.0)


def _transfer_tables(build: LPBuild, reg, x: np.ndarray) -> tuple[pd.Series, pd.Series]:
    index = pd.MultiIndex.from_product([build.demand.index, build.nodes], names=["timestamp", "node_id"])
    if not reg.has("flow") or not build.edges:
        empty = pd.Series(0.0, index=index, dtype=float)
        return empty, empty.copy()

    flows = reg.values(x, "flow")
    transfer_in = np.zeros((len(build.demand.index), len(build.nodes)), dtype=float)
    transfer_out = np.zeros_like(transfer_in)
    for edge_id, (src, dst) in enumerate(build.edges):
        edge_flow = flows[:, edge_id]
        forward = np.maximum(edge_flow, 0.0)
        reverse = np.maximum(-edge_flow, 0.0)
        transfer_out[:, src] += forward
        transfer_in[:, dst] += forward
        transfer_out[:, dst] += reverse
        transfer_in[:, src] += reverse

    in_series = pd.DataFrame(transfer_in, index=build.demand.index, columns=build.nodes).stack().reindex(index).fillna(0.0)
    out_series = pd.DataFrame(transfer_out, index=build.demand.index, columns=build.nodes).stack().reindex(index).fillna(0.0)
    return in_series, out_series
