"""Sparse SciPy/HiGHS LP model for Paper05 experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, linprog
from scipy.sparse import coo_matrix

from paper05.analysis.references import grid_only_emissions_ton
from paper05.data.schema import ModelInputs
from paper05.model.constraints import label
from paper05.model.index import VarRegistry
from paper05.model.objective import annualized_capex
from paper05.model.technology import technology_config
from paper05.temporal.chronological import select_time


@dataclass
class LPBuild:
    registry: VarRegistry
    c: np.ndarray
    A_ub: coo_matrix
    b_ub: np.ndarray
    A_eq: coo_matrix
    b_eq: np.ndarray
    ub_labels: list[str]
    eq_labels: list[str]
    demand: pd.DataFrame
    weights: pd.Series
    price: pd.Series
    co2: pd.Series
    nodes: list[str]
    edges: list[tuple[int, int]]
    config: dict
    tech_config: dict
    e_ref_grid_only_ton: float
    co2_cap_ton: float | None


def solve_model(inputs: ModelInputs, config: dict) -> tuple[LPBuild, OptimizeResult, float]:
    build = build_lp(inputs, config)
    start = time.perf_counter()
    res = linprog(
        build.c,
        A_ub=build.A_ub,
        b_ub=build.b_ub,
        A_eq=build.A_eq,
        b_eq=build.b_eq,
        bounds=build.registry.bounds,
        method="highs",
    )
    runtime = time.perf_counter() - start
    return build, res, runtime


def build_lp(inputs: ModelInputs, config: dict) -> LPBuild:
    idx, weights, _ = select_time(inputs.demand_kw, config)
    demand = inputs.demand_kw.reindex(idx).ffill().bfill().fillna(0.0)
    cfg = technology_config(config)
    aggregate_nodes = bool(config.get("aggregate_nodes", False)) or cfg["network_mode"] == "copperplate"
    if aggregate_nodes:
        demand = pd.DataFrame({"system": demand.sum(axis=1)}, index=demand.index)

    nodes = list(demand.columns)
    T, N = demand.shape
    weights = weights.reindex(demand.index).astype(float)
    price = inputs.price_yuan_per_kwh.reindex(demand.index).ffill().bfill().astype(float)
    co2 = inputs.grid_co2_kg_per_kwh.reindex(demand.index).ffill().bfill().astype(float)

    demand_np = demand.to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    price_np = price.to_numpy(dtype=float)
    co2_np = co2.to_numpy(dtype=float)

    if aggregate_nodes:
        pv_max = np.array([float(inputs.pv_cap_max_kw.sum())])
        wind_max = np.array([float(inputs.wind_cap_max_kw.sum())])
        existing_grid = np.array([float(inputs.grid_capacity["P_grid_existing_kW"].sum())])
        max_grid = np.array([float(inputs.grid_capacity.get("P_grid_max_kW", inputs.grid_capacity["P_grid_existing_kW"] * 2).sum())])
        grid_cost = np.array([float(inputs.grid_capacity["grid_connection_cost_yuan_per_kW"].mean())])
    else:
        pv_max = inputs.pv_cap_max_kw.reindex(nodes).fillna(0.0).to_numpy(dtype=float)
        wind_max = inputs.wind_cap_max_kw.reindex(nodes).fillna(0.0).to_numpy(dtype=float)
        gc = inputs.grid_capacity.reindex(nodes)
        existing_grid = gc["P_grid_existing_kW"].fillna(demand.max(axis=0) * 1.2).to_numpy(dtype=float)
        max_grid = gc.get("P_grid_max_kW", gc["P_grid_existing_kW"] * 2).fillna(demand.max(axis=0) * 2.2).to_numpy(dtype=float)
        grid_cost = gc["grid_connection_cost_yuan_per_kW"].fillna(800.0).to_numpy(dtype=float)

    reg = VarRegistry()
    reg.add("pv_cap", (N,), 0.0, pv_max if cfg["pv"] else np.zeros(N))
    reg.add("wind_cap", (N,), 0.0, wind_max if cfg["wind"] else np.zeros(N))
    reg.add("bess_power", (N,), 0.0, None if cfg["bess"] else np.zeros(N))
    reg.add("bess_energy", (N,), 0.0, None if cfg["bess"] else np.zeros(N))
    reg.add("grid_expansion", (N,), 0.0, np.maximum(max_grid - existing_grid, 0.0))
    reg.add("ldes_power", (N,), 0.0, None if cfg["ldes"] else np.zeros(N))
    reg.add("ldes_energy", (N,), 0.0, None if cfg["ldes"] else np.zeros(N))

    for name in ["pv_gen", "wind_gen", "bess_charge", "bess_discharge", "bess_soc", "grid_import", "load_shedding", "pv_curtail", "wind_curtail"]:
        reg.add(name, (T, N), 0.0, None)
    if cfg["ldes"]:
        for name in ["ldes_charge", "ldes_discharge", "ldes_soc"]:
            reg.add(name, (T, N), 0.0, None)
    if cfg["dr_fraction"] > 0:
        ub = (cfg["dr_fraction"] * demand_np).reshape(-1)
        reg.add("dr_pos", (T, N), 0.0, ub)
        reg.add("dr_neg", (T, N), 0.0, ub)
    if cfg["thermal_fraction"] > 0:
        ub = (cfg["thermal_fraction"] * demand_np).reshape(-1)
        reg.add("thermal_pos", (T, N), 0.0, ub)
        reg.add("thermal_neg", (T, N), 0.0, ub)
    if cfg["ev_mode"] != "disabled":
        ev_bounds = _ev_power_bounds(demand, cfg)
        reg.add("ev_charge", (T, N), 0.0, ev_bounds["charge"])
        reg.add("ev_discharge", (T, N), 0.0, ev_bounds["discharge"])
        reg.add("ev_soc", (T, N), 0.0, ev_bounds["soc"])

    fixed = config.get("fixed_capacities_kw") or {}
    for key, var_name in [
        ("pv_kw", "pv_cap"),
        ("wind_kw", "wind_cap"),
        ("bess_power_kw", "bess_power"),
        ("bess_energy_kwh", "bess_energy"),
        ("ldes_power_kw", "ldes_power"),
        ("ldes_energy_kwh", "ldes_energy"),
        ("grid_expansion_kw", "grid_expansion"),
    ]:
        if key in fixed:
            reg.set_bounds(var_name, _fixed_vector(fixed[key], nodes, N))

    edges = _network_edges(inputs, nodes, config) if cfg["network_mode"] == "spatial_transfer_proxy" and N > 1 else []
    if edges:
        reg.add("flow", (T, len(edges)), -cfg["transfer_cap_kw"], cfg["transfer_cap_kw"])

    c = np.zeros(reg.size)
    _fill_objective(c, reg, inputs, config, cfg, N, T, w, price_np, co2_np, grid_cost)

    eq_rows: list[tuple[list[int], list[float]]] = []
    eq_rhs: list[float] = []
    eq_labels: list[str] = []
    ub_rows: list[tuple[list[int], list[float]]] = []
    ub_rhs: list[float] = []
    ub_labels: list[str] = []

    def add_eq(cols: list[int], vals: list[float], rhs: float, row_label: str) -> None:
        eq_rows.append((cols, vals))
        eq_rhs.append(float(rhs))
        eq_labels.append(row_label)

    def add_ub(cols: list[int], vals: list[float], rhs: float, row_label: str) -> None:
        ub_rows.append((cols, vals))
        ub_rhs.append(float(rhs))
        ub_labels.append(row_label)

    # Power balance: grid + PV + wind + storage discharge + load shedding + inflow
    # = demand + storage charge + outflow + flexible positive shifts.
    edge_incidence = _edge_incidence(edges, nodes)
    for t in range(T):
        for n in range(N):
            cols = [
                reg.idx("pv_gen", t, n),
                reg.idx("wind_gen", t, n),
                reg.idx("bess_discharge", t, n),
                reg.idx("grid_import", t, n),
                reg.idx("load_shedding", t, n),
                reg.idx("bess_charge", t, n),
            ]
            vals = [1.0, 1.0, 1.0, 1.0, 1.0, -1.0]
            if reg.has("ldes_discharge"):
                cols += [reg.idx("ldes_discharge", t, n), reg.idx("ldes_charge", t, n)]
                vals += [1.0, -1.0]
            for pos, neg in [("dr_pos", "dr_neg"), ("thermal_pos", "thermal_neg")]:
                if reg.has(pos):
                    cols += [reg.idx(pos, t, n), reg.idx(neg, t, n)]
                    vals += [-1.0, 1.0]
            if reg.has("ev_charge"):
                cols += [reg.idx("ev_discharge", t, n), reg.idx("ev_charge", t, n)]
                vals += [1.0, -1.0]
            for e, sign in edge_incidence.get(n, []):
                cols.append(reg.idx("flow", t, e))
                vals.append(sign)
            add_eq(cols, vals, demand_np[t, n], label("balance", t, nodes[n]))

    pv_cf = inputs.pv_cf.reindex(demand.index).ffill().bfill().to_numpy(dtype=float)
    wind_cf = inputs.wind_cf.reindex(demand.index).ffill().bfill().to_numpy(dtype=float)
    for t in range(T):
        for n in range(N):
            add_eq(
                [reg.idx("pv_gen", t, n), reg.idx("pv_curtail", t, n), reg.idx("pv_cap", n)],
                [1.0, 1.0, -pv_cf[t]],
                0.0,
                label("pv_availability", t, nodes[n]),
            )
            add_eq(
                [reg.idx("wind_gen", t, n), reg.idx("wind_curtail", t, n), reg.idx("wind_cap", n)],
                [1.0, 1.0, -wind_cf[t]],
                0.0,
                label("wind_availability", t, nodes[n]),
            )
            add_ub([reg.idx("bess_charge", t, n), reg.idx("bess_power", n)], [1.0, -1.0], 0.0, label("bess_charge_power", t, nodes[n]))
            add_ub([reg.idx("bess_discharge", t, n), reg.idx("bess_power", n)], [1.0, -1.0], 0.0, label("bess_discharge_power", t, nodes[n]))
            add_ub([reg.idx("bess_soc", t, n), reg.idx("bess_energy", n)], [1.0, -1.0], 0.0, label("bess_energy_bound", t, nodes[n]))
            add_ub([reg.idx("grid_import", t, n), reg.idx("grid_expansion", n)], [1.0, -1.0], existing_grid[n], label("grid_interface", t, nodes[n]))

    _add_storage_dynamics(add_eq, add_ub, reg, "bess", T, N, nodes, cfg["bess_eta_ch"], cfg["bess_eta_dis"], cfg["bess_h_max"])
    if reg.has("ldes_soc"):
        _add_storage_dynamics(add_eq, add_ub, reg, "ldes", T, N, nodes, cfg["ldes_eta_ch"], cfg["ldes_eta_dis"], cfg["ldes_h_max"])

    _add_shift_constraints(
        add_eq,
        add_ub,
        reg,
        demand_np,
        demand.index,
        nodes,
        "dr",
        cfg["dr_window_h"],
        cfg["dr_ramp_fraction_per_h"],
    )
    _add_shift_constraints(
        add_eq,
        add_ub,
        reg,
        demand_np,
        demand.index,
        nodes,
        "thermal",
        cfg["thermal_window_h"],
        cfg["thermal_ramp_fraction_per_h"],
    )
    _add_ev_constraints(add_eq, reg, demand, weights, cfg, nodes)

    e_ref = grid_only_emissions_ton(demand, co2, weights)
    cap = config.get("co2_cap_ton")
    if cap is None and config.get("co2_cap_ratio") is not None:
        cap = e_ref * float(config["co2_cap_ratio"])
    if cap is not None:
        cols = []
        vals = []
        for t in range(T):
            for n in range(N):
                cols.append(reg.idx("grid_import", t, n))
                vals.append(float(co2_np[t] * w[t] / 1000.0))
        add_ub(cols, vals, float(cap), "co2_cap")

    A_eq = _rows_to_coo(eq_rows, reg.size)
    A_ub = _rows_to_coo(ub_rows, reg.size)
    return LPBuild(
        registry=reg,
        c=c,
        A_ub=A_ub,
        b_ub=np.asarray(ub_rhs, dtype=float),
        A_eq=A_eq,
        b_eq=np.asarray(eq_rhs, dtype=float),
        ub_labels=ub_labels,
        eq_labels=eq_labels,
        demand=demand,
        weights=weights,
        price=price,
        co2=co2,
        nodes=nodes,
        edges=edges,
        config=config,
        tech_config=cfg,
        e_ref_grid_only_ton=e_ref,
        co2_cap_ton=float(cap) if cap is not None else None,
    )
def _fill_objective(c: np.ndarray, reg: VarRegistry, inputs: ModelInputs, config: dict, cfg: dict, N: int, T: int, w: np.ndarray, price: np.ndarray, co2: np.ndarray, grid_cost: np.ndarray) -> None:
    discount = inputs.discount_rate
    crf_grid = annualized_capex(1.0, discount, inputs.planning_horizon_years)
    crf_pv = annualized_capex(1.0, discount, inputs.pv_cost["lifetime_years"])
    crf_wind = annualized_capex(1.0, discount, inputs.wind_cost["lifetime_years"])
    crf_bess = annualized_capex(1.0, discount, inputs.bess_cost["lifetime_years"])
    crf_ldes = annualized_capex(1.0, discount, cfg["ldes_lifetime"])

    objective = config.get("objective", "min_cost")
    for n in range(N):
        c[reg.idx("pv_cap", n)] = inputs.pv_cost["capex_yuan_per_kW"] * (crf_pv + inputs.pv_cost.get("om_fix_fraction", 0.0))
        c[reg.idx("wind_cap", n)] = inputs.wind_cost["capex_yuan_per_kW"] * crf_wind + inputs.wind_cost.get("om_fix_yuan_per_kW_per_year", 0.0)
        c[reg.idx("bess_power", n)] = inputs.bess_cost["capex_power_yuan_per_kW"] * (crf_bess + inputs.bess_cost.get("om_fix_fraction", 0.0))
        c[reg.idx("bess_energy", n)] = inputs.bess_cost["capex_energy_yuan_per_kWh"] * (crf_bess + inputs.bess_cost.get("om_fix_fraction", 0.0))
        c[reg.idx("bess_energy", n)] += cfg["bess_calendar_ageing_cost_yuan_per_kwh_year"]
        c[reg.idx("grid_expansion", n)] = grid_cost[n] * crf_grid
        c[reg.idx("ldes_power", n)] = cfg["ldes_power_capex"] * crf_ldes
        c[reg.idx("ldes_energy", n)] = cfg["ldes_energy_capex"] * crf_ldes
    for t in range(T):
        for n in range(N):
            c[reg.idx("grid_import", t, n)] = price[t] * w[t]
            c[reg.idx("load_shedding", t, n)] = cfg["ls_penalty"] * w[t]
            if cfg["bess_degradation_mode"] in {"throughput", "throughput_plus_calendar"}:
                c[reg.idx("bess_charge", t, n)] += cfg["bess_deg_cost"] * w[t]
                c[reg.idx("bess_discharge", t, n)] += cfg["bess_deg_cost"] * w[t]
            for name, unit_cost in [
                ("dr_pos", cfg["demand_response_cost"]),
                ("dr_neg", cfg["demand_response_cost"]),
                ("thermal_pos", cfg["thermal_flexibility_cost"]),
                ("thermal_neg", cfg["thermal_flexibility_cost"]),
                ("ev_charge", cfg["ev_flexibility_cost"]),
                ("ev_discharge", cfg["ev_flexibility_cost"]),
            ]:
                if reg.has(name):
                    c[reg.idx(name, t, n)] += unit_cost * w[t]
            if objective == "min_emissions":
                c[reg.idx("grid_import", t, n)] = c[reg.idx("grid_import", t, n)] * 1e-9
    if objective == "min_emissions":
        c[:] = 0.0
        # Objective in tons CO2 plus a large load-shedding deterrent.
        for t in range(T):
            for n in range(N):
                c[reg.idx("grid_import", t, n)] = co2[t] * w[t] / 1000.0
                c[reg.idx("load_shedding", t, n)] = cfg["ls_penalty"] * w[t]


def _add_storage_dynamics(add_eq, add_ub, reg: VarRegistry, prefix: str, T: int, N: int, nodes: list[str], eta_ch: float, eta_dis: float, h_max: float) -> None:
    ch = f"{prefix}_charge"
    dis = f"{prefix}_discharge"
    soc = f"{prefix}_soc"
    power = f"{prefix}_power"
    energy = f"{prefix}_energy"
    if not reg.has(soc):
        return
    for t in range(T):
        prev = (t - 1) % T
        for n in range(N):
            add_eq(
                [reg.idx(soc, t, n), reg.idx(soc, prev, n), reg.idx(ch, t, n), reg.idx(dis, t, n)],
                [1.0, -1.0, -eta_ch, 1.0 / eta_dis],
                0.0,
                label(f"{prefix}_soc", t, nodes[n]),
            )
            add_ub([reg.idx(ch, t, n), reg.idx(power, n)], [1.0, -1.0], 0.0, label(f"{prefix}_charge_power", t, nodes[n]))
            add_ub([reg.idx(dis, t, n), reg.idx(power, n)], [1.0, -1.0], 0.0, label(f"{prefix}_discharge_power", t, nodes[n]))
            add_ub([reg.idx(soc, t, n), reg.idx(energy, n)], [1.0, -1.0], 0.0, label(f"{prefix}_energy_bound", t, nodes[n]))
    for n in range(N):
        add_ub([reg.idx(energy, n), reg.idx(power, n)], [1.0, -h_max], 0.0, label(f"{prefix}_duration", nodes[n]))


def _add_shift_constraints(
    add_eq,
    add_ub,
    reg: VarRegistry,
    demand_np: np.ndarray,
    index: pd.DatetimeIndex,
    nodes: list[str],
    prefix: str,
    window_h: int,
    ramp_fraction_per_h: float,
) -> None:
    pos = f"{prefix}_pos"
    neg = f"{prefix}_neg"
    if not reg.has(pos):
        return
    for start in range(0, len(index), max(1, window_h)):
        locs = range(start, min(start + max(1, window_h), len(index)))
        for n, node in enumerate(nodes):
            cols = []
            vals = []
            for t in locs:
                cols += [reg.idx(pos, t, n), reg.idx(neg, t, n)]
                vals += [1.0, -1.0]
            add_eq(cols, vals, 0.0, label(f"{prefix}_energy_conservation", start, node))
            ramp = float(ramp_fraction_per_h) * float(np.max(demand_np[:, n]))
            for prev, t in zip(locs, list(locs)[1:]):
                add_ub(
                    [reg.idx(pos, t, n), reg.idx(neg, t, n), reg.idx(pos, prev, n), reg.idx(neg, prev, n)],
                    [1.0, -1.0, -1.0, 1.0],
                    ramp,
                    label(f"{prefix}_ramp_up", t, node),
                )
                add_ub(
                    [reg.idx(pos, t, n), reg.idx(neg, t, n), reg.idx(pos, prev, n), reg.idx(neg, prev, n)],
                    [-1.0, 1.0, 1.0, -1.0],
                    ramp,
                    label(f"{prefix}_ramp_down", t, node),
                )


def _ev_power_bounds(demand: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    profile = str(cfg.get("ev_penetration", "medium"))
    scale = {"low": 0.03, "medium": 0.06, "high": 0.10}.get(profile, 0.06)
    max_power = (demand.max(axis=0).to_numpy(dtype=float) * scale).clip(min=0.0)
    charge = np.zeros(demand.shape)
    discharge = np.zeros(demand.shape)
    available = {18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7}
    for t, ts in enumerate(demand.index):
        if ts.hour in available:
            charge[t, :] = max_power
            if cfg["ev_mode"] == "smart_charging_plus_V2G":
                discharge[t, :] = max_power * 0.6
    soc = np.tile((max_power * 4.0), (len(demand.index), 1))
    return {"charge": charge.reshape(-1), "discharge": discharge.reshape(-1), "soc": soc.reshape(-1)}


def _add_ev_constraints(add_eq, reg: VarRegistry, demand: pd.DataFrame, weights: pd.Series, cfg: dict, nodes: list[str]) -> None:
    if not reg.has("ev_charge"):
        return
    shares = demand.sum(axis=0)
    shares = shares / shares.sum() if shares.sum() > 0 else shares * 0
    daily_mwh = {"low": 0.03, "medium": 0.06, "high": 0.10}.get(str(cfg.get("ev_penetration", "medium")), 0.06)
    total_kwh = float(demand.sum().sum() * 8760.0 / max(len(demand.index), 1) * daily_mwh)
    w = weights.to_numpy(dtype=float)
    for n, node in enumerate(nodes):
        cols = []
        vals = []
        for t in range(len(demand.index)):
            cols += [reg.idx("ev_charge", t, n), reg.idx("ev_discharge", t, n)]
            vals += [w[t], -w[t]]
        add_eq(cols, vals, total_kwh * float(shares.iloc[n]), label("ev_annual_net_charge", node))
        if cfg["ev_mode"] != "smart_charging_plus_V2G":
            continue
        for t in range(len(demand.index)):
            prev = (t - 1) % len(demand.index)
            add_eq(
                [reg.idx("ev_soc", t, n), reg.idx("ev_soc", prev, n), reg.idx("ev_charge", t, n), reg.idx("ev_discharge", t, n)],
                [1.0, -1.0, -0.92, 1.0 / 0.92],
                0.0,
                label("ev_soc", t, node),
            )


def _network_edges(inputs: ModelInputs, nodes: list[str], config: dict) -> list[tuple[int, int]]:
    k = int(config.get("network_k_nearest", 3))
    if inputs.node_summary is None or "node_id" not in inputs.node_summary.columns:
        return [(i, i + 1) for i in range(len(nodes) - 1)]
    df = inputs.node_summary.set_index("node_id").reindex(nodes)
    numeric = df.select_dtypes(include=[np.number])
    if {"centroid_x", "centroid_y"}.issubset(numeric.columns):
        coords = numeric[["centroid_x", "centroid_y"]].fillna(0.0).to_numpy()
    else:
        coords = np.column_stack([np.arange(len(nodes), dtype=float), np.zeros(len(nodes))])
    edges = set()
    for i in range(len(nodes)):
        d = np.linalg.norm(coords - coords[i], axis=1)
        for j in np.argsort(d)[1 : k + 1]:
            src, dst = sorted((i, int(j)))
            edges.add((src, dst))
    return sorted(edges)


def _fixed_vector(value, nodes: list[str], n: int) -> np.ndarray:
    if isinstance(value, dict):
        return np.asarray([float(value.get(node, 0.0)) for node in nodes], dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        return np.full(n, float(arr))
    return arr.reshape(-1)


def _edge_incidence(edges: list[tuple[int, int]], nodes: list[str]) -> dict[int, list[tuple[int, float]]]:
    inc: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(nodes))}
    for e, (i, j) in enumerate(edges):
        inc[i].append((e, -1.0))
        inc[j].append((e, 1.0))
    return inc


def _rows_to_coo(rows: list[tuple[list[int], list[float]]], n_cols: int) -> coo_matrix:
    data: list[float] = []
    row_idx: list[int] = []
    col_idx: list[int] = []
    for r, (cols, vals) in enumerate(rows):
        data.extend(vals)
        col_idx.extend(cols)
        row_idx.extend([r] * len(cols))
    if not data:
        return coo_matrix((0, n_cols))
    return coo_matrix((data, (row_idx, col_idx)), shape=(len(rows), n_cols))
