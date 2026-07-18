"""Load Paper05 model inputs from workflow or public CSV packages."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from paper05.data.schema import ModelInputs, ProjectPaths


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _normalize_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    out = pd.to_datetime(index)
    if getattr(out, "tz", None) is not None:
        out = out.tz_convert(None)
    return pd.DatetimeIndex(out)


def repeat_to_index(series: pd.Series, target_index: pd.Index) -> pd.Series:
    target = _normalize_datetime_index(target_index)
    src = series.copy()
    src.index = _normalize_datetime_index(src.index)
    if len(src) == len(target):
        return pd.Series(src.to_numpy(), index=target)
    arr = src.to_numpy()
    reps = math.ceil(len(target) / len(arr))
    return pd.Series(np.tile(arr, reps)[: len(target)], index=target)


def _load_demand(paths: ProjectPaths, k: str) -> pd.DataFrame:
    path = paths.workflow / "01_demand_integration" / "Grid_Clustering" / "outputs" / k / "xinwu_nodes_energy_hourly.parquet"
    df = pd.read_parquet(path, columns=["timestamp", "node_id", "P_total_el_node_kW"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    demand = df.pivot(index="timestamp", columns="node_id", values="P_total_el_node_kW")
    demand.index = _normalize_datetime_index(demand.index)
    return demand.sort_index().fillna(0.0)


def _load_cf(path: Path, value_col: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    idx = _normalize_datetime_index(df["timestamp"])
    return pd.Series(df[value_col].to_numpy(dtype=float), index=idx)


def _load_grid_co2(paths: ProjectPaths, scenario: str) -> pd.Series:
    path = paths.workflow / "02_supply_side" / "6ElectricityCO2" / "grid_co2_hourly.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[df["scenario"].astype(str) == str(scenario)].copy()
    if df.empty:
        raise ValueError(f"Missing grid CO2 scenario: {scenario}")
    idx = _normalize_datetime_index(df["timestamp"])
    return pd.Series(df["grid_co2_factor_kg_per_kWh"].to_numpy(dtype=float), index=idx)


def _load_price(paths: ProjectPaths, scenario: str) -> pd.Series:
    path = paths.workflow / "02_supply_side" / "5ElectricityPrice" / "grid_price_hourly.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    idx = _normalize_datetime_index(df["timestamp"])
    price = pd.Series(df["price_yuan_per_kWh"].to_numpy(dtype=float), index=idx)
    if scenario in (None, "tou_base"):
        return price
    if scenario == "tou_high_peak":
        return price * 1.2
    if scenario == "tou_low_peak":
        return price * 0.9
    if scenario == "flat":
        return pd.Series(float(price.mean()), index=price.index)
    if scenario == "steep_peak":
        q80 = price.quantile(0.8)
        q20 = price.quantile(0.2)
        out = price.copy()
        out[price >= q80] *= 1.3
        out[price <= q20] *= 0.8
        return out
    raise ValueError(f"Unknown tariff scenario: {scenario}")


def _load_grid_capacity(paths: ProjectPaths, demand: pd.DataFrame, k: str) -> pd.DataFrame:
    path = paths.workflow / "02_supply_side" / "7CapPrice" / "node_grid_connection.csv"
    raw = pd.read_csv(path).set_index("node_id")
    nodes = list(demand.columns)
    if set(nodes).issubset(set(raw.index)):
        return raw.loc[nodes].copy()

    peak = demand.max(axis=0).reindex(nodes).fillna(0.0)
    # k70/k150 did not have a dedicated grid-capacity table in the inherited
    # data. Derive a documented proxy from node peak demand rather than mixing
    # incompatible k100 node IDs.
    return pd.DataFrame(
        {
            "P_grid_existing_kW": peak * 1.2,
            "P_grid_max_kW": peak * 2.2,
            "grid_connection_cost_yuan_per_kW": 800.0,
        },
        index=nodes,
    )


def _load_potential(paths: ProjectPaths, k: str, demand: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    pv_path = paths.workflow / "02_supply_side" / "1PVPotential" / "outputs" / f"{k}_node_pv_potential.csv"
    pv = pd.read_csv(pv_path).set_index("node_id")["pv_potential_kwp"].astype(float)

    wind_path = paths.workflow / "02_supply_side" / "0Wind" / "processed" / "node_wind_potential.csv"
    wind_raw = pd.read_csv(wind_path).set_index("node_id")["wind_cap_max_kW"].astype(float)
    if set(demand.columns).issubset(set(wind_raw.index)):
        wind = wind_raw.reindex(demand.columns).fillna(0.0)
    else:
        # Wind potential is only available for k100. Preserve total wind
        # potential and allocate by annual demand share for alternate spatial
        # aggregation robustness runs.
        shares = demand.sum(axis=0)
        shares = shares / shares.sum() if shares.sum() > 0 else shares * 0
        wind = shares * float(wind_raw.sum())
    return pv.reindex(demand.columns).fillna(0.0), wind.reindex(demand.columns).fillna(0.0)


def _load_node_summary(paths: ProjectPaths, k: str) -> pd.DataFrame | None:
    path = paths.workflow / "01_demand_integration" / "Grid_Clustering" / "outputs" / k / "node_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(paths: ProjectPaths, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else paths.root / path


def _is_public_config(config: dict[str, Any]) -> bool:
    return bool(config.get("files") or config.get("data_root") or config.get("dataset_id") == "xinwu_public")


def _public_dataset_config(paths: ProjectPaths, config: dict[str, Any]) -> dict[str, Any]:
    if not _is_public_config(config):
        return config
    dataset_path = paths.configs / "datasets" / "xinwu_public.yaml"
    dataset_cfg = read_yaml(dataset_path) if dataset_path.exists() else {}
    merged = dict(dataset_cfg)
    merged.update(config)
    files = dict(dataset_cfg.get("files", {}))
    files.update(config.get("files", {}))
    if files:
        merged["files"] = files
    return merged


def _public_file(paths: ProjectPaths, config: dict[str, Any], key: str) -> Path:
    files = config.get("files", {})
    if key not in files:
        raise ValueError(f"Missing public data file mapping: {key}")
    return _resolve_path(paths, files[key])


def _load_public_demand(paths: ProjectPaths, config: dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(_public_file(paths, config, "demand"), parse_dates=["time"])
    if "demand_kw" in df.columns:
        df["demand_kw"] = pd.to_numeric(df["demand_kw"], errors="coerce")
    elif "demand_mwh" in df.columns:
        # Hourly energy in MWh is equivalent to average MW over the hour.
        df["demand_kw"] = pd.to_numeric(df["demand_mwh"], errors="coerce") * 1000.0
    else:
        raise ValueError("Public demand table requires demand_mwh or demand_kw.")
    demand = df.pivot(index="time", columns="node_id", values="demand_kw")
    demand.index = _normalize_datetime_index(demand.index)
    return demand.sort_index().fillna(0.0)


def _load_public_availability(paths: ProjectPaths, config: dict[str, Any], key: str, value_col: str, target_index: pd.Index) -> pd.Series:
    df = pd.read_csv(_public_file(paths, config, key), parse_dates=["time"])
    if value_col not in df.columns:
        raise ValueError(f"Public availability table {key} is missing {value_col}.")
    if "node_id" in df.columns:
        series = df.groupby("time", dropna=False)[value_col].mean()
    else:
        series = df.set_index("time")[value_col]
    series.index = _normalize_datetime_index(series.index)
    return repeat_to_index(pd.to_numeric(series, errors="coerce").ffill().bfill(), target_index)


def _load_public_price(paths: ProjectPaths, config: dict[str, Any], target_index: pd.Index) -> pd.Series:
    df = pd.read_csv(_public_file(paths, config, "grid_prices"), parse_dates=["time"])
    price = pd.Series(pd.to_numeric(df["price_yuan_per_kwh"], errors="coerce").to_numpy(dtype=float), index=_normalize_datetime_index(df["time"]))
    scenario = str(config.get("tariff_scenario", "tou_base"))
    if scenario in {"tou_base", "base", "None"}:
        out = price
    elif scenario == "tou_high_peak":
        out = price * 1.2
    elif scenario == "tou_low_peak":
        out = price * 0.9
    elif scenario == "flat":
        out = pd.Series(float(price.mean()), index=price.index)
    elif scenario == "steep_peak":
        q80 = price.quantile(0.8)
        q20 = price.quantile(0.2)
        out = price.copy()
        out[price >= q80] *= 1.3
        out[price <= q20] *= 0.8
    else:
        raise ValueError(f"Unknown tariff scenario: {scenario}")
    return repeat_to_index(out.ffill().bfill(), target_index)


def _load_public_co2(paths: ProjectPaths, config: dict[str, Any], target_index: pd.Index) -> pd.Series:
    df = pd.read_csv(_public_file(paths, config, "carbon_factors"), parse_dates=["time"])
    scenario = str(config.get("grid_carbon_scenario", "baseline"))
    df = df[df["scenario"].astype(str) == scenario].copy()
    if df.empty:
        raise ValueError(f"Missing public carbon-factor scenario: {scenario}")
    value_col = "emission_factor_kg_per_kwh" if "emission_factor_kg_per_kwh" in df.columns else "emission_factor"
    series = pd.Series(pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float), index=_normalize_datetime_index(df["time"]))
    return repeat_to_index(series.ffill().bfill(), target_index)


def _load_public_potential(paths: ProjectPaths, config: dict[str, Any], demand: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    pv = pd.read_csv(_public_file(paths, config, "pv_potential")).set_index("node_id")["pv_capacity_mw"].astype(float) * 1000.0
    wind = pd.read_csv(_public_file(paths, config, "wind_potential")).set_index("node_id")["wind_capacity_mw"].astype(float) * 1000.0
    nodes = demand.columns
    return pv.reindex(nodes).fillna(0.0), wind.reindex(nodes).fillna(0.0)


def _load_public_grid_capacity(paths: ProjectPaths, config: dict[str, Any], demand: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(_public_file(paths, config, "grid_interface")).set_index("node_id")
    out = pd.DataFrame(index=demand.columns)
    out["P_grid_existing_kW"] = pd.to_numeric(raw["grid_capacity_mw"], errors="coerce").reindex(demand.columns).fillna(0.0) * 1000.0
    if "grid_capacity_max_mw" in raw.columns:
        out["P_grid_max_kW"] = pd.to_numeric(raw["grid_capacity_max_mw"], errors="coerce").reindex(demand.columns).fillna(0.0) * 1000.0
    else:
        out["P_grid_max_kW"] = out["P_grid_existing_kW"] * 2.0
    if "grid_connection_cost_yuan_per_kW" in raw.columns:
        out["grid_connection_cost_yuan_per_kW"] = pd.to_numeric(raw["grid_connection_cost_yuan_per_kW"], errors="coerce").reindex(demand.columns).fillna(800.0)
    else:
        out["grid_connection_cost_yuan_per_kW"] = 800.0
    return out


def _load_public_node_summary(paths: ProjectPaths, config: dict[str, Any]) -> pd.DataFrame | None:
    files = config.get("files", {})
    if "nodes" not in files:
        return None
    return pd.read_csv(_public_file(paths, config, "nodes"))


def _load_public_costs(paths: ProjectPaths, config: dict[str, Any], technology: str, scenario_key: str) -> dict:
    scenario = str(config.get(scenario_key, "base"))
    df = pd.read_csv(_public_file(paths, config, "technology_costs"))
    rows = df[(df["technology"].astype(str) == technology) & (df["scenario"].astype(str) == scenario)]
    if rows.empty:
        raise ValueError(f"Missing public {technology} cost scenario: {scenario}")
    return {str(row.parameter): float(row.value) for row in rows.itertuples(index=False)}


def _load_public_model_inputs(paths: ProjectPaths, config: dict[str, Any]) -> ModelInputs:
    config = _public_dataset_config(paths, config)
    dataset_id = str(config.get("dataset_id", "xinwu_public"))
    k = str(config.get("spatial_resolution", "k100"))
    demand = _load_public_demand(paths, config)
    pv_cf = _load_public_availability(paths, config, "pv_availability", "pv_availability", demand.index)
    wind_cf = _load_public_availability(paths, config, "wind_availability", "wind_availability", demand.index)
    price = _load_public_price(paths, config, demand.index)
    co2 = _load_public_co2(paths, config, demand.index)
    pv_cap_max, wind_cap_max = _load_public_potential(paths, config, demand)
    grid_capacity = _load_public_grid_capacity(paths, config, demand)
    return ModelInputs(
        dataset_id=dataset_id,
        spatial_resolution=k,
        demand_kw=demand,
        pv_cf=pv_cf,
        wind_cf=wind_cf,
        price_yuan_per_kwh=price,
        grid_co2_kg_per_kwh=co2,
        pv_cap_max_kw=pv_cap_max,
        wind_cap_max_kw=wind_cap_max,
        grid_capacity=grid_capacity,
        node_summary=_load_public_node_summary(paths, config),
        pv_cost=_load_public_costs(paths, config, "pv", "pv_cost_scenario"),
        wind_cost=_load_public_costs(paths, config, "wind", "wind_cost_scenario"),
        bess_cost=_load_public_costs(paths, config, "bess", "bess_cost_scenario"),
        discount_rate=float(config.get("discount_rate", 0.06)),
        planning_horizon_years=float(config.get("planning_horizon_years", 20)),
    )


def load_model_inputs(config: dict) -> ModelInputs:
    paths = ProjectPaths.discover()
    if _is_public_config(config):
        return _load_public_model_inputs(paths, config)

    dataset_id = config.get("dataset_id", "xinwu_k100")
    k = config.get("spatial_resolution") or dataset_id.split("_")[-1]
    grid_carbon_scenario = str(config.get("grid_carbon_scenario", "baseline"))
    tariff_scenario = str(config.get("tariff_scenario", "tou_base"))
    pv_cost_scenario = str(config.get("pv_cost_scenario", "base"))
    wind_cost_scenario = str(config.get("wind_cost_scenario", "base"))
    bess_cost_scenario = str(config.get("bess_cost_scenario", "base"))

    demand = _load_demand(paths, k)
    pv_cf = repeat_to_index(
        _load_cf(paths.workflow / "02_supply_side" / "2CF_PV_t" / "pv_capacity_factor_hourly.csv", "CF_pv"),
        demand.index,
    )
    wind_cf = repeat_to_index(
        _load_cf(paths.workflow / "02_supply_side" / "0Wind" / "processed" / "wind_capacity_factor_hourly.csv", "CF_wind"),
        demand.index,
    )
    price = repeat_to_index(_load_price(paths, tariff_scenario), demand.index)
    co2 = repeat_to_index(_load_grid_co2(paths, grid_carbon_scenario), demand.index)
    pv_cap_max, wind_cap_max = _load_potential(paths, k, demand)
    grid_capacity = _load_grid_capacity(paths, demand, k)

    pv_cost = _load_json(paths.workflow / "02_supply_side" / "3PVPrice" / "pv_price_scenarios.json")["scenarios"][pv_cost_scenario]
    wind_cost = _load_json(paths.workflow / "02_supply_side" / "0Wind" / "processed" / "wind_tech_params.json")["scenarios"][wind_cost_scenario]
    bess_cost = _load_json(paths.workflow / "02_supply_side" / "4ESS" / "storage_tech_params.json")["scenarios"][bess_cost_scenario]

    return ModelInputs(
        dataset_id=dataset_id,
        spatial_resolution=k,
        demand_kw=demand,
        pv_cf=pv_cf,
        wind_cf=wind_cf,
        price_yuan_per_kwh=price,
        grid_co2_kg_per_kwh=co2,
        pv_cap_max_kw=pv_cap_max,
        wind_cap_max_kw=wind_cap_max,
        grid_capacity=grid_capacity,
        node_summary=_load_node_summary(paths, k),
        pv_cost=pv_cost,
        wind_cost=wind_cost,
        bess_cost=bess_cost,
        discount_rate=float(config.get("discount_rate", 0.06)),
        planning_horizon_years=float(config.get("planning_horizon_years", 20)),
    )


def source_input_files(config: dict) -> list[Path]:
    paths = ProjectPaths.discover()
    if _is_public_config(config):
        public_config = _public_dataset_config(paths, config)
        return [_resolve_path(paths, value) for value in public_config.get("files", {}).values()]

    k = config.get("spatial_resolution") or config.get("dataset_id", "xinwu_k100").split("_")[-1]
    return [
        paths.workflow / "01_demand_integration" / "Grid_Clustering" / "outputs" / k / "xinwu_nodes_energy_hourly.parquet",
        paths.workflow / "02_supply_side" / "2CF_PV_t" / "pv_capacity_factor_hourly.csv",
        paths.workflow / "02_supply_side" / "0Wind" / "processed" / "wind_capacity_factor_hourly.csv",
        paths.workflow / "02_supply_side" / "5ElectricityPrice" / "grid_price_hourly.csv",
        paths.workflow / "02_supply_side" / "6ElectricityCO2" / "grid_co2_hourly.csv",
        paths.workflow / "02_supply_side" / "1PVPotential" / "outputs" / f"{k}_node_pv_potential.csv",
        paths.workflow / "02_supply_side" / "0Wind" / "processed" / "node_wind_potential.csv",
        paths.workflow / "02_supply_side" / "7CapPrice" / "node_grid_connection.csv",
    ]
