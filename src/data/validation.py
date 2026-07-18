"""Input data validation summaries for Paper05."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paper05.data.loaders import load_model_inputs, read_yaml
from paper05.data.schema import ProjectPaths


def build_validation_report(config: dict) -> tuple[dict, pd.DataFrame, str]:
    inputs = load_model_inputs(config)
    demand = inputs.demand_kw
    annual_kwh_by_node = demand.sum(axis=0)
    peak_by_node = demand.max(axis=0)
    pv_cf = inputs.pv_cf
    wind_cf = inputs.wind_cf
    grid_ratio = inputs.grid_capacity.reindex(demand.columns)["P_grid_existing_kW"].fillna(0) / peak_by_node.replace(0, pd.NA)
    building_eui = _building_eui_summary(inputs.node_summary)
    report = {
        "dataset_id": inputs.dataset_id,
        "spatial_resolution": inputs.spatial_resolution,
        "annual_electricity_demand_mwh": float(demand.sum().sum() / 1000),
        "peak_demand_mw": float(demand.sum(axis=1).max() / 1000),
        "load_factor": float(demand.sum().sum() / (8760 * demand.sum(axis=1).max())) if demand.sum(axis=1).max() > 0 else None,
        "node_annual_demand_mwh_min": float(annual_kwh_by_node.min() / 1000),
        "node_annual_demand_mwh_median": float(annual_kwh_by_node.median() / 1000),
        "node_annual_demand_mwh_max": float(annual_kwh_by_node.max() / 1000),
        "building_EUI_distribution": building_eui,
        "PV_potential_mw_total": float(inputs.pv_cap_max_kw.sum() / 1000),
        "PV_potential_mw_median": float(inputs.pv_cap_max_kw.median() / 1000),
        "PV_annual_capacity_factor": float(pv_cf.mean()),
        "PV_full_load_hours": float(pv_cf.sum()),
        "wind_potential_mw_total": float(inputs.wind_cap_max_kw.sum() / 1000),
        "wind_potential_mw_median": float(inputs.wind_cap_max_kw.median() / 1000),
        "wind_eligible_node_count": int((inputs.wind_cap_max_kw > 0).sum()),
        "wind_annual_capacity_factor": float(wind_cf.mean()),
        "grid_interface_capacity_mw_total": float(inputs.grid_capacity["P_grid_existing_kW"].sum() / 1000),
        "grid_capacity_to_peak_load_ratio_median": float(grid_ratio.median()),
        "TOU_price_annual_average": float(inputs.price_yuan_per_kwh.mean()),
        "grid_CO2_factor_annual_average": float(inputs.grid_co2_kg_per_kwh.mean()),
    }
    rows = [{"metric": k, "value": json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v} for k, v in report.items()]
    md = "# E11 Data Validation Report\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in report.items())
    return report, pd.DataFrame(rows), md


def write_validation_report(config_path: Path) -> None:
    cfg = read_yaml(config_path)
    paths = ProjectPaths.discover()
    out_dir = paths.results / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    report, table, md = build_validation_report(cfg)
    (out_dir / "E11_data_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    table.to_csv(out_dir / "E11_data_validation.csv", index=False)
    (out_dir / "E11_data_validation.md").write_text(md, encoding="utf-8")


def _building_eui_summary(node_summary: pd.DataFrame | None = None) -> dict:
    if node_summary is not None and {"E_annual", "total_floor_area"}.issubset(node_summary.columns):
        floor_area = pd.to_numeric(node_summary["total_floor_area"], errors="coerce")
        annual = pd.to_numeric(node_summary["E_annual"], errors="coerce")
        eui = (annual / floor_area.replace(0, pd.NA)).dropna()
        if not eui.empty:
            return {
                "status": "available",
                "source": "node_summary",
                "definition": "E_annual divided by total_floor_area",
                "min": float(eui.min()),
                "median": float(eui.median()),
                "max": float(eui.max()),
            }

    path = ProjectPaths.discover().workflow / "01_demand_integration" / "outputs" / "xinwu_building_metadata.parquet"
    if not path.exists():
        return {"status": "missing", "reason": "xinwu_building_metadata.parquet not found"}
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return {"status": "unreadable", "reason": str(exc)}
    numeric = df.select_dtypes("number")
    if numeric.empty:
        return {"status": "no_numeric_columns"}
    candidates = [c for c in numeric.columns if "eui" in c.lower()]
    if not candidates:
        return {"status": "missing_eui_column", "numeric_columns": list(numeric.columns[:20])}
    s = numeric[candidates[0]].dropna()
    return {
        "status": "available",
        "column": candidates[0],
        "min": float(s.min()),
        "median": float(s.median()),
        "max": float(s.max()),
    }
