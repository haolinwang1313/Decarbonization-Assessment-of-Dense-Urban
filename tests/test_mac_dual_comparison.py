from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paper05.analysis.metrics import dual_based_mac
from paper05.experiments.summarize import summarize_results


def test_dual_based_mac_sign_and_binding() -> None:
    assert dual_based_mac(pd.DataFrame([{"constraint": "co2_cap", "shadow_price": -12.0, "is_binding": True}])) == 12.0
    assert dual_based_mac(pd.DataFrame([{"constraint": "co2_cap", "shadow_price": -12.0, "is_binding": False}])) is None


def test_summarize_mac_dual_comparison_uses_valid_runs(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_dual_run(root, "valid_loose", 0.40, 90.0, 100.0, -3.0, True)
    _write_dual_run(root, "valid_tight", 0.25, 70.0, 200.0, -7.0, True)
    _write_dual_run(root, "invalid", 0.14, 60.0, 300.0, -9.0, True, valid=False)

    summarize_results(root)

    comp = pd.read_csv(root / "summaries" / "all_mac_dual_comparison.csv")

    assert set(comp["run_id"]) == {"valid_loose", "valid_tight"}
    assert set(comp["dual_MAC_CNY_per_tCO2"]) == {3.0, 7.0}
    assert comp["is_valid_result"].astype(str).str.lower().eq("true").all()


def _write_dual_run(
    root: Path,
    run_id: str,
    cap: float,
    emissions: float,
    cost: float,
    shadow: float,
    binding: bool,
    *,
    valid: bool = True,
) -> None:
    run_dir = root / "runs" / "E9_dual_bottleneck" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "experiment_id": "E9_dual_bottleneck",
        "run_id": run_id,
        "solver_status": "optimal",
        "objective_name": "min_cost",
        "objective_value": cost,
        "solver_objective_value": cost,
        "is_valid_result": valid,
        "invalid_reason": "" if valid else "load_shedding",
    }
    config = {
        "experiment_id": "E9_dual_bottleneck",
        "run_id": run_id,
        "technology_set": "base_with_storage_degradation",
        "grid_carbon_scenario": "baseline",
        "co2_cap_ratio": cap,
        "temporal_mode": "repday_opt",
        "objective": "min_cost",
    }
    qa = {
        "solver_success": True,
        "objective_name": "min_cost",
        "solver_objective_value": cost,
        "total_load_shedding_mwh": 0.0 if valid else 1.0,
        "cap_violation_ton": 0.0,
        "load_shedding_tolerance_mwh": 1e-6,
        "co2_tolerance_ton": 1e-4,
        "cost_tolerance_yuan": 1e-4,
        "objective_cost_consistency_error": 0.0,
        "is_valid_result": valid,
        "invalid_reason": "" if valid else "load_shedding",
    }
    (run_dir / "run_status.json").write_text(json.dumps(status), encoding="utf-8")
    (run_dir / "config_resolved.yaml").write_text("\n".join(f"{k}: {v}" for k, v in config.items()), encoding="utf-8")
    (run_dir / "qa_checks.json").write_text(json.dumps(qa), encoding="utf-8")
    pd.DataFrame([{"pv_mw": 0.0, "wind_mw": 0.0, "bess_power_mw": 0.0, "bess_energy_mwh": 0.0}]).to_csv(run_dir / "capacities.csv", index=False)
    pd.DataFrame([{"E_ref_grid_only_ton": 100.0, "co2_cap_ton": cap * 100.0, "actual_emissions_ton": emissions, "cap_violation_ton": 0.0}]).to_csv(run_dir / "emissions.csv", index=False)
    pd.DataFrame([{"grid_purchase_cost": cost, "total_system_cost": cost}]).to_csv(run_dir / "costs.csv", index=False)
    pd.DataFrame([{"pv_curtailment_mwh": 0.0, "wind_curtailment_mwh": 0.0}]).to_csv(run_dir / "curtailment.csv", index=False)
    pd.DataFrame([{"bess_charge_mwh": 0.0, "bess_discharge_mwh": 0.0, "bess_throughput_mwh": 0.0}]).to_csv(run_dir / "storage_cycles.csv", index=False)
    pd.DataFrame([{"grid_import_mwh": 1.0, "peak_grid_import_mw": 1.0}]).to_csv(run_dir / "grid_import.csv", index=False)
    pd.DataFrame([{"DR_shifted_mwh": 0.0}]).to_csv(run_dir / "flexibility.csv", index=False)
    pd.DataFrame([{"constraint": "co2_cap", "shadow_price": shadow, "slack": 0.0 if binding else 1.0, "is_binding": binding}]).to_csv(run_dir / "duals.csv", index=False)
    pd.DataFrame().to_csv(run_dir / "node_metrics.csv", index=False)
    pd.DataFrame().to_csv(run_dir / "network_edges.csv", index=False)
