from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper05.analysis.emissions_recast import add_absolute_emission_fields, fixed_baseline_reference_ton
from paper05.experiments.summarize import summarize_results


def test_fixed_baseline_reference_uses_baseline_grid_only() -> None:
    ref = pd.DataFrame(
        [
            {"grid_carbon_scenario": "baseline", "technology_set": "grid_only_reference", "E_ref_grid_only_ton": 100.0},
            {"grid_carbon_scenario": "2030", "technology_set": "grid_only_reference", "E_ref_grid_only_ton": 80.0},
        ]
    )
    df = add_absolute_emission_fields(
        pd.DataFrame({"co2_cap_ton": [50.0, 40.0], "actual_emissions_ton": [45.0, 35.0], "co2_cap_ratio": [0.5, 0.5]}),
        fixed_reference_ton=fixed_baseline_reference_ton(ref),
    )

    assert df["scenario_relative_cap_ratio"].tolist() == [0.5, 0.5]
    assert df["fixed_baseline_cap_ratio"].tolist() == [0.5, 0.4]
    assert df["co2_cap_mt"].tolist() == [0.00005, 0.00004]
    assert df["actual_emissions_mt"].tolist() == [0.000045, 0.000035]


def test_summarize_writes_valid_only_absolute_emissions_recast(tmp_path: Path) -> None:
    root = tmp_path / "results"
    summaries = root / "summaries"
    summaries.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "experiment_id": "E1_reference_floor",
                "run_id": "ref",
                "technology_set": "grid_only_reference",
                "grid_carbon_scenario": "baseline",
                "co2_cap_ratio": 1.0,
                "co2_cap_ton": 100.0,
                "actual_emissions_ton": 100.0,
                "E_ref_grid_only_ton": 100.0,
                "total_system_cost": 1.0,
                "objective_name": "min_cost",
                "temporal_mode": "repday_opt",
                "solver_status": "optimal",
                "is_valid_result": True,
                "invalid_reason": "",
                "total_load_shedding_mwh": 0.0,
            },
            {
                "experiment_id": "E0_legacy_audit",
                "run_id": "valid_2030",
                "technology_set": "base",
                "grid_carbon_scenario": "2030",
                "co2_cap_ratio": 0.5,
                "co2_cap_ton": 40.0,
                "actual_emissions_ton": 38.0,
                "E_ref_grid_only_ton": 80.0,
                "total_system_cost": 2_000_000_000.0,
                "objective_name": "min_cost",
                "temporal_mode": "repday_opt",
                "solver_status": "optimal",
                "is_valid_result": True,
                "invalid_reason": "",
                "total_load_shedding_mwh": 0.0,
            },
            {
                "experiment_id": "E0_legacy_audit",
                "run_id": "invalid",
                "technology_set": "base",
                "grid_carbon_scenario": "baseline",
                "co2_cap_ratio": 0.5,
                "co2_cap_ton": 50.0,
                "actual_emissions_ton": 60.0,
                "E_ref_grid_only_ton": 100.0,
                "total_system_cost": 3.0,
                "objective_name": "min_cost",
                "temporal_mode": "repday_opt",
                "solver_status": "optimal",
                "is_valid_result": False,
                "invalid_reason": "load_shedding",
                "total_load_shedding_mwh": 1.0,
            },
        ]
    ).to_csv(summaries / "all_run_summary.csv", index=False)

    summarize_results(root)

    valid = pd.read_csv(summaries / "absolute_emissions_recast_valid_runs.csv")
    row = valid[valid["run_id"].eq("valid_2030")].iloc[0]

    assert "invalid" not in set(valid["run_id"])
    assert float(row["fixed_baseline_reference_ton"]) == 100.0
    assert float(row["scenario_relative_cap_ratio"]) == 0.5
    assert float(row["fixed_baseline_cap_ratio"]) == 0.4
