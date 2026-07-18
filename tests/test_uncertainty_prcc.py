from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paper05.analysis.sensitivity import partial_rank_correlation
from paper05.experiments.summarize import summarize_results


def test_partial_rank_correlation_schema_and_dominant_parameter() -> None:
    samples = pd.DataFrame(
        {
            "dominant": np.arange(1, 9, dtype=float),
            "noise": [2, 1, 4, 3, 6, 5, 8, 7],
            "weak": [1, 1, 2, 2, 3, 3, 4, 4],
        }
    )
    outputs = pd.DataFrame({"target": samples["dominant"] * 10 + samples["noise"] * 0.01})

    out = partial_rank_correlation(samples, outputs, target="target")

    assert {"parameter", "prcc", "target", "n_samples"} <= set(out.columns)
    assert out.iloc[0]["parameter"] == "dominant"
    assert out.iloc[0]["target"] == "target"
    assert int(out.iloc[0]["n_samples"]) == 8


def test_summarize_e8_prcc_top5_by_target_uses_valid_rows(tmp_path: Path) -> None:
    root = tmp_path / "results"
    summaries = root / "summaries"
    summaries.mkdir(parents=True)
    rows = []
    for cap in [0.25, 0.14]:
        for i in range(8):
            rows.append(
                {
                    "experiment_id": "E8_uncertainty_lhs",
                    "run_id": f"valid_{cap}_{i}",
                    "lhs_sample_id": i,
                    "co2_cap_ratio": cap,
                    "demand_scale": float(i),
                    "PV_potential_scale": float(8 - i),
                    "PV_capacity_factor_scale": float(i % 3),
                    "BESS_energy_capex": float(i + 1),
                    "total_system_cost": float(i * 100 + cap),
                    "actual_emissions_ton": float((8 - i) * 10),
                    "bess_energy_mwh": float(i * 5),
                    "grid_import_mwh": float((8 - i) * 7),
                    "technology_set": "base_with_storage_degradation",
                    "grid_carbon_scenario": "baseline",
                    "temporal_mode": "repday_opt",
                    "objective_name": "min_cost",
                    "solver_status": "optimal",
                    "is_valid_result": True,
                    "invalid_reason": "",
                    "total_load_shedding_mwh": 0.0,
                }
            )
        rows.append(
            {
                "experiment_id": "E8_uncertainty_lhs",
                "run_id": f"invalid_{cap}",
                "lhs_sample_id": 99,
                "co2_cap_ratio": cap,
                "demand_scale": 99.0,
                "PV_potential_scale": 99.0,
                "total_system_cost": -9999.0,
                "actual_emissions_ton": -9999.0,
                "bess_energy_mwh": -9999.0,
                "grid_import_mwh": -9999.0,
                "technology_set": "base_with_storage_degradation",
                "grid_carbon_scenario": "baseline",
                "temporal_mode": "repday_opt",
                "objective_name": "min_cost",
                "solver_status": "optimal",
                "is_valid_result": False,
                "invalid_reason": "load_shedding",
                "total_load_shedding_mwh": 1.0,
            }
        )
    pd.DataFrame(rows).to_csv(summaries / "all_run_summary.csv", index=False)

    summarize_results(root)

    prcc = pd.read_csv(summaries / "E8_prcc_by_target.csv")
    top5 = pd.read_csv(summaries / "E8_uncertainty_top5_by_target.csv")

    assert {"parameter", "prcc", "target", "n_samples", "co2_cap_ratio"} <= set(prcc.columns)
    assert set(prcc["target"]) >= {"total_system_cost", "actual_emissions_ton"}
    assert prcc["source_valid_runs_only"].astype(str).str.lower().eq("true").all()
    assert int(prcc["source_invalid_or_failed_runs_excluded"].max()) == 2
    assert top5.groupby(["co2_cap_ratio", "target"]).size().max() <= 5
