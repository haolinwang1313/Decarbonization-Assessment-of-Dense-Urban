from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper05.data.schema import ModelInputs


ROOT = Path(__file__).resolve().parents[1]


def tiny_inputs(*, price: list[float], demand: list[float], grid_co2: list[float] | None = None) -> ModelInputs:
    idx = pd.date_range("2025-01-01", periods=len(price), freq="h")
    co2 = grid_co2 or [0.6] * len(price)
    zeros = [0.0] * len(price)
    return ModelInputs(
        dataset_id="tiny",
        spatial_resolution="k1",
        demand_kw=pd.DataFrame({"N1": demand}, index=idx),
        pv_cf=pd.Series(zeros, index=idx),
        wind_cf=pd.Series(zeros, index=idx),
        price_yuan_per_kwh=pd.Series(price, index=idx),
        grid_co2_kg_per_kwh=pd.Series(co2, index=idx),
        pv_cap_max_kw=pd.Series({"N1": 0.0}),
        wind_cap_max_kw=pd.Series({"N1": 0.0}),
        grid_capacity=pd.DataFrame(
            {
                "P_grid_existing_kW": [100.0],
                "P_grid_max_kW": [100.0],
                "grid_connection_cost_yuan_per_kW": [800.0],
            },
            index=["N1"],
        ),
        node_summary=None,
        pv_cost={"capex_yuan_per_kW": 3000.0, "lifetime_years": 20.0, "om_fix_fraction": 0.01},
        wind_cost={"capex_yuan_per_kW": 5000.0, "lifetime_years": 20.0, "om_fix_yuan_per_kW_per_year": 50.0},
        bess_cost={
            "capex_power_yuan_per_kW": 0.0,
            "capex_energy_yuan_per_kWh": 0.0,
            "lifetime_years": 15.0,
            "om_fix_fraction": 0.0,
        },
        discount_rate=0.06,
        planning_horizon_years=20.0,
    )


def base_config() -> dict:
    return {
        "experiment_id": "TEST",
        "run_id": "tiny",
        "dataset_id": "tiny",
        "spatial_resolution": "k1",
        "temporal_mode": "full8760_opt",
        "technology_set": "base_corrected",
        "use_existing_weights": False,
        "load_shedding_penalty_yuan_per_kwh": 1e6,
    }
