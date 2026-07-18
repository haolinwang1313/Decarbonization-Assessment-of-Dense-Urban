from __future__ import annotations

import pandas as pd

from paper05.experiments.run_one import _apply_input_multipliers
from tests.runtime_helpers import tiny_inputs


def test_node_specific_relaxation_only_changes_target_nodes() -> None:
    inputs = tiny_inputs(price=[1.0, 1.0], demand=[10.0, 10.0])
    inputs.pv_cap_max_kw = pd.Series({"N1": 10.0, "N2": 20.0})
    inputs.wind_cap_max_kw = pd.Series({"N1": 30.0, "N2": 40.0})
    inputs.grid_capacity = pd.DataFrame(
        {
            "P_grid_existing_kW": [100.0, 200.0],
            "P_grid_max_kW": [110.0, 220.0],
            "grid_connection_cost_yuan_per_kW": [800.0, 800.0],
        },
        index=["N1", "N2"],
    )

    _apply_input_multipliers(
        inputs,
        {
            "node_pv_potential_scale": {"N1": 1.01},
            "node_wind_potential_scale": {"N2": 1.01},
            "node_grid_capacity_scale": {"N1": 1.01},
        },
    )

    assert float(inputs.pv_cap_max_kw.loc["N1"]) == 10.1
    assert float(inputs.pv_cap_max_kw.loc["N2"]) == 20.0
    assert float(inputs.wind_cap_max_kw.loc["N1"]) == 30.0
    assert float(inputs.wind_cap_max_kw.loc["N2"]) == 40.4
    assert float(inputs.grid_capacity.loc["N1", "P_grid_existing_kW"]) == 101.0
    assert float(inputs.grid_capacity.loc["N1", "P_grid_max_kW"]) == 111.1
    assert float(inputs.grid_capacity.loc["N2", "P_grid_existing_kW"]) == 200.0
    assert float(inputs.grid_capacity.loc["N2", "P_grid_max_kW"]) == 220.0
