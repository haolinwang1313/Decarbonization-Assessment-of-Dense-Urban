from __future__ import annotations

import pandas as pd

from paper05.model.result import extract_tables
from paper05.model.solve import solve_model

from tests.runtime_helpers import base_config, tiny_inputs


REQUIRED_CAPACITY_FIELDS = [
    "installed_bess_power_mw",
    "installed_bess_energy_mwh",
    "installed_ldes_power_mw",
    "installed_ldes_energy_mwh",
    "installed_grid_expansion_mw",
    "existing_grid_interface_mw",
    "total_grid_interface_mw",
]


def test_node_metrics_include_node_level_capacity_fields() -> None:
    inputs = tiny_inputs(price=[1.0, 10.0, 1.0, 10.0], demand=[10.0, 10.0, 10.0, 10.0])
    config = base_config() | {"fixed_capacities_kw": {"bess_power_kw": 10.0, "bess_energy_kwh": 40.0}}

    build, res, runtime = solve_model(inputs, config)
    tables = extract_tables(build, res, inputs, runtime, {})

    assert res.success
    assert set(REQUIRED_CAPACITY_FIELDS).issubset(tables.node_metrics.columns)

    capacity = tables.node_metrics[REQUIRED_CAPACITY_FIELDS]
    assert all(pd.api.types.is_numeric_dtype(capacity[column]) for column in REQUIRED_CAPACITY_FIELDS)
    assert bool((capacity >= 0.0).all().all())

    grid_total_error = (
        tables.node_metrics["total_grid_interface_mw"]
        - tables.node_metrics["existing_grid_interface_mw"]
        - tables.node_metrics["installed_grid_expansion_mw"]
    ).abs()
    assert float(grid_total_error.max()) < 1e-6
