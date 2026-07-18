from __future__ import annotations

import pytest

from paper05.model.result import extract_tables
from paper05.model.solve import solve_model

from tests.runtime_helpers import base_config, tiny_inputs


def test_dr_energy_conservation() -> None:
    inputs = tiny_inputs(price=[1.0, 10.0, 1.0, 10.0], demand=[10.0, 10.0, 10.0, 10.0])
    config = base_config() | {
        "technology_set": "base_with_DR10",
        "dr_fraction": 0.5,
        "dr_window_h": 2,
        "demand_response_cost_yuan_per_kwh": 0.0,
    }

    build, res, runtime = solve_model(inputs, config)
    tables = extract_tables(build, res, inputs, runtime, {})
    dispatch = tables.dispatch

    assert res.success
    grouped = dispatch.groupby(dispatch["timestamp"].dt.floor("2h"))[["dr_pos_kwh", "dr_neg_kwh"]].sum()
    assert float(dispatch["dr_pos_kwh"].sum()) > 0.0
    assert grouped["dr_pos_kwh"].tolist() == pytest.approx(grouped["dr_neg_kwh"].tolist())
