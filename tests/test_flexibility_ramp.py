from __future__ import annotations

import numpy as np
import pytest

from paper05.model.result import extract_tables
from paper05.model.solve import solve_model

from tests.runtime_helpers import base_config, tiny_inputs


def test_dr_and_thermal_ramp_constraints_are_labeled() -> None:
    inputs = tiny_inputs(price=[1.0, 10.0, 1.0, 10.0], demand=[10.0, 10.0, 10.0, 10.0])
    config = base_config() | {
        "technology_set": "base_with_DR10_thermal10",
        "dr_fraction": 0.5,
        "thermal_flex_fraction": 0.5,
        "dr_window_h": 4,
        "thermal_window_h": 4,
    }

    build, res, _ = solve_model(inputs, config)

    assert res.success
    assert any(label.startswith("dr_ramp_up") for label in build.ub_labels)
    assert any(label.startswith("dr_ramp_down") for label in build.ub_labels)
    assert any(label.startswith("thermal_ramp_up") for label in build.ub_labels)
    assert any(label.startswith("thermal_ramp_down") for label in build.ub_labels)


def test_dr_ramp_zero_prevents_net_adjustment_jump_and_preserves_window_energy() -> None:
    inputs = tiny_inputs(price=[1.0, 10.0, 1.0, 10.0], demand=[10.0, 10.0, 10.0, 10.0])
    config = base_config() | {
        "technology_set": "base_with_DR10",
        "dr_fraction": 0.5,
        "dr_window_h": 4,
        "dr_ramp_fraction_per_h": 0.0,
        "demand_response_cost_yuan_per_kwh": 0.0,
    }

    build, res, runtime = solve_model(inputs, config)
    tables = extract_tables(build, res, inputs, runtime, {})
    dispatch = tables.dispatch
    net = dispatch["dr_pos_kwh"].to_numpy(dtype=float) - dispatch["dr_neg_kwh"].to_numpy(dtype=float)

    assert res.success
    assert np.diff(net).tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert float(dispatch["dr_pos_kwh"].sum()) == pytest.approx(float(dispatch["dr_neg_kwh"].sum()))
