from __future__ import annotations

import pytest

from paper05.model.result import extract_tables
from paper05.model.solve import solve_model

from tests.runtime_helpers import base_config, tiny_inputs


def test_co2_reference() -> None:
    inputs = tiny_inputs(
        price=[1.0, 1.0, 1.0, 1.0],
        demand=[10.0, 10.0, 10.0, 10.0],
        grid_co2=[0.6, 0.6, 0.6, 0.6],
    )
    config = base_config()

    build, res, runtime = solve_model(inputs, config)
    tables = extract_tables(build, res, inputs, runtime, {})

    assert res.success
    assert build.e_ref_grid_only_ton == pytest.approx(0.024)
    assert float(tables.emissions["E_ref_grid_only_ton"].iloc[0]) == pytest.approx(0.024)
