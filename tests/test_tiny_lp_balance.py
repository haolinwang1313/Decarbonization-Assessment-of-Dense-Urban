from __future__ import annotations

from paper05.model.checks import check_dispatch_balance
from paper05.model.result import extract_tables
from paper05.model.solve import solve_model

from tests.runtime_helpers import base_config, tiny_inputs


def test_tiny_lp_balance() -> None:
    inputs = tiny_inputs(price=[1.0, 10.0, 1.0, 10.0], demand=[10.0, 10.0, 10.0, 10.0])
    config = base_config() | {"fixed_capacities_kw": {"bess_power_kw": 10.0, "bess_energy_kwh": 40.0}}

    build, res, runtime = solve_model(inputs, config)
    tables = extract_tables(build, res, inputs, runtime, {})

    assert res.success
    balance = check_dispatch_balance(tables.dispatch, tolerance=1e-9)
    assert balance.passed
    assert balance.max_abs_error <= 1e-9
