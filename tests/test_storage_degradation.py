from __future__ import annotations

from paper05.model.result import extract_tables, safe_efc
from paper05.model.solve import solve_model

from tests.runtime_helpers import base_config, tiny_inputs


def test_storage_degradation() -> None:
    inputs = tiny_inputs(price=[1.0, 10.0, 1.0, 10.0], demand=[10.0, 10.0, 10.0, 10.0])
    config = base_config() | {"fixed_capacities_kw": {"bess_power_kw": 10.0, "bess_energy_kwh": 40.0}}

    build, res, runtime = solve_model(inputs, config)
    tables = extract_tables(build, res, inputs, runtime, {})

    assert res.success
    assert safe_efc(100.0, 0.0, 0.8) == 0.0
    assert float(tables.costs["bess_degradation_cost"].iloc[0]) > 0.0
    assert float(tables.storage_cycles["bess_equivalent_full_cycles"].iloc[0]) > 0.0


def test_storage_degradation_counts_charge_discharge_and_calendar_ageing() -> None:
    inputs = tiny_inputs(price=[1.0, 10.0, 1.0, 10.0], demand=[10.0, 10.0, 10.0, 10.0])
    config = base_config() | {
        "fixed_capacities_kw": {"bess_power_kw": 10.0, "bess_energy_kwh": 40.0},
        "bess_degradation_cost_yuan_per_kwh": 0.05,
        "bess_calendar_ageing_cost_yuan_per_kwh_year": 2.0,
    }

    build, res, runtime = solve_model(inputs, config)
    tables = extract_tables(build, res, inputs, runtime, {})

    assert res.success
    cycling = float(tables.storage_cycles["bess_cycling_degradation_cost_yuan"].iloc[0])
    calendar = float(tables.storage_cycles["bess_calendar_ageing_cost_yuan"].iloc[0])
    degradation = float(tables.costs["bess_degradation_cost"].iloc[0])
    total = float(tables.costs["total_system_cost"].iloc[0])
    component_sum = float(tables.costs.drop(columns=["total_system_cost"]).iloc[0].sum())

    assert cycling > 0.0
    assert calendar > 0.0
    assert degradation == cycling + calendar
    assert total == component_sum
