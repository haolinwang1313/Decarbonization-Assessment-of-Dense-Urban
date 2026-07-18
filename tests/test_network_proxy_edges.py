from __future__ import annotations

import pandas as pd

from paper05.data.schema import ModelInputs
from paper05.model.result import extract_tables
from paper05.model.solve import solve_model

from tests.runtime_helpers import base_config, tiny_inputs


def _two_node_inputs() -> ModelInputs:
    one = tiny_inputs(price=[1.0, 2.0], demand=[10.0, 11.0])
    idx = one.demand_kw.index
    return ModelInputs(
        dataset_id="tiny",
        spatial_resolution="k2",
        demand_kw=pd.DataFrame({"N1": [10.0, 11.0], "N2": [8.0, 9.0]}, index=idx),
        pv_cf=one.pv_cf,
        wind_cf=one.wind_cf,
        price_yuan_per_kwh=one.price_yuan_per_kwh,
        grid_co2_kg_per_kwh=one.grid_co2_kg_per_kwh,
        pv_cap_max_kw=pd.Series({"N1": 0.0, "N2": 0.0}),
        wind_cap_max_kw=pd.Series({"N1": 0.0, "N2": 0.0}),
        grid_capacity=pd.DataFrame(
            {
                "P_grid_existing_kW": [100.0, 100.0],
                "P_grid_max_kW": [100.0, 100.0],
                "grid_connection_cost_yuan_per_kW": [800.0, 800.0],
            },
            index=["N1", "N2"],
        ),
        node_summary=pd.DataFrame(
            {
                "node_id": ["N1", "N2"],
                "centroid_x": [0.0, 3.0],
                "centroid_y": [0.0, 4.0],
            }
        ),
        pv_cost=one.pv_cost,
        wind_cost=one.wind_cost,
        bess_cost=one.bess_cost,
        discount_rate=one.discount_rate,
        planning_horizon_years=one.planning_horizon_years,
    )


def test_spatial_transfer_proxy_outputs_edge_list() -> None:
    inputs = _two_node_inputs()
    config = base_config() | {
        "technology_set": "base_with_storage_degradation",
        "network_mode": "spatial_transfer_proxy",
        "network_k_nearest": 1,
        "transfer_cap_mw": 5.0,
    }

    build, res, runtime = solve_model(inputs, config)
    tables = extract_tables(build, res, inputs, runtime, {})

    assert res.success
    assert not tables.network_edges.empty
    assert {"src_node", "dst_node", "distance", "transfer_cap_mw", "edge_construction_rule"} <= set(tables.network_edges.columns)
    assert float(tables.network_edges.loc[0, "distance"]) == 5.0
    assert float(tables.network_edges.loc[0, "transfer_cap_mw"]) == 5.0


def test_non_proxy_network_modes_do_not_emit_proxy_edges() -> None:
    inputs = _two_node_inputs()
    for mode in ["node_interface", "copperplate"]:
        build, res, runtime = solve_model(inputs, base_config() | {"network_mode": mode})
        tables = extract_tables(build, res, inputs, runtime, {})
        assert res.success
        assert tables.network_edges.empty
