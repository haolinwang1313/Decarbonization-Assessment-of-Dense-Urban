# Data Dictionary

The released CSV tables use UTF-8 text with a header row. Exact input requirements are defined in `docs/input_schema.md`, and result families are summarized in `docs/results_schema.md`.

## Xinwu Model Inputs

| File | Principal fields | Units or role |
|---|---|---|
| `carbon_factors.csv` | `scenario`, `time`, `emission_factor_kg_per_kwh` | kg CO2e/kWh |
| `flexibility_parameters.csv` | `parameter`, `value`, `unit` | Demand response, EV, and thermal-flexibility parameters |
| `grid_interface_by_node.csv` | `node_id`, capacity and connection-cost fields | MW and yuan/kW |
| `grid_prices.csv` | `time`, `price_yuan_per_kwh` | yuan/kWh |
| `network_edges_spatial_proxy.csv` | `from_node`, `to_node`, `transfer_capacity_mw` | MW |
| `node_hour_demand.csv` | `node_id`, `time`, `demand_mwh` | MWh |
| `nodes.csv` | `node_id`, `area`, `land_use` | Planning-node descriptors |
| `pv_availability_by_hour.csv` | `time`, `node_id`, `pv_availability` | Per-unit availability |
| `pv_potential_by_node.csv` | `node_id`, `pv_capacity_mw` | MW |
| `representative_periods.csv` | `period_id`, `date`, `weight` | Representative-period definition |
| `storage_parameters.csv` | `storage_family`, `scenario`, `parameter`, `value` | Storage technology parameters |
| `technology_costs.csv` | `technology`, `scenario`, `parameter`, `value`, `unit` | Technology cost assumptions |
| `temporal_weights.csv` | `time`, `weight` | Hourly representative-period weights |
| `uncertainty_ranges.csv` | `parameter`, `low`, `high`, `unit` | Uncertainty sampling bounds |
| `wind_availability_by_hour.csv` | `time`, `node_id`, `wind_availability` | Per-unit availability |
| `wind_potential_by_node.csv` | `node_id`, `wind_capacity_mw` | MW |
| `public_input_manifest.csv` | `file`, `rows`, `bytes` | Input-package inventory |

## Reported Results

CSV files under `results/summary_public/reported/` contain reported frontier, storage, flexibility, uncertainty, temporal, spatial, network, validity, and marginal-abatement-cost summaries. Column definitions are carried by each table header and grouped by analytical role in `docs/results_schema.md`.

## Test Fixture

`tests/fixtures/sample_dispatch.csv` is a compact dispatch table used by the public test suite.

## License

All tables described here are available under the MIT License.
