# Input Schema

This document defines the expected schema of the public Xinwu input package.

## Required Tables

| File | Required columns | Description |
|---|---|---|
| `nodes.csv` | `node_id`, `area`, `land_use` | Planning-node descriptors |
| `node_hour_demand.csv` | `node_id`, `time`, `demand_mwh` | Hourly node-level electricity demand |
| `pv_potential_by_node.csv` | `node_id`, `pv_capacity_mw` | Node-level PV technical capacity bound |
| `wind_potential_by_node.csv` | `node_id`, `wind_capacity_mw` | Node-level wind technical capacity bound |
| `pv_availability_by_hour.csv` | `time`, `node_id`, `pv_availability` | Hourly PV availability factor |
| `wind_availability_by_hour.csv` | `time`, `node_id`, `wind_availability` | Hourly wind availability factor |
| `grid_interface_by_node.csv` | `node_id`, `grid_capacity_mw`, `grid_capacity_max_mw`, `grid_connection_cost_yuan_per_kW` | Existing and expandable grid-interface capacity |
| `grid_prices.csv` | `time`, `price_yuan_per_kwh` | Imported-electricity tariff series |
| `network_edges_spatial_proxy.csv` | `from_node`, `to_node`, `transfer_capacity_mw` | Spatial-transfer proxy edge list |
| `representative_periods.csv` | `period_id`, `date`, `weight` | Representative-period inventory |
| `temporal_weights.csv` | `time`, `weight` | Hourly weights used by the representative-period configuration |
| `carbon_factors.csv` | `scenario`, `time`, `emission_factor_kg_per_kwh` | Imported-electricity emissions factors |
| `technology_costs.csv` | `technology`, `scenario`, `parameter`, `value`, `unit` | Technology cost parameters |
| `storage_parameters.csv` | `storage_family`, `scenario`, `parameter`, `value` | Storage technology parameters |
| `flexibility_parameters.csv` | `parameter`, `value`, `unit` | Demand response, thermal flexibility, and EV flexibility parameters |
| `uncertainty_ranges.csv` | `parameter`, `low`, `high`, `unit` | LHS uncertainty ranges |

The dataset configuration in `configs/datasets/xinwu_public.yaml` maps each logical input to its committed CSV file.
