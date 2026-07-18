# Data Provenance

This release includes processed Xinwu case inputs prepared for the urban power-system assessment workflow. The tables provide the spatial, temporal, technology, flexibility, cost, and emissions parameters consumed by the public configurations.

## Processed Input Categories

| Category | Modelling role | Released tables |
|---|---|---|
| Building demand | Hourly node-level electricity demand | `node_hour_demand.csv`, `nodes.csv` |
| Solar resources | Node-level capacity bounds and hourly availability | `pv_potential_by_node.csv`, `pv_availability_by_hour.csv` |
| Wind resources | Node-level capacity bounds and hourly availability | `wind_potential_by_node.csv`, `wind_availability_by_hour.csv` |
| Grid interface | Existing and expandable import capacity and tariffs | `grid_interface_by_node.csv`, `grid_prices.csv` |
| Grid emissions | Imported-electricity emissions scenarios | `carbon_factors.csv` |
| Technology and storage | Cost, efficiency, duration, and lifecycle parameters | `technology_costs.csv`, `storage_parameters.csv` |
| Flexibility | Demand response, EV, and thermal-flexibility parameters | `flexibility_parameters.csv` |
| Temporal representation | Representative periods and hourly weights | `representative_periods.csv`, `temporal_weights.csv` |
| Network abstraction | Spatial-transfer proxy topology and capacity | `network_edges_spatial_proxy.csv` |
| Uncertainty | Parameter ranges used by uncertainty configurations | `uncertainty_ranges.csv` |

## Release Records

- `configs/datasets/xinwu_public.yaml` maps logical model inputs to file paths.
- `data/xinwu_public/public_input_manifest.csv` records row counts and byte sizes.
- `data/catalog.yaml` records release roles, SHA-256 checksums, and MIT licensing.
- `docs/input_schema.md` defines required fields and modelling meanings.

## License

The processed inputs are available under the repository-wide MIT License.
