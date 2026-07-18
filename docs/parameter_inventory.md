# Parameter Inventory

This file records the main parameter families used by the public model configuration. Exact numeric values are stored in the committed CSV and YAML files listed below.

## Emissions

- Fixed-baseline operational-emissions cap labels are configured through `co2_cap_ratio` in `configs/experiments/*.yaml`.
- Imported-electricity carbon-factor scenarios are stored in `data/xinwu_public/carbon_factors.csv`.
- Operational emissions are computed from imported grid electricity multiplied by hourly carbon factors.

## Technologies

- PV capacity bounds are stored in `data/xinwu_public/pv_potential_by_node.csv`.
- Wind capacity bounds are stored in `data/xinwu_public/wind_potential_by_node.csv`.
- PV, wind, and BESS cost scenarios are stored in `data/xinwu_public/technology_costs.csv`.
- Storage duration, efficiency, and long-duration storage defaults are stored in `data/xinwu_public/storage_parameters.csv`.
- Grid-interface expansion capacity and cost parameters are stored in `data/xinwu_public/grid_interface_by_node.csv`.

## Flexibility

- Demand-response fraction is encoded by technology-set names or explicit config keys.
- Thermal flexibility fraction is encoded by technology-set names or explicit config keys.
- Energy-neutral balancing windows and ramp constraints are listed in `data/xinwu_public/flexibility_parameters.csv`.

## Temporal Representation

- Representative periods are listed in `data/xinwu_public/representative_periods.csv`.
- Hourly weights are listed in `data/xinwu_public/temporal_weights.csv`.
- Public smoke checks use a short chronological slice configured in `configs/experiments/public_smoke.yaml`.

## Uncertainty

- LHS sampled parameters and ranges are listed in `data/xinwu_public/uncertainty_ranges.csv`.
- PRCC rankings use the sampled parameters under the independence assumption encoded by the LHS design.
