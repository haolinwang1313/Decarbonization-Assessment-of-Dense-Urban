"""Public variable-name constants for the LP model."""

CAPACITY_VARS = [
    "pv_cap",
    "wind_cap",
    "bess_power",
    "bess_energy",
    "ldes_power",
    "ldes_energy",
    "grid_expansion",
]

DISPATCH_VARS = [
    "pv_gen",
    "wind_gen",
    "bess_charge",
    "bess_discharge",
    "bess_soc",
    "ldes_charge",
    "ldes_discharge",
    "ldes_soc",
    "grid_import",
    "load_shedding",
    "pv_curtail",
    "wind_curtail",
]
