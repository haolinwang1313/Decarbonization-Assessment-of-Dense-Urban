"""Default experiment registry and top-level runtime manifest."""

from __future__ import annotations

from pathlib import Path

from paper05.data.loaders import write_yaml
from paper05.experiments.run_matrix import run_matrix_file


MANIFEST_FILE = "e0_e11.yaml"
EXPERIMENT_REGISTRY = [
    {"id": "E0", "kind": "matrix", "file": "00_legacy_audit.yaml", "purpose": "Legacy audit and corrected baseline rerun."},
    {"id": "E1", "kind": "matrix", "file": "01_reference_floor.yaml", "purpose": "Reference emissions and feasibility floor."},
    {"id": "E2", "kind": "matrix", "file": "02_8760_validation.yaml", "purpose": "Representative-period sensitivity and selected chronological stress diagnostics."},
    {"id": "E3", "kind": "matrix", "file": "03_repday_sensitivity.yaml", "purpose": "Representative-day sensitivity."},
    {"id": "E4", "kind": "matrix", "file": "04_storage_lifecycle.yaml", "purpose": "Storage lifecycle and LDES."},
    {"id": "E5", "kind": "matrix", "file": "05_flexibility_portfolio.yaml", "purpose": "Demand response, EV, and thermal flexibility."},
    {"id": "E6", "kind": "matrix", "file": "06_network_spatial_robustness.yaml", "purpose": "Network abstraction robustness."},
    {"id": "E7", "kind": "matrix", "file": "07_spatial_aggregation.yaml", "purpose": "Spatial aggregation robustness."},
    {"id": "E8", "kind": "matrix", "file": "07_uncertainty_lhs.yaml", "purpose": "Input uncertainty LHS."},
    {"id": "E9", "kind": "matrix", "file": "08_dual_bottleneck.yaml", "purpose": "Dual variables and bottleneck decomposition."},
    {"id": "E10", "kind": "matrix", "file": "09_node_typology.yaml", "purpose": "Node typology."},
    {"id": "E11", "kind": "validation", "file": "10_data_validation.yaml", "purpose": "Input data validation report."},
]
EXPERIMENT_FILES = [entry["file"] for entry in EXPERIMENT_REGISTRY]


def ensure_default_configs(overwrite: bool = False) -> None:
    base = Path("configs")
    (base / "datasets").mkdir(parents=True, exist_ok=True)
    (base / "experiments").mkdir(parents=True, exist_ok=True)
    _write_dataset_configs(base, overwrite)
    _write_experiment_configs(base, overwrite)


def run_all_configs() -> None:
    run_matrix_file(Path("configs") / "experiments" / MANIFEST_FILE)


def _write_if(path: Path, data: dict, overwrite: bool) -> None:
    if overwrite or not path.exists():
        write_yaml(path, data)


def _write_dataset_configs(base: Path, overwrite: bool) -> None:
    public_files = {
        "nodes": "data/xinwu_public/nodes.csv",
        "demand": "data/xinwu_public/node_hour_demand.csv",
        "pv_potential": "data/xinwu_public/pv_potential_by_node.csv",
        "wind_potential": "data/xinwu_public/wind_potential_by_node.csv",
        "pv_availability": "data/xinwu_public/pv_availability_by_hour.csv",
        "wind_availability": "data/xinwu_public/wind_availability_by_hour.csv",
        "grid_interface": "data/xinwu_public/grid_interface_by_node.csv",
        "network_edges": "data/xinwu_public/network_edges_spatial_proxy.csv",
        "representative_periods": "data/xinwu_public/representative_periods.csv",
        "temporal_weights": "data/xinwu_public/temporal_weights.csv",
        "carbon_factors": "data/xinwu_public/carbon_factors.csv",
        "grid_prices": "data/xinwu_public/grid_prices.csv",
        "technology_costs": "data/xinwu_public/technology_costs.csv",
        "storage_parameters": "data/xinwu_public/storage_parameters.csv",
        "flexibility_parameters": "data/xinwu_public/flexibility_parameters.csv",
        "uncertainty_ranges": "data/xinwu_public/uncertainty_ranges.csv",
    }
    _write_if(
        base / "datasets" / "xinwu_public.yaml",
        {
            "dataset_id": "xinwu_public",
            "description": "Processed Xinwu public reproducibility input package",
            "data_root": "data/xinwu_public",
            "spatial_resolution": "k100",
            "files": public_files,
        },
        overwrite,
    )
    _write_if(
        base / "defaults.yaml",
        {
            "dataset_id": "xinwu_public",
            "spatial_resolution": "k100",
            "temporal_mode": "repday_opt",
            "technology_set": "base_corrected",
            "grid_carbon_scenario": "baseline",
            "weights_csv": "data/xinwu_public/temporal_weights.csv",
            "use_existing_weights": True,
            "n_rep_days": 8,
            "load_shedding_penalty_yuan_per_kwh": 100000.0,
        },
        overwrite,
    )


def _write_experiment_configs(base: Path, overwrite: bool) -> None:
    exp = base / "experiments"
    defaults = {
        "dataset_id": "xinwu_public",
        "spatial_resolution": "k100",
        "temporal_mode": "repday_opt",
        "weights_csv": "data/xinwu_public/temporal_weights.csv",
        "use_existing_weights": True,
        "n_rep_days": 8,
    }
    _write_if(
        exp / "00_legacy_audit.yaml",
        {
            "experiment_id": "E0_legacy_audit",
            "defaults": {**defaults, "technology_set": "base_corrected"},
            "matrix": {"co2_cap_ratio": [1.0, 0.40, 0.25, 0.15, 0.14], "grid_carbon_scenario": ["baseline", "2030", "2050"]},
        },
        overwrite,
    )
    _write_if(
        exp / "01_reference_floor.yaml",
        {
            "experiment_id": "E1_reference_floor",
            "defaults": defaults,
            "matrix": {
                "objective": ["min_cost", "min_emissions"],
                "technology_set": ["grid_only_reference", "base_corrected", "base_with_storage_degradation", "base_with_DR10", "base_with_EV_medium", "base_with_thermal10", "base_with_DR10_EV_medium_thermal10", "base_with_DR10_EV_medium_thermal10_LDES"],
                "grid_carbon_scenario": ["baseline", "2030", "2050"],
            },
        },
        overwrite,
    )
    _write_if(
        exp / "02_8760_validation.yaml",
        {
            "experiment_id": "E2_8760_validation",
            "defaults": {**defaults, "technology_set": "base_with_storage_degradation"},
            "matrix": {"temporal_mode": ["repday_opt", "full8760_dispatch_validation", "full8760_opt"], "co2_cap_ratio": [1.0, 0.40, 0.25, 0.15, 0.14], "grid_carbon_scenario": ["baseline", "2030", "2050"]},
        },
        overwrite,
    )
    _write_if(
        exp / "03_repday_sensitivity.yaml",
        {"experiment_id": "E3_repday_sensitivity", "defaults": {**defaults, "technology_set": "base_with_storage_degradation", "use_existing_weights": False}, "matrix": {"n_rep_days": [7, 14, 21, 30], "co2_cap_ratio": [1.0, 0.40, 0.25, 0.15, 0.14], "grid_carbon_scenario": ["baseline"]}},
        overwrite,
    )
    _write_if(
        exp / "04_storage_lifecycle.yaml",
        {"experiment_id": "E4_storage_lifecycle", "defaults": defaults, "matrix": {"storage_model": ["legacy_no_degradation", "liion_4h_with_degradation", "liion_8h_with_degradation", "liion_8h_with_degradation_plus_LDES"], "co2_cap_ratio": [1.0, 0.40, 0.25, 0.15, 0.14], "grid_carbon_scenario": ["baseline", "2030"]}},
        overwrite,
    )
    _write_if(
        exp / "05_flexibility_portfolio.yaml",
        {"experiment_id": "E5_flexibility_portfolio", "defaults": defaults, "matrix": {"technology_set": ["base_with_storage_degradation", "base_with_DR05", "base_with_DR10", "base_with_DR15", "base_with_EV_low_smart", "base_with_EV_medium_smart", "base_with_EV_high_smart", "base_with_EV_medium_smart_V2G", "base_with_thermal05", "base_with_thermal10", "base_with_thermal15", "base_with_DR10_EV_medium_thermal10", "base_with_DR10_EV_medium_thermal10_LDES"], "co2_cap_ratio": [0.40, 0.25, 0.15, 0.14], "grid_carbon_scenario": ["baseline"]}},
        overwrite,
    )
    _write_if(
        exp / "06_network_spatial_robustness.yaml",
        {"experiment_id": "E6_network_spatial_robustness", "defaults": defaults, "matrix": {"network_mode": ["node_interface", "copperplate", "spatial_transfer_proxy"], "technology_set": ["base_with_storage_degradation", "base_with_DR10_EV_medium_thermal10"], "co2_cap_ratio": [0.40, 0.25, 0.14], "grid_carbon_scenario": ["baseline"]}},
        overwrite,
    )
    _write_if(
        exp / "07_spatial_aggregation.yaml",
        {"experiment_id": "E7_spatial_aggregation", "defaults": defaults, "matrix": {"spatial_resolution": ["k100"], "technology_set": ["base_with_storage_degradation", "base_with_DR10_EV_medium_thermal10"], "co2_cap_ratio": [1.0, 0.40, 0.25, 0.15, 0.14], "grid_carbon_scenario": ["baseline"]}},
        overwrite,
    )
    _write_if(
        exp / "07_uncertainty_lhs.yaml",
        {"experiment_id": "E8_uncertainty_lhs", "defaults": {**defaults, "technology_set": "base_with_storage_degradation"}, "lhs_samples": 150, "matrix": {"co2_cap_ratio": [0.25, 0.14], "grid_carbon_scenario": ["baseline"]}},
        overwrite,
    )
    _write_if(
        exp / "08_dual_bottleneck.yaml",
        {"experiment_id": "E9_dual_bottleneck", "defaults": {**defaults, "technology_set": "base_with_DR10_EV_medium_thermal10"}, "matrix": {"co2_cap_ratio": [1.0, 0.40, 0.25, 0.15, 0.14], "grid_carbon_scenario": ["baseline"]}},
        overwrite,
    )
    _write_if(
        exp / "09_node_typology.yaml",
        {"experiment_id": "E10_node_typology", "defaults": defaults, "matrix": {"technology_set": ["base_with_storage_degradation", "base_with_DR10_EV_medium_thermal10"], "co2_cap_ratio": [0.40, 0.25, 0.14], "grid_carbon_scenario": ["baseline"]}},
        overwrite,
    )
    _write_if(
        exp / "10_data_validation.yaml",
        {"experiment_id": "E11_data_validation", "dataset_id": "xinwu_public", "spatial_resolution": "k100", "grid_carbon_scenario": "baseline"},
        overwrite,
    )
    _write_if(
        exp / MANIFEST_FILE,
        {
            "schema_version": 1,
            "matrix_id": "paper05_e0_e11",
            "description": "Paper05 runtime manifest for the PRD-required E0-E11 experiment families.",
            "experiments": EXPERIMENT_REGISTRY,
        },
        overwrite,
    )
