from __future__ import annotations

from pathlib import Path

from paper05.data.loaders import read_yaml
from paper05.experiments.registry import EXPERIMENT_FILES


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG_DIR = ROOT / "configs" / "experiments"
PUBLIC_ENTRYPOINT_CONFIGS = {"public_smoke.yaml", "revision_public.yaml"}


def test_runtime_registry_and_configs_cover_e0_to_e11_execution_surface():
    excluded = {"e0_e11.yaml"} | PUBLIC_ENTRYPOINT_CONFIGS
    runtime_paths = sorted(path for path in RUNTIME_CONFIG_DIR.glob("*.yaml") if path.name not in excluded)
    runtime_ids = {read_yaml(path)["experiment_id"] for path in runtime_paths}

    assert runtime_ids == {
        "E0_legacy_audit",
        "E1_reference_floor",
        "E2_8760_validation",
        "E3_repday_sensitivity",
        "E4_storage_lifecycle",
        "E5_flexibility_portfolio",
        "E6_network_spatial_robustness",
        "E7_spatial_aggregation",
        "E8_uncertainty_lhs",
        "E9_dual_bottleneck",
        "E10_node_typology",
        "E11_data_validation",
    }
    assert EXPERIMENT_FILES == [
        "00_legacy_audit.yaml",
        "01_reference_floor.yaml",
        "02_8760_validation.yaml",
        "03_repday_sensitivity.yaml",
        "04_storage_lifecycle.yaml",
        "05_flexibility_portfolio.yaml",
        "06_network_spatial_robustness.yaml",
        "07_spatial_aggregation.yaml",
        "07_uncertainty_lhs.yaml",
        "08_dual_bottleneck.yaml",
        "09_node_typology.yaml",
        "10_data_validation.yaml",
    ]
    assert read_yaml(RUNTIME_CONFIG_DIR / "10_data_validation.yaml")["experiment_id"] == "E11_data_validation"
