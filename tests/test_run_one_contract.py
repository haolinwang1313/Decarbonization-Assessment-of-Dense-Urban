from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def require_public_function(module, name):
    if not hasattr(module, name):
        pytest.xfail(f"paper05 public API pending: {module.__name__}.{name}")
    return getattr(module, name)


@pytest.fixture(scope="module")
def manifest_config():
    with (ROOT / "configs" / "experiments" / "e0_e11.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def legacy_audit_config():
    with (ROOT / "configs" / "experiments" / "00_legacy_audit.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_expand_manifest_resolves_all_runtime_configs(manifest_config):
    matrix = pytest.importorskip("paper05.experiments.run_matrix")
    expand_manifest = require_public_function(matrix, "expand_manifest")

    plan = expand_manifest(manifest_config, ROOT / "configs" / "experiments")

    assert [item["id"] for item in plan] == [f"E{i}" for i in range(12)]
    assert plan[-1]["kind"] == "validation"
    assert plan[-1]["path"].name == "10_data_validation.yaml"


def test_expand_matrix_generates_runnable_legacy_audit_run_ids(legacy_audit_config):
    matrix = pytest.importorskip("paper05.experiments.run_matrix")
    expand_matrix = require_public_function(matrix, "expand_matrix")

    runs = expand_matrix(legacy_audit_config)

    assert len(runs) == 15
    assert runs[0]["experiment_id"] == "E0_legacy_audit"
    assert runs[0]["run_id"] == "r0001_k100_repday_opt_base_corrected_baseline_cap1.00"
