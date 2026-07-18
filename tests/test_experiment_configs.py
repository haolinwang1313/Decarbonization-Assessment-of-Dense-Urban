from pathlib import Path

import pytest
import yaml

from paper05.experiments import registry

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs" / "experiments" / "e0_e11.yaml"
EXPECTED_IDS = {f"E{i}" for i in range(12)}
VALID_KINDS = {"matrix", "validation"}


@pytest.fixture(scope="module")
def experiment_manifest():
    with MATRIX_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_e0_e11_manifest_declares_complete_unique_ids(experiment_manifest):
    experiments = experiment_manifest["experiments"]
    ids = [item["id"] for item in experiments]

    assert set(ids) == EXPECTED_IDS
    assert len(ids) == len(set(ids)) == 12


def test_manifest_references_relative_existing_runtime_configs(experiment_manifest):
    for item in experiment_manifest["experiments"]:
        path = item["file"]

        assert item["kind"] in VALID_KINDS
        assert not Path(path).is_absolute()
        assert "\\" not in path
        assert (MATRIX_PATH.parent / path).exists()


def test_manifest_matches_registry_runtime_order(experiment_manifest):
    actual = [(item["id"], item["kind"], item["file"]) for item in experiment_manifest["experiments"]]
    expected = [(item["id"], item["kind"], item["file"]) for item in registry.EXPERIMENT_REGISTRY]

    assert actual == expected


def test_split_runtime_configs_declare_matching_experiment_families(experiment_manifest):
    for item in experiment_manifest["experiments"]:
        config = yaml.safe_load((MATRIX_PATH.parent / item["file"]).read_text(encoding="utf-8"))
        runtime_id = str(config["experiment_id"])

        if item["kind"] == "validation":
            assert runtime_id == "E11_data_validation"
        else:
            assert runtime_id.startswith(f"{item['id']}_")
