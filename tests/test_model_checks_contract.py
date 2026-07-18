from pathlib import Path

import pandas as pd
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
def runtime_config():
    with (ROOT / "configs" / "experiments" / "00_legacy_audit.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_public_config_validator_accepts_e0_e11_manifest(manifest_config):
    checks = pytest.importorskip("paper05.model.checks")
    validate_experiment_config = require_public_function(checks, "validate_experiment_config")

    errors = validate_experiment_config(manifest_config)

    assert errors == []


def test_public_config_validator_accepts_runtime_matrix(runtime_config):
    checks = pytest.importorskip("paper05.model.checks")
    validate_experiment_config = require_public_function(checks, "validate_experiment_config")

    errors = validate_experiment_config(runtime_config)

    assert errors == []


def test_public_dispatch_balance_check_accepts_balanced_fixture():
    checks = pytest.importorskip("paper05.model.checks")
    check_dispatch_balance = require_public_function(checks, "check_dispatch_balance")
    dispatch = pd.read_csv(ROOT / "tests" / "fixtures" / "sample_dispatch.csv")

    result = check_dispatch_balance(dispatch, tolerance=1e-9)

    assert result.passed
    assert result.max_abs_error <= 1e-9
