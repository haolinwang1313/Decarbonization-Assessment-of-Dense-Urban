from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def require_public_function(module, name):
    if not hasattr(module, name):
        pytest.xfail(f"paper05 public API pending: {module.__name__}.{name}")
    return getattr(module, name)


def test_public_metric_summary_matches_weighted_fixture_totals():
    metrics = pytest.importorskip("paper05.analysis.metrics")
    summarize_dispatch = require_public_function(metrics, "summarize_dispatch")
    dispatch = pd.read_csv(ROOT / "tests" / "fixtures" / "sample_dispatch.csv")

    summary = summarize_dispatch(dispatch)

    assert summary["load_shedding_kwh"] == pytest.approx(0.0)
    assert summary["grid_import_kwh"] == pytest.approx(997.0)
    assert summary["co2_ton"] == pytest.approx(0.5960066)
    assert summary["curtailment_kwh"] == pytest.approx(46.0)


def test_public_metric_summary_reports_required_paper_fields():
    metrics = pytest.importorskip("paper05.analysis.metrics")
    summarize_dispatch = require_public_function(metrics, "summarize_dispatch")
    dispatch = pd.read_csv(ROOT / "tests" / "fixtures" / "sample_dispatch.csv")

    summary = summarize_dispatch(dispatch)

    for key in [
        "grid_import_kwh",
        "renewable_generation_kwh",
        "load_shedding_kwh",
        "curtailment_kwh",
        "co2_ton",
    ]:
        assert key in summary
