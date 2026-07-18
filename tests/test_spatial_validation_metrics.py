from __future__ import annotations

import pandas as pd

from paper05.analysis.spatial_validation import spatial_aggregation_metrics


def test_spatial_validation_metrics_conserve_identical_aggregation() -> None:
    ts = pd.date_range("2025-01-01", periods=4, freq="h")
    grid = pd.DataFrame(
        {
            "grid250_id": ["g1"] * 4 + ["g2"] * 4,
            "timestamp": list(ts) * 2,
            "load_kw": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    mapping = pd.DataFrame({"grid250_id": ["g1", "g2"], "node_id": ["n1", "n1"]})
    node = grid.merge(mapping, on="grid250_id").groupby(["node_id", "timestamp"], as_index=False)["load_kw"].sum()

    metrics = spatial_aggregation_metrics(grid, node, mapping)

    assert metrics["annual_load_conservation_error"] == 0.0
    assert metrics["peak_load_relative_error"] == 0.0
    assert metrics["top_5pct_peak_hour_overlap_ratio"] == 1.0
