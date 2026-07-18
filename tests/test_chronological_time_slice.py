from __future__ import annotations

import pandas as pd

from paper05.temporal.chronological import select_time


def test_full8760_dispatch_validation_respects_time_slice_start_and_hours() -> None:
    idx = pd.date_range("2017-01-01 00:00:00", periods=8760, freq="h")
    demand = pd.DataFrame({"N1": range(len(idx))}, index=idx)

    selected, weights, mode = select_time(
        demand,
        {
            "temporal_mode": "full8760_dispatch_validation",
            "time_slice_start": "2017-07-01 00:00:00",
            "time_slice_hours": 168,
        },
    )

    assert mode == "full8760_dispatch_validation"
    assert len(selected) == 168
    assert selected[0] == pd.Timestamp("2017-07-01 00:00:00")
    assert (weights == 1.0).all()

