"""Chronological time selection modes."""

from __future__ import annotations

import pandas as pd

from paper05.temporal.representative_days import select_representative_days
from paper05.temporal.weights import explicit_weights, uniform_annual_weights


def select_time(demand: pd.DataFrame, config: dict) -> tuple[pd.DatetimeIndex, pd.Series, str]:
    mode = config.get("temporal_mode", "repday_opt")
    if mode in {"full8760_opt", "full8760_dispatch_validation"}:
        start = config.get("time_slice_start")
        hours = config.get("time_slice_hours") or config.get("max_hours")
        if start is not None:
            available = demand.loc[pd.Timestamp(start) :].index
            idx = available[: int(hours)] if hours is not None else available
        else:
            idx = demand.index if hours is None else demand.index[: int(hours)]
        return pd.DatetimeIndex(idx), explicit_weights(idx, 1.0), mode

    n_days = int(config.get("n_rep_days", 8))
    if config.get("use_existing_weights", True) and n_days == 8 and config.get("spatial_resolution", "k100") == "k100":
        path = config.get("weights_csv")
        if path:
            wdf = pd.read_csv(path, parse_dates=["timestamp"])
            idx = pd.DatetimeIndex(pd.to_datetime(wdf["timestamp"]))
            return idx, pd.Series(wdf["weight"].to_numpy(dtype=float), index=idx), mode

    idx, weights = select_representative_days(demand, n_days)
    return idx, weights, mode


def fallback_sample_time(demand: pd.DataFrame, hours: int) -> tuple[pd.DatetimeIndex, pd.Series]:
    idx = pd.DatetimeIndex(demand.index[:hours])
    return idx, uniform_annual_weights(idx)
