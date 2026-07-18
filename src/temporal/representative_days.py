"""Representative-day selection for annual demand/resource series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_representative_days(demand_kw: pd.DataFrame, n_days: int) -> tuple[pd.DatetimeIndex, pd.Series]:
    if n_days <= 0:
        raise ValueError("n_days must be positive.")
    hourly = demand_kw.sum(axis=1).sort_index()
    daily = hourly.groupby(hourly.index.date)
    profiles = []
    dates = []
    for date, values in daily:
        if len(values) == 24:
            profiles.append(values.to_numpy(dtype=float))
            dates.append(pd.Timestamp(date))
    if not profiles:
        raise ValueError("No complete daily profiles available for representative-day selection.")
    X = np.vstack(profiles)
    n = min(n_days, len(X))
    if n == len(X):
        selected = np.arange(len(X))
        labels = np.arange(len(X))
    else:
        daily_energy = X.sum(axis=1)
        order = np.argsort(daily_energy)
        quantile_positions = np.linspace(0, len(order) - 1, n).round().astype(int)
        selected = np.array(sorted(order[quantile_positions]))
        centers = daily_energy[selected]
        labels = np.argmin(np.abs(daily_energy[:, None] - centers[None, :]), axis=1)

    selected_dates = [dates[i] for i in selected]
    full_index_parts = [pd.date_range(d, periods=24, freq="h") for d in selected_dates]
    rep_index = pd.DatetimeIndex(np.concatenate([idx.to_numpy() for idx in full_index_parts]))
    weight_by_date = {dates[i].date(): int(np.sum(labels == selected_pos)) for selected_pos, i in enumerate(selected)}
    weights = []
    for ts in rep_index:
        weights.append(weight_by_date.get(ts.date(), 1))
    scale = 8760.0 / float(np.sum(weights))
    return rep_index, pd.Series(np.asarray(weights, dtype=float) * scale, index=rep_index)
