"""Temporal weighting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def uniform_annual_weights(index: pd.Index) -> pd.Series:
    if len(index) == 0:
        raise ValueError("Cannot build weights for an empty time index.")
    return pd.Series(np.full(len(index), 8760.0 / len(index)), index=index)


def explicit_weights(index: pd.Index, weight: float = 1.0) -> pd.Series:
    return pd.Series(np.full(len(index), weight), index=index)


def daily_groups(index: pd.DatetimeIndex) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for _, locs in pd.Series(np.arange(len(index)), index=index).groupby(index.date):
        groups.append(locs.to_numpy())
    return groups


def rolling_window_groups(index: pd.DatetimeIndex, window_h: int) -> list[np.ndarray]:
    if window_h <= 0:
        raise ValueError("window_h must be positive.")
    groups = []
    for start in range(0, len(index), window_h):
        groups.append(np.arange(start, min(start + window_h, len(index))))
    return groups
