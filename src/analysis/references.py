"""Reference-emissions utilities."""

from __future__ import annotations

import pandas as pd


def grid_only_emissions_ton(demand_kw: pd.DataFrame, co2_kg_per_kwh: pd.Series, weights: pd.Series) -> float:
    return float((demand_kw.mul(co2_kg_per_kwh, axis=0).mul(weights, axis=0) / 1000.0).sum().sum())
