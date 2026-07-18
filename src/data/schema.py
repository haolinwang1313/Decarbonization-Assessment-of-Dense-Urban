"""Shared data contracts for model inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    workflow: Path
    results: Path
    configs: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        root = Path(__file__).resolve().parents[3]
        return cls(
            root=root,
            workflow=root / "workflow",
            results=root / "results",
            configs=root / "configs",
        )


@dataclass
class ModelInputs:
    dataset_id: str
    spatial_resolution: str
    demand_kw: pd.DataFrame
    pv_cf: pd.Series
    wind_cf: pd.Series
    price_yuan_per_kwh: pd.Series
    grid_co2_kg_per_kwh: pd.Series
    pv_cap_max_kw: pd.Series
    wind_cap_max_kw: pd.Series
    grid_capacity: pd.DataFrame
    node_summary: pd.DataFrame | None
    pv_cost: dict
    wind_cost: dict
    bess_cost: dict
    discount_rate: float
    planning_horizon_years: float


@dataclass
class SolveResult:
    status: str
    message: str
    objective_value: float | None
    x: object | None
    raw: object | None
