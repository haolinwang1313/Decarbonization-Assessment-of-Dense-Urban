"""Input uncertainty sampling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc


LHS_PARAMETERS = [
    "demand_scale",
    "PV_potential_scale",
    "PV_capacity_factor_scale",
    "wind_potential_scale",
    "wind_capacity_factor_scale",
    "grid_existing_capacity_scale",
    "grid_carbon_factor_scale",
    "PV_capex",
    "wind_capex",
    "BESS_power_capex",
    "BESS_energy_capex",
    "BESS_degradation_cost",
    "TOU_price_scale",
    "DR_fraction",
]


def lhs_samples(n_samples: int, seed: int = 42) -> pd.DataFrame:
    sampler = qmc.LatinHypercube(d=len(LHS_PARAMETERS), seed=seed)
    raw = sampler.random(n_samples)
    # Conservative +/- ranges around current defaults. They are inputs to the
    # model, not paper conclusions.
    low = np.array([0.90, 0.80, 0.90, 0.70, 0.80, 0.80, 0.80, 4000, 9000, 500, 1000, 0.00, 0.85, 0.00])
    high = np.array([1.10, 1.20, 1.10, 1.30, 1.20, 1.20, 1.20, 6500, 12000, 1000, 1500, 0.12, 1.20, 0.15])
    vals = qmc.scale(raw, low, high)
    return pd.DataFrame(vals, columns=LHS_PARAMETERS)


def rank_correlation(samples: pd.DataFrame, outputs: pd.DataFrame, target: str = "total_system_cost") -> pd.DataFrame:
    merged = pd.concat([samples.reset_index(drop=True), outputs[[target]].reset_index(drop=True)], axis=1)
    rows = []
    for col in samples.columns:
        rows.append({"parameter": col, "spearman": merged[col].corr(merged[target], method="spearman")})
    return pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)


def partial_rank_correlation(samples: pd.DataFrame, outputs: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Rank-transform inputs and target, then correlate residuals after
    controlling for the remaining sampled parameters.
    """

    if target not in outputs.columns:
        raise ValueError(f"missing PRCC target column: {target}")
    numeric_samples = samples.apply(pd.to_numeric, errors="coerce")
    merged = pd.concat(
        [numeric_samples.reset_index(drop=True), pd.to_numeric(outputs[target], errors="coerce").rename(target).reset_index(drop=True)],
        axis=1,
    ).dropna()
    rows = []
    if merged.empty:
        return pd.DataFrame(columns=["parameter", "prcc", "target", "n_samples"])

    ranked = merged.rank(method="average")
    sample_cols = list(numeric_samples.columns)
    y = ranked[target].to_numpy(dtype=float)
    for col in sample_cols:
        x = ranked[col].to_numpy(dtype=float)
        controls = [other for other in sample_cols if other != col]
        if len(ranked) < 3 or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
            prcc = np.nan
        elif controls:
            X = ranked[controls].to_numpy(dtype=float)
            x_resid = _residualize(x, X)
            y_resid = _residualize(y, X)
            prcc = _safe_corr(x_resid, y_resid)
        else:
            prcc = _safe_corr(x, y)
        rows.append({"parameter": col, "prcc": prcc, "target": target, "n_samples": int(len(ranked))})
    return pd.DataFrame(rows).sort_values("prcc", key=lambda s: s.abs(), ascending=False)


def _residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(controls)), controls])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
