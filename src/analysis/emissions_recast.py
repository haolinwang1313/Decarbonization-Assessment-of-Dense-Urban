"""Absolute-emissions recast helpers for reviewer-facing evidence tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fixed_baseline_reference_ton(reference: pd.DataFrame) -> float:
    """Return the baseline grid-only reference emissions used as fixed denominator."""

    required = {"grid_carbon_scenario", "technology_set", "E_ref_grid_only_ton"}
    missing = required - set(reference.columns)
    if missing:
        raise ValueError(f"missing reference-emissions columns: {sorted(missing)}")

    mask = (
        reference["grid_carbon_scenario"].astype(str).str.lower().eq("baseline")
        & reference["technology_set"].astype(str).str.lower().eq("grid_only_reference")
    )
    candidates = pd.to_numeric(reference.loc[mask, "E_ref_grid_only_ton"], errors="coerce").dropna()
    if candidates.empty:
        raise ValueError("missing baseline grid_only_reference E_ref_grid_only_ton")
    value = float(candidates.iloc[0])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid fixed baseline reference emissions: {value}")
    return value


def add_absolute_emission_fields(
    df: pd.DataFrame,
    *,
    fixed_reference_ton: float,
    cap_col: str = "co2_cap_ton",
    emissions_col: str = "actual_emissions_ton",
    scenario_ratio_col: str = "co2_cap_ratio",
) -> pd.DataFrame:
    """Add absolute-emissions and fixed-baseline cap fields without changing rows."""

    if not np.isfinite(float(fixed_reference_ton)) or float(fixed_reference_ton) <= 0:
        raise ValueError("fixed_reference_ton must be positive and finite")

    out = df.copy()
    cap = pd.to_numeric(out[cap_col], errors="coerce") if cap_col in out.columns else pd.Series(np.nan, index=out.index)
    emissions = (
        pd.to_numeric(out[emissions_col], errors="coerce")
        if emissions_col in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    scenario_ratio = (
        pd.to_numeric(out[scenario_ratio_col], errors="coerce")
        if scenario_ratio_col in out.columns
        else pd.Series(np.nan, index=out.index)
    )

    out["co2_cap_mt"] = cap / 1_000_000.0
    out["actual_emissions_mt"] = emissions / 1_000_000.0
    out["scenario_relative_cap_ratio"] = scenario_ratio
    out["fixed_baseline_reference_ton"] = float(fixed_reference_ton)
    out["fixed_baseline_cap_ratio"] = cap / float(fixed_reference_ton)
    return out
