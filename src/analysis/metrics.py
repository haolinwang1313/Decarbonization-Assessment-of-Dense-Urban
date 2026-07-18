"""Analysis metrics shared by experiment summaries and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _series(dispatch: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in dispatch.columns:
            return pd.to_numeric(dispatch[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=dispatch.index, dtype=float)


def summarize_dispatch(dispatch: pd.DataFrame) -> dict[str, float]:
    weight = _series(dispatch, "weight")
    grid = _series(dispatch, "grid_import_kwh", "grid_import")
    pv = _series(dispatch, "pv_kwh", "pv_gen")
    wind = _series(dispatch, "wind_kwh", "wind_gen")
    ls = _series(dispatch, "load_shedding_kwh", "load_shedding")
    curtail = _series(dispatch, "curtailment_kwh")
    if "curtailment_kwh" not in dispatch.columns:
        curtail = _series(dispatch, "pv_curtailment_kwh", "pv_curtail") + _series(dispatch, "wind_curtailment_kwh", "wind_curtail")
    co2 = _series(dispatch, "co2_kg_per_kwh")
    return {
        "grid_import_kwh": float((grid * weight).sum()),
        "renewable_generation_kwh": float(((pv + wind) * weight).sum()),
        "load_shedding_kwh": float((ls * weight).sum()),
        "curtailment_kwh": float((curtail * weight).sum()),
        "co2_ton": float((grid * co2 * weight / 1000.0).sum()),
    }


def marginal_abatement_cost(frontier: pd.DataFrame, cost_col: str = "total_system_cost", emissions_col: str = "actual_emissions_ton") -> pd.DataFrame:
    df = frontier.sort_values(emissions_col, ascending=False).reset_index(drop=True)
    rows = []
    for i in range(len(df) - 1):
        e_i = float(df.loc[i, emissions_col])
        e_j = float(df.loc[i + 1, emissions_col])
        c_i = float(df.loc[i, cost_col])
        c_j = float(df.loc[i + 1, cost_col])
        d_e = e_i - e_j
        valid = d_e > 0
        rows.append(
            {
                "from_index": i,
                "to_index": i + 1,
                "emissions_reduction_ton": d_e,
                "cost_change_cny": c_j - c_i,
                "MAC_CNY_per_tCO2": (c_j - c_i) / d_e if valid else np.nan,
                "is_valid_mac": valid and (c_j - c_i) >= 0,
            }
        )
    return pd.DataFrame(rows)


def dual_based_mac(duals: pd.DataFrame) -> float | None:
    """
    Return the dual-based MAC from the binding CO2 cap inequality.

    For SciPy/HiGHS minimization with A_ub x <= b_ub, increasing the cap
    relaxes the constraint, so MAC = -shadow_price when the cap is binding.
    """

    if duals.empty or not {"constraint", "shadow_price", "is_binding"}.issubset(duals.columns):
        return None
    rows = duals[
        duals["constraint"].astype(str).eq("co2_cap")
        & duals["is_binding"].astype(str).str.lower().isin({"true", "1", "yes"})
    ].copy()
    if rows.empty:
        return None
    shadow = pd.to_numeric(rows["shadow_price"], errors="coerce").dropna()
    if shadow.empty:
        return None
    return float(max(0.0, -float(shadow.iloc[0])))


def reference_metrics(
    actual_emissions_ton: float,
    e_ref_grid_only_ton: float,
    e_lc_unconstrained_ton: float | None,
    co2_cap_ton: float | None,
    *,
    e_min_techset_ton: float | None = None,
    co2_cap_ratio: float | None = None,
) -> dict[str, float]:
    cap_violation = max(0.0, actual_emissions_ton - co2_cap_ton) if co2_cap_ton is not None else 0.0
    return {
        "E_ref_grid_only_ton": e_ref_grid_only_ton,
        "E_lc_unconstrained_ton": e_lc_unconstrained_ton if e_lc_unconstrained_ton is not None else np.nan,
        "E_min_techset_ton": e_min_techset_ton if e_min_techset_ton is not None else np.nan,
        "co2_cap_ratio": co2_cap_ratio if co2_cap_ratio is not None else (co2_cap_ton / e_ref_grid_only_ton if co2_cap_ton is not None and e_ref_grid_only_ton > 0 else np.nan),
        "co2_cap_ton": co2_cap_ton if co2_cap_ton is not None else np.nan,
        "actual_emissions_ton": actual_emissions_ton,
        "reduction_vs_grid_only": 1 - actual_emissions_ton / e_ref_grid_only_ton if e_ref_grid_only_ton > 0 else np.nan,
        "reduction_vs_unconstrained_lc": 1 - actual_emissions_ton / e_lc_unconstrained_ton if e_lc_unconstrained_ton and e_lc_unconstrained_ton > 0 else np.nan,
        "cap_violation_ton": cap_violation,
        "cap_violation_ratio": cap_violation / co2_cap_ton if co2_cap_ton and co2_cap_ton > 0 else 0.0,
    }
