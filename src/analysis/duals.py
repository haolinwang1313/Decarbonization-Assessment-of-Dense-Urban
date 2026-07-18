"""Dual/shadow-price post-processing."""

from __future__ import annotations

import pandas as pd


def binding_score(duals: pd.DataFrame) -> pd.DataFrame:
    if duals.empty:
        return pd.DataFrame(columns=["constraint_family", "binding_count", "max_shadow_price"])
    df = duals.copy()
    df["constraint_family"] = df["constraint"].astype(str).str.split(":").str[0]
    return (
        df.groupby("constraint_family", as_index=False)
        .agg(binding_count=("is_binding", "sum"), max_shadow_price=("shadow_price", "max"))
        .sort_values("binding_count", ascending=False)
    )
