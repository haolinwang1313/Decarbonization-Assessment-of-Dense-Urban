"""Rule-based node typology diagnostics."""

from __future__ import annotations

import pandas as pd


def assign_node_typology(node_metrics: pd.DataFrame) -> pd.DataFrame:
    df = node_metrics.copy()

    def classify(row) -> str:
        if row.get("node_binding_score", 0) >= 4 or row.get("SDI", 0) >= 2:
            return "storage-critical nodes"
        if row.get("RSI", 0) >= 0.6 and row.get("PV_potential_mw", 0) >= df["PV_potential_mw"].median():
            return "renewable-rich industrial nodes"
        if row.get("GDI", 0) >= 0.8 and row.get("peak_load_mw", 0) >= df["peak_load_mw"].median():
            return "load-dense import-dependent nodes"
        if row.get("wind_potential_mw", 0) > 0:
            return "wind-eligible peripheral nodes"
        return "mixed nodes"

    df["node_type"] = df.apply(classify, axis=1)
    return df


def typology_summary(typology: pd.DataFrame) -> pd.DataFrame:
    numeric = typology.select_dtypes("number").columns.tolist()
    return typology.groupby("node_type", as_index=False)[numeric].mean()
