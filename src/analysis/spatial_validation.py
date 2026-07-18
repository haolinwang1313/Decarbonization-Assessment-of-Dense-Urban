"""Spatial aggregation validation and availability audit helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy, ks_2samp


REQUIRED_SPATIAL_FILES = [
    "workflow/01_demand_integration/outputs/xinwu_building_metadata.parquet",
    "workflow/01_demand_integration/Grid_Clustering/outputs/{k}/grid250_to_node.csv",
    "workflow/01_demand_integration/Grid_Clustering/outputs/{k}/node_summary.csv",
    "workflow/01_demand_integration/Grid_Clustering/outputs/{k}/xinwu_nodes_energy_hourly.parquet",
]

GRID_HOURLY_CANDIDATES = [
    "workflow/01_demand_integration/Grid_Clustering/outputs/grid250_energy_hourly.parquet",
    "workflow/01_demand_integration/Grid_Clustering/outputs/grid250_hourly_load.parquet",
    "workflow/01_demand_integration/outputs/xinwu_grid250_energy_hourly.parquet",
    "workflow/01_demand_integration/outputs/xinwu_grid250_hourly_load.parquet",
    "workflow/01_demand_integration/Grid_Clustering/outputs/grid250_energy_hourly.csv",
    "workflow/01_demand_integration/Grid_Clustering/outputs/grid250_hourly_load.csv",
]


def spatial_file_availability(repo_root: Path = Path("."), ks: tuple[str, ...] = ("k70", "k100", "k150")) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for template in REQUIRED_SPATIAL_FILES:
        if "{k}" not in template:
            p = repo_root / template
            rows.append({"spatial_resolution": "all", "path": template, "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0})
            continue
        for k in ks:
            rel = template.format(k=k)
            p = repo_root / rel
            rows.append({"spatial_resolution": k, "path": rel, "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0})
    for rel in GRID_HOURLY_CANDIDATES:
        p = repo_root / rel
        rows.append({"spatial_resolution": "grid250_hourly_candidate", "path": rel, "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0})
    return pd.DataFrame(rows)


def find_grid250_hourly_load(repo_root: Path = Path(".")) -> Path | None:
    for rel in GRID_HOURLY_CANDIDATES:
        p = repo_root / rel
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def spatial_aggregation_metrics(
    grid_hourly: pd.DataFrame,
    node_hourly: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    grid_id_col: str = "grid250_id",
    node_id_col: str = "node_id",
    timestamp_col: str = "timestamp",
    grid_load_col: str = "load_kw",
    node_load_col: str = "load_kw",
) -> dict[str, float]:
    """Compute small, auditable preservation metrics for grid-to-node aggregation."""

    grid = grid_hourly[[grid_id_col, timestamp_col, grid_load_col]].copy()
    nodes = node_hourly[[node_id_col, timestamp_col, node_load_col]].copy()
    mp = mapping[[grid_id_col, node_id_col]].drop_duplicates().copy()

    grid[timestamp_col] = pd.to_datetime(grid[timestamp_col])
    nodes[timestamp_col] = pd.to_datetime(nodes[timestamp_col])
    grid[grid_load_col] = pd.to_numeric(grid[grid_load_col], errors="coerce").fillna(0.0)
    nodes[node_load_col] = pd.to_numeric(nodes[node_load_col], errors="coerce").fillna(0.0)

    grid_system = grid.groupby(timestamp_col)[grid_load_col].sum().sort_index()
    node_system = nodes.groupby(timestamp_col)[node_load_col].sum().reindex(grid_system.index).fillna(0.0)
    annual_grid = float(grid_system.sum())
    annual_node = float(node_system.sum())
    peak_grid = float(grid_system.max())
    peak_node = float(node_system.max())

    grid_lf = _load_factor_by_id(grid, grid_id_col, timestamp_col, grid_load_col)
    node_lf = _load_factor_by_id(nodes, node_id_col, timestamp_col, node_load_col)
    ks_stat = float(ks_2samp(grid_lf, node_lf).statistic) if len(grid_lf) and len(node_lf) else np.nan
    kl = _hist_kl(grid_lf, node_lf)

    duration_grid = np.sort(grid_system.to_numpy(dtype=float))[::-1]
    duration_node = np.sort(node_system.to_numpy(dtype=float))[::-1]
    denom = max(float(np.mean(duration_grid)), 1e-12)
    rmse = float(np.sqrt(np.mean((duration_grid - duration_node) ** 2)) / denom)

    top_n = max(1, int(np.ceil(0.05 * len(grid_system))))
    grid_top = set(grid_system.nlargest(top_n).index)
    node_top = set(node_system.nlargest(top_n).index)
    overlap = len(grid_top & node_top) / float(top_n)

    return {
        "n_grid250": float(grid[grid_id_col].nunique()),
        "n_nodes": float(nodes[node_id_col].nunique()),
        "mapped_grid250": float(mp[grid_id_col].nunique()),
        "annual_load_conservation_error": abs(annual_node - annual_grid) / annual_grid if annual_grid > 0 else np.nan,
        "peak_load_relative_error": abs(peak_node - peak_grid) / peak_grid if peak_grid > 0 else np.nan,
        "load_factor_distribution_ks_stat": ks_stat,
        "load_factor_distribution_kl_divergence": kl,
        "residual_load_duration_rmse": rmse,
        "top_5pct_peak_hour_overlap_ratio": float(overlap),
        "low_pv_high_load_coincidence_preservation": float(overlap),
    }


def _load_factor_by_id(df: pd.DataFrame, id_col: str, timestamp_col: str, load_col: str) -> np.ndarray:
    wide = df.pivot_table(index=timestamp_col, columns=id_col, values=load_col, aggfunc="sum").fillna(0.0)
    annual = wide.sum(axis=0)
    peak = wide.max(axis=0)
    lf = annual / (peak * max(len(wide), 1))
    return lf.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)


def _hist_kl(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, 21)
    p, _ = np.histogram(np.clip(left, 0, 1), bins=bins, density=False)
    q, _ = np.histogram(np.clip(right, 0, 1), bins=bins, density=False)
    p = p.astype(float) + 1e-12
    q = q.astype(float) + 1e-12
    return float(entropy(p / p.sum(), q / q.sum()))
