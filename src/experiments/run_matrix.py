"""Run experiment matrices from YAML configuration files."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pandas as pd

from paper05.data.loaders import read_yaml
from paper05.data.validation import write_validation_report
from paper05.analysis.sensitivity import lhs_samples
from paper05.experiments.run_one import run_one


def expand_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    base = dict(config.get("defaults", {}))
    experiment_id = config["experiment_id"]
    matrix = config.get("matrix", {})
    extra_runs = config.get("extra_runs", [])
    lhs = config.get("lhs_samples")
    if not matrix:
        runs = [base]
    else:
        keys = list(matrix.keys())
        runs = []
        for combo in itertools.product(*[matrix[k] for k in keys]):
            item = dict(base)
            item.update(dict(zip(keys, combo)))
            runs.append(item)
    for item in extra_runs:
        run = dict(base)
        run.update(dict(item))
        runs.append(run)
    if lhs:
        samples = lhs_samples(int(lhs))
        expanded = []
        for run in runs:
            for sample_id, row in samples.iterrows():
                item = dict(run)
                item.update(row.to_dict())
                item["lhs_sample_id"] = int(sample_id)
                expanded.append(item)
        runs = expanded
    for i, run in enumerate(runs, start=1):
        run["experiment_id"] = experiment_id
        run.setdefault("run_id", _run_id(run, i))
    return runs


def run_matrix_file(path: Path) -> list[dict[str, Any]]:
    cfg = read_yaml(path)
    if _is_manifest_config(cfg):
        return _run_manifest_file(path, cfg)

    statuses = []
    for run_cfg in expand_matrix(cfg):
        statuses.append(run_one(run_cfg))
    out = Path("results") / "summaries" / f"{cfg['experiment_id']}_run_status.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(statuses).to_csv(out, index=False)
    return statuses


def expand_manifest(config: dict[str, Any], base_dir: Path) -> list[dict[str, Any]]:
    entries = []
    for item in config.get("experiments", []):
        entry = dict(item)
        entry["path"] = base_dir / str(item["file"])
        entries.append(entry)
    return entries


def _run_id(config: dict[str, Any], index: int) -> str:
    parts = [
        f"r{index:04d}",
        str(config.get("spatial_resolution", "k100")),
        str(config.get("temporal_mode", "repday")),
        str(config.get("technology_set", "base")),
        str(config.get("grid_carbon_scenario", "baseline")),
    ]
    if config.get("co2_cap_ratio") is not None:
        parts.append(f"cap{float(config['co2_cap_ratio']):.2f}")
    if config.get("network_mode"):
        parts.append(str(config["network_mode"]))
    if config.get("storage_model"):
        parts.append(str(config["storage_model"]))
    if config.get("lhs_sample_id") is not None:
        parts.append(f"lhs{int(config['lhs_sample_id']):03d}")
    return "_".join(p.replace("/", "-") for p in parts)


def _is_manifest_config(config: dict[str, Any]) -> bool:
    experiments = config.get("experiments")
    return isinstance(experiments, list) and bool(experiments) and all(isinstance(item, dict) and "file" in item for item in experiments)


def _run_manifest_file(path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_id = str(config.get("matrix_id", path.stem))
    statuses: list[dict[str, Any]] = []
    for entry in expand_manifest(config, path.parent):
        if str(entry.get("kind", "matrix")) == "validation":
            write_validation_report(entry["path"])
            statuses.append(
                {
                    "manifest_id": manifest_id,
                    "manifest_entry_id": entry["id"],
                    "manifest_entry_kind": entry["kind"],
                    "config_file": entry["file"],
                    "experiment_id": entry["id"],
                    "run_id": "validation",
                    "solver_status": "not_applicable",
                    "objective_value": None,
                    "is_valid_result": True,
                    "invalid_reason": None,
                }
            )
            continue

        for status in run_matrix_file(entry["path"]):
            item = dict(status)
            item.setdefault("manifest_id", manifest_id)
            item.setdefault("manifest_entry_id", entry["id"])
            item.setdefault("manifest_entry_kind", entry.get("kind", "matrix"))
            item.setdefault("config_file", entry["file"])
            statuses.append(item)

    out = Path("results") / "summaries" / f"{manifest_id}_run_status.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(statuses).to_csv(out, index=False)
    return statuses
