"""Experiment package public API."""

from __future__ import annotations

from paper05.experiments.run_one import run_one as _run_one


def run_one(config: dict, dry_run: bool = False):
    if dry_run:
        exp = config.get("experiment", config)
        defaults = config.get("defaults", {})
        return {
            "experiment_id": exp.get("id", exp.get("experiment_id")),
            "runner": defaults.get("runner", "paper05.experiments.run_one"),
            "solver": defaults.get("model", {}).get("solver", config.get("solver_name", "highs")),
            "status": "planned",
        }
    return _run_one(config)
