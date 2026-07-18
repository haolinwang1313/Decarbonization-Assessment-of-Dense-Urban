# Solver Settings

This document records the optimization and validation settings used by the public reproducibility package.

## Solver

The public Python package uses SciPy `linprog` with the HiGHS method through the configured solver label `scipy-linprog-highs`. The exact SciPy version is resolved by `uv.lock`.

## Feasibility Screening

The model writes run-level QA checks including solver status, load-shedding amount, carbon-cap violation, objective-cost consistency, and storage-cost consistency. Load shedding is penalized through `load_shedding_penalty_yuan_per_kwh` in the run configuration.

## Reproducibility Checks

The public branch supports:

```powershell
uv run pytest -q
uv run python -m paper05 validate-data configs/datasets/xinwu_public.yaml
uv run python -m paper05 run-matrix configs/experiments/public_smoke.yaml
```

The public smoke configuration uses an open-source SciPy/HiGHS solve path and a short chronological slice to verify the runtime path.
