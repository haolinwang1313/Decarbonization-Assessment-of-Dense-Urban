# Reproducibility

Run all commands from the repository root.

## Environment

Create the Python 3.11-or-newer environment from the locked dependency set:

```powershell
uv sync --frozen --extra test
```

Run the public test suite:

```powershell
uv run pytest -q
```

## Input Validation

Validate schemas, ranges, temporal coverage, and file mappings for the processed Xinwu inputs:

```powershell
uv run python -m paper05 validate-data configs/datasets/xinwu_public.yaml
```

## Lightweight End-to-End Run

Exercise data loading, model construction, optimization, result writing, and QA checks with the short chronological configuration:

```powershell
uv run python -m paper05 run-matrix configs/experiments/public_smoke.yaml
```

## Complete Public Experiment Matrix

Execute the public representative-period, technology, emissions-cap, and network configuration matrix:

```powershell
uv run python -m paper05 run-matrix configs/experiments/revision_public.yaml
```

## Summary Generation

Build machine-readable summaries from generated run records:

```powershell
uv run python -m paper05 summarize
```

Generated run records are written under `results/runs/`, and generated summaries are written under `results/summaries/`. The accepted-study tables are available under `results/summary_public/reported/` for direct inspection.

## Integrity Records

`data/catalog.yaml` records checksums for released CSV tables. `docs/checksums_sha256.csv` records SHA-256 checksums and byte sizes for the remaining release files.
