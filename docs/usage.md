# Usage

Run all commands from the repository root.

## Prepare the Environment

```powershell
uv sync --frozen --extra test
```

## Validate Inputs

```powershell
uv run python -m paper05 validate-data configs/datasets/xinwu_public.yaml
```

## Run the Lightweight Configuration

```powershell
uv run python -m paper05 run-matrix configs/experiments/public_smoke.yaml
```

## Run the Complete Public Matrix

```powershell
uv run python -m paper05 run-matrix configs/experiments/revision_public.yaml
```

## Summarize Generated Runs

```powershell
uv run python -m paper05 summarize
```

## Run Tests

```powershell
uv run pytest -q
```

Generated run records and summaries are written below `results/`. Reported accepted-study tables are available in `results/summary_public/reported/`.
