# Run Inventory

This file lists the primary public entry points.

| Config | Purpose | Workload | Output |
|---|---|---|---|
| `configs/experiments/public_smoke.yaml` | Lightweight end-to-end optimization check | Short chronological slice | `results/runs/` |
| `configs/experiments/revision_public.yaml` | Complete public envelope experiment matrix | Workload-dependent matrix | `results/runs/` |
| `configs/datasets/xinwu_public.yaml` | Processed Xinwu input mapping and validation | Dataset validation | Validation report |

Additional experiment-family configurations under `configs/experiments/` expose reference, temporal, storage, flexibility, spatial, uncertainty, bottleneck, and typology analyses.
