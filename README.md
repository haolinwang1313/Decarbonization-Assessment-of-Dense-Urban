# Conditional cost–emission envelopes for decarbonizing dense urban power systems under spatial renewable scarcity and flexibility constraints

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.apenergy.2026.128401-blue.svg)](https://doi.org/10.1016/j.apenergy.2026.128401)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-3776AB.svg)](https://www.python.org/)
[![Journal](https://img.shields.io/badge/Journal-Applied%20Energy-2f855a.svg)](https://doi.org/10.1016/j.apenergy.2026.128401)

This repository accompanies the accepted Applied Energy article **"Conditional cost–emission envelopes for decarbonizing dense urban power systems under spatial renewable scarcity and flexibility constraints"**. It provides the research software, processed Xinwu model inputs, experiment configurations, validation tests, machine-readable result tables, and Supplementary Information used for reproducible analysis.

## Publication

**Conditional cost–emission envelopes for decarbonizing dense urban power systems under spatial renewable scarcity and flexibility constraints**

[*Applied Energy*](https://doi.org/10.1016/j.apenergy.2026.128401)

2026-11 | Journal article

DOI: [10.1016/j.apenergy.2026.128401](https://doi.org/10.1016/j.apenergy.2026.128401)

Authors: Haolin Wang; Zhi Wu; Wei Gu; Pengxiang Liu; Qirun Sun; Wei Wang

## Supplementary Information

The final author-provided Supplementary Information associated with the accepted article is archived here because it was inadvertently omitted from the journal's online article page.

- [Download the Supplementary Information (PDF)](supplementary_material.pdf)

## Overview

The workflow evaluates conditional cost–emission envelopes for a dense urban power system under spatial renewable limits and multiple flexibility options. It combines processed node-level demand and renewable-potential inputs with capacity-expansion, dispatch, temporal-representation, storage, flexibility, uncertainty, and network-abstraction configurations.

The repository supports four reproducibility activities:

- inspect the model, assumptions, schemas, and reported tables;
- validate the processed Xinwu input package;
- run a lightweight end-to-end optimization check;
- execute the complete public experiment matrix and summarize generated outputs.

## Repository Structure

```text
configs/                     Dataset and experiment configurations
data/xinwu_public/           Processed Xinwu model inputs
data/catalog.yaml            Machine-readable inventory and checksums
docs/                        Provenance, schemas, parameters, and run guides
results/summary_public/      Reported machine-readable result tables
src/                        Python package root (installed as `paper05`)
├── analysis/               Metrics, sensitivity, and spatial analysis
├── data/                   Input loading, schemas, hashing, and validation
├── experiments/            Experiment registry, execution, and summaries
├── extensions/             Flexibility and network extensions
├── model/                  Optimization model and result extraction
├── temporal/               Chronological and representative-day utilities
├── __main__.py             Module entry point
└── cli.py                  Command-line interface
tests/                       Public unit and contract tests
supplementary_material.pdf   Supplementary Information for the associated article
CITATION.cff                 Citation metadata
LICENSE                      MIT license
pyproject.toml               Package and dependency metadata
uv.lock                      Locked Python environment
```

## Dependencies and Installation

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required. From the repository root, create the locked environment:

```powershell
uv sync --frozen --extra test
```

## Usage

Validate the processed input package:

```powershell
uv run python -m paper05 validate-data configs/datasets/xinwu_public.yaml
```

Run the lightweight end-to-end configuration:

```powershell
uv run python -m paper05 run-matrix configs/experiments/public_smoke.yaml
```

Run the complete public experiment matrix:

```powershell
uv run python -m paper05 run-matrix configs/experiments/revision_public.yaml
```

Summarize generated outputs:

```powershell
uv run python -m paper05 summarize
```

Run the public test suite:

```powershell
uv run pytest -q
```

See `docs/usage.md` and `docs/reproducibility.md` for the complete command sequence and output locations.

## Data

Processed model inputs are stored in `data/xinwu_public/`. Their schemas are documented in `docs/input_schema.md`, and `data/catalog.yaml` records the role, size, SHA-256 checksum, and license of each released CSV table.

## Results

Reported machine-readable tables are stored in `results/summary_public/reported/`. `docs/results_schema.md` describes their analytical categories, including frontier, storage, flexibility, uncertainty, temporal, spatial, network, and marginal-abatement-cost outputs.

## Citation

If this repository or workflow is useful in your research, please cite:

```bibtex
@article{Wang2026Conditional,
  title   = {Conditional cost--emission envelopes for decarbonizing dense urban power systems under spatial renewable scarcity and flexibility constraints},
  author  = {Wang, Haolin and Wu, Zhi and Gu, Wei and Liu, Pengxiang and Sun, Qirun and Wang, Wei},
  journal = {Applied Energy},
  year    = {2026},
  doi     = {10.1016/j.apenergy.2026.128401},
  url     = {https://doi.org/10.1016/j.apenergy.2026.128401}
}
```

## License

All content in this repository, including the Supplementary Information, code, documentation, processed data, and result tables, is available under the [MIT License](LICENSE).
