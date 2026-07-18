# Data

This directory contains the machine-readable inventory and documentation for the released data tables.

## Contents

- `xinwu_public/`: processed Xinwu model inputs used by the public configurations.
- `catalog.yaml`: path, role, source category, license, size, and SHA-256 checksum for each released CSV table.
- `data_dictionary.md`: table-level fields, units, and modelling roles.

The dataset configuration `configs/datasets/xinwu_public.yaml` maps the model inputs to the committed CSV files. Run the following validation from the repository root:

```powershell
uv run python -m paper05 validate-data configs/datasets/xinwu_public.yaml
```

## License

The processed inputs, reported result tables, and public test fixture listed in `catalog.yaml` are available under the repository-wide MIT License.
