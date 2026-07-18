"""Command line entry points for the Paper05 experiment engine."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper05.data.validation import write_validation_report
from paper05.experiments.registry import ensure_default_configs, run_all_configs
from paper05.experiments.run_matrix import run_matrix_file
from paper05.experiments.summarize import summarize_results


def main() -> None:
    parser = argparse.ArgumentParser(prog="paper05")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run_matrix = sub.add_parser("run-matrix")
    p_run_matrix.add_argument("config", type=Path)

    sub.add_parser("run-all")
    sub.add_parser("summarize")

    p_validate = sub.add_parser("validate-data")
    p_validate.add_argument("config", type=Path)

    p_init = sub.add_parser("init-configs")
    p_init.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.command == "run-matrix":
        run_matrix_file(args.config)
    elif args.command == "run-all":
        ensure_default_configs(overwrite=False)
        run_all_configs()
    elif args.command == "summarize":
        summarize_results()
    elif args.command == "validate-data":
        write_validation_report(args.config)
    elif args.command == "init-configs":
        ensure_default_configs(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
