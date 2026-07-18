"""QA checks used by tests and run finalization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pandas as pd


@dataclass
class CheckResult:
    passed: bool
    max_abs_error: float
    details: dict


def _series(dispatch: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in dispatch.columns:
            return pd.to_numeric(dispatch[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=dispatch.index, dtype=float)


def total_cost_matches_components(costs: dict[str, float], tolerance: float = 1e-6) -> bool:
    total = float(costs.get("total_system_cost", 0.0))
    parts = sum(float(v) for k, v in costs.items() if k != "total_system_cost")
    return math.isclose(total, parts, rel_tol=0.0, abs_tol=tolerance)


def valid_result(solver_success: bool, cap_violation_ton: float, load_shedding_mwh: float, tolerance: float = 1e-6) -> tuple[bool, str]:
    reasons = []
    if not solver_success:
        reasons.append("solver_failed")
    if cap_violation_ton > tolerance:
        reasons.append("co2_cap_violation")
    if load_shedding_mwh > tolerance:
        reasons.append("load_shedding")
    return len(reasons) == 0, ";".join(reasons)


def check_dispatch_balance(dispatch: pd.DataFrame, tolerance: float = 1e-6) -> CheckResult:
    supply = (
        _series(dispatch, "pv_kwh", "pv_gen")
        + _series(dispatch, "wind_kwh", "wind_gen")
        + _series(dispatch, "grid_import_kwh", "grid_import")
        + _series(dispatch, "ess_discharge_kwh", "bess_discharge")
        + _series(dispatch, "ldes_discharge_kwh", "ldes_discharge")
        + _series(dispatch, "load_shedding_kwh", "load_shedding")
        + _series(dispatch, "dr_neg_kwh")
        + _series(dispatch, "thermal_neg_kwh")
        + _series(dispatch, "ev_discharge_kwh", "ev_discharge")
        + _series(dispatch, "transfer_in_kwh")
    )
    demand = (
        _series(dispatch, "load_kwh")
        + _series(dispatch, "ess_charge_kwh", "bess_charge")
        + _series(dispatch, "ldes_charge_kwh", "ldes_charge")
        + _series(dispatch, "dr_pos_kwh")
        + _series(dispatch, "thermal_pos_kwh")
        + _series(dispatch, "ev_charge_kwh", "ev_charge")
        + _series(dispatch, "transfer_out_kwh")
    )
    residual = supply - demand
    max_abs = float(residual.abs().max()) if len(residual) else 0.0
    return CheckResult(passed=max_abs <= tolerance, max_abs_error=max_abs, details={"rows": int(len(dispatch))})


def validate_experiment_config(config: dict) -> list[str]:
    if _is_manifest_config(config):
        return _validate_manifest_config(config)

    errors: list[str] = []
    if "experiment_id" not in config:
        errors.append("missing experiment_id")
    defaults = config.get("defaults", {})
    if defaults and not isinstance(defaults, dict):
        errors.append("defaults must be a mapping")
    if "lhs_samples" in config and int(config["lhs_samples"]) <= 0:
        errors.append("lhs_samples must be positive")
    return errors


def _is_manifest_config(config: dict) -> bool:
    experiments = config.get("experiments")
    return isinstance(experiments, list) and bool(experiments) and all(isinstance(item, dict) and "file" in item for item in experiments)


def _validate_manifest_config(config: dict) -> list[str]:
    errors: list[str] = []
    if "matrix_id" not in config:
        errors.append("missing matrix_id")

    ids = [str(item.get("id")) for item in config.get("experiments", [])]
    if len(ids) != len(set(ids)):
        errors.append("duplicate experiment ids")

    for item in config.get("experiments", []):
        exp_id = str(item.get("id"))
        kind = str(item.get("kind", ""))
        file_path = str(item.get("file", ""))
        if kind not in {"matrix", "validation"}:
            errors.append(f"{exp_id}: invalid kind")
        if not file_path:
            errors.append(f"{exp_id}: missing file")
        elif Path(file_path).is_absolute() or "\\" in file_path:
            errors.append(f"{exp_id}: file must be repo-relative")
    return errors
