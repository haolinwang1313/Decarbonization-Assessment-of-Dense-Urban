"""Shared validity evaluation for run outputs and summary regeneration."""

from __future__ import annotations

import math
from typing import Any


DEFAULT_LOAD_SHEDDING_TOLERANCE_MWH = 1e-6
DEFAULT_CO2_TOLERANCE_TON = 1e-4
DEFAULT_COST_TOLERANCE_YUAN = 1e-4
DEFAULT_COST_RELATIVE_TOLERANCE = 1e-10

KNOWN_INVALID_REASON_TOKENS = {
    "solver_failed",
    "load_shedding",
    "co2_cap_violation",
    "cost_inconsistency",
    "objective_cost_mismatch",
    "ess_zero_capacity_nonzero_cost",
}


def split_invalid_reasons(reason: Any) -> list[str]:
    if reason is None:
        return []
    text = str(reason).strip()
    if not text or text.lower() == "nan":
        return []
    parts = [item.strip() for item in text.split(";") if item.strip()]
    if len(parts) <= 1:
        return parts
    if all(part in KNOWN_INVALID_REASON_TOKENS for part in parts):
        return parts
    return [text]


def objective_cost_tolerance(
    total_cost: float | None,
    solver_objective_value: float | None,
    *,
    cost_tolerance_yuan: float = DEFAULT_COST_TOLERANCE_YUAN,
    cost_relative_tolerance: float = DEFAULT_COST_RELATIVE_TOLERANCE,
) -> float:
    scale = max(abs(_coerce_float(total_cost)), abs(_coerce_float(solver_objective_value)), 1.0)
    return max(float(cost_tolerance_yuan), float(cost_relative_tolerance) * scale)


def evaluate_result_validity(
    *,
    solver_success: bool,
    objective_name: str,
    total_load_shedding_mwh: float | None,
    cap_violation_ton: float | None,
    total_cost: float | None,
    cost_component_sum_error: float | None,
    solver_objective_value: float | None,
    ess_zero_capacity_zero_cost: bool | None,
    load_shedding_tolerance_mwh: float = DEFAULT_LOAD_SHEDDING_TOLERANCE_MWH,
    co2_tolerance_ton: float = DEFAULT_CO2_TOLERANCE_TON,
    cost_tolerance_yuan: float = DEFAULT_COST_TOLERANCE_YUAN,
    cost_relative_tolerance: float = DEFAULT_COST_RELATIVE_TOLERANCE,
) -> dict[str, float | bool | str]:
    total_load_shedding_mwh = _coerce_float(total_load_shedding_mwh)
    cap_violation_ton = _coerce_float(cap_violation_ton)
    total_cost = _coerce_float(total_cost)
    cost_component_sum_error = _coerce_float(cost_component_sum_error)
    solver_objective_value = _coerce_float(solver_objective_value)
    objective_cost_consistency_error = 0.0
    objective_cost_consistency_rel_error = 0.0
    if objective_name == "min_cost" and math.isfinite(total_cost) and math.isfinite(solver_objective_value):
        objective_cost_consistency_error = abs(solver_objective_value - total_cost)
        objective_cost_consistency_rel_error = objective_cost_consistency_error / max(abs(solver_objective_value), abs(total_cost), 1.0)
    gap_tolerance = objective_cost_tolerance(
        total_cost,
        solver_objective_value,
        cost_tolerance_yuan=cost_tolerance_yuan,
        cost_relative_tolerance=cost_relative_tolerance,
    )

    reasons = []
    if not solver_success:
        reasons.append("solver_failed")
    if total_load_shedding_mwh > load_shedding_tolerance_mwh:
        reasons.append("load_shedding")
    if cap_violation_ton > co2_tolerance_ton:
        reasons.append("co2_cap_violation")
    if cost_component_sum_error > cost_tolerance_yuan:
        reasons.append("cost_inconsistency")
    if objective_name == "min_cost" and objective_cost_consistency_error > gap_tolerance:
        reasons.append("objective_cost_mismatch")
    if ess_zero_capacity_zero_cost is False:
        reasons.append("ess_zero_capacity_nonzero_cost")

    return {
        "total_load_shedding_mwh": total_load_shedding_mwh,
        "cap_violation_ton": cap_violation_ton,
        "load_shedding_tolerance_mwh": float(load_shedding_tolerance_mwh),
        "co2_tolerance_ton": float(co2_tolerance_ton),
        "cost_tolerance_yuan": float(cost_tolerance_yuan),
        "cost_relative_tolerance": float(cost_relative_tolerance),
        "cost_component_sum_error": cost_component_sum_error,
        "objective_cost_consistency_error": objective_cost_consistency_error,
        "objective_cost_consistency_rel_error": objective_cost_consistency_rel_error,
        "objective_cost_tolerance": gap_tolerance,
        "ess_zero_capacity_zero_cost": bool(ess_zero_capacity_zero_cost) if ess_zero_capacity_zero_cost is not None else False,
        "is_valid_result": len(reasons) == 0,
        "invalid_reason": ";".join(reasons),
    }


def _coerce_float(value: float | int | str | None) -> float:
    if value is None:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(result):
        return 0.0
    return result
