"""Run one resolved experiment case and write the required artifact set."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

from paper05.data.hashing import input_hash
from paper05.data.loaders import load_model_inputs, source_input_files, write_yaml
from paper05.data.schema import ModelInputs, ProjectPaths
from paper05.model.result import RunTables, extract_tables, write_tables
from paper05.model.solve import solve_model
from paper05.model.technology import technology_config
from paper05.model.validity import (
    DEFAULT_CO2_TOLERANCE_TON,
    DEFAULT_COST_RELATIVE_TOLERANCE,
    DEFAULT_COST_TOLERANCE_YUAN,
    DEFAULT_LOAD_SHEDDING_TOLERANCE_MWH,
)

_REFERENCE_CACHE: dict[str, dict[str, float]] = {}


def run_one(config: dict[str, Any], output_root: Path | None = None) -> dict[str, Any]:
    paths = ProjectPaths.discover()
    experiment_id = str(config["experiment_id"])
    run_id = str(config["run_id"])
    run_dir = (output_root or paths.results / "runs") / experiment_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = _normalize_config(dict(config))

    write_yaml(run_dir / "config_resolved.yaml", resolved)
    hash_payload = input_hash(resolved, source_input_files(resolved))
    (run_dir / "input_hash.json").write_text(json.dumps(hash_payload, indent=2), encoding="utf-8")

    start = time.perf_counter()
    try:
        _guard_problem_size(resolved)
        inputs = load_model_inputs(resolved)
        _apply_input_multipliers(inputs, resolved)
        build, solver_result, solver_runtime = solve_model(inputs, resolved)
        if not getattr(solver_result, "success", False):
            status = _solver_failure_status(resolved, solver_result, solver_runtime, hash_payload)
            _write_failure_tables(run_dir, resolved, status)
        else:
            reference_case, reference_case_error = _load_reference_case(inputs, resolved)
            tables = extract_tables(build, solver_result, inputs, solver_runtime, reference_case)
            if reference_case_error is not None:
                tables.qa_checks["reference_case_error"] = reference_case_error
            write_tables(run_dir, tables)
            status = _status_from_tables(resolved, solver_result, tables, solver_runtime, hash_payload, reference_case_error)
    except Exception as exc:
        status = _failure_status(resolved, exc, time.perf_counter() - start, hash_payload)
        _write_failure_tables(run_dir, resolved, status)

    (run_dir / "run_status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    (run_dir / "solver_summary.json").write_text(
        json.dumps(
            {
                "objective_name": status.get("objective_name"),
                "solver_name": status["solver_name"],
                "solver_status": status["solver_status"],
                "objective_value": status["objective_value"],
                "solver_objective_value": status.get("solver_objective_value"),
                "runtime_seconds": status["runtime_seconds"],
                "invalid_reason": status["invalid_reason"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return status


def _guard_problem_size(config: dict[str, Any]) -> None:
    mode = config.get("temporal_mode")
    network_mode = config.get("network_mode")
    # Full-year 100-node capacity optimization is too large for this local
    # SciPy sparse LP path in the current environment. The run is still recorded
    # as required instead of silently skipped.
    if mode == "full8760_opt" and not bool(config.get("allow_large_full8760_opt", False)):
        raise RuntimeError("failed_full8760_opt: disabled by local size guard for SciPy/HiGHS sparse LP path")
    if mode == "full8760_dispatch_validation" and not bool(config.get("allow_large_full8760_validation", False)):
        raise RuntimeError("failed_full8760_dispatch_validation: disabled by local size guard for SciPy/HiGHS sparse LP path")
    if network_mode == "spatial_transfer_proxy" and not bool(config.get("allow_large_spatial_transfer_proxy", False)):
        raise RuntimeError("failed_spatial_transfer_proxy: disabled by local size guard for SciPy/HiGHS sparse LP path")


def _apply_input_multipliers(inputs: ModelInputs, config: dict[str, Any]) -> None:
    if "demand_scale" in config:
        inputs.demand_kw *= float(config["demand_scale"])
    if "PV_potential_scale" in config:
        inputs.pv_cap_max_kw *= float(config["PV_potential_scale"])
    if "PV_capacity_factor_scale" in config:
        inputs.pv_cf = (inputs.pv_cf * float(config["PV_capacity_factor_scale"])).clip(0, 1)
    if "wind_potential_scale" in config:
        inputs.wind_cap_max_kw *= float(config["wind_potential_scale"])
    if "wind_capacity_factor_scale" in config:
        inputs.wind_cf = (inputs.wind_cf * float(config["wind_capacity_factor_scale"])).clip(0, 1)
    if "grid_existing_capacity_scale" in config:
        scale = float(config["grid_existing_capacity_scale"])
        inputs.grid_capacity["P_grid_existing_kW"] *= scale
        if "P_grid_max_kW" in inputs.grid_capacity:
            inputs.grid_capacity["P_grid_max_kW"] *= scale
    if "node_pv_potential_scale" in config:
        for node, scale in config["node_pv_potential_scale"].items():
            if node in inputs.pv_cap_max_kw.index:
                inputs.pv_cap_max_kw.loc[node] *= float(scale)
    if "node_wind_potential_scale" in config:
        for node, scale in config["node_wind_potential_scale"].items():
            if node in inputs.wind_cap_max_kw.index:
                inputs.wind_cap_max_kw.loc[node] *= float(scale)
    if "node_grid_capacity_scale" in config:
        for node, scale in config["node_grid_capacity_scale"].items():
            if node in inputs.grid_capacity.index:
                inputs.grid_capacity.loc[node, "P_grid_existing_kW"] *= float(scale)
                if "P_grid_max_kW" in inputs.grid_capacity.columns:
                    inputs.grid_capacity.loc[node, "P_grid_max_kW"] *= float(scale)
    if "grid_carbon_factor_scale" in config:
        inputs.grid_co2_kg_per_kwh *= float(config["grid_carbon_factor_scale"])
    if "TOU_price_scale" in config:
        inputs.price_yuan_per_kwh *= float(config["TOU_price_scale"])
    if "PV_capex" in config:
        inputs.pv_cost["capex_yuan_per_kW"] = float(config["PV_capex"])
    if "wind_capex" in config:
        inputs.wind_cost["capex_yuan_per_kW"] = float(config["wind_capex"])
    if "BESS_power_capex" in config:
        inputs.bess_cost["capex_power_yuan_per_kW"] = float(config["BESS_power_capex"])
    if "BESS_energy_capex" in config:
        inputs.bess_cost["capex_energy_yuan_per_kWh"] = float(config["BESS_energy_capex"])


def _status_from_tables(
    config: dict,
    solver_result: OptimizeResult,
    tables: RunTables,
    runtime: float,
    hash_payload: dict,
    reference_case_error: str | None,
) -> dict[str, Any]:
    qa = tables.qa_checks
    objective_name = str(config.get("objective", "min_cost"))
    solver_objective = float(getattr(solver_result, "fun", np.nan)) if getattr(solver_result, "fun", None) is not None else None
    objective_value: float | None
    if objective_name == "min_cost" and not tables.costs.empty:
        objective_value = float(tables.costs["total_system_cost"].iloc[0])
    else:
        objective_value = solver_objective
    status = {
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "dataset_id": config.get("dataset_id"),
        "spatial_resolution": config.get("spatial_resolution"),
        "temporal_mode": config.get("temporal_mode"),
        "technology_set": config.get("technology_set"),
        "grid_carbon_scenario": config.get("grid_carbon_scenario"),
        "co2_cap_ratio": config.get("co2_cap_ratio"),
        "objective_name": objective_name,
        "solver_name": config.get("solver_name", "scipy-linprog-highs"),
        "solver_status": "optimal" if getattr(solver_result, "success", False) else "failed",
        "objective_value": objective_value,
        "solver_objective_value": solver_objective,
        "is_valid_result": bool(qa["is_valid_result"]),
        "invalid_reason": qa["invalid_reason"],
        "runtime_seconds": runtime,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": hash_payload["combined_hash"],
    }
    if reference_case_error is not None:
        status["reference_case_error"] = reference_case_error
    return status


def _solver_failure_status(config: dict, solver_result: OptimizeResult, runtime: float, hash_payload: dict) -> dict[str, Any]:
    solver_objective = float(getattr(solver_result, "fun", np.nan)) if getattr(solver_result, "fun", None) is not None and np.isfinite(getattr(solver_result, "fun", np.nan)) else None
    message = str(getattr(solver_result, "message", "unknown solver failure"))
    return {
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "dataset_id": config.get("dataset_id"),
        "spatial_resolution": config.get("spatial_resolution"),
        "temporal_mode": config.get("temporal_mode"),
        "technology_set": config.get("technology_set"),
        "grid_carbon_scenario": config.get("grid_carbon_scenario"),
        "co2_cap_ratio": config.get("co2_cap_ratio"),
        "objective_name": str(config.get("objective", "min_cost")),
        "solver_name": config.get("solver_name", "scipy-linprog-highs"),
        "solver_status": "failed",
        "objective_value": solver_objective,
        "solver_objective_value": solver_objective,
        "is_valid_result": False,
        "invalid_reason": f"solver_failed: {message}" if message else "solver_failed",
        "runtime_seconds": runtime,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": hash_payload["combined_hash"],
    }


def _failure_status(config: dict, exc: Exception, runtime: float, hash_payload: dict) -> dict[str, Any]:
    return {
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "dataset_id": config.get("dataset_id"),
        "spatial_resolution": config.get("spatial_resolution"),
        "temporal_mode": config.get("temporal_mode"),
        "technology_set": config.get("technology_set"),
        "grid_carbon_scenario": config.get("grid_carbon_scenario"),
        "co2_cap_ratio": config.get("co2_cap_ratio"),
        "objective_name": str(config.get("objective", "min_cost")),
        "solver_name": config.get("solver_name", "scipy-linprog-highs"),
        "solver_status": "failed",
        "objective_value": None,
        "solver_objective_value": None,
        "is_valid_result": False,
        "invalid_reason": str(exc),
        "runtime_seconds": runtime,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": hash_payload["combined_hash"],
    }


def _write_failure_tables(run_dir: Path, config: dict[str, Any], status: dict[str, Any]) -> None:
    pd.DataFrame().to_csv(run_dir / "capacities.csv", index=False)
    pd.DataFrame().to_csv(run_dir / "emissions.csv", index=False)
    pd.DataFrame().to_csv(run_dir / "costs.csv", index=False)
    pd.DataFrame().to_csv(run_dir / "curtailment.csv", index=False)
    pd.DataFrame().to_csv(run_dir / "storage_cycles.csv", index=False)
    pd.DataFrame().to_csv(run_dir / "grid_import.csv", index=False)
    pd.DataFrame().to_csv(run_dir / "flexibility.csv", index=False)
    pd.DataFrame().to_csv(run_dir / "duals.csv", index=False)
    pd.DataFrame().to_csv(run_dir / "node_metrics.csv", index=False)
    pd.DataFrame().to_csv(run_dir / "network_edges.csv", index=False)
    dispatch_name = "dispatch_8760.csv" if str(config.get("temporal_mode", "repday_opt")).startswith("full8760") else "dispatch_repday.csv"
    pd.DataFrame().to_csv(run_dir / dispatch_name, index=False)
    (run_dir / "qa_checks.json").write_text(
        json.dumps(
            {
                "solver_success": False,
                "objective_name": str(config.get("objective", "min_cost")),
                "solver_objective_value": None,
                "is_valid_result": False,
                "invalid_reason": status["invalid_reason"],
                "total_load_shedding_mwh": None,
                "cap_violation_ton": None,
                "cap_violation_ratio": None,
                "load_shedding_tolerance_mwh": float(config.get("load_shedding_tolerance_mwh", DEFAULT_LOAD_SHEDDING_TOLERANCE_MWH)),
                "co2_tolerance_ton": float(config.get("co2_tolerance_ton", DEFAULT_CO2_TOLERANCE_TON)),
                "cost_tolerance_yuan": float(config.get("cost_tolerance_yuan", DEFAULT_COST_TOLERANCE_YUAN)),
                "cost_relative_tolerance": float(config.get("cost_relative_tolerance", DEFAULT_COST_RELATIVE_TOLERANCE)),
                "cost_component_sum_error": None,
                "objective_cost_consistency_error": None,
                "objective_cost_consistency_rel_error": None,
                "objective_cost_tolerance": None,
                "ess_zero_capacity_zero_cost": None,
                "reference_case_error": status.get("reference_case_error"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _reference_case(inputs: ModelInputs, config: dict[str, Any]) -> dict[str, float]:
    if config.get("compute_reference_context", True) is False:
        return {}

    key = _reference_cache_key(config)
    cached = _REFERENCE_CACHE.get(key)
    if cached is not None:
        return dict(cached)

    unc_cfg = _reference_variant(config, "min_cost")
    min_cfg = _reference_variant(config, "min_emissions")
    reference_case = {
        "E_lc_unconstrained_ton": _reference_emissions(inputs, unc_cfg),
        "E_min_techset_ton": _reference_emissions(inputs, min_cfg),
    }
    _REFERENCE_CACHE[key] = reference_case
    return dict(reference_case)


def _load_reference_case(inputs: ModelInputs, config: dict[str, Any]) -> tuple[dict[str, float], str | None]:
    try:
        return _reference_case(inputs, config), None
    except Exception as exc:
        return {}, str(exc)


def _reference_variant(config: dict[str, Any], objective: str) -> dict[str, Any]:
    variant = dict(config)
    variant.pop("experiment_id", None)
    variant.pop("run_id", None)
    variant.pop("co2_cap_ratio", None)
    variant.pop("co2_cap_ton", None)
    variant["objective"] = objective
    variant["compute_reference_context"] = False
    return variant


def _reference_emissions(inputs: ModelInputs, config: dict[str, Any]) -> float:
    build, solver_result, _ = solve_model(inputs, config)
    if not getattr(solver_result, "success", False):
        raise RuntimeError(f"failed_reference_{config['objective']}: {getattr(solver_result, 'message', 'unknown solver failure')}")
    grid = build.registry.values(np.asarray(solver_result.x, dtype=float), "grid_import")
    return float((pd.DataFrame(grid, index=build.demand.index).mul(build.co2, axis=0).mul(build.weights, axis=0) / 1000.0).sum().sum())


def _reference_cache_key(config: dict[str, Any]) -> str:
    excluded = {
        "experiment_id",
        "run_id",
        "co2_cap_ratio",
        "co2_cap_ton",
        "objective",
        "solver_name",
        "allow_large_full8760_opt",
        "allow_large_full8760_validation",
        "compute_reference_context",
    }
    payload = {key: value for key, value in config.items() if key not in excluded}
    return json.dumps(payload, sort_keys=True, default=str)


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    config.setdefault("dataset_id", "xinwu_k100")
    config.setdefault("spatial_resolution", str(config["dataset_id"]).split("_")[-1])
    config.setdefault("temporal_mode", "repday_opt")
    config.setdefault("technology_set", "base_corrected")
    config.setdefault("grid_carbon_scenario", "baseline")
    config.setdefault("co2_cap_ratio", None)
    config.setdefault("solver_name", "scipy-linprog-highs")
    config.setdefault("load_shedding_tolerance_mwh", DEFAULT_LOAD_SHEDDING_TOLERANCE_MWH)
    config.setdefault("co2_tolerance_ton", DEFAULT_CO2_TOLERANCE_TON)
    config.setdefault("cost_tolerance_yuan", DEFAULT_COST_TOLERANCE_YUAN)
    config.setdefault("cost_relative_tolerance", DEFAULT_COST_RELATIVE_TOLERANCE)
    if "BESS_degradation_cost" in config and "bess_degradation_cost_yuan_per_kwh" not in config:
        config["bess_degradation_cost_yuan_per_kwh"] = float(config["BESS_degradation_cost"])
    if "DR_fraction" in config and "dr_fraction" not in config:
        config["dr_fraction"] = float(config["DR_fraction"])

    tech_cfg = technology_config(config)
    if tech_cfg["ev_mode"] != "disabled":
        config.setdefault("ev_mode", tech_cfg["ev_mode"])
        config.setdefault("ev_penetration", tech_cfg["ev_penetration"])
    if tech_cfg["dr_fraction"] > 0:
        config.setdefault("dr_fraction", tech_cfg["dr_fraction"])
    if tech_cfg["thermal_fraction"] > 0:
        config.setdefault("thermal_flex_fraction", tech_cfg["thermal_fraction"])
    return config
