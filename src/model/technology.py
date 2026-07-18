"""Resolve technology-set-derived runtime defaults."""

from __future__ import annotations

from typing import Any, Mapping


def fraction_from_name(name: str, token: str) -> float:
    for suffix, value in [("05", 0.05), ("10", 0.10), ("15", 0.15)]:
        if f"{token}{suffix}" in name:
            return value
    return 0.0


def ev_penetration_from_name(name: str) -> str | None:
    if "EV_low" in name:
        return "low"
    if "EV_high" in name:
        return "high"
    if "EV_medium" in name or "EV" in name:
        return "medium"
    return None


def ev_mode_from_name(name: str, explicit_mode: Any = None) -> str:
    if explicit_mode in {"smart_charging", "smart_charging_plus_V2G", "disabled"}:
        return str(explicit_mode)
    if "V2G" in name:
        return "smart_charging_plus_V2G"
    if "EV" in name:
        return "smart_charging"
    return "disabled"


def technology_config(config: Mapping[str, Any]) -> dict[str, Any]:
    tech = str(config.get("technology_set", "base_corrected"))
    storage_model = str(config.get("storage_model", "liion_4h_with_degradation"))
    ev_mode = ev_mode_from_name(tech, config.get("ev_mode"))
    ev_penetration = str(config.get("ev_penetration") or ev_penetration_from_name(tech) or "medium")
    return {
        "pv": not config.get("disable_pv", False) and tech != "grid_only_reference",
        "wind": not config.get("disable_wind", False) and tech != "grid_only_reference",
        "bess": not config.get("disable_bess", False) and tech != "grid_only_reference",
        "ldes": "LDES" in tech or "LDES" in storage_model or bool(config.get("enable_ldes", False)),
        "dr_fraction": float(config.get("dr_fraction", fraction_from_name(tech, "DR"))),
        "thermal_fraction": float(config.get("thermal_flex_fraction", fraction_from_name(tech, "thermal"))),
        "ev_mode": ev_mode,
        "ev_penetration": ev_penetration,
        "network_mode": str(config.get("network_mode", "node_interface")),
        "transfer_cap_kw": float(config.get("transfer_cap_mw", 30.0)) * 1000.0,
        "bess_h_max": float(config.get("h_max_liion", 8.0 if "8h" in storage_model else 4.0)),
        "bess_eta_ch": float(config.get("bess_eta_ch", 0.95)),
        "bess_eta_dis": float(config.get("bess_eta_dis", 0.95)),
        "bess_deg_cost": float(config.get("bess_degradation_cost_yuan_per_kwh", 0.05 if "degradation" in storage_model or "storage_degradation" in tech else 0.0)),
        "bess_calendar_ageing_cost_yuan_per_kwh_year": float(config.get("bess_calendar_ageing_cost_yuan_per_kwh_year", 0.0)),
        "bess_degradation_mode": str(config.get("bess_degradation_mode", "throughput_plus_calendar")),
        "ldes_h_max": float(config.get("ldes_h_max", 24.0)),
        "ldes_eta_ch": float(config.get("ldes_eta_ch", 0.82)),
        "ldes_eta_dis": float(config.get("ldes_eta_dis", 0.82)),
        "ldes_power_capex": float(config.get("ldes_power_capex_yuan_per_kw", 1600.0)),
        "ldes_energy_capex": float(config.get("ldes_energy_capex_yuan_per_kwh", 240.0)),
        "ldes_lifetime": float(config.get("ldes_lifetime_years", 25.0)),
        "dr_window_h": int(config.get("dr_window_h", 4)),
        "thermal_window_h": int(config.get("thermal_window_h", 4)),
        "dr_ramp_fraction_per_h": float(config.get("dr_ramp_fraction_per_h", 0.05)),
        "thermal_ramp_fraction_per_h": float(config.get("thermal_ramp_fraction_per_h", 0.05)),
        "demand_response_cost": float(config.get("demand_response_cost_yuan_per_kwh", 0.02)),
        "thermal_flexibility_cost": float(config.get("thermal_flexibility_cost_yuan_per_kwh", 0.015)),
        "ev_flexibility_cost": float(config.get("ev_flexibility_cost_yuan_per_kwh", 0.01)),
        "ls_penalty": float(config.get("load_shedding_penalty_yuan_per_kwh", 1e5)),
    }
