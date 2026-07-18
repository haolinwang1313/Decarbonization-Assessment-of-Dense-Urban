"""Linear storage throughput degradation helpers."""

from __future__ import annotations


def degradation_cost(discharge_kwh: float, cost_yuan_per_kwh: float) -> float:
    return float(discharge_kwh) * float(cost_yuan_per_kwh)
