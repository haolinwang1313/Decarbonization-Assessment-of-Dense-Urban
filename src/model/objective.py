"""Cost helper functions."""

from __future__ import annotations


def annuity_factor(discount: float, years: float) -> float:
    if years <= 0:
        raise ValueError("years must be positive.")
    if discount == 0:
        return 1.0 / years
    return discount * (1.0 + discount) ** years / ((1.0 + discount) ** years - 1.0)


def annualized_capex(capex: float, discount: float, years: float) -> float:
    return capex * annuity_factor(discount, years)
