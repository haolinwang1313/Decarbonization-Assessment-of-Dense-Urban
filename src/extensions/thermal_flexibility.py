"""Thermal flexibility helpers."""

from __future__ import annotations


def is_heating_or_cooling_month(month: int) -> bool:
    return month in {1, 2, 6, 7, 8, 12}
