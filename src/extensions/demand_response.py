"""Demand response configuration helpers."""

from __future__ import annotations


def fraction_from_technology_set(name: str) -> float:
    for suffix, value in [("05", 0.05), ("10", 0.10), ("15", 0.15)]:
        if f"DR{suffix}" in name:
            return value
    return 0.0
