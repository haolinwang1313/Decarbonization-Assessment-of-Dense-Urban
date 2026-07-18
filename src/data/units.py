"""Unit conversion helpers."""

KW_PER_MW = 1000.0
KWH_PER_MWH = 1000.0
KG_PER_TON = 1000.0
HOURS_PER_YEAR = 8760.0


def kw_to_mw(value: float) -> float:
    return value / KW_PER_MW


def kwh_to_mwh(value: float) -> float:
    return value / KWH_PER_MWH
