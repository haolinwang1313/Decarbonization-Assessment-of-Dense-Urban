"""Constraint label helpers."""

from __future__ import annotations


def label(kind: str, *parts: object) -> str:
    suffix = ":".join(str(p) for p in parts)
    return f"{kind}:{suffix}" if suffix else kind
