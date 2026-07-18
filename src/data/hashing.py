"""Hash resolved configuration and source input files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def input_hash(config: dict, files: list[Path]) -> dict:
    existing = [p for p in files if p.exists()]
    return {
        "config_hash": stable_json_hash(config),
        "files": {str(p): file_sha256(p) for p in existing},
        "combined_hash": stable_json_hash(
            {
                "config": stable_json_hash(config),
                "files": {str(p): file_sha256(p) for p in existing},
            }
        ),
    }
