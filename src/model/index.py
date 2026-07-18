"""Variable indexing for sparse LP construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VarBlock:
    name: str
    shape: tuple[int, ...]
    start: int
    stop: int


class VarRegistry:
    def __init__(self) -> None:
        self.blocks: dict[str, VarBlock] = {}
        self.bounds: list[tuple[float | None, float | None]] = []

    @property
    def size(self) -> int:
        return len(self.bounds)

    def add(self, name: str, shape: tuple[int, ...], lb: float | np.ndarray | None = 0.0, ub: float | np.ndarray | None = None) -> VarBlock:
        count = int(np.prod(shape))
        block = VarBlock(name=name, shape=shape, start=self.size, stop=self.size + count)
        self.blocks[name] = block
        lb_arr = _as_array(lb, count)
        ub_arr = _as_array(ub, count)
        self.bounds.extend((float(lb_arr[i]) if lb_arr[i] is not None else None, float(ub_arr[i]) if ub_arr[i] is not None else None) for i in range(count))
        return block

    def idx(self, name: str, *pos: int) -> int:
        block = self.blocks[name]
        if not pos:
            offset = 0
        else:
            offset = int(np.ravel_multi_index(pos, block.shape))
        return block.start + offset

    def values(self, x: np.ndarray, name: str) -> np.ndarray:
        block = self.blocks[name]
        return x[block.start : block.stop].reshape(block.shape)

    def set_bounds(self, name: str, values: np.ndarray) -> None:
        block = self.blocks[name]
        flat = np.asarray(values, dtype=float).reshape(-1)
        if len(flat) != block.stop - block.start:
            raise ValueError(f"Bound array for {name} has the wrong size.")
        for i, value in enumerate(flat):
            self.bounds[block.start + i] = (float(value), float(value))

    def has(self, name: str) -> bool:
        return name in self.blocks


def _as_array(value: float | np.ndarray | None, count: int) -> np.ndarray:
    if value is None:
        return np.array([None] * count, dtype=object)
    arr = np.asarray(value, dtype=object)
    if arr.shape == ():
        return np.array([arr.item()] * count, dtype=object)
    return arr.reshape(-1)
