"""Shared numerical definitions required by the published studies."""
from __future__ import annotations
import numpy as np

def _boundary(mask: np.ndarray, start: int, stop: int, protected_row: int) -> list[tuple[int, int]]:
    neighbor_solid = np.roll(mask, 1, axis=1) | np.roll(mask, -1, axis=1)
    neighbor_void = np.roll(~mask, 1, axis=1) | np.roll(~mask, -1, axis=1)
    neighbor_solid[1:] |= mask[:-1]
    neighbor_solid[:-1] |= mask[1:]
    neighbor_void[1:] |= ~mask[:-1]
    neighbor_void[:-1] |= ~mask[1:]
    boundary = mask & neighbor_void | ~mask & neighbor_solid
    boundary[:start] = False
    boundary[stop:] = False
    boundary[protected_row] = False
    return [(int(row), int(col)) for row, col in np.argwhere(boundary)]
