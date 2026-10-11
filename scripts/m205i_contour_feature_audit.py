"""Shared numerical definitions required by the published studies."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import distance_transform_edt, label
FOUR = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

def _spans_array(mask: np.ndarray) -> bool:
    nz, nx = mask.shape
    tiled = np.tile(mask, (1, 3))
    labels, _ = label(tiled, structure=FOUR)
    left = set((int(v) for v in np.unique(labels[:, 0]) if v))
    right = set((int(v) for v in np.unique(labels[:, 3 * nx - 1]) if v))
    return bool(left & right)

def _widest_spanning_width(mask: np.ndarray, cell_um: float) -> dict[str, float]:
    nx = mask.shape[1]
    distance = distance_transform_edt(np.tile(mask, (1, 3)), sampling=(cell_um, cell_um))
    central = distance[:, nx:2 * nx]
    if not _spans_array(mask):
        return {'spans_before_erosion': False, 'estimated_maximum_spanning_neck_um': 0.0}
    lo, hi = (0.0, float(np.max(central)))
    for _ in range(12):
        trial = (lo + hi) / 2
        if _spans_array(central >= trial):
            lo = trial
        else:
            hi = trial
    return {'spans_before_erosion': True, 'estimated_maximum_spanning_neck_um': float(2 * lo), 'eroded_radius_bracket_um': [float(lo), float(hi)], 'resolution_um': float(cell_um)}
