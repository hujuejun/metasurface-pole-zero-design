"""Topology checks for an extruded, laterally periodic two-phase print.

Periodic connected-component counts alone are insufficient: an isolated rod
straddling the chosen unit-cell boundary is one component on the periodic
quotient, but becomes a sequence of separate rods when the cell is tiled.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import distance_transform_edt, label

from pzid.fabrication import periodic_component_labels


_FOUR_NEIGHBORS = np.asarray(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool
)


def _validated_mask(mask: ArrayLike) -> NDArray[np.bool_]:
    result = np.asarray(mask, dtype=bool)
    if result.ndim != 2 or min(result.shape) < 3:
        raise ValueError("mask must be a 2D array with at least three cells per axis")
    return result


def fill_axially_enclosed_voids(mask: ArrayLike) -> NDArray[np.bool_]:
    """Fill void components that cannot reach either axial exterior face.

    Lateral boundaries are periodic for this operation. It deliberately does
    not require the top and bottom exterior void regions to connect to each
    other: a continuous polymer web separates them while leaving both vented.
    """

    solid = _validated_mask(mask).copy()
    labels, count = periodic_component_labels(~solid)
    for index in range(1, count + 1):
        if not (np.any(labels[0] == index) or np.any(labels[-1] == index)):
            solid[labels == index] = True
    return solid


def tiled_topology_audit(mask: ArrayLike, *, repeats: int = 5) -> dict[str, Any]:
    """Audit a finite unwrapped array, counting only central-cell defects.

    The odd repeat count keeps the measured cell away from artificial lateral
    ends. Four-neighbor connectivity is used for both phases. An open void
    must reach the top or bottom, not merely a distant lateral array edge.
    """

    solid = _validated_mask(mask)
    if repeats < 3 or repeats % 2 != 1:
        raise ValueError("repeats must be odd and at least three")
    nz, nx = solid.shape
    tiled = np.tile(solid, (1, repeats))
    solids, solid_count = label(tiled, structure=_FOUR_NEIGHBORS)
    voids, void_count = label(~tiled, structure=_FOUR_NEIGHBORS)
    middle = repeats // 2
    central = slice(middle * nx, (middle + 1) * nx)
    central_solid = sorted(int(i) for i in np.unique(solids[:, central]) if i)
    central_void = sorted(int(i) for i in np.unique(voids[:, central]) if i)

    # A genuine inter-period scaffold contains one component spanning the
    # middle three cells, not merely a rod crossing one chosen cell boundary.
    left_probe = (middle - 1) * nx
    right_probe = (middle + 2) * nx - 1
    across_left = set(int(i) for i in np.unique(solids[:, left_probe]) if i)
    across_right = set(int(i) for i in np.unique(solids[:, right_probe]) if i)
    spanning = sorted(across_left & across_right)
    unvented_sizes: list[int] = []
    through_z = False
    for index in central_void:
        reaches_bottom = bool(np.any(voids[0] == index))
        reaches_top = bool(np.any(voids[-1] == index))
        if not (reaches_bottom or reaches_top):
            unvented_sizes.append(int(np.count_nonzero(voids[:, central] == index)))
        if reaches_bottom and reaches_top:
            through_z = True

    return {
        "repeats": repeats,
        "total_solid_components": int(solid_count),
        "total_void_components": int(void_count),
        "central_solid_component_count": len(central_solid),
        "central_solid_component_labels": central_solid,
        "array_spanning_solid_component_count": len(spanning),
        "array_spanning_solid_component_labels": spanning,
        "central_unvented_void_count": len(unvented_sizes),
        "central_unvented_void_cell_counts": unvented_sizes,
        "axially_through_void_present": through_z,
        "passes_periodic_cell_topology": bool(
            len(central_solid) == 1
            and len(spanning) == 1
            and not unvented_sizes
            and not through_z
        ),
    }


def backbone_width_audit(
    mask: ArrayLike,
    *,
    dx_um: float,
    dz_um: float,
    minimum_neck_um: float,
) -> dict[str, Any]:
    """Check whether a solid path spans the array after neck-radius erosion.

    The distance transform is computed on three periodic copies, then the
    central copy is extracted before the unwrapped connectivity test. This is
    a conservative raster gate, not an elasticity or printer-voxel simulation.
    """

    solid = _validated_mask(mask)
    if min(dx_um, dz_um, minimum_neck_um) <= 0:
        raise ValueError("physical spacings and minimum neck must be positive")
    nx = solid.shape[1]
    tiled = np.tile(solid, (1, 3))
    distance = distance_transform_edt(tiled, sampling=(dz_um, dx_um))
    surviving = distance[:, nx : 2 * nx] >= minimum_neck_um / 2.0
    topology = tiled_topology_audit(surviving)
    # A surviving cross-cell web alone is not enough: every *substantial*
    # feature that survives neck-radius erosion must belong to that same web.
    # Thin non-load-bearing surface tips may vanish under erosion; they are
    # covered separately by the minimum-feature opening check.
    core_connected = bool(
        topology["central_solid_component_count"] == 1
        and topology["array_spanning_solid_component_count"] == 1
    )
    return {
        "minimum_neck_um": float(minimum_neck_um),
        "surviving_solid_cells": int(np.count_nonzero(surviving)),
        "eroded_scaffold_spans_array": bool(
            topology["array_spanning_solid_component_count"] >= 1
        ),
        "eroded_core_central_component_count": int(
            topology["central_solid_component_count"]
        ),
        "all_load_bearing_cores_connected": core_connected,
    }
