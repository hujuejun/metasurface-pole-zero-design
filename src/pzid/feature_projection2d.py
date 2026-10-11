"""Periodic two-phase minimum-feature projection for a binary 2D contour.

The optical material remains binary. A raster pixel is an 8-nm-scale contour
element, not an intermediate refractive index; coarse FDFD permittivity is
obtained by area averaging the exact two-material indicator.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def periodic_disk_opening(mask: np.ndarray, *, cell_um: float, diameter_um: float) -> np.ndarray:
    """Open one phase by a Euclidean disk with periodic transverse boundary.

    Axial boundaries are not periodic. For metasurface use, assess the design
    window away from the absorbing-domain ends, not the exterior air padding.
    """
    phase = np.asarray(mask, dtype=bool)
    if phase.ndim != 2 or min(phase.shape) < 3:
        raise ValueError("mask must be a nontrivial 2D array")
    if cell_um <= 0 or diameter_um <= 0:
        raise ValueError("physical cell and feature diameter must be positive")
    nx = phase.shape[1]
    tiled = np.tile(phase, (1, 3))
    radius = diameter_um / 2.0
    eroded = distance_transform_edt(tiled, sampling=cell_um) >= radius
    opened = distance_transform_edt(~eroded, sampling=cell_um) <= radius
    return np.asarray(opened[:, nx:2 * nx] & phase)


def feature_opening_audit(
    solid: np.ndarray, *, cell_um: float, diameter_um: float,
    design_rows: tuple[int, int],
) -> dict[str, int | bool | float]:
    """Count fine pixels rejected by solid and air feature-size openings."""
    mask = np.asarray(solid, dtype=bool)
    start, stop = (int(value) for value in design_rows)
    if not 0 <= start < stop <= mask.shape[0]:
        raise ValueError("invalid axial design rows")
    inside = slice(start, stop)
    solid_removed = int(np.count_nonzero((mask & ~periodic_disk_opening(
        mask, cell_um=cell_um, diameter_um=diameter_um))[inside]))
    air = ~mask
    air_removed = int(np.count_nonzero((air & ~periodic_disk_opening(
        air, cell_um=cell_um, diameter_um=diameter_um))[inside]))
    return {
        "diameter_um": float(diameter_um),
        "contour_cell_um": float(cell_um),
        "solid_removed_pixels": solid_removed,
        "air_removed_pixels": air_removed,
        "passes_dual_phase_opening": bool(solid_removed == 0 and air_removed == 0),
    }


def project_dual_phase_minimum_feature(
    solid: np.ndarray, *, cell_um: float, diameter_um: float,
    design_rows: tuple[int, int], max_cycles: int = 6,
) -> tuple[np.ndarray, dict[str, int | bool | float]]:
    """Alternately remove undersize polymer and air tips to a fixed point.

    A successful return has zero solid *and* air opening violations in the
    physical design window. No topology promise is made here; callers must
    separately audit connectivity, venting, and load-bearing width.
    """
    if max_cycles < 1:
        raise ValueError("max_cycles must be positive")
    result = np.asarray(solid, dtype=bool).copy()
    initial = result.copy()
    for cycle in range(max_cycles):
        result = periodic_disk_opening(result, cell_um=cell_um, diameter_um=diameter_um)
        result = ~periodic_disk_opening(~result, cell_um=cell_um, diameter_um=diameter_um)
        audit = feature_opening_audit(
            result, cell_um=cell_um, diameter_um=diameter_um,
            design_rows=design_rows,
        )
        if audit["passes_dual_phase_opening"]:
            return result, {
                **audit,
                "cycles": cycle + 1,
                "changed_contour_pixels": int(np.count_nonzero(result != initial)),
            }
    raise RuntimeError("dual-phase 100-nm feature projection did not reach a fixed point")


def area_averaged_binary_permittivity(
    fine_solid: np.ndarray, *, coarse_shape: tuple[int, int],
    low_index: float, high_index: float,
) -> np.ndarray:
    """Cell-area fraction of a binary polymer/air contour on a coarse grid."""
    mask = np.asarray(fine_solid, dtype=bool)
    nz, nx = (int(value) for value in coarse_shape)
    if mask.shape[0] % nz or mask.shape[1] % nx:
        raise ValueError("fine binary contour must be an integer refinement")
    fz, fx = mask.shape[0] // nz, mask.shape[1] // nx
    fraction = mask.reshape(nz, fz, nx, fx).mean(axis=(1, 3))
    return low_index**2 + (high_index**2 - low_index**2) * fraction


def conservative_binary_permittivity(
    fine_solid: np.ndarray, *, target_shape: tuple[int, int],
    low_index: float, high_index: float,
) -> np.ndarray:
    """Integrate a fixed binary contour into any aligned 2D FDFD grid.

    The fine contour is interpreted as piecewise constant in physical space.
    Thus the material interface is not interpolated from a previous FDFD
    permittivity array, and noninteger mesh ratios preserve material area.
    """
    mask = np.asarray(fine_solid, dtype=bool)
    if mask.ndim != 2 or len(target_shape) != 2:
        raise ValueError("fine contour and target shape must be two dimensional")
    nz, nx = (int(value) for value in target_shape)
    if nz < 1 or nx < 1 or not 0.0 < low_index < high_index:
        raise ValueError("invalid target shape or refractive indices")
    if mask.shape[0] % nz == 0 and mask.shape[1] % nx == 0:
        return area_averaged_binary_permittivity(
            mask, coarse_shape=(nz, nx),
            low_index=low_index, high_index=high_index,
        )

    def average_axis(values: np.ndarray, target_cells: int, axis: int) -> np.ndarray:
        source_cells = values.shape[axis]
        boundaries = np.linspace(0.0, float(source_cells), target_cells + 1)
        whole = np.floor(boundaries).astype(int)
        fractional = boundaries - whole
        cumulative = np.concatenate((
            np.zeros_like(np.take(values, [0], axis=axis)),
            np.cumsum(values, axis=axis),
        ), axis=axis)
        shape = [1] * values.ndim
        shape[axis] = target_cells + 1
        integral = np.take(cumulative, whole, axis=axis) + (
            fractional.reshape(shape)
            * np.take(values, np.minimum(whole, source_cells - 1), axis=axis)
        )
        return np.diff(integral, axis=axis) / (source_cells / target_cells)

    fraction = average_axis(mask.astype(float), nz, axis=0)
    fraction = average_axis(fraction, nx, axis=1)
    if np.any(fraction < -1e-10) or np.any(fraction > 1.0 + 1e-10):
        raise RuntimeError("binary-contour integration left the physical bounds")
    fraction = np.clip(fraction, 0.0, 1.0)
    if abs(float(np.mean(fraction)) - float(np.mean(mask))) > 1e-11:
        raise RuntimeError("binary-contour integration did not conserve area")
    return low_index**2 + (high_index**2 - low_index**2) * fraction
