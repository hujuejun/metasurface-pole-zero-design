"""Fabrication-continuation diagnostics for density-based metasurfaces."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


RealArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]


def projected_density_from_permittivity(
    epsilon_r: ArrayLike, low_index: float, high_index: float
) -> RealArray:
    """Recover the projected index-interpolation density from permittivity."""

    if not 0.0 < low_index < high_index:
        raise ValueError("indices must satisfy 0 < low_index < high_index")
    epsilon = np.asarray(epsilon_r, dtype=float)
    if np.any(epsilon < 0.0):
        raise ValueError("relative permittivity must be nonnegative")
    density = (np.sqrt(epsilon) - low_index) / (high_index - low_index)
    return np.clip(density, 0.0, 1.0)


def binarity_index(projected_density: ArrayLike) -> float:
    """Return zero for binary density and one for uniform half density."""

    density = np.asarray(projected_density, dtype=float)
    if np.any((density < 0.0) | (density > 1.0)):
        raise ValueError("projected density must lie within [0, 1]")
    return float(np.mean(4.0 * density * (1.0 - density)))


def gray_fraction(
    projected_density: ArrayLike, *, lower: float = 0.1, upper: float = 0.9
) -> float:
    """Return the fraction of density strictly between two thresholds."""

    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("gray thresholds must satisfy 0 <= lower < upper <= 1")
    density = np.asarray(projected_density, dtype=float)
    if np.any((density < 0.0) | (density > 1.0)):
        raise ValueError("projected density must lie within [0, 1]")
    return float(np.mean((density > lower) & (density < upper)))


def periodic_component_labels(
    material_mask: ArrayLike,
) -> tuple[IntegerArray, int]:
    """Label four-connected components with periodic transverse boundaries.

    The first axis is axial and is not periodic. The second axis is the
    periodic direction of the metasurface cell. Background has label zero.
    """

    mask = np.asarray(material_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("material_mask must be two-dimensional")
    labels = np.zeros(mask.shape, dtype=np.int64)
    component = 0
    nz, nx = mask.shape
    for seed_z, seed_x in np.argwhere(mask):
        if labels[seed_z, seed_x] != 0:
            continue
        component += 1
        labels[seed_z, seed_x] = component
        stack = [(int(seed_z), int(seed_x))]
        while stack:
            z_index, x_index = stack.pop()
            neighbors = (
                (z_index - 1, x_index),
                (z_index + 1, x_index),
                (z_index, (x_index - 1) % nx),
                (z_index, (x_index + 1) % nx),
            )
            for neighbor_z, neighbor_x in neighbors:
                if not 0 <= neighbor_z < nz:
                    continue
                if (
                    mask[neighbor_z, neighbor_x]
                    and labels[neighbor_z, neighbor_x] == 0
                ):
                    labels[neighbor_z, neighbor_x] = component
                    stack.append((neighbor_z, neighbor_x))
    return labels, component


def periodic_component_sizes(material_mask: ArrayLike) -> tuple[int, ...]:
    """Return descending component areas for a transverse-periodic mask."""

    labels, count = periodic_component_labels(material_mask)
    if count == 0:
        return ()
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    return tuple(int(value) for value in np.sort(sizes)[::-1])


def is_periodically_connected(material_mask: ArrayLike) -> bool:
    """Return whether all material belongs to exactly one periodic component."""

    mask = np.asarray(material_mask, dtype=bool)
    return bool(np.any(mask) and len(periodic_component_sizes(mask)) == 1)


def periodic_bilinear_resample(
    field: ArrayLike, target_shape: tuple[int, int]
) -> RealArray:
    """Resample a cell-centered axial-periodic field on the same physical cell.

    The second axis is periodic. The first axis uses constant edge extension;
    metasurface density is already zero at the axial computational boundaries.
    """

    source = np.asarray(field, dtype=float)
    if source.ndim != 2:
        raise ValueError("field must be two-dimensional")
    target_nz, target_nx = (int(value) for value in target_shape)
    if target_nz < 1 or target_nx < 1:
        raise ValueError("target dimensions must be positive")
    source_nz, source_nx = source.shape
    z_coordinate = (
        (np.arange(target_nz, dtype=float) + 0.5) * source_nz / target_nz
        - 0.5
    )
    x_coordinate = (
        (np.arange(target_nx, dtype=float) + 0.5) * source_nx / target_nx
        - 0.5
    )
    z_floor = np.floor(z_coordinate).astype(int)
    x0 = np.floor(x_coordinate).astype(int)
    wz = z_coordinate - z_floor
    wx = x_coordinate - x0
    z0 = np.clip(z_floor, 0, source_nz - 1)
    z1 = np.clip(z_floor + 1, 0, source_nz - 1)
    x0 %= source_nx
    x1 = (x0 + 1) % source_nx
    lower = (
        (1.0 - wx)[np.newaxis, :] * source[z0[:, np.newaxis], x0]
        + wx[np.newaxis, :] * source[z0[:, np.newaxis], x1]
    )
    upper = (
        (1.0 - wx)[np.newaxis, :] * source[z1[:, np.newaxis], x0]
        + wx[np.newaxis, :] * source[z1[:, np.newaxis], x1]
    )
    return (1.0 - wz)[:, np.newaxis] * lower + wz[:, np.newaxis] * upper


def periodic_bilinear_resample_adjoint(
    target_gradient: ArrayLike, source_shape: tuple[int, int]
) -> RealArray:
    """Apply the exact transpose of :func:`periodic_bilinear_resample`.

    Leading axes are retained so a stack of material sensitivities can be
    transferred from a validation grid to one shared physical design grid.
    """

    gradient = np.asarray(target_gradient)
    if gradient.ndim < 2:
        raise ValueError("target_gradient must have at least two dimensions")
    source_nz, source_nx = (int(value) for value in source_shape)
    if source_nz < 1 or source_nx < 1:
        raise ValueError("source dimensions must be positive")
    target_nz, target_nx = gradient.shape[-2:]
    z_coordinate = (
        (np.arange(target_nz, dtype=float) + 0.5) * source_nz / target_nz
        - 0.5
    )
    x_coordinate = (
        (np.arange(target_nx, dtype=float) + 0.5) * source_nx / target_nx
        - 0.5
    )
    z_floor = np.floor(z_coordinate).astype(int)
    x_floor = np.floor(x_coordinate).astype(int)
    wz = z_coordinate - z_floor
    wx = x_coordinate - x_floor
    z0 = np.clip(z_floor, 0, source_nz - 1)
    z1 = np.clip(z_floor + 1, 0, source_nz - 1)
    x0 = x_floor % source_nx
    x1 = (x0 + 1) % source_nx

    leading_shape = gradient.shape[:-2]
    flattened = gradient.reshape(-1, target_nz * target_nx)
    output = np.zeros(
        (flattened.shape[0], source_nz * source_nx), dtype=gradient.dtype
    )
    target_flat = np.arange(target_nz * target_nx)
    entries = (
        (
            z0[:, np.newaxis],
            x0[np.newaxis, :],
            (1.0 - wz)[:, np.newaxis] * (1.0 - wx)[np.newaxis, :],
        ),
        (
            z0[:, np.newaxis],
            x1[np.newaxis, :],
            (1.0 - wz)[:, np.newaxis] * wx[np.newaxis, :],
        ),
        (
            z1[:, np.newaxis],
            x0[np.newaxis, :],
            wz[:, np.newaxis] * (1.0 - wx)[np.newaxis, :],
        ),
        (
            z1[:, np.newaxis],
            x1[np.newaxis, :],
            wz[:, np.newaxis] * wx[np.newaxis, :],
        ),
    )
    for source_z, source_x, weight in entries:
        source_flat = np.broadcast_to(
            source_z * source_nx + source_x, (target_nz, target_nx)
        ).ravel()
        weighted = flattened[:, target_flat] * weight.ravel()[np.newaxis, :]
        for row in range(flattened.shape[0]):
            np.add.at(output[row], source_flat, weighted[row])
    return output.reshape(*leading_shape, source_nz, source_nx)


def binary_permittivity(
    projected_density: ArrayLike,
    threshold: float,
    low_index: float,
    high_index: float,
) -> RealArray:
    """Threshold a projected density and return a two-material permittivity."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly inside (0, 1)")
    density = np.asarray(projected_density, dtype=float)
    binary = density >= threshold
    return np.where(binary, high_index**2, low_index**2).astype(float)


def subpixel_binary_permittivity(
    source_density: ArrayLike,
    target_shape: tuple[int, int],
    threshold: float,
    low_index: float,
    high_index: float,
    *,
    subpixels: int = 8,
) -> tuple[RealArray, RealArray]:
    """Rasterize one fixed binary contour by subpixel area integration.

    ``source_density == threshold`` defines the mesh-independent material
    boundary.  The returned fill fraction is sampled on a uniformly refined
    version of each target cell; permittivity, rather than refractive index, is
    then area averaged for the cell-centered FDFD discretization.
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly inside (0, 1)")
    if not 0.0 < low_index < high_index:
        raise ValueError("indices must satisfy 0 < low_index < high_index")
    if int(subpixels) != subpixels or subpixels < 1:
        raise ValueError("subpixels must be a positive integer")
    target_nz, target_nx = (int(value) for value in target_shape)
    if target_nz < 1 or target_nx < 1:
        raise ValueError("target dimensions must be positive")
    refined = periodic_bilinear_resample(
        source_density,
        (target_nz * int(subpixels), target_nx * int(subpixels)),
    )
    mask = refined >= threshold
    fill_fraction = mask.reshape(
        target_nz, int(subpixels), target_nx, int(subpixels)
    ).mean(axis=(1, 3))
    low_epsilon = low_index**2
    epsilon = low_epsilon + (high_index**2 - low_epsilon) * fill_fraction
    return np.asarray(epsilon), np.asarray(fill_fraction)
