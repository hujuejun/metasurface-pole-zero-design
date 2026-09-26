"""Static structure visualization for three-dimensional unit cells."""

from __future__ import annotations

from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from pzid.grid3d import UniformGrid3D


AxesArray: TypeAlias = NDArray[np.object_]


def plot_refractive_index_slices_3d(
    epsilon_r: ArrayLike,
    grid: UniformGrid3D,
    *,
    slice_indices: tuple[int, int, int] | None = None,
    length_unit: str = "normalized length",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[Figure, AxesArray]:
    """Plot central ``xy``, ``xz``, and ``yz`` refractive-index sections."""

    epsilon = np.asarray(epsilon_r, dtype=np.complex128)
    if epsilon.shape != grid.shape:
        raise ValueError(f"epsilon_r must have shape {grid.shape}")
    if np.max(np.abs(epsilon.imag)) > 1e-12:
        raise ValueError("structure plotting requires effectively real epsilon")
    if np.any(epsilon.real <= 0.0):
        raise ValueError("epsilon_r must be strictly positive")
    if slice_indices is None:
        iz, iy, ix = grid.nz // 2, grid.ny // 2, grid.nx // 2
    else:
        iz, iy, ix = slice_indices
    if not (0 <= iz < grid.nz and 0 <= iy < grid.ny and 0 <= ix < grid.nx):
        raise ValueError("slice indices must lie inside the grid")

    refractive_index = np.sqrt(epsilon.real)
    shared_min = float(np.min(refractive_index)) if vmin is None else vmin
    shared_max = float(np.max(refractive_index)) if vmax is None else vmax
    figure, axes = plt.subplots(
        1, 3, figsize=(11.2, 3.5), constrained_layout=True
    )
    sections = (
        (
            refractive_index[iz],
            (0.0, grid.period_x, 0.0, grid.period_y),
            f"xy at z-index {iz}",
            f"x ({length_unit})",
            f"y ({length_unit})",
        ),
        (
            refractive_index[:, iy, :],
            (0.0, grid.period_x, 0.0, grid.height),
            f"xz at y-index {iy}",
            f"x ({length_unit})",
            f"z ({length_unit})",
        ),
        (
            refractive_index[:, :, ix],
            (0.0, grid.period_y, 0.0, grid.height),
            f"yz at x-index {ix}",
            f"y ({length_unit})",
            f"z ({length_unit})",
        ),
    )
    image = None
    for axis, (section, extent, title, xlabel, ylabel) in zip(
        axes, sections, strict=True
    ):
        image = axis.imshow(
            section,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            aspect="equal",
            cmap=cmap,
            vmin=shared_min,
            vmax=shared_max,
        )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
    assert image is not None
    figure.colorbar(image, ax=axes, label="Refractive index, n", shrink=0.86)
    return figure, np.asarray(axes, dtype=object)
