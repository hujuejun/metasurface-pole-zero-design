"""Publication-oriented static figures for structures and scalar channels."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import ArrayLike, NDArray

from pzid.grid import Grid2D, UniformGrid2D


RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _new_axis(
    ax: Axes | None,
    *,
    figsize: tuple[float, float],
) -> tuple[Figure, Axes]:
    if ax is None:
        figure, created_axis = plt.subplots(figsize=figsize, constrained_layout=True)
        return figure, created_axis
    return ax.figure, ax


def _as_complex_response(
    values: ArrayLike,
    shape: tuple[int, ...],
    *,
    name: str,
) -> ComplexArray:
    response = np.asarray(values, dtype=np.complex128)
    if response.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.any(np.isfinite(response)):
        raise ValueError(f"{name} must contain at least one finite value")
    return response


def plot_refractive_index(
    epsilon_r: ArrayLike,
    grid: Grid2D,
    *,
    ax: Axes | None = None,
    length_unit: str = "normalized length",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    design_rows: tuple[int, int] | None = None,
) -> tuple[Figure, Axes, AxesImage | QuadMesh]:
    """Plot the two-dimensional refractive-index profile ``n(x,z)``.

    The initial framework is lossless, so this function requires an effectively
    real, strictly positive relative permittivity. Pixel edges follow the
    cell-centered simulation grid exactly; no visual interpolation is applied.
    """

    epsilon = np.asarray(epsilon_r, dtype=np.complex128)
    if epsilon.shape != grid.shape:
        raise ValueError(f"epsilon_r must have shape {grid.shape}")
    if np.max(np.abs(epsilon.imag)) > 1e-12:
        raise ValueError("refractive-index plotting currently requires real epsilon")
    if np.any(epsilon.real <= 0.0):
        raise ValueError("epsilon_r must be strictly positive")
    refractive_index = np.sqrt(epsilon.real)

    figure, axis = _new_axis(ax, figsize=(6.2, 3.6))
    if isinstance(grid, UniformGrid2D):
        image: AxesImage | QuadMesh = axis.imshow(
            refractive_index,
            origin="lower",
            extent=(0.0, grid.period, 0.0, grid.height),
            interpolation="nearest",
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        image = axis.pcolormesh(
            np.arange(grid.nx + 1, dtype=float) * grid.dx,
            grid.z_edges,
            refractive_index,
            shading="flat",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        axis.set_aspect("equal")
    axis.set_xlabel(f"x ({length_unit})")
    axis.set_ylabel(f"z ({length_unit})")
    axis.set_title("Refractive-index profile")
    if design_rows is not None:
        start, stop = design_rows
        if not 0 <= start < stop <= grid.nz:
            raise ValueError("design_rows must lie inside the grid")
        for row in (start, stop):
            axis.axhline(
                grid.z_edges[row], color="white", linewidth=0.8, linestyle="--"
            )
    if colorbar:
        figure.colorbar(image, ax=axis, label="Refractive index, n")
    return figure, axis, image


def _scaled_magnitude_squared(
    response: ComplexArray,
    *,
    scale: Literal["linear", "db"],
    db_floor: float,
) -> RealArray:
    if scale not in {"linear", "db"}:
        raise ValueError("scale must be 'linear' or 'db'")
    if not np.isfinite(db_floor):
        raise ValueError("db_floor must be finite")
    magnitude_squared = np.abs(response) ** 2
    if scale == "linear":
        return np.asarray(magnitude_squared, dtype=float)
    floor = 10.0 ** (db_floor / 10.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scaled = 10.0 * np.log10(np.maximum(magnitude_squared, floor))
    finite = scaled[np.isfinite(scaled)]
    if finite.size:
        scaled = np.where(np.isposinf(scaled), np.max(finite), scaled)
    return np.asarray(scaled, dtype=float)


def _overlay_singularities(
    axis: Axes,
    poles: Sequence[complex],
    zeros: Sequence[complex],
) -> None:
    if poles:
        pole_values = np.asarray(poles, dtype=np.complex128)
        axis.scatter(
            pole_values.real,
            pole_values.imag,
            marker="x",
            s=42,
            linewidths=1.3,
            color="cyan",
            label="Pole",
        )
    if zeros:
        zero_values = np.asarray(zeros, dtype=np.complex128)
        axis.scatter(
            zero_values.real,
            zero_values.imag,
            marker="o",
            s=42,
            linewidths=1.2,
            facecolors="none",
            edgecolors="white",
            label="Channel zero",
        )
    if poles or zeros:
        axis.legend(loc="best", frameon=True)


def plot_complex_frequency_channels(
    real_omega: ArrayLike,
    imag_omega: ArrayLike,
    transmission: ArrayLike,
    reflection: ArrayLike,
    *,
    frequency_unit: str = "normalized angular frequency",
    scale: Literal["linear", "db"] = "db",
    db_floor: float = -80.0,
    vmin: float | None = None,
    vmax: float | None = None,
    poles: Sequence[complex] = (),
    transmission_zeros: Sequence[complex] = (),
    reflection_zeros: Sequence[complex] = (),
    cmap: str = "magma",
) -> tuple[Figure, NDArray[np.object_]]:
    r"""Plot analytic continuations of ``|t|^2`` and ``|r|^2`` over omega.

    Rows of the response arrays correspond to ``imag_omega`` and columns to
    ``real_omega``. These off-real-axis magnitudes locate singularities and are
    not physical transmittance or reflectance. Optional pole and scalar-zero
    markers are overlaid without altering the sampled data.
    """

    real_values = np.asarray(real_omega, dtype=float)
    imag_values = np.asarray(imag_omega, dtype=float)
    if real_values.ndim != 1 or imag_values.ndim != 1:
        raise ValueError("real_omega and imag_omega must be one-dimensional")
    if real_values.size < 2 or imag_values.size < 2:
        raise ValueError("both frequency axes require at least two samples")
    if (
        not np.all(np.isfinite(real_values))
        or not np.all(np.isfinite(imag_values))
        or np.any(np.diff(real_values) <= 0.0)
        or np.any(np.diff(imag_values) <= 0.0)
    ):
        raise ValueError("frequency axes must be finite and strictly increasing")
    shape = (imag_values.size, real_values.size)
    transmission_values = _as_complex_response(
        transmission, shape, name="transmission"
    )
    reflection_values = _as_complex_response(reflection, shape, name="reflection")
    maps = (
        _scaled_magnitude_squared(
            transmission_values, scale=scale, db_floor=db_floor
        ),
        _scaled_magnitude_squared(reflection_values, scale=scale, db_floor=db_floor),
    )
    finite_values = np.concatenate(
        [values[np.isfinite(values)] for values in maps]
    )
    if finite_values.size == 0:
        raise ValueError("scaled channel maps contain no finite values")
    shared_vmin = float(np.min(finite_values)) if vmin is None else vmin
    shared_vmax = float(np.max(finite_values)) if vmax is None else vmax

    figure, axes = plt.subplots(
        1, 2, figsize=(9.0, 3.8), sharex=True, sharey=True, constrained_layout=True
    )
    titles = (r"Transmission continuation, $|t|^2$", r"Reflection continuation, $|r|^2$")
    meshes = []
    for axis, values, title in zip(axes, maps, titles, strict=True):
        mesh = axis.pcolormesh(
            real_values,
            imag_values,
            values,
            shading="auto",
            cmap=cmap,
            vmin=shared_vmin,
            vmax=shared_vmax,
            rasterized=True,
        )
        meshes.append(mesh)
        axis.axhline(0.0, color="white", linewidth=0.7, alpha=0.8)
        axis.set_xlabel(rf"Re($\omega$) ({frequency_unit})")
        axis.set_title(title)
    axes[0].set_ylabel(rf"Im($\omega$) ({frequency_unit})")
    _overlay_singularities(axes[0], poles, transmission_zeros)
    _overlay_singularities(axes[1], poles, reflection_zeros)
    label = "Magnitude squared" if scale == "linear" else r"$10\log_{10}|s|^2$ (dB)"
    figure.colorbar(meshes[0], ax=axes, label=label, shrink=0.92)
    return figure, np.asarray(axes, dtype=object)


def plot_real_frequency_transmission_phase(
    omega: ArrayLike,
    transmission: ArrayLike,
    *,
    reflection: ArrayLike | None = None,
    frequency_unit: str = "normalized angular frequency",
    phase_unit: Literal["radians", "pi", "degrees"] = "pi",
    unwrap_phase: bool = True,
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot real-axis power transmittance and transmission phase."""

    frequency = np.asarray(omega, dtype=float)
    if frequency.ndim != 1 or frequency.size < 2:
        raise ValueError("omega must be a one-dimensional real-axis sample")
    if not np.all(np.isfinite(frequency)) or np.any(np.diff(frequency) <= 0.0):
        raise ValueError("omega must be finite and strictly increasing")
    if phase_unit not in {"radians", "pi", "degrees"}:
        raise ValueError("phase_unit must be 'radians', 'pi', or 'degrees'")
    transmission_values = _as_complex_response(
        transmission, frequency.shape, name="transmission"
    )
    reflection_values = None
    if reflection is not None:
        reflection_values = _as_complex_response(
            reflection, frequency.shape, name="reflection"
        )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.4),
        sharex=True,
        constrained_layout=True,
        height_ratios=(1.0, 1.0),
    )
    axes[0].plot(frequency, np.abs(transmission_values) ** 2, label="T", linewidth=1.7)
    if reflection_values is not None:
        axes[0].plot(
            frequency,
            np.abs(reflection_values) ** 2,
            label="R",
            linewidth=1.4,
            linestyle="--",
        )
        axes[0].legend(loc="best", frameon=False)
    axes[0].set_ylabel("Power coefficient")
    axes[0].set_title("Real-frequency transmission and phase")
    axes[0].grid(True, linewidth=0.4, alpha=0.35)

    phase = np.angle(transmission_values)
    if unwrap_phase:
        phase = np.unwrap(phase)
    if phase_unit == "pi":
        phase = phase / np.pi
        phase_label = r"Transmission phase / $\pi$"
    elif phase_unit == "degrees":
        phase = np.rad2deg(phase)
        phase_label = "Transmission phase (degrees)"
    else:
        phase_label = "Transmission phase (radians)"
    axes[1].plot(frequency, phase, linewidth=1.7)
    axes[1].set_xlabel(rf"$\omega$ ({frequency_unit})")
    axes[1].set_ylabel(phase_label)
    axes[1].grid(True, linewidth=0.4, alpha=0.35)
    return figure, np.asarray(axes, dtype=object)


def save_figure(
    figure: Figure,
    path: str | Path,
    *,
    dpi: int = 300,
    transparent: bool = False,
) -> Path:
    """Save a figure with stable metadata, creating its parent directory."""

    destination = Path(path)
    if not destination.suffix:
        raise ValueError("figure path must include a file extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        dpi=dpi,
        transparent=transparent,
        bbox_inches="tight",
        metadata={"Creator": "pzid"},
    )
    return destination
