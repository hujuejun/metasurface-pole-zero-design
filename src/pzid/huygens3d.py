"""Geometry and channel diagnostics for a symmetric Huygens nanodisk cell.

The helpers in this module are deliberately independent of the optimizer.  A
fixed physical geometry can therefore be replayed on several meshes before it
is admitted to an adjoint run.  The cylinder is represented by deterministic
subcell quadrature and linear volume averaging of permittivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pzid.grid3d import UniformGrid3D


RealArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]


@dataclass(frozen=True)
class SmoothNanodiskMapping3D:
    """Differentiable nanodisk geometry and its physical-coordinate Jacobian."""

    raw: RealArray
    epsilon_r: RealArray
    epsilon_derivative: RealArray
    radius: float
    height: float
    disk_index: float


@dataclass(frozen=True)
class SmoothNanodiskParameterization3D:
    """Map three unit-box variables to radius, height, and disk index.

    A mesh-scale smooth Heaviside supplies a differentiable boundary for the
    physical-coordinate adjoint audit.  Each coordinate is normalized to
    ``[0, 1]`` so it can use the repository's bounded LM implementation.
    """

    grid: UniformGrid3D
    host_index: float
    radius_range: tuple[float, float]
    height_range: tuple[float, float]
    disk_index_range: tuple[float, float]
    transverse_boundary_width: float
    axial_boundary_width: float

    def __post_init__(self) -> None:
        for name, bounds in (
            ("radius", self.radius_range),
            ("height", self.height_range),
            ("disk index", self.disk_index_range),
        ):
            if len(bounds) != 2 or not 0.0 < bounds[0] < bounds[1]:
                raise ValueError(f"{name}_range must be increasing and positive")
        if not self.host_index < self.disk_index_range[0]:
            raise ValueError("disk-index range must exceed the host index")
        if min(self.transverse_boundary_width, self.axial_boundary_width) <= 0.0:
            raise ValueError("boundary widths must be positive")
        if self.radius_range[1] >= 0.5 * min(
            self.grid.period_x, self.grid.period_y
        ):
            raise ValueError("maximum disk radius must remain inside the cell")

    @property
    def design_shape(self) -> tuple[int]:
        return (3,)

    @property
    def variable_count(self) -> int:
        return 3

    @staticmethod
    def _smooth_step(value: RealArray) -> tuple[RealArray, RealArray]:
        tangent = np.tanh(0.5 * value)
        return 0.5 * (1.0 + tangent), 0.25 * (1.0 - tangent**2)

    def map(
        self,
        raw: ArrayLike,
        *,
        background_epsilon: ArrayLike | float | None = None,
    ) -> SmoothNanodiskMapping3D:
        values = np.asarray(raw, dtype=float).reshape(-1)
        if values.shape != (3,) or np.any(~np.isfinite(values)):
            raise ValueError("raw nanodisk coordinates must contain three finite values")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("raw nanodisk coordinates must lie in [0, 1]")
        if background_epsilon is not None:
            raise ValueError("the smooth nanodisk uses its declared homogeneous host")

        ranges = (
            self.radius_range,
            self.height_range,
            self.disk_index_range,
        )
        physical = np.asarray(
            [low + value * (high - low) for value, (low, high) in zip(values, ranges, strict=True)]
        )
        scales = np.asarray([high - low for low, high in ranges])
        radius, height, disk_index = physical
        cx = 0.5 * self.grid.period_x
        cy = 0.5 * self.grid.period_y
        cz = 0.5 * self.grid.height
        dx = (self.grid.x - cx + 0.5 * self.grid.period_x) % self.grid.period_x
        dx -= 0.5 * self.grid.period_x
        dy = (self.grid.y - cy + 0.5 * self.grid.period_y) % self.grid.period_y
        dy -= 0.5 * self.grid.period_y
        radial = np.sqrt(dy[:, np.newaxis] ** 2 + dx[np.newaxis, :] ** 2)
        transverse, dtransverse = self._smooth_step(
            (radius - radial) / self.transverse_boundary_width
        )
        axial_distance = np.abs(self.grid.z - cz)
        axial, daxial = self._smooth_step(
            (0.5 * height - axial_distance) / self.axial_boundary_width
        )
        # Compactly truncate only the numerically negligible far tail.  Exact
        # zero support is required in source/monitor/PML cells; eight smoothing
        # widths remains well separated from every admissible disk interface.
        axial_support = axial_distance <= (
            0.5 * self.height_range[1] + 8.0 * self.axial_boundary_width
        )
        axial = np.where(axial_support, axial, 0.0)
        daxial = np.where(axial_support, daxial, 0.0)
        fill = axial[:, np.newaxis, np.newaxis] * transverse[np.newaxis, :, :]
        contrast = disk_index**2 - self.host_index**2
        epsilon = self.host_index**2 + contrast * fill

        derivative = np.empty((3, *self.grid.shape), dtype=float)
        derivative[0] = (
            contrast
            * scales[0]
            * axial[:, np.newaxis, np.newaxis]
            * dtransverse[np.newaxis, :, :]
            / self.transverse_boundary_width
        )
        derivative[1] = (
            contrast
            * scales[1]
            * transverse[np.newaxis, :, :]
            * daxial[:, np.newaxis, np.newaxis]
            / (2.0 * self.axial_boundary_width)
        )
        derivative[2] = 2.0 * disk_index * scales[2] * fill
        return SmoothNanodiskMapping3D(
            raw=np.asarray(values),
            epsilon_r=np.asarray(epsilon),
            epsilon_derivative=derivative,
            radius=float(radius),
            height=float(height),
            disk_index=float(disk_index),
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: SmoothNanodiskMapping3D,
    ) -> ComplexArray:
        gradient = np.asarray(epsilon_gradient, dtype=np.complex128)
        if gradient.shape[-3:] != self.grid.shape:
            raise ValueError("epsilon_gradient trailing axes must match the grid")
        return np.asarray(
            np.einsum(
                "...zyx,qzyx->...q",
                gradient,
                mapping.epsilon_derivative,
            )
        )


@dataclass(frozen=True)
class EmbeddedNanodisk3D:
    """One circular high-index disk in a homogeneous embedding medium."""

    radius: float
    height: float
    disk_index: float
    host_index: float
    center_x: float | None = None
    center_y: float | None = None
    center_z: float | None = None
    transverse_supersampling: int = 8

    def __post_init__(self) -> None:
        if self.radius <= 0.0 or self.height <= 0.0:
            raise ValueError("disk radius and height must be positive")
        if not 0.0 < self.host_index < self.disk_index:
            raise ValueError("indices must satisfy 0 < host_index < disk_index")
        if self.transverse_supersampling < 1:
            raise ValueError("transverse_supersampling must be positive")

    def fill_fraction(self, grid: UniformGrid3D) -> RealArray:
        """Return the disk volume fraction in every Cartesian cell."""

        cx = 0.5 * grid.period_x if self.center_x is None else self.center_x
        cy = 0.5 * grid.period_y if self.center_y is None else self.center_y
        cz = 0.5 * grid.height if self.center_z is None else self.center_z
        if not 0.0 <= cx <= grid.period_x or not 0.0 <= cy <= grid.period_y:
            raise ValueError("disk center must lie in the transverse unit cell")
        if not 0.0 <= cz <= grid.height:
            raise ValueError("disk center must lie in the axial grid")

        samples = self.transverse_supersampling
        offsets = (np.arange(samples, dtype=float) + 0.5) / samples - 0.5
        x_samples = grid.x[:, np.newaxis] + grid.dx * offsets[np.newaxis, :]
        y_samples = grid.y[:, np.newaxis] + grid.dy * offsets[np.newaxis, :]
        # Minimum-image distances retain a centered cylinder if a future
        # parameter sweep approaches a transverse periodic boundary.
        dx = (x_samples - cx + 0.5 * grid.period_x) % grid.period_x
        dx -= 0.5 * grid.period_x
        dy = (y_samples - cy + 0.5 * grid.period_y) % grid.period_y
        dy -= 0.5 * grid.period_y
        radial_squared = (
            dy[:, :, np.newaxis, np.newaxis] ** 2
            + dx[np.newaxis, np.newaxis, :, :] ** 2
        )
        transverse = np.mean(
            radial_squared <= self.radius**2, axis=(1, 3), dtype=float
        )

        lower = cz - 0.5 * self.height
        upper = cz + 0.5 * self.height
        cell_lower = grid.z - 0.5 * grid.dz
        cell_upper = grid.z + 0.5 * grid.dz
        axial = np.maximum(
            0.0, np.minimum(cell_upper, upper) - np.maximum(cell_lower, lower)
        ) / grid.dz
        return np.asarray(
            axial[:, np.newaxis, np.newaxis]
            * transverse[np.newaxis, :, :],
            dtype=float,
        )

    def epsilon_r(self, grid: UniformGrid3D) -> RealArray:
        """Return volume-averaged relative permittivity on ``grid``."""

        fill = self.fill_fraction(grid)
        return np.asarray(
            self.host_index**2
            + fill * (self.disk_index**2 - self.host_index**2),
            dtype=float,
        )


def copolarized_coefficients(
    scattering_matrix: ArrayLike,
    *,
    polarization: str = "x",
) -> tuple[complex, complex]:
    """Return lower-incidence co-polarized reflection and transmission."""

    matrix = np.asarray(scattering_matrix, dtype=np.complex128)
    if matrix.shape != (4, 4):
        raise ValueError("scattering_matrix must have shape (4, 4)")
    if polarization == "x":
        lower, upper = 0, 2
    elif polarization == "y":
        lower, upper = 1, 3
    else:
        raise ValueError("polarization must be 'x' or 'y'")
    return complex(matrix[lower, lower]), complex(matrix[upper, lower])


def axial_eigenchannels(
    scattering_matrix: ArrayLike,
    *,
    polarization: str = "x",
) -> tuple[complex, complex]:
    """Return symmetric and antisymmetric axial scattering eigenchannels.

    The result assumes an axially mirror-symmetric reciprocal cell.  The
    symmetry defect should be checked independently before these scalar
    combinations are interpreted as modal-parity channels.
    """

    matrix = np.asarray(scattering_matrix, dtype=np.complex128)
    if matrix.shape != (4, 4):
        raise ValueError("scattering_matrix must have shape (4, 4)")
    if polarization == "x":
        lower, upper = 0, 2
    elif polarization == "y":
        lower, upper = 1, 3
    else:
        raise ValueError("polarization must be 'x' or 'y'")
    block = matrix[np.ix_((lower, upper), (lower, upper))]
    symmetric = 0.5 * (
        block[0, 0] + block[1, 1] + block[0, 1] + block[1, 0]
    )
    antisymmetric = 0.5 * (
        block[0, 0] + block[1, 1] - block[0, 1] - block[1, 0]
    )
    return complex(symmetric), complex(antisymmetric)


def axial_symmetry_defect(
    scattering_matrix: ArrayLike,
    *,
    polarization: str = "x",
) -> float:
    """Return the maximum reciprocity/mirror-symmetry defect of one block."""

    matrix = np.asarray(scattering_matrix, dtype=np.complex128)
    if polarization == "x":
        indices = (0, 2)
    elif polarization == "y":
        indices = (1, 3)
    else:
        raise ValueError("polarization must be 'x' or 'y'")
    block = matrix[np.ix_(indices, indices)]
    reference = 0.5 * (block[0, 0] + block[1, 1])
    exchange = 0.5 * (block[0, 1] + block[1, 0])
    return float(
        max(
            abs(block[0, 0] - reference),
            abs(block[1, 1] - reference),
            abs(block[0, 1] - exchange),
            abs(block[1, 0] - exchange),
        )
    )
