"""Filtered density parameterization for periodic 3D design volumes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import SuperLU, splu

from pzid.grid3d import UniformGrid3D


ComplexArray: TypeAlias = NDArray[np.complex128]
RealArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class DensityMapping3D:
    """Intermediates required for the exact 3D density pullback."""

    raw: RealArray
    filtered: RealArray
    projected: RealArray
    epsilon_r: RealArray
    projection_derivative: RealArray
    epsilon_derivative: RealArray


@dataclass
class DensityParameterization3D:
    r"""Map a bounded periodic design volume to isotropic permittivity.

    The design occupies every transverse cell in the half-open axial range
    ``design_layers``.  A three-dimensional Helmholtz filter is periodic in
    ``x`` and ``y`` and has homogeneous-Neumann boundaries in ``z``.  Tanh
    projection and refractive-index interpolation follow the existing 2D
    implementation.
    """

    grid: UniformGrid3D
    design_layers: tuple[int, int]
    low_index: float
    high_index: float
    filter_radius: float
    projection_beta: float = 8.0
    projection_threshold: float = 0.5
    filter_operator: csc_matrix = field(init=False, repr=False)
    _filter_factorization: SuperLU = field(init=False, repr=False)

    def __post_init__(self) -> None:
        start, stop = self.design_layers
        if not 0 <= start < stop <= self.grid.nz:
            raise ValueError("design_layers must be a nonempty axial range")
        if not 0.0 < self.low_index < self.high_index:
            raise ValueError("indices must satisfy 0 < low_index < high_index")
        if self.filter_radius < 0.0:
            raise ValueError("filter_radius must be nonnegative")
        if self.projection_beta <= 0.0:
            raise ValueError("projection_beta must be positive")
        if not 0.0 < self.projection_threshold < 1.0:
            raise ValueError("projection_threshold must lie in (0, 1)")
        self.filter_operator = self._assemble_filter_operator()
        self._filter_factorization = splu(self.filter_operator)

    @property
    def design_shape(self) -> tuple[int, int, int]:
        start, stop = self.design_layers
        return stop - start, self.grid.ny, self.grid.nx

    @property
    def variable_count(self) -> int:
        return int(np.prod(self.design_shape))

    def _flat(self, iz: int, iy: int, ix: int) -> int:
        _, ny, nx = self.design_shape
        return (iz * ny + iy) * nx + ix

    def _assemble_filter_operator(self) -> csc_matrix:
        nz, ny, nx = self.design_shape
        size = self.variable_count
        diagonal = np.ones(size, dtype=np.complex128)
        rows: list[int] = []
        columns: list[int] = []
        data: list[complex] = []
        radius_squared = self.filter_radius**2

        def add_face(first: int, second: int, weight: float) -> None:
            diagonal[first] += weight
            diagonal[second] += weight
            rows.extend((first, second))
            columns.extend((second, first))
            data.extend((-weight, -weight))

        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    cell = self._flat(iz, iy, ix)
                    add_face(
                        cell,
                        self._flat(iz, iy, (ix + 1) % nx),
                        radius_squared / self.grid.dx**2,
                    )
                    add_face(
                        cell,
                        self._flat(iz, (iy + 1) % ny, ix),
                        radius_squared / self.grid.dy**2,
                    )
                    if iz + 1 < nz:
                        add_face(
                            cell,
                            self._flat(iz + 1, iy, ix),
                            radius_squared / self.grid.dz**2,
                        )
        indices = np.arange(size)
        rows.extend(indices.tolist())
        columns.extend(indices.tolist())
        data.extend(diagonal.tolist())
        return coo_matrix(
            (data, (rows, columns)), shape=(size, size), dtype=np.complex128
        ).tocsc()

    def _as_raw(self, raw: ArrayLike) -> RealArray:
        values = np.asarray(raw, dtype=float)
        if values.size != self.variable_count:
            raise ValueError(
                f"raw density must contain {self.variable_count} variables"
            )
        values = values.reshape(self.design_shape)
        if not np.all(np.isfinite(values)):
            raise ValueError("raw density must be finite")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("raw density must lie in [0, 1]")
        return values

    def map(
        self,
        raw: ArrayLike,
        *,
        background_epsilon: ArrayLike | float | None = None,
    ) -> DensityMapping3D:
        raw_values = self._as_raw(raw)
        filtered = np.asarray(
            self._filter_factorization.solve(raw_values.ravel())
        ).real.reshape(self.design_shape)
        beta = self.projection_beta
        threshold = self.projection_threshold
        denominator = np.tanh(beta * threshold) + np.tanh(
            beta * (1.0 - threshold)
        )
        shifted = beta * (filtered - threshold)
        projected = (
            np.tanh(beta * threshold) + np.tanh(shifted)
        ) / denominator
        projection_derivative = (
            beta * (1.0 - np.tanh(shifted) ** 2) / denominator
        )

        index_difference = self.high_index - self.low_index
        refractive_index = self.low_index + index_difference * projected
        design_epsilon = refractive_index**2
        epsilon_derivative = 2.0 * refractive_index * index_difference

        if background_epsilon is None:
            epsilon = np.full(self.grid.shape, self.low_index**2, dtype=float)
        else:
            background = np.asarray(background_epsilon, dtype=float)
            if background.ndim == 0:
                epsilon = np.full(self.grid.shape, background, dtype=float)
            elif background.shape == self.grid.shape:
                epsilon = background.copy()
            else:
                raise ValueError(
                    "background_epsilon must be scalar or match the full grid"
                )
        start, stop = self.design_layers
        epsilon[start:stop] = design_epsilon
        return DensityMapping3D(
            raw=raw_values,
            filtered=np.asarray(filtered),
            projected=np.asarray(projected),
            epsilon_r=epsilon,
            projection_derivative=np.asarray(projection_derivative),
            epsilon_derivative=np.asarray(epsilon_derivative),
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: DensityMapping3D,
    ) -> ComplexArray:
        gradient = np.asarray(epsilon_gradient, dtype=np.complex128)
        if gradient.shape[-3:] != self.grid.shape:
            raise ValueError("epsilon_gradient trailing axes must match the grid")
        leading_shape = gradient.shape[:-3]
        start, stop = self.design_layers
        local = gradient[..., start:stop, :, :].reshape(-1, self.variable_count)
        local = local * (
            mapping.epsilon_derivative * mapping.projection_derivative
        ).ravel()[np.newaxis, :]
        pulled = np.asarray(
            self._filter_factorization.solve(local.T, trans="T")
        ).T
        return pulled.reshape(*leading_shape, *self.design_shape)
