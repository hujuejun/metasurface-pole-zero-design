"""Filtered and projected density parameterization for periodic TM cells."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import SuperLU, splu

from pzid.fabrication import (
    periodic_bilinear_resample,
    periodic_bilinear_resample_adjoint,
)
from pzid.grid import UniformGrid2D


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


@dataclass(frozen=True)
class DensityMapping:
    """Intermediate fields needed for an exact parameterization pullback."""

    raw: RealArray
    filtered: RealArray
    projected: RealArray
    epsilon_r: RealArray
    projection_derivative: RealArray
    epsilon_derivative: RealArray


@dataclass(frozen=True)
class NestedTwoTierDensityMapping:
    """Intermediates for two nested, laterally patterned etch heights."""

    lower: DensityMapping
    upper_control: DensityMapping
    projected_lower: RealArray
    projected_upper_control: RealArray
    projected_upper_material: RealArray
    upper_epsilon_derivative: RealArray
    epsilon_r: RealArray


@dataclass(frozen=True)
class SeparatedTwoLayerDensityMapping:
    """Intermediates for two independently patterned separated layers."""

    lower: DensityMapping
    upper: DensityMapping
    epsilon_r: RealArray


@dataclass(frozen=True)
class PairedRibDensityMapping:
    """Intermediates for one lateral mask copied to mirrored rib layers."""

    lower: DensityMapping
    upper: DensityMapping
    epsilon_r: RealArray


@dataclass(frozen=True)
class PairedPhaseLayerDensityMapping:
    """Intermediates for paired-sheet density plus a centered phase film."""

    base: object
    phase_index: float
    phase_thickness: float
    coverage: RealArray
    index_epsilon_derivative: RealArray
    thickness_epsilon_derivative: RealArray
    epsilon_r: RealArray


@dataclass(frozen=True)
class PairedPatternedPhaseLayerDensityMapping:
    """Intermediates for paired sheets plus an independent central mask."""

    base: object
    pattern: DensityMapping
    projected_pattern: RealArray
    phase_index: float
    phase_thickness: float
    coverage: RealArray
    index_epsilon_derivative: RealArray
    thickness_epsilon_derivative: RealArray
    pattern_epsilon_derivative: RealArray
    epsilon_r: RealArray


@dataclass(frozen=True)
class PairedNestedRibDensityMapping:
    """Intermediates for two nested rib tiers mirrored about a fixed slab."""

    inner: DensityMapping
    outer_control: DensityMapping
    projected_inner: RealArray
    projected_outer_control: RealArray
    projected_outer_material: RealArray
    outer_epsilon_derivative: RealArray
    epsilon_r: RealArray


@dataclass(frozen=True)
class CrossGridPairedNestedReliefDensityMapping:
    """Intermediates for one physical two-tier relief on a target mesh."""

    inner: DensityMapping
    outer_control: DensityMapping
    projected_inner: RealArray
    projected_outer_control: RealArray
    projected_outer_material: RealArray
    outer_epsilon_derivative: RealArray
    epsilon_r: RealArray


@dataclass(frozen=True)
class PairedNestedRibSlabExpansionDensityMapping:
    """Intermediates for a nested rib map plus symmetric slab expansion."""

    base: PairedNestedRibDensityMapping
    expansion_fraction: float
    interface_rows: tuple[int, int]
    epsilon_r: RealArray


@dataclass(frozen=True)
class PairedNestedRibSlabContractionDensityMapping:
    """Intermediates for a nested rib map plus symmetric slab contraction."""

    base: PairedNestedRibDensityMapping
    contraction_fraction: float
    slab_interface_rows: tuple[int, int]
    rib_interface_rows: tuple[int, int]
    epsilon_r: RealArray


@dataclass(frozen=True)
class DBRTerminationContractionDensityMapping:
    """Intermediates for symmetric contraction of an innermost DBR low layer."""

    base: object
    contraction_fraction: float
    interface_rows: tuple[int, int]
    epsilon_r: RealArray


@dataclass(frozen=True)
class SymmetricDBRTerminationShiftDensityMapping:
    """Intermediates for a signed shift of the innermost DBR interface."""

    base: object
    shift_cells: float
    low_interface_rows: tuple[int, int]
    high_interface_rows: tuple[int, int]
    shift_epsilon_derivative: RealArray
    epsilon_r: RealArray


@dataclass
class DensityParameterization:
    r"""Map bounded raw densities to a material distribution.

    The raw variables occupy all transverse cells in the half-open axial range
    ``design_rows``. Filtering solves

    ``(I - filter_radius**2 * Laplacian) rho_filtered = rho``

    with periodic transverse and homogeneous-Neumann axial boundaries. A tanh
    projection is followed by linear interpolation of refractive index and
    squaring to obtain relative permittivity.
    """

    grid: UniformGrid2D
    design_rows: tuple[int, int]
    low_index: float
    high_index: float
    filter_radius: float
    projection_beta: float = 8.0
    projection_threshold: float = 0.5
    filter_operator: csc_matrix = field(init=False, repr=False)
    _filter_factorization: SuperLU = field(init=False, repr=False)

    def __post_init__(self) -> None:
        start, stop = self.design_rows
        if not 0 <= start < stop <= self.grid.nz:
            raise ValueError("design_rows must be a nonempty range inside the grid")
        if not 0.0 < self.low_index < self.high_index:
            raise ValueError("indices must satisfy 0 < low_index < high_index")
        if self.filter_radius < 0.0:
            raise ValueError("filter_radius must be nonnegative")
        if self.projection_beta <= 0.0:
            raise ValueError("projection_beta must be positive")
        if not 0.0 < self.projection_threshold < 1.0:
            raise ValueError("projection_threshold must lie strictly inside (0, 1)")
        self.filter_operator = self._assemble_filter_operator()
        self._filter_factorization = splu(self.filter_operator)

    @property
    def design_shape(self) -> tuple[int, int]:
        start, stop = self.design_rows
        return stop - start, self.grid.nx

    @property
    def variable_count(self) -> int:
        rows, columns = self.design_shape
        return rows * columns

    def _flat_index(self, iz: int, ix: int) -> int:
        return iz * self.grid.nx + ix

    def _assemble_filter_operator(self) -> csc_matrix:
        nz, nx = self.design_shape
        size = nz * nx
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
            for ix in range(nx):
                cell = self._flat_index(iz, ix)
                neighbor = self._flat_index(iz, (ix + 1) % nx)
                add_face(cell, neighbor, radius_squared / self.grid.dx**2)
                if iz < nz - 1:
                    neighbor = self._flat_index(iz + 1, ix)
                    add_face(cell, neighbor, radius_squared / self.grid.dz**2)

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
            raise ValueError("raw density must lie within [0, 1]")
        return values

    def map(
        self,
        raw: ArrayLike,
        *,
        background_epsilon: ArrayLike | float | None = None,
    ) -> DensityMapping:
        """Return the full-grid permittivity and chain-rule intermediates."""

        raw_values = self._as_raw(raw)
        filtered = np.asarray(
            self._filter_factorization.solve(raw_values.ravel()), dtype=np.complex128
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
        projection_derivative = beta * (1.0 - np.tanh(shifted) ** 2) / denominator

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
        start, stop = self.design_rows
        epsilon[start:stop] = design_epsilon
        return DensityMapping(
            raw=raw_values,
            filtered=filtered,
            projected=np.asarray(projected),
            epsilon_r=epsilon,
            projection_derivative=np.asarray(projection_derivative),
            epsilon_derivative=np.asarray(epsilon_derivative),
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: DensityMapping,
    ) -> ComplexArray:
        """Apply material, projection, and transpose-filter chain rules."""

        gradient = np.asarray(epsilon_gradient, dtype=np.complex128)
        if gradient.shape[-2:] != self.grid.shape:
            raise ValueError("epsilon_gradient trailing axes must match the grid")
        leading_shape = gradient.shape[:-2]
        start, stop = self.design_rows
        local = gradient[..., start:stop, :].reshape(-1, self.variable_count)
        local = local * (
            mapping.epsilon_derivative * mapping.projection_derivative
        ).ravel()[np.newaxis, :]
        pulled = np.asarray(
            self._filter_factorization.solve(local.T, trans="T")
        ).T
        return pulled.reshape(*leading_shape, *self.design_shape)


@dataclass
class AxiallySymmetricDensityParameterization:
    """Reduce a centered design region to exact axial-mirror variables.

    The wrapped parameterization performs the filter, projection, and index
    interpolation. This adapter expands one independent axial half before
    mapping and sums mirrored sensitivities during the pullback.
    """

    base: DensityParameterization

    def __post_init__(self) -> None:
        start, stop = self.base.design_rows
        if start != self.base.grid.nz - stop:
            raise ValueError("base design_rows must be centered on the axial grid")

    @property
    def grid(self) -> UniformGrid2D:
        return self.base.grid

    @property
    def design_rows(self) -> tuple[int, int]:
        return self.base.design_rows

    @property
    def low_index(self) -> float:
        return self.base.low_index

    @property
    def high_index(self) -> float:
        return self.base.high_index

    @property
    def projection_beta(self) -> float:
        return self.base.projection_beta

    @property
    def projection_threshold(self) -> float:
        return self.base.projection_threshold

    @property
    def full_design_shape(self) -> tuple[int, int]:
        return self.base.design_shape

    @property
    def design_shape(self) -> tuple[int, int]:
        rows, columns = self.full_design_shape
        return (rows + 1) // 2, columns

    @property
    def variable_count(self) -> int:
        rows, columns = self.design_shape
        return rows * columns

    def expand(self, raw: ArrayLike) -> RealArray:
        independent = np.asarray(raw, dtype=float)
        if independent.size != self.variable_count:
            raise ValueError(
                f"raw density must contain {self.variable_count} variables"
            )
        independent = independent.reshape(self.design_shape)
        full_rows, _ = self.full_design_shape
        if full_rows % 2 == 0:
            return np.concatenate((independent, np.flipud(independent)), axis=0)
        return np.concatenate(
            (independent, np.flipud(independent[:-1])), axis=0
        )

    def reduce(self, full_raw: ArrayLike) -> RealArray:
        values = np.asarray(full_raw, dtype=float)
        if values.shape != self.full_design_shape:
            raise ValueError(
                f"full raw density must have shape {self.full_design_shape}"
            )
        if not np.allclose(values, np.flipud(values), rtol=0.0, atol=1e-12):
            raise ValueError("full raw density must be axially mirror symmetric")
        independent_rows, _ = self.design_shape
        return np.asarray(values[:independent_rows])

    def map(
        self,
        raw: ArrayLike,
        *,
        background_epsilon: ArrayLike | float | None = None,
    ) -> DensityMapping:
        return self.base.map(
            self.expand(raw), background_epsilon=background_epsilon
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: DensityMapping,
    ) -> ComplexArray:
        full = np.asarray(self.base.pullback(epsilon_gradient, mapping))
        full_rows, _ = self.full_design_shape
        independent_rows, _ = self.design_shape
        lower = full[..., :independent_rows, :].copy()
        if full_rows % 2 == 0:
            lower += np.flip(full[..., independent_rows:, :], axis=-2)
        else:
            paired = independent_rows - 1
            lower[..., :paired, :] += np.flip(
                full[..., independent_rows:, :], axis=-2
            )
        return np.asarray(lower)


@dataclass
class PlanarLayerDensityParameterization:
    """Extrude an unrestricted lateral density through a fixed layer."""

    base: DensityParameterization
    fixed_background_epsilon: ArrayLike | float | None = None

    @property
    def grid(self) -> UniformGrid2D:
        return self.base.grid

    @property
    def design_rows(self) -> tuple[int, int]:
        return self.base.design_rows

    @property
    def low_index(self) -> float:
        return self.base.low_index

    @property
    def high_index(self) -> float:
        return self.base.high_index

    @property
    def projection_beta(self) -> float:
        return self.base.projection_beta

    @property
    def projection_threshold(self) -> float:
        return self.base.projection_threshold

    @property
    def full_design_shape(self) -> tuple[int, int]:
        return self.base.design_shape

    @property
    def design_shape(self) -> tuple[int, int]:
        return 1, self.grid.nx

    @property
    def variable_count(self) -> int:
        return self.grid.nx

    def expand(self, raw: ArrayLike) -> RealArray:
        lateral = np.asarray(raw, dtype=float)
        if lateral.size != self.variable_count:
            raise ValueError(
                f"raw density must contain {self.variable_count} variables"
            )
        rows, _ = self.full_design_shape
        return np.tile(lateral.reshape(1, -1), (rows, 1))

    def reduce(self, full_raw: ArrayLike) -> RealArray:
        values = np.asarray(full_raw, dtype=float)
        if values.shape != self.full_design_shape:
            raise ValueError(
                f"full raw density must have shape {self.full_design_shape}"
            )
        if not np.allclose(values, values[:1], rtol=0.0, atol=1e-12):
            raise ValueError("full raw density must be uniform through the layer")
        return np.asarray(values[:1])

    def map(
        self,
        raw: ArrayLike,
        *,
        background_epsilon: ArrayLike | float | None = None,
    ) -> DensityMapping:
        if (
            background_epsilon is not None
            and self.fixed_background_epsilon is not None
        ):
            raise ValueError(
                "background_epsilon cannot override a fixed fabrication background"
            )
        background = (
            self.fixed_background_epsilon
            if background_epsilon is None
            else background_epsilon
        )
        return self.base.map(
            self.expand(raw), background_epsilon=background
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: DensityMapping,
    ) -> ComplexArray:
        full = np.asarray(self.base.pullback(epsilon_gradient, mapping))
        return np.sum(full, axis=-2, keepdims=True)


@dataclass
class MirrorSymmetricPlanarLayerDensityParameterization:
    """Extrude one mirror-symmetric lateral pattern through a fixed layer.

    This adapter restricts the wrapped two-dimensional density map to a
    single-etch planar process: one independent density profile varies along
    the periodic transverse axis, has exact mirror symmetry about the unit-cell
    center, and is copied through every row in ``base.design_rows``.  Its
    pullback sums the sensitivities of all copied and mirrored pixels.
    """

    base: DensityParameterization
    fixed_background_epsilon: ArrayLike | float | None = None

    @property
    def grid(self) -> UniformGrid2D:
        return self.base.grid

    @property
    def design_rows(self) -> tuple[int, int]:
        return self.base.design_rows

    @property
    def low_index(self) -> float:
        return self.base.low_index

    @property
    def high_index(self) -> float:
        return self.base.high_index

    @property
    def projection_beta(self) -> float:
        return self.base.projection_beta

    @property
    def projection_threshold(self) -> float:
        return self.base.projection_threshold

    @property
    def full_design_shape(self) -> tuple[int, int]:
        return self.base.design_shape

    @property
    def design_shape(self) -> tuple[int, int]:
        return 1, (self.grid.nx + 1) // 2

    @property
    def variable_count(self) -> int:
        return self.design_shape[1]

    def _expand_lateral(self, raw: ArrayLike) -> RealArray:
        independent = np.asarray(raw, dtype=float)
        if independent.size != self.variable_count:
            raise ValueError(
                f"raw density must contain {self.variable_count} variables"
            )
        independent = independent.reshape(-1)
        if self.grid.nx % 2 == 0:
            return np.concatenate((independent, independent[::-1]))
        return np.concatenate((independent, independent[-2::-1]))

    def expand(self, raw: ArrayLike) -> RealArray:
        lateral = self._expand_lateral(raw)
        rows, _ = self.full_design_shape
        return np.tile(lateral[np.newaxis, :], (rows, 1))

    def reduce(self, full_raw: ArrayLike) -> RealArray:
        values = np.asarray(full_raw, dtype=float)
        if values.shape != self.full_design_shape:
            raise ValueError(
                f"full raw density must have shape {self.full_design_shape}"
            )
        if not np.allclose(values, values[:1], rtol=0.0, atol=1e-12):
            raise ValueError("full raw density must be uniform through the layer")
        lateral = values[0]
        if not np.allclose(lateral, lateral[::-1], rtol=0.0, atol=1e-12):
            raise ValueError("lateral density must be mirror symmetric")
        return lateral[: self.variable_count][np.newaxis, :]

    def map(
        self,
        raw: ArrayLike,
        *,
        background_epsilon: ArrayLike | float | None = None,
    ) -> DensityMapping:
        if (
            background_epsilon is not None
            and self.fixed_background_epsilon is not None
        ):
            raise ValueError(
                "background_epsilon cannot override a fixed fabrication background"
            )
        background = (
            self.fixed_background_epsilon
            if background_epsilon is None
            else background_epsilon
        )
        return self.base.map(
            self.expand(raw), background_epsilon=background
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: DensityMapping,
    ) -> ComplexArray:
        full = np.asarray(self.base.pullback(epsilon_gradient, mapping))
        lateral = np.sum(full, axis=-2)
        count = self.variable_count
        independent = lateral[..., :count].copy()
        if self.grid.nx % 2 == 0:
            independent += np.flip(lateral[..., count:], axis=-1)
        else:
            independent[..., : count - 1] += np.flip(
                lateral[..., count:], axis=-1
            )
        return independent[..., np.newaxis, :]


@dataclass
class LaterallyMirrorSymmetricDensityParameterization:
    """Restrict a 2D design block to exact lateral mirror symmetry.

    Unlike :class:`MirrorSymmetricPlanarLayerDensityParameterization`, every
    axial row retains an independent half-period profile.  The adapter is
    therefore suitable for thickness/sidewall or multilevel topology studies
    while preserving the normal-incidence lateral symmetry exactly.
    """

    base: DensityParameterization
    fixed_background_epsilon: ArrayLike | float | None = None

    @property
    def grid(self) -> UniformGrid2D:
        return self.base.grid

    @property
    def design_rows(self) -> tuple[int, int]:
        return self.base.design_rows

    @property
    def low_index(self) -> float:
        return self.base.low_index

    @property
    def high_index(self) -> float:
        return self.base.high_index

    @property
    def projection_beta(self) -> float:
        return self.base.projection_beta

    @property
    def projection_threshold(self) -> float:
        return self.base.projection_threshold

    @property
    def full_design_shape(self) -> tuple[int, int]:
        return self.base.design_shape

    @property
    def design_shape(self) -> tuple[int, int]:
        rows, _ = self.full_design_shape
        return rows, (self.grid.nx + 1) // 2

    @property
    def variable_count(self) -> int:
        rows, columns = self.design_shape
        return rows * columns

    def expand(self, raw: ArrayLike) -> RealArray:
        independent = np.asarray(raw, dtype=float)
        if independent.size != self.variable_count:
            raise ValueError(
                f"raw density must contain {self.variable_count} variables"
            )
        independent = independent.reshape(self.design_shape)
        if self.grid.nx % 2 == 0:
            return np.concatenate((independent, independent[:, ::-1]), axis=1)
        return np.concatenate((independent, independent[:, -2::-1]), axis=1)

    def reduce(self, full_raw: ArrayLike) -> RealArray:
        values = np.asarray(full_raw, dtype=float)
        if values.shape != self.full_design_shape:
            raise ValueError(
                f"full raw density must have shape {self.full_design_shape}"
            )
        if not np.allclose(values, np.fliplr(values), rtol=0.0, atol=1e-12):
            raise ValueError("density block must be laterally mirror symmetric")
        return np.asarray(values[:, : self.design_shape[1]])

    def map(
        self,
        raw: ArrayLike,
        *,
        background_epsilon: ArrayLike | float | None = None,
    ) -> DensityMapping:
        if (
            background_epsilon is not None
            and self.fixed_background_epsilon is not None
        ):
            raise ValueError(
                "background_epsilon cannot override a fixed fabrication background"
            )
        background = (
            self.fixed_background_epsilon
            if background_epsilon is None
            else background_epsilon
        )
        return self.base.map(
            self.expand(raw), background_epsilon=background
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: DensityMapping,
    ) -> ComplexArray:
        full = np.asarray(self.base.pullback(epsilon_gradient, mapping))
        count = self.design_shape[1]
        independent = full[..., :count].copy()
        if self.grid.nx % 2 == 0:
            independent += np.flip(full[..., count:], axis=-1)
        else:
            independent[..., : count - 1] += np.flip(
                full[..., count:], axis=-1
            )
        return np.asarray(independent)


@dataclass
class PairedPlanarRibDensityParameterization:
    """Copy one unrestricted lateral mask to two axially mirrored ribs.

    The fixed background contains the two uniform slabs, their spacer, and
    the port regions.  Only the two facing relief layers are designable.  A
    single lateral density is extruded through the lower relief and copied
    bit-for-bit to the axially mirrored upper relief.  Unlike
    :class:`PairedMirrorSymmetricRibDensityParameterization`, this adapter does
    not impose lateral mirror symmetry, so it can retain the unequal sites of
    a period-doubled quasi-BIC grating.  Every thresholded relief feature is
    attached to its adjacent fixed slab.
    """

    lower: PlanarLayerDensityParameterization
    upper: PlanarLayerDensityParameterization
    fixed_background_epsilon: ArrayLike | float

    def __post_init__(self) -> None:
        if self.lower.grid.shape != self.upper.grid.shape or not np.isclose(
            self.lower.grid.period, self.upper.grid.period
        ):
            raise ValueError("paired ribs must use the same grid")
        if self.lower.design_rows[1] >= self.upper.design_rows[0]:
            raise ValueError("paired ribs require a nonempty fixed spacer")
        if self.lower.variable_count != self.upper.variable_count:
            raise ValueError("paired rib adapters must have equal variable counts")
        if not np.isclose(self.lower.low_index, self.upper.low_index) or not np.isclose(
            self.lower.high_index, self.upper.high_index
        ):
            raise ValueError("paired ribs must use the same materials")
        if (
            self.lower.fixed_background_epsilon is not None
            or self.upper.fixed_background_epsilon is not None
        ):
            raise ValueError("rib adapters must not define their own backgrounds")
        lower_rows = self.lower.design_rows
        upper_rows = self.upper.design_rows
        if (
            lower_rows[0] != self.grid.nz - upper_rows[1]
            or lower_rows[1] != self.grid.nz - upper_rows[0]
        ):
            raise ValueError("paired rib intervals must be exact axial mirrors")
        background = np.asarray(self.fixed_background_epsilon, dtype=float)
        if background.ndim not in {0, 2}:
            raise ValueError("fixed background must be scalar or two dimensional")
        if background.ndim == 2:
            if background.shape != self.grid.shape:
                raise ValueError("fixed background must match the full grid")
            if not np.array_equal(background, np.flipud(background)):
                raise ValueError("fixed background must be axially symmetric")

    @property
    def grid(self) -> UniformGrid2D:
        return self.lower.grid

    @property
    def design_rows(self) -> tuple[int, int]:
        return self.lower.design_rows[0], self.upper.design_rows[1]

    @property
    def rib_rows(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return self.lower.design_rows, self.upper.design_rows

    @property
    def spacer_rows(self) -> tuple[int, int]:
        return self.lower.design_rows[1], self.upper.design_rows[0]

    @property
    def low_index(self) -> float:
        return self.lower.low_index

    @property
    def high_index(self) -> float:
        return self.lower.high_index

    @property
    def projection_beta(self) -> float:
        return self.lower.projection_beta

    @property
    def projection_threshold(self) -> float:
        return self.lower.projection_threshold

    @property
    def design_shape(self) -> tuple[int, int]:
        return self.lower.design_shape

    @property
    def variable_count(self) -> int:
        return self.lower.variable_count

    def _background(self) -> RealArray:
        background = np.asarray(self.fixed_background_epsilon, dtype=float)
        if background.ndim == 0:
            return np.full(self.grid.shape, float(background), dtype=float)
        return background.copy()

    def map(self, raw: ArrayLike) -> PairedRibDensityMapping:
        background = self._background()
        lower_mapping = self.lower.map(raw, background_epsilon=background)
        upper_mapping = self.upper.map(raw, background_epsilon=background)
        epsilon = background.copy()
        lower_start, lower_stop = self.lower.design_rows
        upper_start, upper_stop = self.upper.design_rows
        epsilon[lower_start:lower_stop] = lower_mapping.epsilon_r[
            lower_start:lower_stop
        ]
        epsilon[upper_start:upper_stop] = np.flipud(
            epsilon[lower_start:lower_stop]
        )
        if not np.array_equal(epsilon, np.flipud(epsilon)):
            raise RuntimeError("paired-rib map broke axial symmetry")
        return PairedRibDensityMapping(
            lower=lower_mapping,
            upper=upper_mapping,
            epsilon_r=epsilon,
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: PairedRibDensityMapping,
    ) -> ComplexArray:
        gradient = np.asarray(epsilon_gradient, dtype=np.complex128)
        if gradient.shape[-2:] != self.grid.shape:
            raise ValueError("epsilon_gradient trailing axes must match the grid")
        lower = np.asarray(self.lower.pullback(gradient, mapping.lower))
        upper = np.asarray(self.upper.pullback(gradient, mapping.upper))
        return lower + upper


@dataclass
class PairedNestedPlanarReliefDensityParameterization:
    """Two unrestricted nested relief masks mirrored across a spacer.

    ``inner`` is the relief tier touching each fixed uniform slab. ``outer``
    is the next tier toward the central spacer. Its material fraction is the
    product of the projected inner density and an independent outer control,
    so thresholded outer material is always supported by inner material. The
    complete lower relief is copied by exact axial reflection to the upper
    slab. Lateral asymmetry is unrestricted, allowing period-doubled qBIC
    perturbations while guaranteeing a connected two-height etch profile.
    """

    inner: PlanarLayerDensityParameterization
    outer: PlanarLayerDensityParameterization
    fixed_background_epsilon: ArrayLike | float

    def __post_init__(self) -> None:
        if self.inner.grid.shape != self.outer.grid.shape or not np.isclose(
            self.inner.grid.period, self.outer.grid.period
        ):
            raise ValueError("nested relief tiers must use the same grid")
        if self.inner.design_rows[1] != self.outer.design_rows[0]:
            raise ValueError("lower inner and outer relief tiers must be contiguous")
        if self.inner.variable_count != self.outer.variable_count:
            raise ValueError("nested relief tiers need equal lateral variables")
        if not np.isclose(self.inner.low_index, self.outer.low_index) or not np.isclose(
            self.inner.high_index, self.outer.high_index
        ):
            raise ValueError("nested relief tiers must use the same materials")
        if (
            self.inner.fixed_background_epsilon is not None
            or self.outer.fixed_background_epsilon is not None
        ):
            raise ValueError("nested relief adapters must not define backgrounds")
        if self.upper_outer_rows[1] != self.upper_inner_rows[0]:
            raise ValueError("mirrored upper outer and inner tiers must be contiguous")
        if self.lower_outer_rows[1] > self.upper_outer_rows[0]:
            raise ValueError("nested reliefs cannot overlap across the spacer")
        background = np.asarray(self.fixed_background_epsilon, dtype=float)
        if background.ndim not in {0, 2}:
            raise ValueError("fixed background must be scalar or two dimensional")
        if background.ndim == 2:
            if background.shape != self.grid.shape:
                raise ValueError("fixed background must match the full grid")
            if not np.array_equal(background, np.flipud(background)):
                raise ValueError("fixed background must be axially symmetric")

    @property
    def grid(self) -> UniformGrid2D:
        return self.inner.grid

    @property
    def lower_inner_rows(self) -> tuple[int, int]:
        return self.inner.design_rows

    @property
    def lower_outer_rows(self) -> tuple[int, int]:
        return self.outer.design_rows

    @property
    def upper_inner_rows(self) -> tuple[int, int]:
        start, stop = self.lower_inner_rows
        return self.grid.nz - stop, self.grid.nz - start

    @property
    def upper_outer_rows(self) -> tuple[int, int]:
        start, stop = self.lower_outer_rows
        return self.grid.nz - stop, self.grid.nz - start

    @property
    def spacer_rows(self) -> tuple[int, int]:
        return self.lower_outer_rows[1], self.upper_outer_rows[0]

    @property
    def design_rows(self) -> tuple[int, int]:
        return self.lower_inner_rows[0], self.upper_inner_rows[1]

    @property
    def tier_rows(
        self,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        return (
            self.lower_inner_rows,
            self.lower_outer_rows,
            self.upper_outer_rows,
            self.upper_inner_rows,
        )

    @property
    def low_index(self) -> float:
        return self.inner.low_index

    @property
    def high_index(self) -> float:
        return self.inner.high_index

    @property
    def projection_beta(self) -> float:
        return self.inner.projection_beta

    @property
    def projection_threshold(self) -> float:
        return self.inner.projection_threshold

    @property
    def design_shape(self) -> tuple[int, int]:
        return 2, self.inner.variable_count

    @property
    def variable_count(self) -> int:
        return 2 * self.inner.variable_count

    def _background(self) -> RealArray:
        background = np.asarray(self.fixed_background_epsilon, dtype=float)
        if background.ndim == 0:
            return np.full(self.grid.shape, float(background), dtype=float)
        return background.copy()

    def map(self, raw: ArrayLike) -> PairedNestedRibDensityMapping:
        values = np.asarray(raw, dtype=float)
        if values.size != self.variable_count:
            raise ValueError(
                f"raw density must contain {self.variable_count} variables"
            )
        values = values.reshape(self.design_shape)
        background = self._background()
        inner_mapping = self.inner.map(values[0], background_epsilon=background)
        outer_mapping = self.outer.map(values[1], background_epsilon=background)
        projected_inner_rows = np.asarray(inner_mapping.projected)
        projected_outer_rows = np.asarray(outer_mapping.projected)
        if not np.allclose(projected_inner_rows, projected_inner_rows[:1]):
            raise RuntimeError("inner relief mask is not axially extruded")
        if not np.allclose(projected_outer_rows, projected_outer_rows[:1]):
            raise RuntimeError("outer relief mask is not axially extruded")
        projected_inner = projected_inner_rows[0]
        projected_outer_control = projected_outer_rows[0]
        projected_outer_material = projected_inner * projected_outer_control
        index_difference = self.high_index - self.low_index
        inner_index = self.low_index + index_difference * projected_inner
        outer_index = self.low_index + index_difference * projected_outer_material
        inner_epsilon = inner_index**2
        outer_epsilon = outer_index**2
        outer_epsilon_derivative = 2.0 * outer_index * index_difference

        epsilon = background.copy()
        epsilon[slice(*self.lower_inner_rows)] = inner_epsilon[np.newaxis, :]
        epsilon[slice(*self.lower_outer_rows)] = outer_epsilon[np.newaxis, :]
        epsilon[slice(*self.upper_outer_rows)] = np.flipud(
            epsilon[slice(*self.lower_outer_rows)]
        )
        epsilon[slice(*self.upper_inner_rows)] = np.flipud(
            epsilon[slice(*self.lower_inner_rows)]
        )
        if not np.array_equal(epsilon, np.flipud(epsilon)):
            raise RuntimeError("nested paired-relief map broke axial symmetry")
        return PairedNestedRibDensityMapping(
            inner=inner_mapping,
            outer_control=outer_mapping,
            projected_inner=projected_inner,
            projected_outer_control=projected_outer_control,
            projected_outer_material=projected_outer_material,
            outer_epsilon_derivative=outer_epsilon_derivative,
            epsilon_r=epsilon,
        )

    def _fold_gradient(
        self,
        gradient: ComplexArray,
        lower_rows: tuple[int, int],
        upper_rows: tuple[int, int],
    ) -> ComplexArray:
        lower = gradient[..., slice(*lower_rows), :]
        upper = np.flip(gradient[..., slice(*upper_rows), :], axis=-2)
        return lower + upper

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: PairedNestedRibDensityMapping,
    ) -> ComplexArray:
        gradient = np.asarray(epsilon_gradient, dtype=np.complex128)
        if gradient.shape[-2:] != self.grid.shape:
            raise ValueError("epsilon_gradient trailing axes must match the grid")
        leading = gradient.shape[:-2]
        inner_gradient = self._fold_gradient(
            gradient, self.lower_inner_rows, self.upper_inner_rows
        )
        outer_gradient = self._fold_gradient(
            gradient, self.lower_outer_rows, self.upper_outer_rows
        )
        inner_pseudo = np.zeros_like(gradient)
        outer_pseudo = np.zeros_like(gradient)
        inner_start, inner_stop = self.lower_inner_rows
        outer_start, outer_stop = self.lower_outer_rows
        inner_pseudo[..., inner_start:inner_stop, :] = inner_gradient

        inner_from_outer = np.sum(
            outer_gradient
            * mapping.outer_epsilon_derivative
            * mapping.projected_outer_control,
            axis=-2,
        )
        inner_rows = inner_stop - inner_start
        inner_pseudo[..., inner_start:inner_stop, :] += (
            inner_from_outer[..., np.newaxis, :]
            / inner_rows
            / mapping.inner.epsilon_derivative
        )
        outer_from_control = np.sum(
            outer_gradient
            * mapping.outer_epsilon_derivative
            * mapping.projected_inner,
            axis=-2,
        )
        outer_rows = outer_stop - outer_start
        outer_pseudo[..., outer_start:outer_stop, :] = (
            outer_from_control[..., np.newaxis, :]
            / outer_rows
            / mapping.outer_control.epsilon_derivative
        )
        pulled_inner = np.asarray(
            self.inner.pullback(inner_pseudo, mapping.inner)
        ).reshape(*leading, -1)
        pulled_outer = np.asarray(
            self.outer.pullback(outer_pseudo, mapping.outer_control)
        ).reshape(*leading, -1)
        return np.stack((pulled_inner, pulled_outer), axis=-2)


@dataclass
class CrossGridPairedNestedReliefDensityParameterization:
    """Map two physical lateral masks to a subpixel-resolved target mesh.

    The shared raw variables live on ``master_nx`` transverse cell centers.
    They are periodically interpolated to the target mesh before target-grid
    filtering and projection. ``lower_inner_fill`` and ``lower_outer_fill``
    are exact axial cell-overlap fractions for two adjacent physical relief
    tiers; their reflected copies enforce axial symmetry.  Fractional values
    permit one fixed physical tier thickness to be represented on meshes for
    which an interface falls inside a finite-difference cell.
    """

    master_nx: int
    inner: DensityParameterization
    outer: DensityParameterization
    lower_inner_fill: ArrayLike
    lower_outer_fill: ArrayLike
    fixed_background_epsilon: ArrayLike | float

    def __post_init__(self) -> None:
        if self.master_nx < 2:
            raise ValueError("master_nx must be at least two")
        if self.inner.grid.shape != self.outer.grid.shape or not np.isclose(
            self.inner.grid.period, self.outer.grid.period
        ):
            raise ValueError("inner and outer filters must use the same grid")
        if self.inner.design_shape != (1, self.grid.nx) or self.outer.design_shape != (
            1,
            self.grid.nx,
        ):
            raise ValueError("inner and outer filters must each contain one row")
        if not np.isclose(self.inner.low_index, self.outer.low_index) or not np.isclose(
            self.inner.high_index, self.outer.high_index
        ):
            raise ValueError("inner and outer filters must use the same materials")
        inner_fill = np.asarray(self.lower_inner_fill, dtype=float).ravel()
        outer_fill = np.asarray(self.lower_outer_fill, dtype=float).ravel()
        if inner_fill.size != self.grid.nz or outer_fill.size != self.grid.nz:
            raise ValueError("axial fill arrays must contain one value per grid row")
        if np.any((inner_fill < 0.0) | (inner_fill > 1.0)) or np.any(
            (outer_fill < 0.0) | (outer_fill > 1.0)
        ):
            raise ValueError("axial fill fractions must lie within [0, 1]")
        total = inner_fill + outer_fill + np.flip(inner_fill) + np.flip(outer_fill)
        if np.any(total > 1.0 + 1e-12):
            raise ValueError("mirrored relief tiers overlap in physical space")
        background = np.asarray(self.fixed_background_epsilon, dtype=float)
        if background.ndim not in {0, 2}:
            raise ValueError("fixed background must be scalar or two dimensional")
        if background.ndim == 2:
            if background.shape != self.grid.shape:
                raise ValueError("fixed background must match the target grid")
            if not np.allclose(background, np.flipud(background), rtol=0.0, atol=1e-12):
                raise ValueError("fixed background must be axially symmetric")

    @property
    def grid(self) -> UniformGrid2D:
        return self.inner.grid

    @property
    def low_index(self) -> float:
        return self.inner.low_index

    @property
    def high_index(self) -> float:
        return self.inner.high_index

    @property
    def projection_beta(self) -> float:
        return self.inner.projection_beta

    @property
    def projection_threshold(self) -> float:
        return self.inner.projection_threshold

    @property
    def design_shape(self) -> tuple[int, int]:
        return 2, self.master_nx

    @property
    def variable_count(self) -> int:
        return 2 * self.master_nx

    @property
    def axial_fill_profiles(self) -> tuple[RealArray, RealArray]:
        inner = np.asarray(self.lower_inner_fill, dtype=float).ravel()
        outer = np.asarray(self.lower_outer_fill, dtype=float).ravel()
        return inner + np.flip(inner), outer + np.flip(outer)

    def _background(self) -> RealArray:
        background = np.asarray(self.fixed_background_epsilon, dtype=float)
        if background.ndim == 0:
            return np.full(self.grid.shape, float(background), dtype=float)
        return background.copy()

    def _target_raw(self, raw: RealArray) -> RealArray:
        return periodic_bilinear_resample(
            raw[np.newaxis, :], (1, self.grid.nx)
        )

    def map(self, raw: ArrayLike) -> CrossGridPairedNestedReliefDensityMapping:
        values = np.asarray(raw, dtype=float)
        if values.size != self.variable_count:
            raise ValueError(
                f"raw density must contain {self.variable_count} variables"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("raw density must be finite and lie within [0, 1]")
        values = values.reshape(self.design_shape)
        inner_mapping = self.inner.map(self._target_raw(values[0]))
        outer_mapping = self.outer.map(self._target_raw(values[1]))
        projected_inner = np.asarray(inner_mapping.projected[0])
        projected_outer_control = np.asarray(outer_mapping.projected[0])
        projected_outer_material = projected_inner * projected_outer_control
        difference = self.high_index - self.low_index
        inner_index = self.low_index + difference * projected_inner
        outer_index = self.low_index + difference * projected_outer_material
        inner_epsilon = inner_index**2
        outer_epsilon = outer_index**2
        outer_epsilon_derivative = 2.0 * outer_index * difference
        inner_fill, outer_fill = self.axial_fill_profiles
        low_epsilon = self.low_index**2
        epsilon = self._background()
        epsilon += inner_fill[:, np.newaxis] * (
            inner_epsilon[np.newaxis, :] - low_epsilon
        )
        epsilon += outer_fill[:, np.newaxis] * (
            outer_epsilon[np.newaxis, :] - low_epsilon
        )
        if not np.allclose(epsilon, np.flipud(epsilon), rtol=0.0, atol=1e-12):
            raise RuntimeError("cross-grid nested relief map broke axial symmetry")
        return CrossGridPairedNestedReliefDensityMapping(
            inner=inner_mapping,
            outer_control=outer_mapping,
            projected_inner=projected_inner,
            projected_outer_control=projected_outer_control,
            projected_outer_material=projected_outer_material,
            outer_epsilon_derivative=outer_epsilon_derivative,
            epsilon_r=epsilon,
        )

    def _master_pullback(self, target: ComplexArray) -> ComplexArray:
        return periodic_bilinear_resample_adjoint(
            target.reshape(*target.shape[:-1], 1, self.grid.nx),
            (1, self.master_nx),
        )[..., 0, :]

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: CrossGridPairedNestedReliefDensityMapping,
    ) -> ComplexArray:
        gradient = np.asarray(epsilon_gradient, dtype=np.complex128)
        if gradient.shape[-2:] != self.grid.shape:
            raise ValueError("epsilon_gradient trailing axes must match the grid")
        leading = gradient.shape[:-2]
        inner_fill, outer_fill = self.axial_fill_profiles
        inner_epsilon_gradient = np.sum(
            gradient * inner_fill.reshape((1,) * len(leading) + (-1, 1)), axis=-2
        )
        outer_epsilon_gradient = np.sum(
            gradient * outer_fill.reshape((1,) * len(leading) + (-1, 1)), axis=-2
        )
        difference = self.high_index - self.low_index
        inner_derivative = 2.0 * (
            self.low_index + difference * mapping.projected_inner
        ) * difference
        inner_projected_gradient = (
            inner_epsilon_gradient * inner_derivative
            + outer_epsilon_gradient
            * mapping.outer_epsilon_derivative
            * mapping.projected_outer_control
        )
        outer_projected_gradient = (
            outer_epsilon_gradient
            * mapping.outer_epsilon_derivative
            * mapping.projected_inner
        )
        inner_pseudo = np.zeros(
            (*leading, *self.grid.shape), dtype=np.complex128
        )
        outer_pseudo = np.zeros_like(inner_pseudo)
        inner_row = self.inner.design_rows[0]
        outer_row = self.outer.design_rows[0]
        inner_pseudo[..., inner_row, :] = (
            inner_projected_gradient / mapping.inner.epsilon_derivative[0]
        )
        outer_pseudo[..., outer_row, :] = (
            outer_projected_gradient / mapping.outer_control.epsilon_derivative[0]
        )
        target_inner = np.asarray(
            self.inner.pullback(inner_pseudo, mapping.inner)
        )[..., 0, :]
        target_outer = np.asarray(
            self.outer.pullback(outer_pseudo, mapping.outer_control)
        )[..., 0, :]
        return np.stack(
            (self._master_pullback(target_inner), self._master_pullback(target_outer)),
            axis=-2,
        )


@dataclass
class PairedMirrorSymmetricRibDensityParameterization:
    """Copy one mirror-symmetric mask to two axially mirrored rib layers.

    A fixed full-grid background retains the uniform central slab, DBRs, port
    regions, and every other non-design cell. The same lateral density is
    extruded through both disjoint rib intervals, which preserves exact axial
    and lateral symmetry throughout an adjoint update. Because either rib is
    directly adjacent to the fixed slab, all patterned high-index material
    remains connected to that slab after hard thresholding.
    """

    lower: MirrorSymmetricPlanarLayerDensityParameterization
    upper: MirrorSymmetricPlanarLayerDensityParameterization
    fixed_background_epsilon: ArrayLike | float

    def __post_init__(self) -> None:
        if self.lower.grid.shape != self.upper.grid.shape or not np.isclose(
            self.lower.grid.period, self.upper.grid.period
        ):
            raise ValueError("paired ribs must use the same grid")
        if self.lower.design_rows[1] >= self.upper.design_rows[0]:
            raise ValueError("paired ribs require a nonempty fixed central slab")
        if self.lower.variable_count != self.upper.variable_count:
            raise ValueError("paired rib adapters must have equal variable counts")
        if not np.isclose(self.lower.low_index, self.upper.low_index) or not np.isclose(
            self.lower.high_index, self.upper.high_index
        ):
            raise ValueError("paired ribs must use the same materials")
        if (
            self.lower.fixed_background_epsilon is not None
            or self.upper.fixed_background_epsilon is not None
        ):
            raise ValueError("rib adapters must not define their own backgrounds")
        lower_rows = self.lower.design_rows
        upper_rows = self.upper.design_rows
        if lower_rows[0] != self.grid.nz - upper_rows[1] or lower_rows[1] != self.grid.nz - upper_rows[0]:
            raise ValueError("paired rib intervals must be exact axial mirrors")
        background = np.asarray(self.fixed_background_epsilon, dtype=float)
        if background.ndim not in {0, 2}:
            raise ValueError("fixed background must be scalar or two dimensional")
        if background.ndim == 2:
            if background.shape != self.grid.shape:
                raise ValueError("fixed background must match the full grid")
            if not np.array_equal(background, np.flipud(background)):
                raise ValueError("fixed background must be axially symmetric")
            if not np.array_equal(background, np.fliplr(background)):
                raise ValueError("fixed background must be laterally symmetric")

    @property
    def grid(self) -> UniformGrid2D:
        return self.lower.grid

    @property
    def design_rows(self) -> tuple[int, int]:
        return self.lower.design_rows[0], self.upper.design_rows[1]

    @property
    def rib_rows(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return self.lower.design_rows, self.upper.design_rows

    @property
    def slab_rows(self) -> tuple[int, int]:
        return self.lower.design_rows[1], self.upper.design_rows[0]

    @property
    def low_index(self) -> float:
        return self.lower.low_index

    @property
    def high_index(self) -> float:
        return self.lower.high_index

    @property
    def projection_beta(self) -> float:
        return self.lower.projection_beta

    @property
    def projection_threshold(self) -> float:
        return self.lower.projection_threshold

    @property
    def design_shape(self) -> tuple[int, int]:
        return self.lower.design_shape

    @property
    def variable_count(self) -> int:
        return self.lower.variable_count

    def _background(self) -> RealArray:
        background = np.asarray(self.fixed_background_epsilon, dtype=float)
        if background.ndim == 0:
            return np.full(self.grid.shape, float(background), dtype=float)
        return background.copy()

    def map(self, raw: ArrayLike) -> PairedRibDensityMapping:
        background = self._background()
        lower_mapping = self.lower.map(raw, background_epsilon=background)
        upper_mapping = self.upper.map(raw, background_epsilon=background)
        epsilon = background.copy()
        lower_start, lower_stop = self.lower.design_rows
        upper_start, upper_stop = self.upper.design_rows
        lower_design = lower_mapping.epsilon_r[lower_start:lower_stop]
        lower_design = 0.5 * (lower_design + np.fliplr(lower_design))
        epsilon[lower_start:lower_stop] = lower_design
        # Copy the already mapped lower rib bit-for-bit. Evaluating the two
        # identical sparse filters separately can differ at roundoff and would
        # unnecessarily spoil exact axial symmetry in the Maxwell matrix.
        epsilon[upper_start:upper_stop] = np.flipud(
            epsilon[lower_start:lower_stop]
        )
        if not np.array_equal(epsilon, np.flipud(epsilon)):
            raise RuntimeError("paired-rib map broke axial symmetry")
        if not np.array_equal(epsilon, np.fliplr(epsilon)):
            raise RuntimeError("paired-rib map broke lateral symmetry")
        return PairedRibDensityMapping(
            lower=lower_mapping,
            upper=upper_mapping,
            epsilon_r=epsilon,
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: PairedRibDensityMapping,
    ) -> ComplexArray:
        gradient = np.asarray(epsilon_gradient, dtype=np.complex128)
        if gradient.shape[-2:] != self.grid.shape:
            raise ValueError("epsilon_gradient trailing axes must match the grid")
        lower_gradient = np.zeros_like(gradient)
        upper_gradient = np.zeros_like(gradient)
        lower_start, lower_stop = self.lower.design_rows
        upper_start, upper_stop = self.upper.design_rows
        lower_local = gradient[..., lower_start:lower_stop, :]
        upper_local = gradient[..., upper_start:upper_stop, :]
        lower_gradient[..., lower_start:lower_stop, :] = 0.5 * (
            lower_local + np.flip(lower_local, axis=-1)
        )
        upper_gradient[..., upper_start:upper_stop, :] = 0.5 * (
            upper_local + np.flip(upper_local, axis=-1)
        )
        lower = np.asarray(self.lower.pullback(lower_gradient, mapping.lower))
        upper = np.asarray(self.upper.pullback(upper_gradient, mapping.upper))
        return lower + upper


@dataclass
class PairedPhaseLayerDensityParameterization:
    """Append centered-film index and thickness to a paired-sheet design.

    The last two raw variables linearly map to refractive index and physical
    film thickness.  The film is centered inside ``phase_rows`` and its cell
    coverage is the exact one-dimensional overlap of a continuous slab with
    each axial grid cell.  Linear permittivity mixing is used only for the two
    partially covered interface cells.  Both the map and pullback preserve
    exact axial and lateral mirror symmetry.
    """

    base: object
    phase_rows: tuple[int, int]
    background_index: float
    phase_index_bounds: tuple[float, float]
    phase_thickness_bounds: tuple[float, float]

    def __post_init__(self) -> None:
        start, stop = self.phase_rows
        if not 0 <= start < stop <= self.grid.nz:
            raise ValueError("phase_rows must lie inside the grid")
        if start != self.grid.nz - stop:
            raise ValueError("phase_rows must be axially centered")
        if self.background_index <= 0.0:
            raise ValueError("background_index must be positive")
        index_low, index_high = self.phase_index_bounds
        if not 0.0 < index_low < index_high:
            raise ValueError("phase index bounds must be positive and increasing")
        thickness_low, thickness_high = self.phase_thickness_bounds
        if not 0.0 <= thickness_low < thickness_high:
            raise ValueError("phase thickness bounds must be nonnegative and increasing")
        available = (stop - start) * self.grid.dz
        if thickness_high > available + 1e-12:
            raise ValueError("maximum phase thickness exceeds phase_rows")

    @property
    def grid(self) -> UniformGrid2D:
        return self.base.grid

    @property
    def design_shape(self) -> tuple[int]:
        return (self.variable_count,)

    @property
    def variable_count(self) -> int:
        return int(self.base.variable_count) + 2

    @property
    def film_variable_indices(self) -> tuple[int, int]:
        return self.variable_count - 2, self.variable_count - 1

    def encode_phase(self, index: float, thickness: float) -> tuple[float, float]:
        index_low, index_high = self.phase_index_bounds
        thickness_low, thickness_high = self.phase_thickness_bounds
        raw_index = (float(index) - index_low) / (index_high - index_low)
        raw_thickness = (float(thickness) - thickness_low) / (
            thickness_high - thickness_low
        )
        if not 0.0 <= raw_index <= 1.0 or not 0.0 <= raw_thickness <= 1.0:
            raise ValueError("initial phase index and thickness must lie within bounds")
        return raw_index, raw_thickness

    def decode_phase(self, raw: ArrayLike) -> tuple[float, float]:
        values = np.asarray(raw, dtype=float).ravel()
        if values.size != self.variable_count:
            raise ValueError(f"raw density must contain {self.variable_count} variables")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("raw density values must lie within [0, 1]")
        index_low, index_high = self.phase_index_bounds
        thickness_low, thickness_high = self.phase_thickness_bounds
        phase_index = index_low + (index_high - index_low) * values[-2]
        phase_thickness = thickness_low + (
            thickness_high - thickness_low
        ) * values[-1]
        return float(phase_index), float(phase_thickness)

    def _coverage(
        self,
        thickness: float,
        *,
        derivative_side: str = "central",
    ) -> tuple[RealArray, RealArray]:
        start, stop = self.phase_rows
        center = 0.5 * self.grid.height
        lower = center - 0.5 * thickness
        upper = center + 0.5 * thickness
        rows = np.arange(start, stop, dtype=float)
        cell_lower = rows * self.grid.dz
        cell_upper = cell_lower + self.grid.dz
        overlap = np.maximum(
            0.0,
            np.minimum(cell_upper, upper) - np.maximum(cell_lower, lower),
        )
        coverage_local = np.clip(overlap / self.grid.dz, 0.0, 1.0)
        if derivative_side not in {"left", "central", "right"}:
            raise ValueError("derivative_side must be 'left', 'central', or 'right'")
        # Cell overlap is piecewise linear but nondifferentiable when a film
        # boundary lies exactly on a grid interface.  At a raw-variable bound
        # only one thickness direction is feasible, so retain its one-sided
        # derivative.  At an interior kink use the centered subgradient.  The
        # upper and lower interfaces move by one half of the total thickness.
        upper_left = (upper > cell_lower) & (upper <= cell_upper)
        lower_left = (lower >= cell_lower) & (lower < cell_upper)
        upper_right = (upper >= cell_lower) & (upper < cell_upper)
        lower_right = (lower > cell_lower) & (lower <= cell_upper)
        left_derivative = 0.5 * (
            upper_left.astype(float) + lower_left.astype(float)
        ) / self.grid.dz
        right_derivative = 0.5 * (
            upper_right.astype(float) + lower_right.astype(float)
        ) / self.grid.dz
        if derivative_side == "left":
            derivative_local = left_derivative
        elif derivative_side == "right":
            derivative_local = right_derivative
        else:
            derivative_local = 0.5 * (left_derivative + right_derivative)
        # Make paired rows bitwise identical even when their floating-point
        # coordinate expressions reached them by different arithmetic paths.
        coverage_local = 0.5 * (coverage_local + coverage_local[::-1])
        derivative_local = 0.5 * (derivative_local + derivative_local[::-1])
        coverage = np.zeros(self.grid.shape, dtype=float)
        derivative = np.zeros(self.grid.shape, dtype=float)
        coverage[start:stop] = coverage_local[:, np.newaxis]
        derivative[start:stop] = derivative_local[:, np.newaxis]
        return coverage, derivative

    def map(self, raw: ArrayLike) -> PairedPhaseLayerDensityMapping:
        values = np.asarray(raw, dtype=float).ravel()
        phase_index, phase_thickness = self.decode_phase(values)
        base_mapping = self.base.map(values[:-2])
        epsilon = np.asarray(base_mapping.epsilon_r, dtype=float).copy()
        thickness_raw = float(values[-1])
        if thickness_raw <= np.finfo(float).eps:
            derivative_side = "right"
        elif thickness_raw >= 1.0 - np.finfo(float).eps:
            derivative_side = "left"
        else:
            derivative_side = "central"
        coverage, coverage_derivative = self._coverage(
            phase_thickness,
            derivative_side=derivative_side,
        )
        background_epsilon = self.background_index**2
        phase_epsilon = phase_index**2
        start, stop = self.phase_rows
        epsilon[start:stop] = (
            background_epsilon
            + coverage[start:stop] * (phase_epsilon - background_epsilon)
        )
        index_span = self.phase_index_bounds[1] - self.phase_index_bounds[0]
        thickness_span = (
            self.phase_thickness_bounds[1] - self.phase_thickness_bounds[0]
        )
        index_derivative = coverage * 2.0 * phase_index * index_span
        thickness_derivative = (
            coverage_derivative
            * thickness_span
            * (phase_epsilon - background_epsilon)
        )
        if not np.array_equal(epsilon, np.flipud(epsilon)):
            raise RuntimeError("centered phase-film map broke axial symmetry")
        if not np.array_equal(epsilon, np.fliplr(epsilon)):
            raise RuntimeError("centered phase-film map broke lateral symmetry")
        return PairedPhaseLayerDensityMapping(
            base=base_mapping,
            phase_index=phase_index,
            phase_thickness=phase_thickness,
            coverage=coverage,
            index_epsilon_derivative=index_derivative,
            thickness_epsilon_derivative=thickness_derivative,
            epsilon_r=epsilon,
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: PairedPhaseLayerDensityMapping,
    ) -> ComplexArray:
        gradient = np.asarray(epsilon_gradient, dtype=np.complex128)
        if gradient.shape[-2:] != self.grid.shape:
            raise ValueError("epsilon_gradient trailing axes must match the grid")
        leading_shape = gradient.shape[:-2]
        base = np.asarray(self.base.pullback(gradient, mapping.base)).reshape(
            *leading_shape, -1
        )
        index = np.sum(
            gradient * mapping.index_epsilon_derivative,
            axis=(-2, -1),
        )
        thickness = np.sum(
            gradient * mapping.thickness_epsilon_derivative,
            axis=(-2, -1),
        )
        return np.concatenate(
            (base, index[..., np.newaxis], thickness[..., np.newaxis]),
            axis=-1,
        )


@dataclass
class PairedPatternedPhaseLayerDensityParameterization:
    """Add an independently patterned centered film to a paired-sheet cell.

    The design vector contains the base paired-sheet variables, one separate
    mirror-symmetric lateral mask, film index, and continuous film thickness.
    The central mask is extruded only through the continuously covered film
    thickness and is independent of the two outer patterned sheets.
    """

    base: object
    pattern: MirrorSymmetricPlanarLayerDensityParameterization
    phase_rows: tuple[int, int]
    background_index: float
    phase_index_bounds: tuple[float, float]
    phase_thickness_bounds: tuple[float, float]

    def __post_init__(self) -> None:
        start, stop = self.phase_rows
        if not 0 <= start < stop <= self.grid.nz:
            raise ValueError("phase_rows must lie inside the grid")
        if start != self.grid.nz - stop:
            raise ValueError("phase_rows must be axially centered")
        if self.pattern.grid != self.grid or self.pattern.design_rows != self.phase_rows:
            raise ValueError("central pattern must use the same grid and phase rows")
        if self.background_index <= 0.0:
            raise ValueError("background_index must be positive")
        index_low, index_high = self.phase_index_bounds
        if not 0.0 < index_low < index_high:
            raise ValueError("phase index bounds must be positive and increasing")
        thickness_low, thickness_high = self.phase_thickness_bounds
        if not 0.0 <= thickness_low < thickness_high:
            raise ValueError("phase thickness bounds must be nonnegative and increasing")
        if thickness_high > (stop - start) * self.grid.dz + 1e-12:
            raise ValueError("maximum phase thickness exceeds phase_rows")

    @property
    def grid(self) -> UniformGrid2D:
        return self.base.grid

    @property
    def variable_count(self) -> int:
        return int(self.base.variable_count) + int(self.pattern.variable_count) + 2

    @property
    def design_shape(self) -> tuple[int]:
        return (self.variable_count,)

    @property
    def pattern_variable_slice(self) -> slice:
        start = int(self.base.variable_count)
        return slice(start, start + int(self.pattern.variable_count))

    @property
    def film_variable_indices(self) -> tuple[int, int]:
        return self.variable_count - 2, self.variable_count - 1

    def encode_phase(self, index: float, thickness: float) -> tuple[float, float]:
        index_low, index_high = self.phase_index_bounds
        thickness_low, thickness_high = self.phase_thickness_bounds
        raw_index = (float(index) - index_low) / (index_high - index_low)
        raw_thickness = (float(thickness) - thickness_low) / (
            thickness_high - thickness_low
        )
        if not 0.0 <= raw_index <= 1.0 or not 0.0 <= raw_thickness <= 1.0:
            raise ValueError("initial phase index and thickness must lie within bounds")
        return float(raw_index), float(raw_thickness)

    def decode_phase(self, raw: ArrayLike) -> tuple[float, float]:
        values = np.asarray(raw, dtype=float).ravel()
        if values.size != self.variable_count:
            raise ValueError(f"raw density must contain {self.variable_count} variables")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("raw density values must lie within [0, 1]")
        index_low, index_high = self.phase_index_bounds
        thickness_low, thickness_high = self.phase_thickness_bounds
        phase_index = index_low + (index_high - index_low) * values[-2]
        phase_thickness = thickness_low + (
            thickness_high - thickness_low
        ) * values[-1]
        return float(phase_index), float(phase_thickness)

    def _coverage(
        self,
        thickness: float,
        *,
        derivative_side: str,
    ) -> tuple[RealArray, RealArray]:
        start, stop = self.phase_rows
        center = 0.5 * self.grid.height
        lower = center - 0.5 * thickness
        upper = center + 0.5 * thickness
        rows = np.arange(start, stop, dtype=float)
        cell_lower = rows * self.grid.dz
        cell_upper = cell_lower + self.grid.dz
        overlap = np.maximum(
            0.0,
            np.minimum(cell_upper, upper) - np.maximum(cell_lower, lower),
        )
        coverage_local = np.clip(overlap / self.grid.dz, 0.0, 1.0)
        upper_left = (upper > cell_lower) & (upper <= cell_upper)
        lower_left = (lower >= cell_lower) & (lower < cell_upper)
        upper_right = (upper >= cell_lower) & (upper < cell_upper)
        lower_right = (lower > cell_lower) & (lower <= cell_upper)
        left = 0.5 * (upper_left.astype(float) + lower_left.astype(float)) / self.grid.dz
        right = 0.5 * (upper_right.astype(float) + lower_right.astype(float)) / self.grid.dz
        if derivative_side == "left":
            derivative_local = left
        elif derivative_side == "right":
            derivative_local = right
        elif derivative_side == "central":
            derivative_local = 0.5 * (left + right)
        else:
            raise ValueError("invalid coverage derivative side")
        coverage_local = 0.5 * (coverage_local + coverage_local[::-1])
        derivative_local = 0.5 * (derivative_local + derivative_local[::-1])
        coverage = np.zeros(self.grid.shape, dtype=float)
        derivative = np.zeros(self.grid.shape, dtype=float)
        coverage[start:stop] = coverage_local[:, np.newaxis]
        derivative[start:stop] = derivative_local[:, np.newaxis]
        return coverage, derivative

    def map(self, raw: ArrayLike) -> PairedPatternedPhaseLayerDensityMapping:
        values = np.asarray(raw, dtype=float).ravel()
        phase_index, phase_thickness = self.decode_phase(values)
        base_count = int(self.base.variable_count)
        pattern_slice = self.pattern_variable_slice
        base_mapping = self.base.map(values[:base_count])
        pattern_mapping = self.pattern.map(values[pattern_slice])
        projected_rows = np.asarray(pattern_mapping.projected, dtype=float)
        if not np.allclose(projected_rows, projected_rows[:1], rtol=0.0, atol=1e-12):
            raise RuntimeError("central pattern must remain axially extruded")
        projected = 0.5 * (projected_rows[0] + projected_rows[0, ::-1])
        thickness_raw = float(values[-1])
        if thickness_raw <= np.finfo(float).eps:
            side = "right"
        elif thickness_raw >= 1.0 - np.finfo(float).eps:
            side = "left"
        else:
            side = "central"
        coverage, coverage_derivative = self._coverage(
            phase_thickness,
            derivative_side=side,
        )
        material_fraction = coverage * projected[np.newaxis, :]
        background_epsilon = self.background_index**2
        phase_epsilon = phase_index**2
        difference = phase_epsilon - background_epsilon
        epsilon = np.asarray(base_mapping.epsilon_r, dtype=float).copy()
        start, stop = self.phase_rows
        epsilon[start:stop] = (
            background_epsilon + material_fraction[start:stop] * difference
        )
        index_span = self.phase_index_bounds[1] - self.phase_index_bounds[0]
        thickness_span = self.phase_thickness_bounds[1] - self.phase_thickness_bounds[0]
        index_derivative = material_fraction * 2.0 * phase_index * index_span
        thickness_derivative = (
            coverage_derivative
            * projected[np.newaxis, :]
            * thickness_span
            * difference
        )
        pattern_derivative = coverage * difference
        if not np.array_equal(epsilon, np.flipud(epsilon)):
            raise RuntimeError("patterned phase layer broke axial symmetry")
        if not np.array_equal(epsilon, np.fliplr(epsilon)):
            raise RuntimeError("patterned phase layer broke lateral symmetry")
        return PairedPatternedPhaseLayerDensityMapping(
            base=base_mapping,
            pattern=pattern_mapping,
            projected_pattern=projected,
            phase_index=phase_index,
            phase_thickness=phase_thickness,
            coverage=coverage,
            index_epsilon_derivative=index_derivative,
            thickness_epsilon_derivative=thickness_derivative,
            pattern_epsilon_derivative=pattern_derivative,
            epsilon_r=epsilon,
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: PairedPatternedPhaseLayerDensityMapping,
    ) -> ComplexArray:
        gradient = np.asarray(epsilon_gradient, dtype=np.complex128)
        if gradient.shape[-2:] != self.grid.shape:
            raise ValueError("epsilon_gradient trailing axes must match the grid")
        leading_shape = gradient.shape[:-2]
        start, stop = self.phase_rows
        base_gradient = gradient.copy()
        base_gradient[..., start:stop, :] = 0.0
        base = np.asarray(self.base.pullback(base_gradient, mapping.base)).reshape(
            *leading_shape, -1
        )
        pseudo = np.zeros_like(gradient)
        denominator = np.asarray(mapping.pattern.epsilon_derivative)
        local_pattern_gradient = (
            gradient[..., start:stop, :]
            * mapping.pattern_epsilon_derivative[start:stop]
        )
        local_pattern_gradient = 0.5 * (
            local_pattern_gradient + np.flip(local_pattern_gradient, axis=-1)
        )
        pseudo[..., start:stop, :] = local_pattern_gradient / denominator
        pattern = np.asarray(self.pattern.pullback(pseudo, mapping.pattern)).reshape(
            *leading_shape, -1
        )
        index = np.sum(
            gradient * mapping.index_epsilon_derivative,
            axis=(-2, -1),
        )
        thickness = np.sum(
            gradient * mapping.thickness_epsilon_derivative,
            axis=(-2, -1),
        )
        return np.concatenate(
            (base, pattern, index[..., np.newaxis], thickness[..., np.newaxis]),
            axis=-1,
        )


@dataclass
class PairedNestedMirrorSymmetricRibDensityParameterization:
    """Two nested lateral rib masks copied symmetrically around a fixed slab.

    ``inner`` controls the rib tier adjacent to the central slab. ``outer`` is
    a second control whose projected density is multiplied by the inner
    projected density before material interpolation. The outer high-index
    material is therefore a subset of the inner support after hard logical
    thresholding. Both lower-side tiers are copied by exact axial reflection,
    producing a two-etch-depth profile with one uniform connected slab.
    """

    inner: MirrorSymmetricPlanarLayerDensityParameterization
    outer: MirrorSymmetricPlanarLayerDensityParameterization
    fixed_background_epsilon: ArrayLike | float

    def __post_init__(self) -> None:
        if self.inner.grid.shape != self.outer.grid.shape or not np.isclose(
            self.inner.grid.period, self.outer.grid.period
        ):
            raise ValueError("nested paired rib tiers must use the same grid")
        if self.outer.design_rows[1] != self.inner.design_rows[0]:
            raise ValueError("lower outer and inner rib tiers must be contiguous")
        if self.inner.variable_count != self.outer.variable_count:
            raise ValueError("nested paired rib tiers need equal lateral variables")
        if not np.isclose(self.inner.low_index, self.outer.low_index) or not np.isclose(
            self.inner.high_index, self.outer.high_index
        ):
            raise ValueError("nested paired rib tiers must use the same materials")
        if (
            self.inner.fixed_background_epsilon is not None
            or self.outer.fixed_background_epsilon is not None
        ):
            raise ValueError("nested rib adapters must not define backgrounds")
        lower_outer = self.outer.design_rows
        lower_inner = self.inner.design_rows
        upper_inner = (
            self.grid.nz - lower_inner[1],
            self.grid.nz - lower_inner[0],
        )
        upper_outer = (
            self.grid.nz - lower_outer[1],
            self.grid.nz - lower_outer[0],
        )
        if lower_inner[1] >= upper_inner[0]:
            raise ValueError("nested paired ribs require a nonempty central slab")
        if upper_inner[1] != upper_outer[0]:
            raise ValueError("mirrored upper inner and outer tiers must be contiguous")
        background = np.asarray(self.fixed_background_epsilon, dtype=float)
        if background.ndim not in {0, 2}:
            raise ValueError("fixed background must be scalar or two dimensional")
        if background.ndim == 2:
            if background.shape != self.grid.shape:
                raise ValueError("fixed background must match the full grid")
            if not np.array_equal(background, np.flipud(background)):
                raise ValueError("fixed background must be axially symmetric")
            if not np.array_equal(background, np.fliplr(background)):
                raise ValueError("fixed background must be laterally symmetric")

    @property
    def grid(self) -> UniformGrid2D:
        return self.inner.grid

    @property
    def lower_outer_rows(self) -> tuple[int, int]:
        return self.outer.design_rows

    @property
    def lower_inner_rows(self) -> tuple[int, int]:
        return self.inner.design_rows

    @property
    def upper_inner_rows(self) -> tuple[int, int]:
        start, stop = self.lower_inner_rows
        return self.grid.nz - stop, self.grid.nz - start

    @property
    def upper_outer_rows(self) -> tuple[int, int]:
        start, stop = self.lower_outer_rows
        return self.grid.nz - stop, self.grid.nz - start

    @property
    def slab_rows(self) -> tuple[int, int]:
        return self.lower_inner_rows[1], self.upper_inner_rows[0]

    @property
    def design_rows(self) -> tuple[int, int]:
        return self.lower_outer_rows[0], self.upper_outer_rows[1]

    @property
    def tier_rows(
        self,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        return (
            self.lower_outer_rows,
            self.lower_inner_rows,
            self.upper_inner_rows,
            self.upper_outer_rows,
        )

    @property
    def low_index(self) -> float:
        return self.inner.low_index

    @property
    def high_index(self) -> float:
        return self.inner.high_index

    @property
    def projection_beta(self) -> float:
        return self.inner.projection_beta

    @property
    def projection_threshold(self) -> float:
        return self.inner.projection_threshold

    @property
    def design_shape(self) -> tuple[int, int]:
        return 2, self.inner.variable_count

    @property
    def variable_count(self) -> int:
        return 2 * self.inner.variable_count

    def _background(self) -> RealArray:
        background = np.asarray(self.fixed_background_epsilon, dtype=float)
        if background.ndim == 0:
            return np.full(self.grid.shape, float(background), dtype=float)
        return background.copy()

    @staticmethod
    def _lateral_symmetry(values: RealArray) -> RealArray:
        return 0.5 * (values + np.flip(values, axis=-1))

    def map(self, raw: ArrayLike) -> PairedNestedRibDensityMapping:
        values = np.asarray(raw, dtype=float)
        if values.size != self.variable_count:
            raise ValueError(
                f"raw density must contain {self.variable_count} variables"
            )
        values = values.reshape(self.design_shape)
        background = self._background()
        inner_mapping = self.inner.map(values[0], background_epsilon=background)
        outer_mapping = self.outer.map(values[1], background_epsilon=background)
        projected_inner_rows = np.asarray(inner_mapping.projected)
        projected_outer_rows = np.asarray(outer_mapping.projected)
        projected_inner = self._lateral_symmetry(projected_inner_rows[:1])[0]
        projected_outer_control = self._lateral_symmetry(
            projected_outer_rows[:1]
        )[0]
        projected_outer_material = projected_inner * projected_outer_control
        index_difference = self.high_index - self.low_index
        inner_index = self.low_index + index_difference * projected_inner
        outer_index = self.low_index + index_difference * projected_outer_material
        inner_epsilon = inner_index**2
        outer_epsilon = outer_index**2
        outer_epsilon_derivative = 2.0 * outer_index * index_difference

        epsilon = background.copy()
        lower_outer = self.lower_outer_rows
        lower_inner = self.lower_inner_rows
        upper_inner = self.upper_inner_rows
        upper_outer = self.upper_outer_rows
        epsilon[slice(*lower_outer)] = outer_epsilon[np.newaxis, :]
        epsilon[slice(*lower_inner)] = inner_epsilon[np.newaxis, :]
        epsilon[slice(*upper_inner)] = np.flipud(epsilon[slice(*lower_inner)])
        epsilon[slice(*upper_outer)] = np.flipud(epsilon[slice(*lower_outer)])
        if not np.array_equal(epsilon, np.flipud(epsilon)):
            raise RuntimeError("nested paired-rib map broke axial symmetry")
        if not np.array_equal(epsilon, np.fliplr(epsilon)):
            raise RuntimeError("nested paired-rib map broke lateral symmetry")
        return PairedNestedRibDensityMapping(
            inner=inner_mapping,
            outer_control=outer_mapping,
            projected_inner=projected_inner,
            projected_outer_control=projected_outer_control,
            projected_outer_material=projected_outer_material,
            outer_epsilon_derivative=outer_epsilon_derivative,
            epsilon_r=epsilon,
        )

    def _fold_gradient(
        self,
        gradient: ComplexArray,
        lower_rows: tuple[int, int],
        upper_rows: tuple[int, int],
    ) -> ComplexArray:
        lower = gradient[..., slice(*lower_rows), :]
        upper = np.flip(
            gradient[..., slice(*upper_rows), :],
            axis=-2,
        )
        combined = lower + upper
        return 0.5 * (combined + np.flip(combined, axis=-1))

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: PairedNestedRibDensityMapping,
    ) -> ComplexArray:
        gradient = np.asarray(epsilon_gradient, dtype=np.complex128)
        if gradient.shape[-2:] != self.grid.shape:
            raise ValueError("epsilon_gradient trailing axes must match the grid")
        leading = gradient.shape[:-2]
        inner_gradient = self._fold_gradient(
            gradient, self.lower_inner_rows, self.upper_inner_rows
        )
        outer_gradient = self._fold_gradient(
            gradient, self.lower_outer_rows, self.upper_outer_rows
        )

        inner_pseudo = np.zeros_like(gradient)
        outer_pseudo = np.zeros_like(gradient)
        inner_start, inner_stop = self.lower_inner_rows
        outer_start, outer_stop = self.lower_outer_rows
        inner_pseudo[..., inner_start:inner_stop, :] = inner_gradient

        inner_from_outer = np.sum(
            outer_gradient
            * mapping.outer_epsilon_derivative
            * mapping.projected_outer_control,
            axis=-2,
        )
        inner_rows = inner_stop - inner_start
        inner_denominator = mapping.inner.epsilon_derivative
        inner_pseudo[..., inner_start:inner_stop, :] += (
            inner_from_outer[..., np.newaxis, :]
            / inner_rows
            / inner_denominator
        )

        outer_from_control = np.sum(
            outer_gradient
            * mapping.outer_epsilon_derivative
            * mapping.projected_inner,
            axis=-2,
        )
        outer_rows = outer_stop - outer_start
        outer_denominator = mapping.outer_control.epsilon_derivative
        outer_pseudo[..., outer_start:outer_stop, :] = (
            outer_from_control[..., np.newaxis, :]
            / outer_rows
            / outer_denominator
        )
        pulled_inner = np.asarray(
            self.inner.pullback(inner_pseudo, mapping.inner)
        ).reshape(*leading, -1)
        pulled_outer = np.asarray(
            self.outer.pullback(outer_pseudo, mapping.outer_control)
        ).reshape(*leading, -1)
        return np.stack((pulled_inner, pulled_outer), axis=-2)


@dataclass
class PairedNestedRibSlabExpansionDensityParameterization:
    """Add one symmetric subcell slab-thickness variable to nested ribs.

    The final variable expands the uniform central slab into the adjacent
    inner-rib row on both sides. A value of zero recovers the base nested-rib
    map; one converts both adjacent interface rows to the high-index material.
    Intermediate values are standard boundary-cell permittivity averages and
    represent a continuous physical displacement of both slab interfaces.
    """

    base: PairedNestedMirrorSymmetricRibDensityParameterization

    @property
    def grid(self) -> UniformGrid2D:
        return self.base.grid

    @property
    def variable_count(self) -> int:
        return self.base.variable_count + 1

    @property
    def design_shape(self) -> tuple[int]:
        return (self.variable_count,)

    @property
    def interface_rows(self) -> tuple[int, int]:
        return self.base.lower_inner_rows[1] - 1, self.base.upper_inner_rows[0]

    def map(self, raw: ArrayLike) -> PairedNestedRibSlabExpansionDensityMapping:
        values = np.asarray(raw, dtype=float).ravel()
        if values.size != self.variable_count:
            raise ValueError(
                f"raw density must contain {self.variable_count} variables"
            )
        expansion = float(values[-1])
        if not 0.0 <= expansion <= 1.0:
            raise ValueError("slab expansion fraction must lie in [0, 1]")
        base_mapping = self.base.map(values[:-1])
        epsilon = base_mapping.epsilon_r.copy()
        high_epsilon = self.base.high_index**2
        for row in self.interface_rows:
            epsilon[row] += expansion * (high_epsilon - epsilon[row])
        if not np.array_equal(epsilon, np.flipud(epsilon)):
            raise RuntimeError("slab expansion broke axial symmetry")
        if not np.array_equal(epsilon, np.fliplr(epsilon)):
            raise RuntimeError("slab expansion broke lateral symmetry")
        return PairedNestedRibSlabExpansionDensityMapping(
            base=base_mapping,
            expansion_fraction=expansion,
            interface_rows=self.interface_rows,
            epsilon_r=epsilon,
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: PairedNestedRibSlabExpansionDensityMapping,
    ) -> ComplexArray:
        gradient = np.asarray(epsilon_gradient, dtype=np.complex128)
        if gradient.shape[-2:] != self.grid.shape:
            raise ValueError("epsilon_gradient trailing axes must match the grid")
        base_gradient = gradient.copy()
        for row in mapping.interface_rows:
            base_gradient[..., row, :] *= 1.0 - mapping.expansion_fraction
        base_pulled = np.asarray(
            self.base.pullback(base_gradient, mapping.base)
        ).reshape(*gradient.shape[:-2], -1)
        high_epsilon = self.base.high_index**2
        expansion_gradient = sum(
            np.sum(
                gradient[..., row, :]
                * (high_epsilon - mapping.base.epsilon_r[row]),
                axis=-1,
            )
            for row in mapping.interface_rows
        )
        return np.concatenate(
            (base_pulled, np.asarray(expansion_gradient)[..., np.newaxis]),
            axis=-1,
        )


@dataclass
class PairedNestedRibSlabContractionDensityParameterization:
    """Add one symmetric subcell slab-contraction variable to nested ribs.

    The final variable replaces the two outermost rows of the uniform slab by
    the adjacent inner-rib pattern. Zero recovers the base map; one contracts
    the slab by one cell on each side. Intermediate values are boundary-cell
    permittivity averages for a symmetric physical interface displacement.
    """

    base: PairedNestedMirrorSymmetricRibDensityParameterization

    @property
    def grid(self) -> UniformGrid2D:
        return self.base.grid

    @property
    def variable_count(self) -> int:
        return self.base.variable_count + 1

    @property
    def design_shape(self) -> tuple[int]:
        return (self.variable_count,)

    @property
    def slab_interface_rows(self) -> tuple[int, int]:
        start, stop = self.base.slab_rows
        return start, stop - 1

    @property
    def rib_interface_rows(self) -> tuple[int, int]:
        return self.base.lower_inner_rows[1] - 1, self.base.upper_inner_rows[0]

    def map(self, raw: ArrayLike) -> PairedNestedRibSlabContractionDensityMapping:
        values = np.asarray(raw, dtype=float).ravel()
        if values.size != self.variable_count:
            raise ValueError(
                f"raw density must contain {self.variable_count} variables"
            )
        contraction = float(values[-1])
        if not 0.0 <= contraction <= 1.0:
            raise ValueError("slab contraction fraction must lie in [0, 1]")
        base_mapping = self.base.map(values[:-1])
        epsilon = base_mapping.epsilon_r.copy()
        for slab_row, rib_row in zip(
            self.slab_interface_rows, self.rib_interface_rows, strict=True
        ):
            epsilon[slab_row] += contraction * (
                base_mapping.epsilon_r[rib_row] - epsilon[slab_row]
            )
        if not np.array_equal(epsilon, np.flipud(epsilon)):
            raise RuntimeError("slab contraction broke axial symmetry")
        if not np.array_equal(epsilon, np.fliplr(epsilon)):
            raise RuntimeError("slab contraction broke lateral symmetry")
        return PairedNestedRibSlabContractionDensityMapping(
            base=base_mapping,
            contraction_fraction=contraction,
            slab_interface_rows=self.slab_interface_rows,
            rib_interface_rows=self.rib_interface_rows,
            epsilon_r=epsilon,
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: PairedNestedRibSlabContractionDensityMapping,
    ) -> ComplexArray:
        gradient = np.asarray(epsilon_gradient, dtype=np.complex128)
        if gradient.shape[-2:] != self.grid.shape:
            raise ValueError("epsilon_gradient trailing axes must match the grid")
        base_gradient = gradient.copy()
        for slab_row, rib_row in zip(
            mapping.slab_interface_rows, mapping.rib_interface_rows, strict=True
        ):
            base_gradient[..., rib_row, :] += (
                mapping.contraction_fraction * gradient[..., slab_row, :]
            )
        base_pulled = np.asarray(
            self.base.pullback(base_gradient, mapping.base)
        ).reshape(*gradient.shape[:-2], -1)
        high_epsilon = self.base.high_index**2
        contraction_gradient = sum(
            np.sum(
                gradient[..., slab_row, :]
                * (mapping.base.epsilon_r[rib_row] - high_epsilon),
                axis=-1,
            )
            for slab_row, rib_row in zip(
                mapping.slab_interface_rows, mapping.rib_interface_rows, strict=True
            )
        )
        return np.concatenate(
            (base_pulled, np.asarray(contraction_gradient)[..., np.newaxis]),
            axis=-1,
        )


@dataclass
class DBRTerminationContractionDensityParameterization:
    """Add one symmetric subcell DBR-termination thickness coordinate.

    ``interface_rows`` identify the outermost cell of the innermost low-index
    DBR layer on each side. The final variable converts those two cells from
    low to high index, moving the high/low DBR interface toward the cavity
    without changing the simulation-domain or cavity row count.
    """

    base: object
    interface_rows: tuple[int, int]
    low_index: float
    high_index: float

    def __post_init__(self) -> None:
        if len(self.interface_rows) != 2:
            raise ValueError("exactly two mirrored DBR interface rows are required")
        lower, upper = (int(value) for value in self.interface_rows)
        if lower < 0 or upper >= self.grid.nz or lower + upper != self.grid.nz - 1:
            raise ValueError("DBR interface rows must be valid axial mirrors")
        if not float(self.high_index) > float(self.low_index) > 0.0:
            raise ValueError("DBR indices must satisfy high > low > 0")

    @property
    def grid(self) -> UniformGrid2D:
        return self.base.grid

    @property
    def variable_count(self) -> int:
        return int(self.base.variable_count) + 1

    @property
    def design_shape(self) -> tuple[int]:
        return (self.variable_count,)

    def map(self, raw: ArrayLike) -> DBRTerminationContractionDensityMapping:
        values = np.asarray(raw, dtype=float).ravel()
        if values.size != self.variable_count:
            raise ValueError(
                f"raw density must contain {self.variable_count} variables"
            )
        contraction = float(values[-1])
        if not 0.0 <= contraction <= 1.0:
            raise ValueError("DBR contraction fraction must lie in [0, 1]")
        base_mapping = self.base.map(values[:-1])
        epsilon = np.asarray(base_mapping.epsilon_r).copy()
        difference = float(self.high_index) ** 2 - float(self.low_index) ** 2
        for row in self.interface_rows:
            epsilon[row] += contraction * difference
        if not np.array_equal(epsilon, np.flipud(epsilon)):
            raise RuntimeError("DBR termination contraction broke axial symmetry")
        if not np.array_equal(epsilon, np.fliplr(epsilon)):
            raise RuntimeError("DBR termination contraction broke lateral symmetry")
        return DBRTerminationContractionDensityMapping(
            base=base_mapping,
            contraction_fraction=contraction,
            interface_rows=self.interface_rows,
            epsilon_r=epsilon,
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: DBRTerminationContractionDensityMapping,
    ) -> ComplexArray:
        gradient = np.asarray(epsilon_gradient, dtype=np.complex128)
        if gradient.shape[-2:] != self.grid.shape:
            raise ValueError("epsilon_gradient trailing axes must match the grid")
        base_pulled = np.asarray(
            self.base.pullback(gradient, mapping.base)
        ).reshape(*gradient.shape[:-2], -1)
        difference = float(self.high_index) ** 2 - float(self.low_index) ** 2
        contraction_gradient = difference * sum(
            np.sum(gradient[..., row, :], axis=-1) for row in mapping.interface_rows
        )
        return np.concatenate(
            (base_pulled, np.asarray(contraction_gradient)[..., np.newaxis]),
            axis=-1,
        )


@dataclass
class SymmetricDBRTerminationShiftDensityParameterization:
    """Append a signed, symmetry-preserving DBR-interface coordinate.

    Positive shift converts the outermost row of the innermost low-index DBR
    layer toward high index, moving the low/high interface toward the cavity.
    Negative shift converts the adjacent high-index row toward low index,
    moving the same interface away from the cavity.  A magnitude of one is one
    axial grid cell.  At the unshifted, cell-aligned interface the discrete map
    is nondifferentiable; its pullback uses the centered subgradient across the
    two adjacent rows.
    """

    base: object
    low_interface_rows: tuple[int, int]
    high_interface_rows: tuple[int, int]
    low_index: float
    high_index: float
    shift_bounds_cells: tuple[float, float] = (-1.0, 1.0)

    def __post_init__(self) -> None:
        if len(self.low_interface_rows) != 2 or len(self.high_interface_rows) != 2:
            raise ValueError("exactly two mirrored low and high interface rows are required")
        lower_low, upper_low = (int(value) for value in self.low_interface_rows)
        lower_high, upper_high = (int(value) for value in self.high_interface_rows)
        if (
            lower_low < 0
            or upper_low >= self.grid.nz
            or lower_high < 0
            or upper_high >= self.grid.nz
        ):
            raise ValueError("DBR interface rows must lie inside the grid")
        if (
            lower_low + upper_low != self.grid.nz - 1
            or lower_high + upper_high != self.grid.nz - 1
        ):
            raise ValueError("DBR interface rows must form axial mirror pairs")
        if lower_high + 1 != lower_low or upper_low + 1 != upper_high:
            raise ValueError("high and low DBR interface rows must be adjacent")
        if not float(self.high_index) > float(self.low_index) > 0.0:
            raise ValueError("DBR indices must satisfy high > low > 0")
        lower, upper = (float(value) for value in self.shift_bounds_cells)
        if not -1.0 <= lower < 0.0 < upper <= 1.0:
            raise ValueError("shift bounds must straddle zero inside [-1, 1]")

    @property
    def grid(self) -> UniformGrid2D:
        return self.base.grid

    @property
    def variable_count(self) -> int:
        return int(self.base.variable_count) + 1

    @property
    def design_shape(self) -> tuple[int]:
        return (self.variable_count,)

    @property
    def shift_variable_index(self) -> int:
        return self.variable_count - 1

    def encode_shift_cells(self, shift_cells: float) -> float:
        lower, upper = (float(value) for value in self.shift_bounds_cells)
        raw = (float(shift_cells) - lower) / (upper - lower)
        if not 0.0 <= raw <= 1.0:
            raise ValueError("initial termination shift lies outside configured bounds")
        return float(raw)

    def decode_shift_cells(self, raw: ArrayLike) -> float:
        values = np.asarray(raw, dtype=float).ravel()
        if values.size != self.variable_count:
            raise ValueError(
                f"raw density must contain {self.variable_count} variables"
            )
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("raw density values must lie within [0, 1]")
        lower, upper = (float(value) for value in self.shift_bounds_cells)
        return float(lower + (upper - lower) * values[-1])

    def map(self, raw: ArrayLike) -> SymmetricDBRTerminationShiftDensityMapping:
        values = np.asarray(raw, dtype=float).ravel()
        shift = self.decode_shift_cells(values)
        base_mapping = self.base.map(values[:-1])
        epsilon = np.asarray(base_mapping.epsilon_r, dtype=float).copy()
        difference = float(self.high_index) ** 2 - float(self.low_index) ** 2
        derivative = np.zeros(self.grid.shape, dtype=float)
        span = float(self.shift_bounds_cells[1] - self.shift_bounds_cells[0])
        if shift > np.finfo(float).eps:
            for row in self.low_interface_rows:
                epsilon[row] += shift * difference
                derivative[row] = span * difference
        elif shift < -np.finfo(float).eps:
            for row in self.high_interface_rows:
                epsilon[row] += shift * difference
                derivative[row] = span * difference
        else:
            for row in (*self.low_interface_rows, *self.high_interface_rows):
                derivative[row] = 0.5 * span * difference
        if not np.array_equal(epsilon, np.flipud(epsilon)):
            raise RuntimeError("DBR termination shift broke axial symmetry")
        if not np.array_equal(epsilon, np.fliplr(epsilon)):
            raise RuntimeError("DBR termination shift broke lateral symmetry")
        return SymmetricDBRTerminationShiftDensityMapping(
            base=base_mapping,
            shift_cells=shift,
            low_interface_rows=self.low_interface_rows,
            high_interface_rows=self.high_interface_rows,
            shift_epsilon_derivative=derivative,
            epsilon_r=epsilon,
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: SymmetricDBRTerminationShiftDensityMapping,
    ) -> ComplexArray:
        gradient = np.asarray(epsilon_gradient, dtype=np.complex128)
        if gradient.shape[-2:] != self.grid.shape:
            raise ValueError("epsilon_gradient trailing axes must match the grid")
        base_pulled = np.asarray(
            self.base.pullback(gradient, mapping.base)
        ).reshape(*gradient.shape[:-2], -1)
        shift_gradient = np.sum(
            gradient * mapping.shift_epsilon_derivative,
            axis=(-2, -1),
        )
        return np.concatenate(
            (base_pulled, shift_gradient[..., np.newaxis]),
            axis=-1,
        )


@dataclass
class NestedTwoTierMirrorSymmetricPlanarDensityParameterization:
    """Two nested lateral masks defining two independently etched heights.

    The lower mask is extruded through the lower patterned tier.  Material in
    the upper tier is the product of the projected lower mask and a second
    projected control mask.  Consequently the upper material is always a
    subset of the lower support, as required for two sequential etch depths in
    one initially uniform film.  A fixed residual slab or other fabrication
    background may be supplied separately and is excluded from both masks.

    Each lateral mask is mirror symmetric.  The pullback includes the product
    rule through the nested upper material and the exact transpose of both
    Helmholtz filters.
    """

    lower: MirrorSymmetricPlanarLayerDensityParameterization
    upper: MirrorSymmetricPlanarLayerDensityParameterization
    fixed_background_epsilon: ArrayLike | float

    def __post_init__(self) -> None:
        if self.lower.grid.shape != self.upper.grid.shape or not np.isclose(
            self.lower.grid.period, self.upper.grid.period
        ):
            raise ValueError("lower and upper tiers must use the same grid")
        if self.lower.design_rows[1] != self.upper.design_rows[0]:
            raise ValueError("lower and upper tier row intervals must be contiguous")
        if self.lower.variable_count != self.upper.variable_count:
            raise ValueError("lower and upper masks must have equal variable counts")
        if not np.isclose(self.lower.low_index, self.upper.low_index) or not np.isclose(
            self.lower.high_index, self.upper.high_index
        ):
            raise ValueError("lower and upper tiers must use the same materials")
        if (
            self.lower.fixed_background_epsilon is not None
            or self.upper.fixed_background_epsilon is not None
        ):
            raise ValueError(
                "tier adapters must not define their own fixed backgrounds"
            )
        background = np.asarray(self.fixed_background_epsilon, dtype=float)
        if background.ndim not in {0, 2}:
            raise ValueError("fixed background must be scalar or two dimensional")
        if background.ndim == 2 and background.shape != self.grid.shape:
            raise ValueError("fixed background must match the full grid")

    @property
    def grid(self) -> UniformGrid2D:
        return self.lower.grid

    @property
    def design_rows(self) -> tuple[int, int]:
        return self.lower.design_rows[0], self.upper.design_rows[1]

    @property
    def tier_rows(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return self.lower.design_rows, self.upper.design_rows

    @property
    def low_index(self) -> float:
        return self.lower.low_index

    @property
    def high_index(self) -> float:
        return self.lower.high_index

    @property
    def projection_beta(self) -> float:
        return self.lower.projection_beta

    @property
    def projection_threshold(self) -> float:
        return self.lower.projection_threshold

    @property
    def design_shape(self) -> tuple[int, int]:
        return 2, self.lower.variable_count

    @property
    def variable_count(self) -> int:
        return 2 * self.lower.variable_count

    def _background(self) -> RealArray:
        background = np.asarray(self.fixed_background_epsilon, dtype=float)
        if background.ndim == 0:
            return np.full(self.grid.shape, float(background), dtype=float)
        return background.copy()

    def map(self, raw: ArrayLike) -> NestedTwoTierDensityMapping:
        values = np.asarray(raw, dtype=float)
        if values.size != self.variable_count:
            raise ValueError(
                f"raw density must contain {self.variable_count} variables"
            )
        values = values.reshape(self.design_shape)
        background = self._background()
        lower_mapping = self.lower.map(
            values[0], background_epsilon=background
        )
        upper_mapping = self.upper.map(
            values[1], background_epsilon=background
        )
        projected_lower_rows = np.asarray(lower_mapping.projected)
        projected_upper_rows = np.asarray(upper_mapping.projected)
        if not np.allclose(
            projected_lower_rows, projected_lower_rows[:1], rtol=0.0, atol=1e-11
        ) or not np.allclose(
            projected_upper_rows, projected_upper_rows[:1], rtol=0.0, atol=1e-11
        ):
            raise RuntimeError("extruded tier filtering must remain axially uniform")
        projected_lower = projected_lower_rows[0]
        projected_upper_control = projected_upper_rows[0]
        projected_upper_material = projected_lower * projected_upper_control
        index_difference = self.high_index - self.low_index
        upper_index = self.low_index + index_difference * projected_upper_material
        upper_epsilon = upper_index**2
        upper_epsilon_derivative = 2.0 * upper_index * index_difference

        epsilon = background.copy()
        lower_start, lower_stop = self.lower.design_rows
        upper_start, upper_stop = self.upper.design_rows
        epsilon[lower_start:lower_stop] = lower_mapping.epsilon_r[
            lower_start:lower_stop
        ]
        epsilon[upper_start:upper_stop] = upper_epsilon[np.newaxis, :]
        return NestedTwoTierDensityMapping(
            lower=lower_mapping,
            upper_control=upper_mapping,
            projected_lower=projected_lower,
            projected_upper_control=projected_upper_control,
            projected_upper_material=projected_upper_material,
            upper_epsilon_derivative=upper_epsilon_derivative,
            epsilon_r=epsilon,
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: NestedTwoTierDensityMapping,
    ) -> ComplexArray:
        gradient = np.asarray(epsilon_gradient, dtype=np.complex128)
        if gradient.shape[-2:] != self.grid.shape:
            raise ValueError("epsilon_gradient trailing axes must match the grid")
        leading_shape = gradient.shape[:-2]
        lower_start, lower_stop = self.lower.design_rows
        upper_start, upper_stop = self.upper.design_rows

        lower_direct = np.asarray(self.lower.pullback(gradient, mapping.lower))
        upper_gradient = gradient[..., upper_start:upper_stop, :]
        top_derivative = mapping.upper_epsilon_derivative

        # Convert sensitivities with respect to the nested projected densities
        # into equivalent permittivity sensitivities for each tier adapter.
        upper_pseudo = np.zeros_like(gradient)
        upper_denominator = mapping.upper_control.epsilon_derivative
        upper_pseudo[..., upper_start:upper_stop, :] = (
            upper_gradient
            * top_derivative
            * mapping.projected_lower
            / upper_denominator
        )
        upper_pulled = np.asarray(
            self.upper.pullback(upper_pseudo, mapping.upper_control)
        )

        lower_from_upper = np.sum(
            upper_gradient
            * top_derivative
            * mapping.projected_upper_control,
            axis=-2,
        )
        lower_rows = lower_stop - lower_start
        lower_pseudo = np.zeros_like(gradient)
        lower_denominator = mapping.lower.epsilon_derivative
        lower_pseudo[..., lower_start:lower_stop, :] = (
            lower_from_upper[..., np.newaxis, :]
            / lower_rows
            / lower_denominator
        )
        lower_pulled = lower_direct + np.asarray(
            self.lower.pullback(lower_pseudo, mapping.lower)
        )

        return np.stack(
            (
                lower_pulled.reshape(*leading_shape, -1),
                upper_pulled.reshape(*leading_shape, -1),
            ),
            axis=-2,
        )


@dataclass
class SeparatedTwoLayerMirrorSymmetricPlanarDensityParameterization:
    """Two independent lateral masks separated by a fixed uniform spacer.

    Each mask is extruded through one non-overlapping high-index layer. The
    axial interval between them and every region outside the two layers retain
    ``fixed_background_epsilon``. This represents two separately patterned
    films with a planarized or deposited homogeneous spacer. No symmetry is
    imposed between the two layers, although each lateral mask is mirror
    symmetric within the periodic unit cell.
    """

    lower: MirrorSymmetricPlanarLayerDensityParameterization
    upper: MirrorSymmetricPlanarLayerDensityParameterization
    fixed_background_epsilon: ArrayLike | float

    def __post_init__(self) -> None:
        if self.lower.grid.shape != self.upper.grid.shape or not np.isclose(
            self.lower.grid.period, self.upper.grid.period
        ):
            raise ValueError("lower and upper layers must use the same grid")
        if self.lower.design_rows[1] >= self.upper.design_rows[0]:
            raise ValueError("the two layers must be separated by a nonempty spacer")
        if self.lower.variable_count != self.upper.variable_count:
            raise ValueError("lower and upper masks must have equal variable counts")
        if not np.isclose(self.lower.low_index, self.upper.low_index) or not np.isclose(
            self.lower.high_index, self.upper.high_index
        ):
            raise ValueError("lower and upper layers must use the same materials")
        if (
            self.lower.fixed_background_epsilon is not None
            or self.upper.fixed_background_epsilon is not None
        ):
            raise ValueError(
                "layer adapters must not define their own fixed backgrounds"
            )
        background = np.asarray(self.fixed_background_epsilon, dtype=float)
        if background.ndim not in {0, 2}:
            raise ValueError("fixed background must be scalar or two dimensional")
        if background.ndim == 2 and background.shape != self.grid.shape:
            raise ValueError("fixed background must match the full grid")

    @property
    def grid(self) -> UniformGrid2D:
        return self.lower.grid

    @property
    def design_rows(self) -> tuple[int, int]:
        return self.lower.design_rows[0], self.upper.design_rows[1]

    @property
    def layer_rows(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return self.lower.design_rows, self.upper.design_rows

    @property
    def spacer_rows(self) -> tuple[int, int]:
        return self.lower.design_rows[1], self.upper.design_rows[0]

    @property
    def low_index(self) -> float:
        return self.lower.low_index

    @property
    def high_index(self) -> float:
        return self.lower.high_index

    @property
    def projection_beta(self) -> float:
        return self.lower.projection_beta

    @property
    def projection_threshold(self) -> float:
        return self.lower.projection_threshold

    @property
    def design_shape(self) -> tuple[int, int]:
        return 2, self.lower.variable_count

    @property
    def variable_count(self) -> int:
        return 2 * self.lower.variable_count

    def _background(self) -> RealArray:
        background = np.asarray(self.fixed_background_epsilon, dtype=float)
        if background.ndim == 0:
            return np.full(self.grid.shape, float(background), dtype=float)
        return background.copy()

    def map(self, raw: ArrayLike) -> SeparatedTwoLayerDensityMapping:
        values = np.asarray(raw, dtype=float)
        if values.size != self.variable_count:
            raise ValueError(
                f"raw density must contain {self.variable_count} variables"
            )
        values = values.reshape(self.design_shape)
        background = self._background()
        lower_mapping = self.lower.map(
            values[0], background_epsilon=background
        )
        upper_mapping = self.upper.map(
            values[1], background_epsilon=background
        )
        epsilon = background.copy()
        lower_start, lower_stop = self.lower.design_rows
        upper_start, upper_stop = self.upper.design_rows
        epsilon[lower_start:lower_stop] = lower_mapping.epsilon_r[
            lower_start:lower_stop
        ]
        epsilon[upper_start:upper_stop] = upper_mapping.epsilon_r[
            upper_start:upper_stop
        ]
        return SeparatedTwoLayerDensityMapping(
            lower=lower_mapping,
            upper=upper_mapping,
            epsilon_r=epsilon,
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: SeparatedTwoLayerDensityMapping,
    ) -> ComplexArray:
        lower = np.asarray(self.lower.pullback(epsilon_gradient, mapping.lower))
        upper = np.asarray(self.upper.pullback(epsilon_gradient, mapping.upper))
        leading_shape = lower.shape[:-2]
        return np.stack(
            (
                lower.reshape(*leading_shape, -1),
                upper.reshape(*leading_shape, -1),
            ),
            axis=-2,
        )


@dataclass
class CrossGridSymmetricDensityParameterization:
    """Map one physical symmetric design to a different discretization.

    Variables live on ``master_grid``. They are expanded by axial symmetry,
    embedded in the full master domain, and physically resampled onto the
    target grid before the target filter and projection are applied. The
    pullback uses the exact transpose interpolation. This gives simultaneous
    multi-mesh optimization one shared design vector.
    """

    master_grid: UniformGrid2D
    master_design_rows: tuple[int, int]
    target: DensityParameterization

    def __post_init__(self) -> None:
        start, stop = self.master_design_rows
        if not 0 <= start < stop <= self.master_grid.nz:
            raise ValueError("master_design_rows must lie inside master_grid")
        if start != self.master_grid.nz - stop:
            raise ValueError("master design interval must be axially centered")
        if not np.isclose(
            self.master_grid.period,
            self.target.grid.period,
            rtol=0.0,
            atol=1e-12,
        ) or not np.isclose(
            self.master_grid.height,
            self.target.grid.height,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("master and target grids must cover the same cell")

    @property
    def grid(self) -> UniformGrid2D:
        return self.target.grid

    @property
    def low_index(self) -> float:
        return self.target.low_index

    @property
    def high_index(self) -> float:
        return self.target.high_index

    @property
    def projection_beta(self) -> float:
        return self.target.projection_beta

    @property
    def projection_threshold(self) -> float:
        return self.target.projection_threshold

    @property
    def full_design_shape(self) -> tuple[int, int]:
        start, stop = self.master_design_rows
        return stop - start, self.master_grid.nx

    @property
    def design_shape(self) -> tuple[int, int]:
        rows, columns = self.full_design_shape
        return (rows + 1) // 2, columns

    @property
    def variable_count(self) -> int:
        rows, columns = self.design_shape
        return rows * columns

    def expand(self, raw: ArrayLike) -> RealArray:
        independent = np.asarray(raw, dtype=float)
        if independent.size != self.variable_count:
            raise ValueError(
                f"raw density must contain {self.variable_count} variables"
            )
        independent = independent.reshape(self.design_shape)
        full_rows, _ = self.full_design_shape
        if full_rows % 2 == 0:
            return np.concatenate((independent, np.flipud(independent)), axis=0)
        return np.concatenate(
            (independent, np.flipud(independent[:-1])), axis=0
        )

    def reduce(self, full_raw: ArrayLike) -> RealArray:
        values = np.asarray(full_raw, dtype=float)
        if values.shape != self.full_design_shape:
            raise ValueError(
                f"full raw density must have shape {self.full_design_shape}"
            )
        if not np.allclose(values, np.flipud(values), rtol=0.0, atol=1e-12):
            raise ValueError("full raw density must be axially mirror symmetric")
        independent_rows, _ = self.design_shape
        return np.asarray(values[:independent_rows])

    def _target_raw(self, raw: ArrayLike) -> RealArray:
        master = np.zeros(self.master_grid.shape, dtype=float)
        start, stop = self.master_design_rows
        master[start:stop] = self.expand(raw)
        target_full = periodic_bilinear_resample(master, self.target.grid.shape)
        target_start, target_stop = self.target.design_rows
        return np.asarray(target_full[target_start:target_stop])

    def map(
        self,
        raw: ArrayLike,
        *,
        background_epsilon: ArrayLike | float | None = None,
    ) -> DensityMapping:
        return self.target.map(
            self._target_raw(raw), background_epsilon=background_epsilon
        )

    def pullback(
        self,
        epsilon_gradient: ArrayLike,
        mapping: DensityMapping,
    ) -> ComplexArray:
        target_local = np.asarray(
            self.target.pullback(epsilon_gradient, mapping)
        )
        leading_shape = target_local.shape[:-2]
        target_full = np.zeros(
            (*leading_shape, *self.target.grid.shape), dtype=target_local.dtype
        )
        target_start, target_stop = self.target.design_rows
        target_full[..., target_start:target_stop, :] = target_local
        master_full = periodic_bilinear_resample_adjoint(
            target_full, self.master_grid.shape
        )
        start, stop = self.master_design_rows
        local = master_full[..., start:stop, :]
        full_rows, _ = self.full_design_shape
        independent_rows, _ = self.design_shape
        lower = local[..., :independent_rows, :].copy()
        if full_rows % 2 == 0:
            lower += np.flip(local[..., independent_rows:, :], axis=-2)
        else:
            paired = independent_rows - 1
            lower[..., :paired, :] += np.flip(
                local[..., independent_rows:, :], axis=-2
            )
        return np.asarray(lower)
