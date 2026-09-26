"""Published geometries used for independent solver validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pzid.grid import UniformGrid2D


RealArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrapezoidalMembraneGrating:
    """One trapezoidal ridge on a full-period membrane slab.

    All lengths use the same unit as the supplied grid. The ridge is centered
    in the periodic cell and sits on top of a homogeneous membrane. Subpixel
    sampling returns an area-averaged relative permittivity for reproducible
    representation of slanted and non-grid-aligned interfaces.
    """

    period: float
    slab_thickness: float
    ridge_height: float
    ridge_bottom_width: float
    ridge_top_width: float
    refractive_index: float
    background_index: float = 1.0

    def __post_init__(self) -> None:
        lengths = (
            self.period,
            self.slab_thickness,
            self.ridge_height,
            self.ridge_bottom_width,
            self.ridge_top_width,
        )
        if any(value <= 0.0 for value in lengths):
            raise ValueError("all grating lengths must be positive")
        if max(self.ridge_bottom_width, self.ridge_top_width) > self.period:
            raise ValueError("ridge widths must not exceed the period")
        if self.refractive_index <= self.background_index or self.background_index <= 0.0:
            raise ValueError("indices must satisfy 0 < background < grating")

    @property
    def total_thickness(self) -> float:
        return self.slab_thickness + self.ridge_height

    def relative_permittivity(
        self,
        grid: UniformGrid2D,
        *,
        base_z: float,
        subpixels: int = 4,
    ) -> RealArray:
        """Rasterize the grating on ``grid`` using deterministic subpixels."""

        if not np.isclose(grid.period, self.period, rtol=0.0, atol=1e-12):
            raise ValueError("grid period must equal the benchmark period")
        if subpixels < 1:
            raise ValueError("subpixels must be positive")
        if base_z <= 0.0 or base_z + self.total_thickness >= grid.height:
            raise ValueError("grating must lie strictly inside the grid")

        offsets = (np.arange(subpixels, dtype=float) + 0.5) / subpixels - 0.5
        sample_x = grid.x[:, None] + offsets[None, :] * grid.dx
        sample_z = grid.z[:, None] + offsets[None, :] * grid.dz
        x_values = np.broadcast_to(
            sample_x[None, :, :, None],
            (grid.nz, grid.nx, subpixels, subpixels),
        )
        z_values = np.broadcast_to(
            sample_z[:, None, None, :],
            (grid.nz, grid.nx, subpixels, subpixels),
        )

        slab_top = base_z + self.slab_thickness
        slab = (z_values >= base_z) & (z_values < slab_top)
        ridge_fraction = np.clip(
            (z_values - slab_top) / self.ridge_height, 0.0, 1.0
        )
        ridge_width = self.ridge_bottom_width + ridge_fraction * (
            self.ridge_top_width - self.ridge_bottom_width
        )
        periodic_distance = np.abs(
            ((x_values - 0.5 * self.period + 0.5 * self.period) % self.period)
            - 0.5 * self.period
        )
        ridge = (
            (z_values >= slab_top)
            & (z_values < slab_top + self.ridge_height)
            & (periodic_distance <= 0.5 * ridge_width)
        )
        fill = np.mean(slab | ridge, axis=(2, 3))
        low_epsilon = self.background_index**2
        high_epsilon = self.refractive_index**2
        return np.asarray(low_epsilon + fill * (high_epsilon - low_epsilon))


@dataclass(frozen=True)
class RectangularDielectricBarArray:
    """One centered rectangular dielectric bar per periodic 2D cell.

    The bar is invariant along ``y`` and centered in both ``x`` and ``z``.
    It is the 2D normal-incidence analog of the high-index dielectric
    particles commonly used for electric/magnetic Huygens metasurfaces.
    """

    period: float
    width: float
    height: float
    refractive_index: float
    background_index: float = 1.0

    def __post_init__(self) -> None:
        if self.period <= 0.0 or self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("period, width, and height must be positive")
        if self.width > self.period:
            raise ValueError("bar width must not exceed the period")
        if (
            self.background_index <= 0.0
            or self.refractive_index <= self.background_index
        ):
            raise ValueError("indices must satisfy 0 < background < bar")

    def relative_permittivity(
        self,
        grid: UniformGrid2D,
        *,
        center_z: float | None = None,
        subpixels: int = 4,
    ) -> RealArray:
        """Rasterize the centered bar with deterministic area averaging."""

        if not np.isclose(grid.period, self.period, rtol=0.0, atol=1e-12):
            raise ValueError("grid period must equal the benchmark period")
        if subpixels < 1:
            raise ValueError("subpixels must be positive")
        center = 0.5 * grid.height if center_z is None else float(center_z)
        if (
            center - 0.5 * self.height <= 0.0
            or center + 0.5 * self.height >= grid.height
        ):
            raise ValueError("bar must lie strictly inside the grid")

        offsets = (np.arange(subpixels, dtype=float) + 0.5) / subpixels - 0.5
        sample_x = grid.x[:, None] + offsets[None, :] * grid.dx
        sample_z = grid.z[:, None] + offsets[None, :] * grid.dz
        periodic_distance = np.abs(
            ((sample_x - 0.5 * self.period + 0.5 * self.period) % self.period)
            - 0.5 * self.period
        )
        inside_x = periodic_distance <= 0.5 * self.width
        inside_z = np.abs(sample_z - center) <= 0.5 * self.height
        fill = np.mean(
            inside_z[:, None, None, :] & inside_x[None, :, :, None],
            axis=(2, 3),
        )
        low_epsilon = self.background_index**2
        high_epsilon = self.refractive_index**2
        epsilon = low_epsilon + fill * (high_epsilon - low_epsilon)
        # Eliminate subpixel tie-breaking at the axial symmetry plane.
        return np.asarray(0.5 * (epsilon + np.flipud(epsilon)))


@dataclass(frozen=True)
class PlanarPatternedLayer:
    """One fixed-thickness, single-etch layer with vertical sidewalls.

    ``radial_intervals`` lists the dielectric intervals in terms of periodic
    distance from the unit-cell center.  For example, ``((0, width / 2),)``
    is a centered bar, while ``((inner / 2, outer / 2),)`` is a centered slot
    splitting that bar into two laterally separated rails.  This radial form
    imposes the mirror symmetry appropriate to normal incidence.
    """

    period: float
    thickness: float
    radial_intervals: tuple[tuple[float, float], ...]
    refractive_index: float
    background_index: float = 1.0

    def __post_init__(self) -> None:
        if self.period <= 0.0 or self.thickness <= 0.0:
            raise ValueError("period and thickness must be positive")
        if not self.radial_intervals:
            raise ValueError("radial_intervals must be nonempty")
        previous = -np.inf
        for lower, upper in self.radial_intervals:
            if lower < 0.0 or upper <= lower or upper > 0.5 * self.period:
                raise ValueError(
                    "radial intervals must satisfy 0 <= lower < upper <= period/2"
                )
            if lower < previous:
                raise ValueError("radial intervals must be sorted and disjoint")
            previous = upper
        if (
            self.background_index <= 0.0
            or self.refractive_index <= self.background_index
        ):
            raise ValueError("indices must satisfy 0 < background < layer")

    def relative_permittivity(
        self,
        grid: UniformGrid2D,
        *,
        center_z: float | None = None,
        subpixels: int = 4,
    ) -> RealArray:
        """Rasterize the planar pattern with deterministic area averaging."""

        if not np.isclose(grid.period, self.period, rtol=0.0, atol=1e-12):
            raise ValueError("grid period must equal the patterned-layer period")
        if subpixels < 1:
            raise ValueError("subpixels must be positive")
        center = 0.5 * grid.height if center_z is None else float(center_z)
        if center - 0.5 * self.thickness <= 0.0 or center + 0.5 * self.thickness >= grid.height:
            raise ValueError("patterned layer must lie strictly inside the grid")

        offsets = (np.arange(subpixels, dtype=float) + 0.5) / subpixels - 0.5
        sample_x = grid.x[:, None] + offsets[None, :] * grid.dx
        sample_z = grid.z[:, None] + offsets[None, :] * grid.dz
        distance = np.abs(
            ((sample_x - 0.5 * self.period + 0.5 * self.period) % self.period)
            - 0.5 * self.period
        )
        inside_x = np.zeros_like(distance, dtype=bool)
        for lower, upper in self.radial_intervals:
            inside_x |= (distance >= lower) & (distance <= upper)
        inside_z = np.abs(sample_z - center) <= 0.5 * self.thickness
        fill = np.mean(
            inside_z[:, None, None, :] & inside_x[None, :, :, None],
            axis=(2, 3),
        )
        low_epsilon = self.background_index**2
        high_epsilon = self.refractive_index**2
        epsilon = low_epsilon + fill * (high_epsilon - low_epsilon)
        return np.asarray(0.5 * (epsilon + np.flipud(epsilon)))


@dataclass(frozen=True)
class PartiallyEtchedPatternedLayer:
    """One patterned film with a uniform residual slab.

    The geometry is the cross-section produced by one lithography step and a
    timed etch of depth ``etch_depth`` into a film of thickness
    ``film_thickness``.  Protected lateral intervals retain the full film;
    exposed intervals retain a uniform slab of thickness
    ``film_thickness - etch_depth``.  ``etch_side`` declares whether the etch
    enters from the upper or lower side of the simulation cell.

    Unlike :class:`PlanarPatternedLayer`, a one-sided partial etch is not
    axially mirror symmetric.  The implementation therefore preserves the
    requested orientation instead of averaging it with its axial reflection.
    """

    period: float
    film_thickness: float
    etch_depth: float
    radial_intervals: tuple[tuple[float, float], ...]
    refractive_index: float
    background_index: float = 1.0
    etch_side: str = "upper"

    def __post_init__(self) -> None:
        if self.period <= 0.0 or self.film_thickness <= 0.0:
            raise ValueError("period and film_thickness must be positive")
        if not 0.0 < self.etch_depth < self.film_thickness:
            raise ValueError(
                "etch_depth must lie strictly between zero and film_thickness"
            )
        if self.etch_side not in {"upper", "lower"}:
            raise ValueError("etch_side must be 'upper' or 'lower'")
        if not self.radial_intervals:
            raise ValueError("radial_intervals must be nonempty")
        previous = -np.inf
        for lower, upper in self.radial_intervals:
            if lower < 0.0 or upper <= lower or upper > 0.5 * self.period:
                raise ValueError(
                    "radial intervals must satisfy 0 <= lower < upper <= period/2"
                )
            if lower < previous:
                raise ValueError("radial intervals must be sorted and disjoint")
            previous = upper
        if (
            self.background_index <= 0.0
            or self.refractive_index <= self.background_index
        ):
            raise ValueError("indices must satisfy 0 < background < layer")

    @property
    def slab_thickness(self) -> float:
        """Thickness remaining in every lateral position after etching."""

        return self.film_thickness - self.etch_depth

    def relative_permittivity(
        self,
        grid: UniformGrid2D,
        *,
        center_z: float | None = None,
        subpixels: int = 4,
    ) -> RealArray:
        """Rasterize the partially etched film by deterministic area averaging."""

        if not np.isclose(grid.period, self.period, rtol=0.0, atol=1e-12):
            raise ValueError("grid period must equal the patterned-film period")
        if subpixels < 1:
            raise ValueError("subpixels must be positive")
        center = 0.5 * grid.height if center_z is None else float(center_z)
        lower_film = center - 0.5 * self.film_thickness
        upper_film = center + 0.5 * self.film_thickness
        if lower_film <= 0.0 or upper_film >= grid.height:
            raise ValueError("patterned film must lie strictly inside the grid")

        offsets = (np.arange(subpixels, dtype=float) + 0.5) / subpixels - 0.5
        sample_x = grid.x[:, None] + offsets[None, :] * grid.dx
        sample_z = grid.z[:, None] + offsets[None, :] * grid.dz
        x_values = np.broadcast_to(
            sample_x[None, :, :, None],
            (grid.nz, grid.nx, subpixels, subpixels),
        )
        z_values = np.broadcast_to(
            sample_z[:, None, None, :],
            (grid.nz, grid.nx, subpixels, subpixels),
        )
        distance = np.abs(
            ((x_values - 0.5 * self.period + 0.5 * self.period) % self.period)
            - 0.5 * self.period
        )
        protected = np.zeros_like(distance, dtype=bool)
        for lower, upper in self.radial_intervals:
            protected |= (distance >= lower) & (distance <= upper)

        inside_film = (z_values >= lower_film) & (z_values < upper_film)
        if self.etch_side == "upper":
            residual_slab = (
                (z_values >= lower_film)
                & (z_values < upper_film - self.etch_depth)
            )
        else:
            residual_slab = (
                (z_values >= lower_film + self.etch_depth)
                & (z_values < upper_film)
            )
        material = residual_slab | (inside_film & protected)
        fill = np.mean(material, axis=(2, 3))
        low_epsilon = self.background_index**2
        high_epsilon = self.refractive_index**2
        return np.asarray(low_epsilon + fill * (high_epsilon - low_epsilon))


@dataclass(frozen=True)
class CoupledRectangularBarArray:
    """Axially coupled pair of centered rectangular dielectric bars.

    The two identical bars are separated by a uniform background-index spacer
    and mirrored about the center of the computational cell.  The spacer is a
    genuine part of the Maxwell geometry rather than an added propagation
    phase, so the same pixel-density adjoint can subsequently reshape both
    bars and the intervening gap.
    """

    period: float
    width: float
    bar_height: float
    central_gap: float
    refractive_index: float
    background_index: float = 1.0

    def __post_init__(self) -> None:
        if (
            self.period <= 0.0
            or self.width <= 0.0
            or self.bar_height <= 0.0
            or self.central_gap <= 0.0
        ):
            raise ValueError("period, width, bar_height, and central_gap must be positive")
        if self.width > self.period:
            raise ValueError("bar width must not exceed the period")
        if (
            self.background_index <= 0.0
            or self.refractive_index <= self.background_index
        ):
            raise ValueError("indices must satisfy 0 < background < bar")

    @property
    def total_thickness(self) -> float:
        return 2.0 * self.bar_height + self.central_gap

    def relative_permittivity(
        self,
        grid: UniformGrid2D,
        *,
        center_z: float | None = None,
        subpixels: int = 4,
    ) -> RealArray:
        """Rasterize the coupled bars with deterministic area averaging."""

        if not np.isclose(grid.period, self.period, rtol=0.0, atol=1e-12):
            raise ValueError("grid period must equal the benchmark period")
        if subpixels < 1:
            raise ValueError("subpixels must be positive")
        center = 0.5 * grid.height if center_z is None else float(center_z)
        if not np.isclose(center, 0.5 * grid.height, rtol=0.0, atol=1e-12):
            raise ValueError("the coupled bars must be centered on the axial grid")
        lower_edge = center - 0.5 * self.total_thickness
        upper_edge = center + 0.5 * self.total_thickness
        if lower_edge <= 0.0 or upper_edge >= grid.height:
            raise ValueError("coupled bars must lie strictly inside the grid")

        offsets = (np.arange(subpixels, dtype=float) + 0.5) / subpixels - 0.5
        sample_x = grid.x[:, None] + offsets[None, :] * grid.dx
        sample_z = grid.z[:, None] + offsets[None, :] * grid.dz
        periodic_distance = np.abs(
            ((sample_x - 0.5 * self.period + 0.5 * self.period) % self.period)
            - 0.5 * self.period
        )
        inside_x = periodic_distance <= 0.5 * self.width
        distance_z = np.abs(sample_z - center)
        inside_z = (
            (distance_z >= 0.5 * self.central_gap)
            & (distance_z <= 0.5 * self.central_gap + self.bar_height)
        )
        fill = np.mean(
            inside_z[:, None, None, :] & inside_x[None, :, :, None],
            axis=(2, 3),
        )
        low_epsilon = self.background_index**2
        high_epsilon = self.refractive_index**2
        epsilon = low_epsilon + fill * (high_epsilon - low_epsilon)
        return np.asarray(0.5 * (epsilon + np.flipud(epsilon)))


@dataclass(frozen=True)
class SeparatedRectangularBarLayers:
    """Two independently dimensioned rectangular layers with a uniform spacer.

    The complete stack is centered axially, but its lower and upper patterned
    films may have different widths and thicknesses.  This is the seed geometry
    for fabrication-level outer sweeps; subsequent density optimization keeps
    the three axial thicknesses fixed and changes only the two lateral masks.
    """

    period: float
    lower_width: float
    upper_width: float
    lower_thickness: float
    upper_thickness: float
    spacer_thickness: float
    refractive_index: float
    background_index: float = 1.0

    def __post_init__(self) -> None:
        lengths = (
            self.period,
            self.lower_width,
            self.upper_width,
            self.lower_thickness,
            self.upper_thickness,
            self.spacer_thickness,
        )
        if any(value <= 0.0 for value in lengths):
            raise ValueError("all separated-layer dimensions must be positive")
        if max(self.lower_width, self.upper_width) > self.period:
            raise ValueError("layer widths must not exceed the period")
        if (
            self.background_index <= 0.0
            or self.refractive_index <= self.background_index
        ):
            raise ValueError("indices must satisfy 0 < background < layer")

    @property
    def total_thickness(self) -> float:
        return (
            self.lower_thickness
            + self.spacer_thickness
            + self.upper_thickness
        )

    def relative_permittivity(
        self,
        grid: UniformGrid2D,
        *,
        center_z: float | None = None,
        subpixels: int = 4,
    ) -> RealArray:
        """Rasterize both films by deterministic subpixel area averaging."""

        if not np.isclose(grid.period, self.period, rtol=0.0, atol=1e-12):
            raise ValueError("grid period must equal the benchmark period")
        if subpixels < 1:
            raise ValueError("subpixels must be positive")
        center = 0.5 * grid.height if center_z is None else float(center_z)
        lower_edge = center - 0.5 * self.total_thickness
        upper_edge = center + 0.5 * self.total_thickness
        if lower_edge <= 0.0 or upper_edge >= grid.height:
            raise ValueError("separated layers must lie strictly inside the grid")

        lower_top = lower_edge + self.lower_thickness
        upper_bottom = lower_top + self.spacer_thickness
        offsets = (np.arange(subpixels, dtype=float) + 0.5) / subpixels - 0.5
        sample_x = grid.x[:, None] + offsets[None, :] * grid.dx
        sample_z = grid.z[:, None] + offsets[None, :] * grid.dz
        distance = np.abs(
            ((sample_x - 0.5 * self.period + 0.5 * self.period) % self.period)
            - 0.5 * self.period
        )
        in_lower_x = distance <= 0.5 * self.lower_width
        in_upper_x = distance <= 0.5 * self.upper_width
        in_lower_z = (sample_z >= lower_edge) & (sample_z < lower_top)
        in_upper_z = (sample_z >= upper_bottom) & (sample_z < upper_edge)
        fill = np.mean(
            (in_lower_z[:, None, None, :] & in_lower_x[None, :, :, None])
            | (in_upper_z[:, None, None, :] & in_upper_x[None, :, :, None]),
            axis=(2, 3),
        )
        low_epsilon = self.background_index**2
        high_epsilon = self.refractive_index**2
        return np.asarray(low_epsilon + fill * (high_epsilon - low_epsilon))


@dataclass(frozen=True)
class CoupledTrapezoidalMembraneGrating:
    """Mirror-symmetric pair of trapezoidal membrane gratings.

    The two ridges face a central uniform gap. Each constituent is the
    :class:`TrapezoidalMembraneGrating` geometry, so the gap tunes the
    splitting of the even and odd supermodes while preserving identical
    exterior ports.
    """

    period: float
    slab_thickness: float
    ridge_height: float
    ridge_bottom_width: float
    ridge_top_width: float
    central_gap: float
    refractive_index: float
    background_index: float = 1.0

    def __post_init__(self) -> None:
        constituent = self.constituent
        if self.central_gap <= 0.0:
            raise ValueError("central_gap must be positive")
        if constituent.total_thickness <= 0.0:  # pragma: no cover - defensive
            raise ValueError("constituent thickness must be positive")

    @property
    def constituent(self) -> TrapezoidalMembraneGrating:
        return TrapezoidalMembraneGrating(
            period=self.period,
            slab_thickness=self.slab_thickness,
            ridge_height=self.ridge_height,
            ridge_bottom_width=self.ridge_bottom_width,
            ridge_top_width=self.ridge_top_width,
            refractive_index=self.refractive_index,
            background_index=self.background_index,
        )

    @property
    def total_thickness(self) -> float:
        return 2.0 * self.constituent.total_thickness + self.central_gap

    def relative_permittivity(
        self,
        grid: UniformGrid2D,
        *,
        center_z: float | None = None,
        subpixels: int = 4,
    ) -> RealArray:
        """Rasterize the facing pair and preserve exact axial mirror symmetry."""

        center = 0.5 * grid.height if center_z is None else float(center_z)
        if not np.isclose(center, 0.5 * grid.height, rtol=0.0, atol=1e-12):
            raise ValueError(
                "the current coupled benchmark requires center_z at grid center"
            )
        lower_base = (
            center
            - 0.5 * self.central_gap
            - self.constituent.total_thickness
        )
        lower = self.constituent.relative_permittivity(
            grid,
            base_z=lower_base,
            subpixels=subpixels,
        )
        upper = np.flipud(lower)
        return np.maximum(lower, upper)
