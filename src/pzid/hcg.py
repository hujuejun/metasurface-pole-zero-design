"""Reduced analytical models for subwavelength high-contrast gratings.

The modal calculation in this module solves the exact transverse dispersion
relation of a binary lamellar grating at normal incidence.  The subsequent
two-mode interference model is deliberately a *screening* model: it captures
the propagation-phase mechanism of an HCG, but it does not include the full
interface mode-mixing matrices needed for quantitative reflectance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from pzid.grid import UniformGrid2D


Polarization = Literal["TE", "TM"]


@dataclass(frozen=True)
class HCGTwoModeResult:
    """Two even HCG modes and their one-pass phase metrics."""

    beta_period: tuple[float, float]
    effective_indices: tuple[float, float]
    common_phase: float
    differential_phase: float
    equal_weight_reflectance_proxy: float


@dataclass(frozen=True)
class SymmetricPairedHCGLayout:
    """Row and column bookkeeping for two identical HCG layers."""

    lower_hcg_rows: tuple[int, int]
    cavity_rows: tuple[int, int]
    upper_hcg_rows: tuple[int, int]
    structure_rows: tuple[int, int]
    grating_columns: tuple[int, int]


@dataclass(frozen=True)
class PairedHCGGeometryMapping:
    """Permittivity and exact subcell derivatives for three HCG coordinates."""

    raw: np.ndarray
    epsilon_r: np.ndarray
    epsilon_derivatives: np.ndarray
    hcg_thickness_um: float
    cavity_thickness_um: float
    grating_width_um: float


@dataclass(frozen=True)
class PairedTwoTierHCGGeometryMapping:
    """Permittivity and subcell derivatives for a two-etch HCG pair."""

    raw: np.ndarray
    epsilon_r: np.ndarray
    epsilon_derivatives: np.ndarray
    inner_tier_height_um: float
    outer_tier_height_um: float
    cavity_thickness_um: float
    inner_tier_width_um: float
    outer_tier_width_um: float


@dataclass
class SymmetricPairedHCGGeometryParameterization:
    """Continuous thickness, separation, and width coordinates for an HCG pair.

    Cell-area overlap fractions move the four axial and two lateral interfaces
    continuously.  The pullback contracts an arbitrary permittivity gradient
    with the exact transpose of these interface derivatives.
    """

    grid: UniformGrid2D
    port_index: float
    grating_index: float
    thickness_bounds_um: tuple[float, float]
    cavity_bounds_um: tuple[float, float]
    width_bounds_um: tuple[float, float]

    def __post_init__(self) -> None:
        if not 0.0 < self.port_index < self.grating_index:
            raise ValueError("indices must satisfy 0 < port_index < grating_index")
        for name, bounds in (
            ("thickness", self.thickness_bounds_um),
            ("cavity", self.cavity_bounds_um),
            ("width", self.width_bounds_um),
        ):
            if not 0.0 < bounds[0] < bounds[1]:
                raise ValueError(f"{name} bounds must be positive and increasing")
        maximum_height = 2.0 * self.thickness_bounds_um[1] + self.cavity_bounds_um[1]
        if maximum_height >= self.grid.height:
            raise ValueError("maximum paired-HCG height must fit inside the grid")
        if self.width_bounds_um[1] >= self.grid.period:
            raise ValueError("maximum grating width must be smaller than the period")

    @property
    def variable_count(self) -> int:
        return 3

    @property
    def design_shape(self) -> tuple[int]:
        return (3,)

    @staticmethod
    def _decode_coordinate(raw: float, bounds: tuple[float, float]) -> float:
        if not 0.0 <= raw <= 1.0:
            raise ValueError("geometry coordinates must lie in [0, 1]")
        return float(bounds[0] + raw * (bounds[1] - bounds[0]))

    @staticmethod
    def _encode_coordinate(value: float, bounds: tuple[float, float]) -> float:
        raw = (float(value) - bounds[0]) / (bounds[1] - bounds[0])
        if not 0.0 <= raw <= 1.0:
            raise ValueError("physical coordinate lies outside its bounds")
        return float(raw)

    def encode(self, thickness_um: float, cavity_um: float, width_um: float) -> np.ndarray:
        return np.asarray(
            (
                self._encode_coordinate(thickness_um, self.thickness_bounds_um),
                self._encode_coordinate(cavity_um, self.cavity_bounds_um),
                self._encode_coordinate(width_um, self.width_bounds_um),
            ),
            dtype=float,
        )

    def decode(self, raw: np.ndarray) -> tuple[float, float, float]:
        values = np.asarray(raw, dtype=float).reshape(-1)
        if values.size != self.variable_count:
            raise ValueError("paired-HCG geometry requires three coordinates")
        return (
            self._decode_coordinate(float(values[0]), self.thickness_bounds_um),
            self._decode_coordinate(float(values[1]), self.cavity_bounds_um),
            self._decode_coordinate(float(values[2]), self.width_bounds_um),
        )

    @staticmethod
    def _interval_coverage(
        lower: float,
        upper: float,
        spacing: float,
        cells: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        edges = np.arange(cells + 1, dtype=float) * spacing
        coverage = np.clip(
            (np.minimum(edges[1:], upper) - np.maximum(edges[:-1], lower)) / spacing,
            0.0,
            1.0,
        )
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(lower), abs(upper))

        def boundary_weights(value: float) -> np.ndarray:
            weights = np.zeros(cells, dtype=float)
            interior = np.flatnonzero(
                (value > edges[:-1] + tolerance) & (value < edges[1:] - tolerance)
            )
            if interior.size:
                weights[int(interior[0])] = 1.0 / spacing
                return weights
            boundary = np.flatnonzero(np.isclose(edges, value, rtol=0.0, atol=tolerance))
            if boundary.size:
                index = int(boundary[0])
                if index > 0:
                    weights[index - 1] += 0.5 / spacing
                if index < cells:
                    weights[index] += 0.5 / spacing
            return weights

        lower_weights = boundary_weights(lower)
        upper_weights = boundary_weights(upper)
        return coverage, -lower_weights, upper_weights

    def map(self, raw: np.ndarray) -> PairedHCGGeometryMapping:
        values = np.asarray(raw, dtype=float).reshape(-1)
        thickness, cavity, width = self.decode(values)
        z_center = 0.5 * self.grid.height
        lower_a = z_center - 0.5 * cavity - thickness
        lower_b = z_center - 0.5 * cavity
        upper_a = z_center + 0.5 * cavity
        upper_b = z_center + 0.5 * cavity + thickness
        lower, lower_da, lower_db = self._interval_coverage(
            lower_a, lower_b, self.grid.dz, self.grid.nz
        )
        upper, upper_da, upper_db = self._interval_coverage(
            upper_a, upper_b, self.grid.dz, self.grid.nz
        )
        axial = lower + upper
        axial_thickness = -lower_da + upper_db
        axial_cavity = -0.5 * lower_da - 0.5 * lower_db + 0.5 * upper_da + 0.5 * upper_db

        x_center = 0.5 * self.grid.period
        lateral, lateral_da, lateral_db = self._interval_coverage(
            x_center - 0.5 * width,
            x_center + 0.5 * width,
            self.grid.dx,
            self.grid.nx,
        )
        lateral_width = -0.5 * lateral_da + 0.5 * lateral_db
        fraction = axial[:, np.newaxis] * lateral[np.newaxis, :]
        contrast = self.grating_index**2 - self.port_index**2
        epsilon = self.port_index**2 + contrast * fraction
        physical_derivatives = np.stack(
            (
                contrast * axial_thickness[:, np.newaxis] * lateral[np.newaxis, :],
                contrast * axial_cavity[:, np.newaxis] * lateral[np.newaxis, :],
                contrast * axial[:, np.newaxis] * lateral_width[np.newaxis, :],
            )
        )
        spans = np.asarray(
            (
                self.thickness_bounds_um[1] - self.thickness_bounds_um[0],
                self.cavity_bounds_um[1] - self.cavity_bounds_um[0],
                self.width_bounds_um[1] - self.width_bounds_um[0],
            )
        )
        derivatives = physical_derivatives * spans[:, np.newaxis, np.newaxis]
        if not np.allclose(epsilon, np.flipud(epsilon), rtol=0.0, atol=1e-12):
            raise RuntimeError("continuous paired-HCG map broke axial symmetry")
        if not np.allclose(epsilon, np.fliplr(epsilon), rtol=0.0, atol=1e-12):
            raise RuntimeError("continuous paired-HCG map broke lateral symmetry")
        return PairedHCGGeometryMapping(
            raw=values.copy(),
            epsilon_r=np.asarray(epsilon),
            epsilon_derivatives=np.asarray(derivatives),
            hcg_thickness_um=thickness,
            cavity_thickness_um=cavity,
            grating_width_um=width,
        )

    def pullback(
        self,
        epsilon_gradient: np.ndarray,
        mapping: PairedHCGGeometryMapping,
    ) -> np.ndarray:
        gradient = np.asarray(epsilon_gradient, dtype=np.complex128)
        if gradient.shape[-2:] != self.grid.shape:
            raise ValueError("epsilon_gradient trailing axes must match the grid")
        return np.einsum(
            "...zx,kzx->...k",
            gradient,
            mapping.epsilon_derivatives,
            optimize=True,
        )


@dataclass
class SymmetricPairedTwoTierHCGGeometryParameterization:
    """Two axially mirrored, two-tier HCGs with exact interface pullbacks.

    Each HCG has an ``inner`` tier facing the cavity and an ``outer`` tier
    facing the port. The two tiers have independent heights and widths, so the
    endpoint is compatible with two aligned lithography/etch steps while
    providing more resonance-residue control than a rectangular ridge.
    """

    grid: UniformGrid2D
    port_index: float
    grating_index: float
    inner_height_bounds_um: tuple[float, float]
    outer_height_bounds_um: tuple[float, float]
    cavity_bounds_um: tuple[float, float]
    inner_width_bounds_um: tuple[float, float]
    outer_width_bounds_um: tuple[float, float]

    def __post_init__(self) -> None:
        if not 0.0 < self.port_index < self.grating_index:
            raise ValueError("indices must satisfy 0 < port_index < grating_index")
        for name, bounds in (
            ("inner height", self.inner_height_bounds_um),
            ("outer height", self.outer_height_bounds_um),
            ("cavity", self.cavity_bounds_um),
            ("inner width", self.inner_width_bounds_um),
            ("outer width", self.outer_width_bounds_um),
        ):
            if not 0.0 < bounds[0] < bounds[1]:
                raise ValueError(f"{name} bounds must be positive and increasing")
        maximum_height = 2.0 * (
            self.inner_height_bounds_um[1] + self.outer_height_bounds_um[1]
        ) + self.cavity_bounds_um[1]
        if maximum_height >= self.grid.height:
            raise ValueError("maximum two-tier pair height must fit inside the grid")
        if max(self.inner_width_bounds_um[1], self.outer_width_bounds_um[1]) >= self.grid.period:
            raise ValueError("maximum tier width must be smaller than the period")

    @property
    def variable_count(self) -> int:
        return 5

    @property
    def design_shape(self) -> tuple[int]:
        return (5,)

    @property
    def _bounds(self) -> tuple[tuple[float, float], ...]:
        return (
            self.inner_height_bounds_um,
            self.outer_height_bounds_um,
            self.cavity_bounds_um,
            self.inner_width_bounds_um,
            self.outer_width_bounds_um,
        )

    def encode(
        self,
        inner_height_um: float,
        outer_height_um: float,
        cavity_um: float,
        inner_width_um: float,
        outer_width_um: float,
    ) -> np.ndarray:
        physical = (
            inner_height_um,
            outer_height_um,
            cavity_um,
            inner_width_um,
            outer_width_um,
        )
        return np.asarray(
            [
                SymmetricPairedHCGGeometryParameterization._encode_coordinate(value, bounds)
                for value, bounds in zip(physical, self._bounds, strict=True)
            ],
            dtype=float,
        )

    def decode(self, raw: np.ndarray) -> tuple[float, float, float, float, float]:
        values = np.asarray(raw, dtype=float).reshape(-1)
        if values.size != self.variable_count:
            raise ValueError("paired two-tier HCG geometry requires five coordinates")
        return tuple(
            SymmetricPairedHCGGeometryParameterization._decode_coordinate(value, bounds)
            for value, bounds in zip(values, self._bounds, strict=True)
        )  # type: ignore[return-value]

    def map(self, raw: np.ndarray) -> PairedTwoTierHCGGeometryMapping:
        values = np.asarray(raw, dtype=float).reshape(-1)
        inner_height, outer_height, cavity, inner_width, outer_width = self.decode(values)
        center = 0.5 * self.grid.height
        z0 = center - 0.5 * cavity - inner_height - outer_height
        z1 = center - 0.5 * cavity - inner_height
        z2 = center - 0.5 * cavity
        z3 = center + 0.5 * cavity
        z4 = center + 0.5 * cavity + inner_height
        z5 = center + 0.5 * cavity + inner_height + outer_height
        coverage = SymmetricPairedHCGGeometryParameterization._interval_coverage
        lower_outer, lo_da, lo_db = coverage(z0, z1, self.grid.dz, self.grid.nz)
        lower_inner, li_da, li_db = coverage(z1, z2, self.grid.dz, self.grid.nz)
        upper_inner, ui_da, ui_db = coverage(z3, z4, self.grid.dz, self.grid.nz)
        upper_outer, uo_da, uo_db = coverage(z4, z5, self.grid.dz, self.grid.nz)
        axial_inner = lower_inner + upper_inner
        axial_outer = lower_outer + upper_outer
        axial_inner_d_inner_height = -li_da + ui_db
        axial_outer_d_inner_height = -lo_da - lo_db + uo_da + uo_db
        axial_outer_d_outer_height = -lo_da + uo_db
        axial_inner_d_cavity = -0.5 * (li_da + li_db) + 0.5 * (ui_da + ui_db)
        axial_outer_d_cavity = -0.5 * (lo_da + lo_db) + 0.5 * (uo_da + uo_db)

        x_center = 0.5 * self.grid.period

        def lateral(width: float) -> tuple[np.ndarray, np.ndarray]:
            result, result_da, result_db = coverage(
                x_center - 0.5 * width,
                x_center + 0.5 * width,
                self.grid.dx,
                self.grid.nx,
            )
            return result, -0.5 * result_da + 0.5 * result_db

        inner_lateral, inner_lateral_d_width = lateral(inner_width)
        outer_lateral, outer_lateral_d_width = lateral(outer_width)
        fraction = (
            axial_inner[:, np.newaxis] * inner_lateral[np.newaxis, :]
            + axial_outer[:, np.newaxis] * outer_lateral[np.newaxis, :]
        )
        contrast = self.grating_index**2 - self.port_index**2
        epsilon = self.port_index**2 + contrast * fraction
        physical_derivatives = np.stack(
            (
                contrast
                * (
                    axial_inner_d_inner_height[:, np.newaxis]
                    * inner_lateral[np.newaxis, :]
                    + axial_outer_d_inner_height[:, np.newaxis]
                    * outer_lateral[np.newaxis, :]
                ),
                contrast
                * axial_outer_d_outer_height[:, np.newaxis]
                * outer_lateral[np.newaxis, :],
                contrast
                * (
                    axial_inner_d_cavity[:, np.newaxis]
                    * inner_lateral[np.newaxis, :]
                    + axial_outer_d_cavity[:, np.newaxis]
                    * outer_lateral[np.newaxis, :]
                ),
                contrast
                * axial_inner[:, np.newaxis]
                * inner_lateral_d_width[np.newaxis, :],
                contrast
                * axial_outer[:, np.newaxis]
                * outer_lateral_d_width[np.newaxis, :],
            )
        )
        spans = np.asarray([upper - lower for lower, upper in self._bounds])
        derivatives = physical_derivatives * spans[:, np.newaxis, np.newaxis]
        if not np.allclose(epsilon, np.flipud(epsilon), rtol=0.0, atol=1e-12):
            raise RuntimeError("continuous two-tier HCG map broke axial symmetry")
        if not np.allclose(epsilon, np.fliplr(epsilon), rtol=0.0, atol=1e-12):
            raise RuntimeError("continuous two-tier HCG map broke lateral symmetry")
        return PairedTwoTierHCGGeometryMapping(
            raw=values.copy(),
            epsilon_r=np.asarray(epsilon),
            epsilon_derivatives=np.asarray(derivatives),
            inner_tier_height_um=inner_height,
            outer_tier_height_um=outer_height,
            cavity_thickness_um=cavity,
            inner_tier_width_um=inner_width,
            outer_tier_width_um=outer_width,
        )

    def pullback(
        self,
        epsilon_gradient: np.ndarray,
        mapping: PairedTwoTierHCGGeometryMapping,
    ) -> np.ndarray:
        gradient = np.asarray(epsilon_gradient, dtype=np.complex128)
        if gradient.shape[-2:] != self.grid.shape:
            raise ValueError("epsilon_gradient trailing axes must match the grid")
        return np.einsum(
            "...zx,kzx->...k", gradient, mapping.epsilon_derivatives, optimize=True
        )


def build_symmetric_paired_hcg(
    *,
    period_um: float,
    cells_per_period: int,
    port_margin_rows: int,
    hcg_rows: int,
    cavity_rows: int,
    grating_width_cells: int,
    port_index: float,
    grating_index: float,
) -> tuple[UniformGrid2D, np.ndarray, SymmetricPairedHCGLayout]:
    """Rasterize two identical binary HCGs separated by a uniform cavity."""

    integer_controls = {
        "cells_per_period": cells_per_period,
        "port_margin_rows": port_margin_rows,
        "hcg_rows": hcg_rows,
        "cavity_rows": cavity_rows,
        "grating_width_cells": grating_width_cells,
    }
    if any(int(value) != value or int(value) < 1 for value in integer_controls.values()):
        raise ValueError("all row and cell controls must be positive integers")
    if period_um <= 0.0:
        raise ValueError("period_um must be positive")
    if not 0.0 < port_index < grating_index:
        raise ValueError("indices must satisfy 0 < port_index < grating_index")

    nx = int(cells_per_period)
    margin = int(port_margin_rows)
    thickness = int(hcg_rows)
    gap = int(cavity_rows)
    width = int(grating_width_cells)
    if width >= nx:
        raise ValueError("grating_width_cells must be smaller than one period")
    if (nx - width) % 2:
        raise ValueError("grating width must preserve lateral mirror symmetry")

    spacing = float(period_um) / nx
    nz = 2 * margin + 2 * thickness + gap
    grid = UniformGrid2D(nx=nx, nz=nz, dx=spacing, dz=spacing)
    epsilon = np.full(grid.shape, float(port_index) ** 2, dtype=float)
    lower_rows = (margin, margin + thickness)
    cavity_interval = (lower_rows[1], lower_rows[1] + gap)
    upper_rows = (cavity_interval[1], cavity_interval[1] + thickness)
    columns = ((nx - width) // 2, (nx + width) // 2)
    epsilon[slice(*lower_rows), slice(*columns)] = float(grating_index) ** 2
    epsilon[slice(*upper_rows), slice(*columns)] = float(grating_index) ** 2

    if upper_rows[1] != nz - margin:
        raise RuntimeError("paired-HCG row construction did not close")
    if not np.array_equal(epsilon, np.flipud(epsilon)):
        raise RuntimeError("paired-HCG structure is not axially symmetric")
    if not np.array_equal(epsilon, np.fliplr(epsilon)):
        raise RuntimeError("paired-HCG structure is not laterally symmetric")
    layout = SymmetricPairedHCGLayout(
        lower_hcg_rows=lower_rows,
        cavity_rows=cavity_interval,
        upper_hcg_rows=upper_rows,
        structure_rows=(lower_rows[0], upper_rows[1]),
        grating_columns=columns,
    )
    return grid, epsilon, layout


def parity_eigenphase_separation(reflectance: float) -> float:
    r"""Return the even/odd eigenphase separation of a lossless mirror.

    A reciprocal, mirror-symmetric, lossless two-port can be written as

    .. math::

       S=e^{i\chi}\begin{pmatrix}i\sin\theta&\cos\theta\\
       \cos\theta&i\sin\theta\end{pmatrix},

    with ``reflectance = sin(theta)**2``.  The phase difference between its
    odd and even scattering eigenchannels is therefore
    ``pi - 2*theta = 2*acos(sqrt(reflectance))``.
    """

    value = float(reflectance)
    if not 0.0 <= value <= 1.0:
        raise ValueError("reflectance must lie in [0, 1]")
    return float(2.0 * np.arccos(np.sqrt(value)))


def symmetric_lossless_mirror(
    reflectance: float,
    common_phase: float = 0.0,
) -> tuple[complex, complex, complex, complex]:
    """Return ``(r, t, c_even, c_odd)`` for a symmetric lossless mirror."""

    value = float(reflectance)
    if not 0.0 <= value <= 1.0:
        raise ValueError("reflectance must lie in [0, 1]")
    theta = float(np.arcsin(np.sqrt(value)))
    phase = np.exp(1j * float(common_phase))
    reflection = 1j * phase * np.sin(theta)
    transmission = phase * np.cos(theta)
    return (
        complex(reflection),
        complex(transmission),
        complex(reflection + transmission),
        complex(reflection - transmission),
    )


def cavity_pole_depth(
    mirror_reflectance: float,
    propagation_delay: float,
    *,
    mirror_group_delay: float = 0.0,
) -> float:
    r"""Return ``-Im(omega_p)`` for two identical, slowly varying mirrors.

    ``propagation_delay`` is the one-way phase derivative through the cavity;
    ``mirror_group_delay`` is ``d arg(r) / d omega`` for one mirror.  The
    expression follows from the complex-frequency round-trip condition
    ``r**2 exp(2 i phi_prop) = 1``.
    """

    value = float(mirror_reflectance)
    if not 0.0 < value <= 1.0:
        raise ValueError("mirror_reflectance must lie in (0, 1]")
    total_delay = float(propagation_delay) + float(mirror_group_delay)
    if total_delay <= 0.0:
        raise ValueError("total one-way phase delay must be positive")
    return float(-np.log(value) / (2.0 * total_delay))


def _even_characteristic(
    beta_period: float,
    wavelength_over_period: float,
    duty_cycle: float,
    high_index: float,
    low_index: float,
    polarization: Polarization,
) -> float:
    """Characteristic determinant for even, normal-Bloch grating modes."""

    k0_period = 2.0 * np.pi / wavelength_over_period
    q_high = np.sqrt((high_index * k0_period) ** 2 - beta_period**2 + 0j)
    q_low = np.sqrt((low_index * k0_period) ** 2 - beta_period**2 + 0j)
    high_half_width = 0.5 * duty_cycle
    low_half_width = 0.5 * (1.0 - duty_cycle)
    if polarization == "TE":
        high_flux = q_high
        low_flux = q_low
    elif polarization == "TM":
        high_flux = q_high / high_index**2
        low_flux = q_low / low_index**2
    else:
        raise ValueError("polarization must be 'TE' or 'TM'")
    determinant = (
        high_flux
        * np.sin(q_high * high_half_width)
        * np.cos(q_low * low_half_width)
        + low_flux
        * np.cos(q_high * high_half_width)
        * np.sin(q_low * low_half_width)
    )
    if abs(determinant.imag) > 2e-9 * max(1.0, abs(determinant.real)):
        raise RuntimeError("even-mode determinant unexpectedly became complex")
    return float(determinant.real)


def even_mode_propagation_constants(
    wavelength_over_period: float,
    duty_cycle: float,
    high_index: float,
    low_index: float = 1.0,
    *,
    polarization: Polarization = "TM",
    samples: int = 8193,
) -> tuple[float, ...]:
    """Solve for real propagation constants ``beta * period``.

    The high-index bar is centered in the unit cell.  Only modes even under
    reflection about the bar center are returned because a normally incident
    zeroth-order plane wave cannot excite the odd transverse family.
    """

    ratio = float(wavelength_over_period)
    duty = float(duty_cycle)
    n_high = float(high_index)
    n_low = float(low_index)
    if ratio <= 0.0:
        raise ValueError("wavelength_over_period must be positive")
    if not 0.0 < duty < 1.0:
        raise ValueError("duty_cycle must lie in (0, 1)")
    if n_low <= 0.0 or n_high <= n_low:
        raise ValueError("indices must satisfy high_index > low_index > 0")
    if samples < 257:
        raise ValueError("samples must be at least 257")

    upper = n_high * 2.0 * np.pi / ratio
    beta = np.linspace(0.0, upper * (1.0 - 1e-11), int(samples))
    values = np.asarray(
        [
            _even_characteristic(
                item,
                ratio,
                duty,
                n_high,
                n_low,
                polarization,
            )
            for item in beta
        ]
    )
    roots: list[float] = []
    for left, right, f_left, f_right in zip(
        beta[:-1], beta[1:], values[:-1], values[1:], strict=True
    ):
        if f_left == 0.0:
            candidate = float(left)
        elif f_left * f_right < 0.0:
            candidate = float(
                brentq(
                    _even_characteristic,
                    float(left),
                    float(right),
                    args=(ratio, duty, n_high, n_low, polarization),
                    xtol=1e-13,
                    rtol=1e-13,
                )
            )
        else:
            continue
        if candidate <= 1e-9:
            continue
        if not roots or abs(candidate - roots[-1]) > 1e-7:
            roots.append(candidate)
    return tuple(sorted(roots, reverse=True))


def two_mode_phase_metrics(
    wavelength_over_period: float,
    duty_cycle: float,
    thickness_over_period: float,
    high_index: float,
    low_index: float = 1.0,
    *,
    polarization: Polarization = "TM",
) -> HCGTwoModeResult:
    """Evaluate the two-mode HCG propagation-phase screening model."""

    thickness = float(thickness_over_period)
    if thickness <= 0.0:
        raise ValueError("thickness_over_period must be positive")
    roots = even_mode_propagation_constants(
        wavelength_over_period,
        duty_cycle,
        high_index,
        low_index,
        polarization=polarization,
    )
    if len(roots) != 2:
        raise ValueError(f"two even propagating modes required; found {len(roots)}")
    beta0, beta2 = roots
    common = 0.5 * (beta0 + beta2) * thickness
    differential = (beta0 - beta2) * thickness
    reflectance_proxy = np.sin(0.5 * differential) ** 2
    k0_period = 2.0 * np.pi / float(wavelength_over_period)
    return HCGTwoModeResult(
        beta_period=(float(beta0), float(beta2)),
        effective_indices=(float(beta0 / k0_period), float(beta2 / k0_period)),
        common_phase=float(common),
        differential_phase=float(differential),
        equal_weight_reflectance_proxy=float(reflectance_proxy),
    )
