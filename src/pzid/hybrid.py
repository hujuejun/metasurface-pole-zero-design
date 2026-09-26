"""Cell-aligned planar hybrid cavities for the M11 symmetry study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from pzid.grid import UniformGrid2D


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SymmetricDBRGratingLayout:
    """Discrete row bookkeeping for a mirror-symmetric DBR/grating cavity."""

    cavity_rows: tuple[int, int]
    lower_dbr_rows: tuple[tuple[int, int], ...]
    upper_dbr_rows: tuple[tuple[int, int], ...]
    structure_rows: tuple[int, int]
    grating_columns: tuple[int, int]


@dataclass(frozen=True)
class SymmetricDBRSlabGratingLayout(SymmetricDBRGratingLayout):
    """Row bookkeeping for a centered slab with mirrored grating ribs."""

    central_slab_rows: tuple[int, int]
    lower_grating_rows: tuple[int, int]
    upper_grating_rows: tuple[int, int]


@dataclass(frozen=True)
class SymmetricDBRIndependentSlabGratingLayout(SymmetricDBRSlabGratingLayout):
    """Layout with independent slab, rib, and DBR-termination thicknesses."""

    lower_termination_rows: tuple[int, int]
    upper_termination_rows: tuple[int, int]


@dataclass(frozen=True)
class SymmetricSurfaceReliefSlabLayout:
    """Bookkeeping for a connected slab with mirrored surface-relief ribs."""

    central_slab_rows: tuple[int, int]
    lower_grating_rows: tuple[int, int]
    upper_grating_rows: tuple[int, int]
    structure_rows: tuple[int, int]
    cavity_rows: tuple[int, int]
    grating_columns: tuple[int, int]


@dataclass(frozen=True)
class PeriodDoubledSurfaceReliefSlabLayout:
    """Bookkeeping for a two-site, period-doubled surface-relief slab."""

    central_slab_rows: tuple[int, int]
    lower_grating_rows: tuple[int, int]
    upper_grating_rows: tuple[int, int]
    structure_rows: tuple[int, int]
    cavity_rows: tuple[int, int]
    left_grating_columns: tuple[int, int]
    right_grating_columns: tuple[int, int]
    parent_half_period_cells: int


@dataclass(frozen=True)
class SymmetricCoupledSurfaceReliefSlabsLayout:
    """Bookkeeping for two mirrored single-mode surface-relief slabs."""

    lower_slab_rows: tuple[int, int]
    lower_grating_rows: tuple[int, int]
    spacer_rows: tuple[int, int]
    upper_grating_rows: tuple[int, int]
    upper_slab_rows: tuple[int, int]
    structure_rows: tuple[int, int]
    cavity_rows: tuple[int, int]
    grating_columns: tuple[int, int]

    @property
    def lower_component_rows(self) -> tuple[int, int]:
        return self.lower_slab_rows[0], self.lower_grating_rows[1]

    @property
    def upper_component_rows(self) -> tuple[int, int]:
        return self.upper_grating_rows[0], self.upper_slab_rows[1]


@dataclass(frozen=True)
class PeriodDoubledCoupledSurfaceReliefSlabsLayout:
    """Bookkeeping for a period-doubled pair of surface-relief slabs."""

    lower_slab_rows: tuple[int, int]
    lower_grating_rows: tuple[int, int]
    spacer_rows: tuple[int, int]
    upper_grating_rows: tuple[int, int]
    upper_slab_rows: tuple[int, int]
    structure_rows: tuple[int, int]
    cavity_rows: tuple[int, int]
    left_grating_columns: tuple[int, int]
    right_grating_columns: tuple[int, int]
    parent_half_period_cells: int

    @property
    def lower_component_rows(self) -> tuple[int, int]:
        return self.lower_slab_rows[0], self.lower_grating_rows[1]

    @property
    def upper_component_rows(self) -> tuple[int, int]:
        return self.upper_grating_rows[0], self.upper_slab_rows[1]


@dataclass(frozen=True)
class SymmetricCoupledDualSurfaceReliefSlabsLayout:
    """Bookkeeping for two slabs with independent inner/outer reliefs."""

    lower_outer_grating_rows: tuple[int, int]
    lower_slab_rows: tuple[int, int]
    lower_inner_grating_rows: tuple[int, int]
    spacer_rows: tuple[int, int]
    upper_inner_grating_rows: tuple[int, int]
    upper_slab_rows: tuple[int, int]
    upper_outer_grating_rows: tuple[int, int]
    structure_rows: tuple[int, int]
    cavity_rows: tuple[int, int]
    inner_grating_columns: tuple[int, int]
    outer_grating_columns: tuple[int, int]
    outer_grating_shift_cells: int

    @property
    def lower_component_rows(self) -> tuple[int, int]:
        return self.lower_outer_grating_rows[0], self.lower_inner_grating_rows[1]

    @property
    def upper_component_rows(self) -> tuple[int, int]:
        return self.upper_inner_grating_rows[0], self.upper_outer_grating_rows[1]


@dataclass(frozen=True)
class SymmetricDBRPairedGratingSheetLayout(SymmetricDBRGratingLayout):
    """Bookkeeping for two mirrored patterned sheets inside one DBR cavity."""

    lower_outer_spacer_rows: tuple[int, int]
    lower_grating_sheet_rows: tuple[int, int]
    central_gap_rows: tuple[int, int]
    upper_grating_sheet_rows: tuple[int, int]
    upper_outer_spacer_rows: tuple[int, int]


@dataclass(frozen=True)
class SymmetricDBRPairedSheetPhaseLayerLayout(
    SymmetricDBRPairedGratingSheetLayout
):
    """Paired-sheet layout with a centered uniform dielectric phase layer."""

    central_phase_layer_rows: tuple[int, int]


def build_symmetric_dbr_grating_cavity(
    *,
    period_um: float,
    cells_per_period: int,
    port_margin_rows: int,
    cavity_rows: int,
    grating_width_cells: int,
    dbr_pairs: int,
    dbr_high_rows: int,
    dbr_low_rows: int,
    port_index: float,
    dbr_high_index: float,
    dbr_low_index: float,
    grating_high_index: float,
    grating_low_index: float,
) -> tuple[UniformGrid2D, FloatArray, SymmetricDBRGratingLayout]:
    """Rasterize a symmetric DBR cavity containing one through-etched grating.

    Starting at either cavity interface, each mirror is ordered low-index then
    high-index.  All boundaries are cell aligned and the returned permittivity
    is exactly invariant under axial reflection.  The central grating is also
    mirror symmetric laterally and axially.
    """

    integer_controls = {
        "cells_per_period": cells_per_period,
        "port_margin_rows": port_margin_rows,
        "cavity_rows": cavity_rows,
        "grating_width_cells": grating_width_cells,
        "dbr_pairs": dbr_pairs,
        "dbr_high_rows": dbr_high_rows,
        "dbr_low_rows": dbr_low_rows,
    }
    if any(int(value) != value or int(value) < 1 for value in integer_controls.values()):
        raise ValueError("all cell and row controls must be positive integers")
    nx = int(cells_per_period)
    margin = int(port_margin_rows)
    cavity_count = int(cavity_rows)
    width = int(grating_width_cells)
    pair_count = int(dbr_pairs)
    high_count = int(dbr_high_rows)
    low_count = int(dbr_low_rows)
    if width >= nx:
        raise ValueError("grating_width_cells must be smaller than one period")
    if (nx - width) % 2:
        raise ValueError("grating width must preserve lateral mirror symmetry")
    indices = (
        port_index,
        dbr_high_index,
        dbr_low_index,
        grating_high_index,
        grating_low_index,
    )
    if any(float(value) <= 0.0 for value in indices):
        raise ValueError("all refractive indices must be positive")
    if period_um <= 0.0:
        raise ValueError("period_um must be positive")

    mirror_rows = pair_count * (low_count + high_count)
    occupied_rows = cavity_count + 2 * mirror_rows
    nz = occupied_rows + 2 * margin
    spacing = float(period_um) / nx
    grid = UniformGrid2D(nx=nx, nz=nz, dx=spacing, dz=spacing)
    epsilon = np.full(grid.shape, float(port_index) ** 2, dtype=float)

    cavity_start = margin + mirror_rows
    cavity_stop = cavity_start + cavity_count
    column_start = (nx - width) // 2
    column_stop = column_start + width
    epsilon[cavity_start:cavity_stop] = float(grating_low_index) ** 2
    epsilon[cavity_start:cavity_stop, column_start:column_stop] = (
        float(grating_high_index) ** 2
    )

    lower_layers: list[tuple[int, int]] = []
    upper_layers: list[tuple[int, int]] = []
    lower_cursor = cavity_start
    upper_cursor = cavity_stop
    for _ in range(pair_count):
        lower_low = (lower_cursor - low_count, lower_cursor)
        upper_low = (upper_cursor, upper_cursor + low_count)
        epsilon[slice(*lower_low)] = float(dbr_low_index) ** 2
        epsilon[slice(*upper_low)] = float(dbr_low_index) ** 2
        lower_layers.append(lower_low)
        upper_layers.append(upper_low)
        lower_cursor = lower_low[0]
        upper_cursor = upper_low[1]

        lower_high = (lower_cursor - high_count, lower_cursor)
        upper_high = (upper_cursor, upper_cursor + high_count)
        epsilon[slice(*lower_high)] = float(dbr_high_index) ** 2
        epsilon[slice(*upper_high)] = float(dbr_high_index) ** 2
        lower_layers.append(lower_high)
        upper_layers.append(upper_high)
        lower_cursor = lower_high[0]
        upper_cursor = upper_high[1]

    if lower_cursor != margin or upper_cursor != nz - margin:
        raise RuntimeError("DBR row construction did not close on the port margins")
    if not np.array_equal(epsilon, np.flipud(epsilon)):
        raise RuntimeError("constructed cavity is not exactly axially symmetric")
    if not np.array_equal(epsilon, np.fliplr(epsilon)):
        raise RuntimeError("constructed grating is not exactly laterally symmetric")

    layout = SymmetricDBRGratingLayout(
        cavity_rows=(cavity_start, cavity_stop),
        lower_dbr_rows=tuple(lower_layers),
        upper_dbr_rows=tuple(upper_layers),
        structure_rows=(margin, nz - margin),
        grating_columns=(column_start, column_stop),
    )
    return grid, epsilon, layout


def build_symmetric_dbr_paired_grating_sheet_cavity(
    *,
    period_um: float,
    cells_per_period: int,
    port_margin_rows: int,
    central_gap_rows: int,
    grating_sheet_rows: int,
    outer_spacer_rows: int,
    grating_width_cells: int,
    dbr_pairs: int,
    dbr_high_rows: int,
    dbr_low_rows: int,
    port_index: float,
    dbr_high_index: float,
    dbr_low_index: float,
    grating_high_index: float,
    grating_low_index: float,
) -> tuple[UniformGrid2D, FloatArray, SymmetricDBRPairedGratingSheetLayout]:
    """Rasterize two identical patterned sheets in a symmetric FP cavity.

    From the lower mirror toward the midplane, the cavity contains a uniform
    low-index outer spacer, one patterned high-index sheet, and half of the
    central low-index gap.  The upper half is its exact axial reflection.  The
    antisymmetric sheet supermode therefore supplies a globally odd resonance,
    while the symmetric cavity also retains an even FP-like resonance.
    """

    positive_controls = {
        "central_gap_rows": central_gap_rows,
        "grating_sheet_rows": grating_sheet_rows,
    }
    if any(int(value) != value or int(value) < 1 for value in positive_controls.values()):
        raise ValueError("central gap and grating sheet rows must be positive integers")
    if int(outer_spacer_rows) != outer_spacer_rows or int(outer_spacer_rows) < 0:
        raise ValueError("outer_spacer_rows must be a nonnegative integer")

    gap_count = int(central_gap_rows)
    sheet_count = int(grating_sheet_rows)
    spacer_count = int(outer_spacer_rows)
    cavity_count = gap_count + 2 * (sheet_count + spacer_count)
    grid, epsilon, base = build_symmetric_dbr_grating_cavity(
        period_um=period_um,
        cells_per_period=cells_per_period,
        port_margin_rows=port_margin_rows,
        cavity_rows=cavity_count,
        grating_width_cells=grating_width_cells,
        dbr_pairs=dbr_pairs,
        dbr_high_rows=dbr_high_rows,
        dbr_low_rows=dbr_low_rows,
        port_index=port_index,
        dbr_high_index=dbr_high_index,
        dbr_low_index=dbr_low_index,
        grating_high_index=grating_high_index,
        grating_low_index=grating_low_index,
    )

    cavity_start, cavity_stop = base.cavity_rows
    lower_spacer = (cavity_start, cavity_start + spacer_count)
    lower_sheet = (lower_spacer[1], lower_spacer[1] + sheet_count)
    gap = (lower_sheet[1], lower_sheet[1] + gap_count)
    upper_sheet = (gap[1], gap[1] + sheet_count)
    upper_spacer = (upper_sheet[1], cavity_stop)

    epsilon[cavity_start:cavity_stop] = float(grating_low_index) ** 2
    column_start, column_stop = base.grating_columns
    epsilon[slice(*lower_sheet), column_start:column_stop] = (
        float(grating_high_index) ** 2
    )
    epsilon[slice(*upper_sheet), column_start:column_stop] = (
        float(grating_high_index) ** 2
    )

    if not np.array_equal(epsilon, np.flipud(epsilon)):
        raise RuntimeError("paired-sheet cavity is not exactly axially symmetric")
    if not np.array_equal(epsilon, np.fliplr(epsilon)):
        raise RuntimeError("paired-sheet grating is not exactly laterally symmetric")

    layout = SymmetricDBRPairedGratingSheetLayout(
        cavity_rows=base.cavity_rows,
        lower_dbr_rows=base.lower_dbr_rows,
        upper_dbr_rows=base.upper_dbr_rows,
        structure_rows=base.structure_rows,
        grating_columns=base.grating_columns,
        lower_outer_spacer_rows=lower_spacer,
        lower_grating_sheet_rows=lower_sheet,
        central_gap_rows=gap,
        upper_grating_sheet_rows=upper_sheet,
        upper_outer_spacer_rows=upper_spacer,
    )
    return grid, epsilon, layout


def build_symmetric_dbr_paired_sheet_phase_layer_cavity(
    *,
    period_um: float,
    cells_per_period: int,
    port_margin_rows: int,
    central_gap_rows: int,
    central_phase_layer_rows: int,
    central_phase_layer_index: float,
    grating_sheet_rows: int,
    outer_spacer_rows: int,
    grating_width_cells: int,
    dbr_pairs: int,
    dbr_high_rows: int,
    dbr_low_rows: int,
    port_index: float,
    dbr_high_index: float,
    dbr_low_index: float,
    grating_high_index: float,
    grating_low_index: float,
) -> tuple[UniformGrid2D, FloatArray, SymmetricDBRPairedSheetPhaseLayerLayout]:
    """Insert a cell-aligned uniform phase layer at the cavity midplane.

    The phase layer replaces a centered subset of the low-index gap between
    the two patterned sheets.  Requiring equal unused rows on both sides keeps
    the full structure exactly invariant under axial reflection.
    """

    if (
        int(central_phase_layer_rows) != central_phase_layer_rows
        or int(central_phase_layer_rows) < 1
    ):
        raise ValueError("central_phase_layer_rows must be a positive integer")
    phase_count = int(central_phase_layer_rows)
    gap_count = int(central_gap_rows)
    if phase_count > gap_count:
        raise ValueError("central phase layer must fit inside the central gap")
    if (gap_count - phase_count) % 2:
        raise ValueError(
            "central gap and phase-layer row counts must have matching parity"
        )
    if float(central_phase_layer_index) <= 0.0:
        raise ValueError("central_phase_layer_index must be positive")

    grid, epsilon, base = build_symmetric_dbr_paired_grating_sheet_cavity(
        period_um=period_um,
        cells_per_period=cells_per_period,
        port_margin_rows=port_margin_rows,
        central_gap_rows=central_gap_rows,
        grating_sheet_rows=grating_sheet_rows,
        outer_spacer_rows=outer_spacer_rows,
        grating_width_cells=grating_width_cells,
        dbr_pairs=dbr_pairs,
        dbr_high_rows=dbr_high_rows,
        dbr_low_rows=dbr_low_rows,
        port_index=port_index,
        dbr_high_index=dbr_high_index,
        dbr_low_index=dbr_low_index,
        grating_high_index=grating_high_index,
        grating_low_index=grating_low_index,
    )
    gap_start, gap_stop = base.central_gap_rows
    phase_start = gap_start + (gap_count - phase_count) // 2
    phase_stop = phase_start + phase_count
    epsilon[phase_start:phase_stop] = float(central_phase_layer_index) ** 2
    if not np.array_equal(epsilon, np.flipud(epsilon)):
        raise RuntimeError("central phase layer broke axial symmetry")
    if not np.array_equal(epsilon, np.fliplr(epsilon)):
        raise RuntimeError("central phase layer broke lateral symmetry")
    layout = SymmetricDBRPairedSheetPhaseLayerLayout(
        cavity_rows=base.cavity_rows,
        lower_dbr_rows=base.lower_dbr_rows,
        upper_dbr_rows=base.upper_dbr_rows,
        structure_rows=base.structure_rows,
        grating_columns=base.grating_columns,
        lower_outer_spacer_rows=base.lower_outer_spacer_rows,
        lower_grating_sheet_rows=base.lower_grating_sheet_rows,
        central_gap_rows=base.central_gap_rows,
        upper_grating_sheet_rows=base.upper_grating_sheet_rows,
        upper_outer_spacer_rows=base.upper_outer_spacer_rows,
        central_phase_layer_rows=(phase_start, phase_stop),
    )
    return grid, epsilon, layout


def build_symmetric_dbr_slab_grating_cavity(
    *,
    period_um: float,
    cells_per_period: int,
    port_margin_rows: int,
    cavity_rows: int,
    central_slab_rows: int,
    grating_width_cells: int,
    dbr_pairs: int,
    dbr_high_rows: int,
    dbr_low_rows: int,
    port_index: float,
    dbr_high_index: float,
    dbr_low_index: float,
    grating_high_index: float,
    grating_low_index: float,
) -> tuple[UniformGrid2D, FloatArray, SymmetricDBRSlabGratingLayout]:
    """Rasterize an exact-symmetry rib--slab--rib DBR cavity.

    The central ``central_slab_rows`` are uniformly high index.  The remaining
    cavity rows are divided equally between lower and upper patterned ribs.
    Consequently the high-index material is one connected component and the
    structure is invariant under both lateral and axial reflection.  Setting
    ``central_slab_rows=0`` recovers the through-grating geometry exactly.
    """

    if int(central_slab_rows) != central_slab_rows or int(central_slab_rows) < 0:
        raise ValueError("central_slab_rows must be a nonnegative integer")
    slab_count = int(central_slab_rows)
    if slab_count >= int(cavity_rows):
        raise ValueError("central_slab_rows must be smaller than cavity_rows")
    if (int(cavity_rows) - slab_count) % 2:
        raise ValueError(
            "cavity_rows - central_slab_rows must be even for axial symmetry"
        )

    grid, epsilon, base = build_symmetric_dbr_grating_cavity(
        period_um=period_um,
        cells_per_period=cells_per_period,
        port_margin_rows=port_margin_rows,
        cavity_rows=cavity_rows,
        grating_width_cells=grating_width_cells,
        dbr_pairs=dbr_pairs,
        dbr_high_rows=dbr_high_rows,
        dbr_low_rows=dbr_low_rows,
        port_index=port_index,
        dbr_high_index=dbr_high_index,
        dbr_low_index=dbr_low_index,
        grating_high_index=grating_high_index,
        grating_low_index=grating_low_index,
    )
    cavity_start, cavity_stop = base.cavity_rows
    wing_count = (int(cavity_rows) - slab_count) // 2
    slab_start = cavity_start + wing_count
    slab_stop = slab_start + slab_count
    if slab_count:
        epsilon[slab_start:slab_stop] = float(grating_high_index) ** 2

    if not np.array_equal(epsilon, np.flipud(epsilon)):
        raise RuntimeError("constructed slab cavity is not exactly axially symmetric")
    if not np.array_equal(epsilon, np.fliplr(epsilon)):
        raise RuntimeError("constructed slab grating is not exactly laterally symmetric")

    layout = SymmetricDBRSlabGratingLayout(
        cavity_rows=base.cavity_rows,
        lower_dbr_rows=base.lower_dbr_rows,
        upper_dbr_rows=base.upper_dbr_rows,
        structure_rows=base.structure_rows,
        grating_columns=base.grating_columns,
        central_slab_rows=(slab_start, slab_stop),
        lower_grating_rows=(cavity_start, slab_start),
        upper_grating_rows=(slab_stop, cavity_stop),
    )
    return grid, epsilon, layout


def build_symmetric_dbr_independent_slab_grating_cavity(
    *,
    period_um: float,
    cells_per_period: int,
    port_margin_rows: int,
    central_slab_rows: int,
    grating_rib_rows: int,
    termination_rows: int,
    grating_width_cells: int,
    dbr_pairs: int,
    dbr_high_rows: int,
    dbr_low_rows: int,
    port_index: float,
    dbr_high_index: float,
    dbr_low_index: float,
    grating_high_index: float,
    grating_low_index: float,
) -> tuple[
    UniformGrid2D, FloatArray, SymmetricDBRIndependentSlabGratingLayout
]:
    """Rasterize a slab with independently thick mirrored grating ribs.

    From the midplane outward, the central region contains a uniform slab,
    one patterned rib on each side, and one uniform low-index termination on
    each side.  The central tooth thickness is therefore
    ``slab + 2 * rib`` and is not tied to the DBR separation.  All intervals
    are cell aligned and the high-index slab/rib material is connected.
    """

    integer_controls = {
        "central_slab_rows": central_slab_rows,
        "grating_rib_rows": grating_rib_rows,
        "termination_rows": termination_rows,
    }
    if any(int(value) != value for value in integer_controls.values()):
        raise ValueError("slab, rib, and termination row controls must be integers")
    slab_count = int(central_slab_rows)
    rib_count = int(grating_rib_rows)
    termination_count = int(termination_rows)
    if slab_count < 1 or rib_count < 1 or termination_count < 0:
        raise ValueError(
            "slab and rib rows must be positive; termination rows may be zero"
        )
    cavity_count = slab_count + 2 * (rib_count + termination_count)
    grid, epsilon, base = build_symmetric_dbr_grating_cavity(
        period_um=period_um,
        cells_per_period=cells_per_period,
        port_margin_rows=port_margin_rows,
        cavity_rows=cavity_count,
        grating_width_cells=grating_width_cells,
        dbr_pairs=dbr_pairs,
        dbr_high_rows=dbr_high_rows,
        dbr_low_rows=dbr_low_rows,
        port_index=port_index,
        dbr_high_index=dbr_high_index,
        dbr_low_index=dbr_low_index,
        grating_high_index=grating_high_index,
        grating_low_index=grating_low_index,
    )
    cavity_start, cavity_stop = base.cavity_rows
    lower_termination = (cavity_start, cavity_start + termination_count)
    lower_rib = (lower_termination[1], lower_termination[1] + rib_count)
    slab = (lower_rib[1], lower_rib[1] + slab_count)
    upper_rib = (slab[1], slab[1] + rib_count)
    upper_termination = (upper_rib[1], cavity_stop)
    if termination_count:
        epsilon[slice(*lower_termination)] = float(grating_low_index) ** 2
        epsilon[slice(*upper_termination)] = float(grating_low_index) ** 2
    epsilon[slice(*slab)] = float(grating_high_index) ** 2

    if upper_termination[1] != cavity_stop:
        raise RuntimeError("independent central-region row construction did not close")
    if not np.array_equal(epsilon, np.flipud(epsilon)):
        raise RuntimeError("independent slab cavity is not exactly axially symmetric")
    if not np.array_equal(epsilon, np.fliplr(epsilon)):
        raise RuntimeError("independent slab grating is not exactly laterally symmetric")

    layout = SymmetricDBRIndependentSlabGratingLayout(
        cavity_rows=base.cavity_rows,
        lower_dbr_rows=base.lower_dbr_rows,
        upper_dbr_rows=base.upper_dbr_rows,
        structure_rows=base.structure_rows,
        grating_columns=base.grating_columns,
        central_slab_rows=slab,
        lower_grating_rows=lower_rib,
        upper_grating_rows=upper_rib,
        lower_termination_rows=lower_termination,
        upper_termination_rows=upper_termination,
    )
    return grid, epsilon, layout


def build_symmetric_surface_relief_slab(
    *,
    period_um: float,
    cells_per_period: int,
    port_margin_rows: int,
    central_slab_rows: int,
    grating_rib_rows: int,
    grating_width_cells: int,
    port_index: float,
    slab_index: float,
) -> tuple[UniformGrid2D, FloatArray, SymmetricSurfaceReliefSlabLayout]:
    """Rasterize an axially symmetric surface-relief slab grating.

    A uniform high-index slab spans the unit cell. Identical high-index ribs
    are attached to its lower and upper surfaces over one centered lateral
    interval. The whole dielectric is connected, and both axial and lateral
    mirror symmetries are exact on the cell-centered grid.
    """

    integer_controls = {
        "cells_per_period": cells_per_period,
        "port_margin_rows": port_margin_rows,
        "central_slab_rows": central_slab_rows,
        "grating_rib_rows": grating_rib_rows,
        "grating_width_cells": grating_width_cells,
    }
    if any(int(value) != value or int(value) < 1 for value in integer_controls.values()):
        raise ValueError("all surface-relief cell and row controls must be positive integers")
    nx = int(cells_per_period)
    margin = int(port_margin_rows)
    slab_count = int(central_slab_rows)
    rib_count = int(grating_rib_rows)
    width = int(grating_width_cells)
    if width >= nx:
        raise ValueError("grating_width_cells must be smaller than one period")
    if (nx - width) % 2:
        raise ValueError("grating width must preserve lateral mirror symmetry")
    if period_um <= 0.0 or not slab_index > port_index > 0.0:
        raise ValueError("require period > 0 and slab_index > port_index > 0")

    spacing = float(period_um) / nx
    structure_count = slab_count + 2 * rib_count
    grid = UniformGrid2D(
        nx=nx,
        nz=2 * margin + structure_count,
        dx=spacing,
        dz=spacing,
    )
    epsilon = np.full(grid.shape, float(port_index) ** 2, dtype=float)
    structure_start = margin
    lower_rib = (structure_start, structure_start + rib_count)
    slab = (lower_rib[1], lower_rib[1] + slab_count)
    upper_rib = (slab[1], slab[1] + rib_count)
    column_start = (nx - width) // 2
    column_stop = column_start + width
    epsilon[slice(*slab)] = float(slab_index) ** 2
    epsilon[slice(*lower_rib), column_start:column_stop] = float(slab_index) ** 2
    epsilon[slice(*upper_rib), column_start:column_stop] = float(slab_index) ** 2

    if upper_rib[1] != grid.nz - margin:
        raise RuntimeError("surface-relief row construction did not close")
    if not np.array_equal(epsilon, np.flipud(epsilon)):
        raise RuntimeError("surface-relief slab is not exactly axially symmetric")
    if not np.array_equal(epsilon, np.fliplr(epsilon)):
        raise RuntimeError("surface-relief slab is not exactly laterally symmetric")

    layout = SymmetricSurfaceReliefSlabLayout(
        central_slab_rows=slab,
        lower_grating_rows=lower_rib,
        upper_grating_rows=upper_rib,
        structure_rows=(structure_start, upper_rib[1]),
        cavity_rows=(structure_start, upper_rib[1]),
        grating_columns=(column_start, column_stop),
    )
    return grid, epsilon, layout


def build_period_doubled_surface_relief_slab(
    *,
    period_um: float,
    cells_per_period: int,
    port_margin_rows: int,
    central_slab_rows: int,
    grating_rib_rows: int,
    left_grating_width_cells: int,
    right_grating_width_cells: int,
    port_index: float,
    slab_index: float,
) -> tuple[UniformGrid2D, FloatArray, PeriodDoubledSurfaceReliefSlabLayout]:
    """Rasterize a connected period-doubled slab with two relief sites.

    The computational period contains two half-period cells.  Each half cell
    carries one centered rib on both slab faces.  Equal rib widths restore the
    parent half-period translation exactly; unequal widths break only that
    translation while preserving axial mirror symmetry.  The latter converts
    parent-zone-edge guided modes into radiating folded quasi-BICs.

    Widths are required to have the same parity as the half-period cell count
    so that each rib is exactly centered on the cell-aligned raster.
    """

    integer_controls = {
        "cells_per_period": cells_per_period,
        "port_margin_rows": port_margin_rows,
        "central_slab_rows": central_slab_rows,
        "grating_rib_rows": grating_rib_rows,
        "left_grating_width_cells": left_grating_width_cells,
        "right_grating_width_cells": right_grating_width_cells,
    }
    if any(int(value) != value or int(value) < 1 for value in integer_controls.values()):
        raise ValueError("all period-doubled cell and row controls must be positive integers")
    nx = int(cells_per_period)
    if nx % 2:
        raise ValueError("cells_per_period must be even for a half-period parent lattice")
    half = nx // 2
    left_width = int(left_grating_width_cells)
    right_width = int(right_grating_width_cells)
    if left_width >= half or right_width >= half:
        raise ValueError("each grating width must be smaller than one parent half period")
    if (half - left_width) % 2 or (half - right_width) % 2:
        raise ValueError("grating widths must be centered within their half-period cells")
    if period_um <= 0.0 or not slab_index > port_index > 0.0:
        raise ValueError("require period > 0 and slab_index > port_index > 0")

    margin = int(port_margin_rows)
    slab_count = int(central_slab_rows)
    rib_count = int(grating_rib_rows)
    spacing = float(period_um) / nx
    structure_count = slab_count + 2 * rib_count
    grid = UniformGrid2D(
        nx=nx,
        nz=2 * margin + structure_count,
        dx=spacing,
        dz=spacing,
    )
    epsilon = np.full(grid.shape, float(port_index) ** 2, dtype=float)
    structure_start = margin
    lower_rib = (structure_start, structure_start + rib_count)
    slab = (lower_rib[1], lower_rib[1] + slab_count)
    upper_rib = (slab[1], slab[1] + rib_count)
    left_columns = ((half - left_width) // 2, (half + left_width) // 2)
    right_columns = (
        half + (half - right_width) // 2,
        half + (half + right_width) // 2,
    )
    high = float(slab_index) ** 2
    epsilon[slice(*slab)] = high
    for rows in (lower_rib, upper_rib):
        epsilon[slice(*rows), slice(*left_columns)] = high
        epsilon[slice(*rows), slice(*right_columns)] = high

    if upper_rib[1] != grid.nz - margin:
        raise RuntimeError("period-doubled surface-relief rows did not close")
    if not np.array_equal(epsilon, np.flipud(epsilon)):
        raise RuntimeError("period-doubled surface-relief slab is not axially symmetric")
    if left_width == right_width and not np.array_equal(
        epsilon, np.roll(epsilon, half, axis=1)
    ):
        raise RuntimeError("equal sites did not restore the parent half-period translation")

    layout = PeriodDoubledSurfaceReliefSlabLayout(
        central_slab_rows=slab,
        lower_grating_rows=lower_rib,
        upper_grating_rows=upper_rib,
        structure_rows=(structure_start, upper_rib[1]),
        cavity_rows=(structure_start, upper_rib[1]),
        left_grating_columns=left_columns,
        right_grating_columns=right_columns,
        parent_half_period_cells=half,
    )
    return grid, epsilon, layout


def build_symmetric_coupled_surface_relief_slabs(
    *,
    period_um: float,
    cells_per_period: int,
    port_margin_rows: int,
    single_slab_rows: int,
    inner_grating_rib_rows: int,
    spacer_rows: int,
    grating_width_cells: int,
    port_index: float,
    slab_index: float,
) -> tuple[
    UniformGrid2D, FloatArray, SymmetricCoupledSurfaceReliefSlabsLayout
]:
    """Rasterize two mirrored single-mode slabs with facing relief gratings.

    From bottom to top the structure contains one uniform slab, its attached
    patterned rib, a homogeneous spacer, the mirrored patterned rib, and the
    mirrored uniform slab.  The two high-index bodies remain disconnected for
    every positive spacer thickness.  Exact axial and lateral mirror symmetry
    makes their coupled supermodes separable into even- and odd-``H_y`` port
    eigenchannels.
    """

    integer_controls = {
        "cells_per_period": cells_per_period,
        "port_margin_rows": port_margin_rows,
        "single_slab_rows": single_slab_rows,
        "inner_grating_rib_rows": inner_grating_rib_rows,
        "spacer_rows": spacer_rows,
        "grating_width_cells": grating_width_cells,
    }
    if any(int(value) != value or int(value) < 1 for value in integer_controls.values()):
        raise ValueError("all coupled-slab cell and row controls must be positive integers")
    nx = int(cells_per_period)
    margin = int(port_margin_rows)
    slab_count = int(single_slab_rows)
    rib_count = int(inner_grating_rib_rows)
    gap_count = int(spacer_rows)
    width = int(grating_width_cells)
    if width >= nx:
        raise ValueError("grating_width_cells must be smaller than one period")
    if (nx - width) % 2:
        raise ValueError("grating width must preserve lateral mirror symmetry")
    if period_um <= 0.0 or not slab_index > port_index > 0.0:
        raise ValueError("require period > 0 and slab_index > port_index > 0")

    spacing = float(period_um) / nx
    structure_count = 2 * (slab_count + rib_count) + gap_count
    grid = UniformGrid2D(
        nx=nx,
        nz=2 * margin + structure_count,
        dx=spacing,
        dz=spacing,
    )
    epsilon = np.full(grid.shape, float(port_index) ** 2, dtype=float)
    structure_start = margin
    lower_slab = (structure_start, structure_start + slab_count)
    lower_rib = (lower_slab[1], lower_slab[1] + rib_count)
    spacer = (lower_rib[1], lower_rib[1] + gap_count)
    upper_rib = (spacer[1], spacer[1] + rib_count)
    upper_slab = (upper_rib[1], upper_rib[1] + slab_count)
    column_start = (nx - width) // 2
    column_stop = column_start + width
    epsilon[slice(*lower_slab)] = float(slab_index) ** 2
    epsilon[slice(*upper_slab)] = float(slab_index) ** 2
    epsilon[slice(*lower_rib), column_start:column_stop] = float(slab_index) ** 2
    epsilon[slice(*upper_rib), column_start:column_stop] = float(slab_index) ** 2

    if upper_slab[1] != grid.nz - margin:
        raise RuntimeError("coupled-slab row construction did not close")
    if not np.array_equal(epsilon, np.flipud(epsilon)):
        raise RuntimeError("coupled slabs are not exactly axially symmetric")
    if not np.array_equal(epsilon, np.fliplr(epsilon)):
        raise RuntimeError("coupled-slab gratings are not exactly laterally symmetric")

    layout = SymmetricCoupledSurfaceReliefSlabsLayout(
        lower_slab_rows=lower_slab,
        lower_grating_rows=lower_rib,
        spacer_rows=spacer,
        upper_grating_rows=upper_rib,
        upper_slab_rows=upper_slab,
        structure_rows=(structure_start, upper_slab[1]),
        cavity_rows=(structure_start, upper_slab[1]),
        grating_columns=(column_start, column_stop),
    )
    return grid, epsilon, layout


def build_period_doubled_coupled_surface_relief_slabs(
    *,
    period_um: float,
    cells_per_period: int,
    port_margin_rows: int,
    single_slab_rows: int,
    inner_grating_rib_rows: int,
    spacer_rows: int,
    left_grating_width_cells: int,
    right_grating_width_cells: int,
    port_index: float,
    slab_index: float,
) -> tuple[
    UniformGrid2D, FloatArray, PeriodDoubledCoupledSurfaceReliefSlabsLayout
]:
    """Rasterize two mirrored slabs with period-doubled facing reliefs.

    Equal left/right site widths restore a parent lattice of half the reported
    computational period.  Width contrast opens the folded guided modes while
    the two slab components produce closely spaced even/odd axial supermodes.
    """

    integer_controls = {
        "cells_per_period": cells_per_period,
        "port_margin_rows": port_margin_rows,
        "single_slab_rows": single_slab_rows,
        "inner_grating_rib_rows": inner_grating_rib_rows,
        "spacer_rows": spacer_rows,
        "left_grating_width_cells": left_grating_width_cells,
        "right_grating_width_cells": right_grating_width_cells,
    }
    if any(int(value) != value or int(value) < 1 for value in integer_controls.values()):
        raise ValueError("all coupled period-doubled controls must be positive integers")
    nx = int(cells_per_period)
    if nx % 2:
        raise ValueError("cells_per_period must be even for a half-period parent lattice")
    half = nx // 2
    left_width = int(left_grating_width_cells)
    right_width = int(right_grating_width_cells)
    if left_width >= half or right_width >= half:
        raise ValueError("each grating width must be smaller than one parent half period")
    if (half - left_width) % 2 or (half - right_width) % 2:
        raise ValueError("grating widths must be centered within their half-period cells")
    if period_um <= 0.0 or not slab_index > port_index > 0.0:
        raise ValueError("require period > 0 and slab_index > port_index > 0")

    margin = int(port_margin_rows)
    slab_count = int(single_slab_rows)
    rib_count = int(inner_grating_rib_rows)
    gap_count = int(spacer_rows)
    spacing = float(period_um) / nx
    structure_count = 2 * (slab_count + rib_count) + gap_count
    grid = UniformGrid2D(
        nx=nx,
        nz=2 * margin + structure_count,
        dx=spacing,
        dz=spacing,
    )
    epsilon = np.full(grid.shape, float(port_index) ** 2, dtype=float)
    lower_slab = (margin, margin + slab_count)
    lower_rib = (lower_slab[1], lower_slab[1] + rib_count)
    spacer = (lower_rib[1], lower_rib[1] + gap_count)
    upper_rib = (spacer[1], spacer[1] + rib_count)
    upper_slab = (upper_rib[1], upper_rib[1] + slab_count)
    left_columns = ((half - left_width) // 2, (half + left_width) // 2)
    right_columns = (
        half + (half - right_width) // 2,
        half + (half + right_width) // 2,
    )
    high = float(slab_index) ** 2
    epsilon[slice(*lower_slab)] = high
    epsilon[slice(*upper_slab)] = high
    for rows in (lower_rib, upper_rib):
        epsilon[slice(*rows), slice(*left_columns)] = high
        epsilon[slice(*rows), slice(*right_columns)] = high

    if upper_slab[1] != grid.nz - margin:
        raise RuntimeError("coupled period-doubled rows did not close")
    if not np.array_equal(epsilon, np.flipud(epsilon)):
        raise RuntimeError("coupled period-doubled slab is not axially symmetric")
    if left_width == right_width and not np.array_equal(
        epsilon, np.roll(epsilon, half, axis=1)
    ):
        raise RuntimeError("equal coupled sites did not restore half-period translation")

    layout = PeriodDoubledCoupledSurfaceReliefSlabsLayout(
        lower_slab_rows=lower_slab,
        lower_grating_rows=lower_rib,
        spacer_rows=spacer,
        upper_grating_rows=upper_rib,
        upper_slab_rows=upper_slab,
        structure_rows=(lower_slab[0], upper_slab[1]),
        cavity_rows=(lower_slab[0], upper_slab[1]),
        left_grating_columns=left_columns,
        right_grating_columns=right_columns,
        parent_half_period_cells=half,
    )
    return grid, epsilon, layout


def build_symmetric_coupled_dual_surface_relief_slabs(
    *,
    period_um: float,
    cells_per_period: int,
    port_margin_rows: int,
    single_slab_rows: int,
    inner_grating_rib_rows: int,
    outer_grating_rib_rows: int,
    spacer_rows: int,
    inner_grating_width_cells: int,
    outer_grating_width_cells: int,
    port_index: float,
    slab_index: float,
    outer_grating_shift_cells: int = 0,
) -> tuple[
    UniformGrid2D, FloatArray, SymmetricCoupledDualSurfaceReliefSlabsLayout
]:
    """Rasterize identical slabs with independently patterned inner/outer faces.

    Each high-index component contains a uniform single-mode slab plus attached
    inner- and outer-face relief ribs.  The upper component is the exact axial
    reflection of the lower component.  Inner and outer relief depths and
    widths are independent, providing two coherent radiation pathways without
    detuning the isolated slab pair or mixing global axial parity.
    """

    integer_controls = {
        "cells_per_period": cells_per_period,
        "port_margin_rows": port_margin_rows,
        "single_slab_rows": single_slab_rows,
        "inner_grating_rib_rows": inner_grating_rib_rows,
        "outer_grating_rib_rows": outer_grating_rib_rows,
        "spacer_rows": spacer_rows,
        "inner_grating_width_cells": inner_grating_width_cells,
        "outer_grating_width_cells": outer_grating_width_cells,
    }
    if any(int(value) != value or int(value) < 1 for value in integer_controls.values()):
        raise ValueError("all dual-relief cell and row controls must be positive integers")
    nx = int(cells_per_period)
    margin = int(port_margin_rows)
    slab_count = int(single_slab_rows)
    inner_count = int(inner_grating_rib_rows)
    outer_count = int(outer_grating_rib_rows)
    gap_count = int(spacer_rows)
    inner_width = int(inner_grating_width_cells)
    outer_width = int(outer_grating_width_cells)
    if int(outer_grating_shift_cells) != outer_grating_shift_cells:
        raise ValueError("outer_grating_shift_cells must be an integer")
    outer_shift = int(outer_grating_shift_cells) % nx
    for name, width in (
        ("inner_grating_width_cells", inner_width),
        ("outer_grating_width_cells", outer_width),
    ):
        if width >= nx:
            raise ValueError(f"{name} must be smaller than one period")
        if (nx - width) % 2:
            raise ValueError(f"{name} must preserve lateral mirror symmetry")
    if period_um <= 0.0 or not slab_index > port_index > 0.0:
        raise ValueError("require period > 0 and slab_index > port_index > 0")

    spacing = float(period_um) / nx
    structure_count = 2 * (outer_count + slab_count + inner_count) + gap_count
    grid = UniformGrid2D(
        nx=nx,
        nz=2 * margin + structure_count,
        dx=spacing,
        dz=spacing,
    )
    epsilon = np.full(grid.shape, float(port_index) ** 2, dtype=float)
    cursor = margin
    lower_outer = (cursor, cursor + outer_count)
    cursor = lower_outer[1]
    lower_slab = (cursor, cursor + slab_count)
    cursor = lower_slab[1]
    lower_inner = (cursor, cursor + inner_count)
    cursor = lower_inner[1]
    spacer = (cursor, cursor + gap_count)
    cursor = spacer[1]
    upper_inner = (cursor, cursor + inner_count)
    cursor = upper_inner[1]
    upper_slab = (cursor, cursor + slab_count)
    cursor = upper_slab[1]
    upper_outer = (cursor, cursor + outer_count)

    inner_start = (nx - inner_width) // 2
    inner_stop = inner_start + inner_width
    outer_start = (nx - outer_width) // 2
    outer_stop = outer_start + outer_width
    high = float(slab_index) ** 2
    epsilon[slice(*lower_slab)] = high
    epsilon[slice(*upper_slab)] = high
    epsilon[slice(*lower_inner), inner_start:inner_stop] = high
    epsilon[slice(*upper_inner), inner_start:inner_stop] = high
    outer_mask = np.zeros(nx, dtype=bool)
    outer_mask[outer_start:outer_stop] = True
    outer_mask = np.roll(outer_mask, outer_shift)
    epsilon[slice(*lower_outer), np.flatnonzero(outer_mask)] = high
    epsilon[slice(*upper_outer), np.flatnonzero(outer_mask)] = high

    if upper_outer[1] != grid.nz - margin:
        raise RuntimeError("dual-relief coupled-slab row construction did not close")
    if not np.array_equal(epsilon, np.flipud(epsilon)):
        raise RuntimeError("dual-relief coupled slabs are not exactly axially symmetric")
    if outer_shift in (0, nx // 2) and not np.array_equal(
        epsilon, np.fliplr(epsilon)
    ):
        raise RuntimeError("dual-relief coupled slabs are not exactly laterally symmetric")

    layout = SymmetricCoupledDualSurfaceReliefSlabsLayout(
        lower_outer_grating_rows=lower_outer,
        lower_slab_rows=lower_slab,
        lower_inner_grating_rows=lower_inner,
        spacer_rows=spacer,
        upper_inner_grating_rows=upper_inner,
        upper_slab_rows=upper_slab,
        upper_outer_grating_rows=upper_outer,
        structure_rows=(lower_outer[0], upper_outer[1]),
        cavity_rows=(lower_outer[0], upper_outer[1]),
        inner_grating_columns=(inner_start, inner_stop),
        outer_grating_columns=(outer_start, outer_stop),
        outer_grating_shift_cells=outer_shift,
    )
    return grid, epsilon, layout


def symmetric_slab_tm_frequencies(
    *,
    period_um: float,
    slab_thickness_um: float,
    core_index: float,
    cladding_index: float,
    samples: int = 5000,
) -> dict[str, list[float]]:
    """Return folded symmetric-slab TM frequencies at beta=2*pi/period.

    The scalar field is ``H_y``.  Returned ``even_hy`` and ``odd_hy`` roots
    solve the exact symmetric, nondispersive three-layer slab equations.  The
    first odd-Hy root is the TM1 reference used by the M11 staged screen.
    """

    if period_um <= 0.0 or slab_thickness_um <= 0.0:
        raise ValueError("period and slab thickness must be positive")
    if not core_index > cladding_index > 0.0:
        raise ValueError("indices must satisfy core_index > cladding_index > 0")
    if int(samples) != samples or int(samples) < 100:
        raise ValueError("samples must be an integer of at least 100")
    beta = 2.0 * np.pi / float(period_um)
    half_thickness = 0.5 * float(slab_thickness_um)
    epsilon_core = float(core_index) ** 2
    epsilon_clad = float(cladding_index) ** 2
    lower = beta / float(core_index) * (1.0 + 1e-9)
    upper = beta / float(cladding_index) * (1.0 - 1e-9)

    def components(omega: float) -> tuple[float, float]:
        h = np.sqrt(max(epsilon_core * omega**2 - beta**2, 0.0))
        q = np.sqrt(max(beta**2 - epsilon_clad * omega**2, 0.0))
        return float(h), float(q)

    def even_residual(omega: float) -> float:
        h, q = components(omega)
        return float(
            epsilon_clad * h * np.sin(h * half_thickness)
            - epsilon_core * q * np.cos(h * half_thickness)
        )

    def odd_residual(omega: float) -> float:
        h, q = components(omega)
        return float(
            epsilon_clad * h * np.cos(h * half_thickness)
            + epsilon_core * q * np.sin(h * half_thickness)
        )

    omega = np.linspace(lower, upper, int(samples))
    h_samples = np.sqrt(np.maximum(epsilon_core * omega**2 - beta**2, 0.0))
    q_samples = np.sqrt(np.maximum(beta**2 - epsilon_clad * omega**2, 0.0))
    even_values = (
        epsilon_clad * h_samples * np.sin(h_samples * half_thickness)
        - epsilon_core * q_samples * np.cos(h_samples * half_thickness)
    )
    odd_values = (
        epsilon_clad * h_samples * np.cos(h_samples * half_thickness)
        + epsilon_core * q_samples * np.sin(h_samples * half_thickness)
    )

    def roots(function: object, values: np.ndarray) -> list[float]:
        result: list[float] = []
        for index in np.flatnonzero(values[:-1] * values[1:] < 0.0):
            root = float(brentq(function, omega[index], omega[index + 1]))
            if not result or abs(root - result[-1]) > 1e-8:
                result.append(root)
        return result

    return {
        "even_hy": roots(even_residual, even_values),
        "odd_hy": roots(odd_residual, odd_values),
    }
