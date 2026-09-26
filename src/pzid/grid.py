"""Cell-centered grids for the finite-volume frequency-domain model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class UniformGrid2D:
    """Cell-centered uniform grid with array shape ``(nz, nx)``."""

    nx: int
    nz: int
    dx: float
    dz: float

    def __post_init__(self) -> None:
        if self.nx < 2 or self.nz < 2:
            raise ValueError("nx and nz must both be at least two")
        if self.dx <= 0.0 or self.dz <= 0.0:
            raise ValueError("dx and dz must be positive")

    @property
    def shape(self) -> tuple[int, int]:
        return self.nz, self.nx

    @property
    def size(self) -> int:
        return self.nx * self.nz

    @property
    def period(self) -> float:
        return self.nx * self.dx

    @property
    def height(self) -> float:
        return self.nz * self.dz

    @property
    def x(self) -> NDArray[np.float64]:
        return (np.arange(self.nx, dtype=float) + 0.5) * self.dx

    @property
    def z(self) -> NDArray[np.float64]:
        return (np.arange(self.nz, dtype=float) + 0.5) * self.dz

    @property
    def z_edges(self) -> NDArray[np.float64]:
        """Axial cell edges, including the two domain boundaries."""

        return np.arange(self.nz + 1, dtype=float) * self.dz

    @property
    def dz_cells(self) -> NDArray[np.float64]:
        """Axial cell widths."""

        return np.full(self.nz, self.dz, dtype=float)

    @property
    def z_center_spacings(self) -> NDArray[np.float64]:
        """Distances between neighboring axial cell centers."""

        return np.full(self.nz - 1, self.dz, dtype=float)

    @property
    def reference_dz(self) -> float:
        """Length used to normalize the volume-weighted weak form."""

        return self.dz

    def boundary_dz(self, side: str) -> float:
        """Return the boundary-cell width on ``side``."""

        if side not in {"lower", "upper"}:
            raise ValueError("side must be 'lower' or 'upper'")
        return self.dz

    def uniform_dz_for_planes(self, planes: tuple[int, ...]) -> float:
        """Return the common local spacing around a set of port planes."""

        if not planes or any(not 0 <= plane < self.nz for plane in planes):
            raise ValueError("planes must lie inside the grid")
        return self.dz

    def flat_index(self, iz: int, ix: int) -> int:
        if not 0 <= iz < self.nz or not 0 <= ix < self.nx:
            raise IndexError("grid index out of range")
        return iz * self.nx + ix


@dataclass(frozen=True)
class NonuniformGrid2D:
    """Cell-centered grid with uniform ``x`` and fixed nonuniform ``z``.

    ``z_edges`` is copied on construction so the grid cannot change while an
    optimization is running.  The finite-volume equations are normalized by
    ``reference_dz``.  This global row scale does not alter a calibrated
    scattering solution; choosing the minimum cell width keeps all volume
    weights at least one and recovers the legacy operator exactly when the
    edges are uniform.
    """

    nx: int
    dx: float
    z_edges: NDArray[np.float64]
    normalization_dz: float | None = None

    def __post_init__(self) -> None:
        edges = np.asarray(self.z_edges, dtype=float).copy()
        if self.nx < 2 or self.dx <= 0.0:
            raise ValueError("nx must be at least two and dx must be positive")
        if edges.ndim != 1 or edges.size < 3:
            raise ValueError("z_edges must contain at least three edges")
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
            raise ValueError("z_edges must be finite and strictly increasing")
        if not np.isclose(edges[0], 0.0, rtol=0.0, atol=1e-14):
            raise ValueError("the first axial edge must be zero")
        if self.normalization_dz is not None and self.normalization_dz <= 0.0:
            raise ValueError("normalization_dz must be positive")
        edges.setflags(write=False)
        object.__setattr__(self, "z_edges", edges)

    @property
    def nz(self) -> int:
        return self.z_edges.size - 1

    @property
    def shape(self) -> tuple[int, int]:
        return self.nz, self.nx

    @property
    def size(self) -> int:
        return self.nx * self.nz

    @property
    def period(self) -> float:
        return self.nx * self.dx

    @property
    def height(self) -> float:
        return float(self.z_edges[-1])

    @property
    def x(self) -> NDArray[np.float64]:
        return (np.arange(self.nx, dtype=float) + 0.5) * self.dx

    @property
    def z(self) -> NDArray[np.float64]:
        return 0.5 * (self.z_edges[:-1] + self.z_edges[1:])

    @property
    def dz_cells(self) -> NDArray[np.float64]:
        return np.diff(self.z_edges)

    @property
    def z_center_spacings(self) -> NDArray[np.float64]:
        return np.diff(self.z)

    @property
    def reference_dz(self) -> float:
        if self.normalization_dz is not None:
            return float(self.normalization_dz)
        return float(np.min(self.dz_cells))

    def boundary_dz(self, side: str) -> float:
        if side == "lower":
            return float(self.dz_cells[0])
        if side == "upper":
            return float(self.dz_cells[-1])
        raise ValueError("side must be 'lower' or 'upper'")

    def uniform_dz_for_planes(self, planes: tuple[int, ...]) -> float:
        """Return a common cell/center spacing for uniform port planes.

        The two-point modal decomposition assumes the discrete port is locally
        translation invariant.  Every cell from the smallest through largest
        requested plane, plus the intervening center spacings, is checked.
        """

        if not planes or any(not 0 <= plane < self.nz for plane in planes):
            raise ValueError("planes must lie inside the grid")
        first = min(planes)
        last = max(planes)
        widths = self.dz_cells[first : last + 1]
        candidate = float(widths[0])
        if not np.allclose(widths, candidate, rtol=1e-10, atol=1e-13):
            raise ValueError("port planes must lie in one uniform axial region")
        if last > first and not np.allclose(
            self.z_center_spacings[first:last],
            candidate,
            rtol=1e-10,
            atol=1e-13,
        ):
            raise ValueError("port-plane center spacings must be uniform")
        return candidate

    def flat_index(self, iz: int, ix: int) -> int:
        if not 0 <= iz < self.nz or not 0 <= ix < self.nx:
            raise IndexError("grid index out of range")
        return iz * self.nx + ix


Grid2D: TypeAlias = UniformGrid2D | NonuniformGrid2D
