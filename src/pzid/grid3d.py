"""Uniform Cartesian grids for the three-dimensional Maxwell solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


FieldComponent = Literal["x", "y", "z"]


@dataclass(frozen=True)
class UniformGrid3D:
    """Cell-centered material grid with logical Yee field arrays.

    Scalar material arrays use ``(nz, ny, nx)`` ordering and ``x`` is the
    fastest flattened axis.  Vector fields are stored component-major as
    ``[E_x.ravel(), E_y.ravel(), E_z.ravel()]``.  The logical component
    arrays are staggered by the discrete curl operators; keeping one common
    array shape makes Bloch-periodic unit cells compact and predictable.
    """

    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float

    def __post_init__(self) -> None:
        if min(self.nx, self.ny, self.nz) < 2:
            raise ValueError("nx, ny, and nz must all be at least two")
        if min(self.dx, self.dy, self.dz) <= 0.0:
            raise ValueError("dx, dy, and dz must be positive")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.nz, self.ny, self.nx

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def field_size(self) -> int:
        return 3 * self.cell_count

    @property
    def period_x(self) -> float:
        return self.nx * self.dx

    @property
    def period_y(self) -> float:
        return self.ny * self.dy

    @property
    def height(self) -> float:
        return self.nz * self.dz

    @property
    def x(self) -> NDArray[np.float64]:
        return (np.arange(self.nx, dtype=float) + 0.5) * self.dx

    @property
    def y(self) -> NDArray[np.float64]:
        return (np.arange(self.ny, dtype=float) + 0.5) * self.dy

    @property
    def z(self) -> NDArray[np.float64]:
        return (np.arange(self.nz, dtype=float) + 0.5) * self.dz

    @property
    def tangential_e_z(self) -> NDArray[np.float64]:
        """Axial coordinates of logical ``E_x`` and ``E_y`` Yee samples."""

        return np.arange(self.nz, dtype=float) * self.dz

    def cell_flat_index(self, iz: int, iy: int, ix: int) -> int:
        if not 0 <= iz < self.nz:
            raise IndexError("z index out of range")
        if not 0 <= iy < self.ny:
            raise IndexError("y index out of range")
        if not 0 <= ix < self.nx:
            raise IndexError("x index out of range")
        return (iz * self.ny + iy) * self.nx + ix

    def component_slice(self, component: FieldComponent | int) -> slice:
        if isinstance(component, str):
            try:
                index = ("x", "y", "z").index(component)
            except ValueError as error:
                raise ValueError("component must be 'x', 'y', or 'z'") from error
        else:
            index = int(component)
            if index not in (0, 1, 2):
                raise ValueError("component index must be 0, 1, or 2")
        start = index * self.cell_count
        return slice(start, start + self.cell_count)

    def component_view(
        self,
        field: NDArray[np.complex128],
        component: FieldComponent | int,
    ) -> NDArray[np.complex128]:
        """Return a ``(nz, ny, nx, ...)`` view of one vector component."""

        values = np.asarray(field)
        if values.shape[0] != self.field_size:
            raise ValueError("the first field axis must equal grid.field_size")
        selected = values[self.component_slice(component)]
        return selected.reshape(*self.shape, *values.shape[1:])
