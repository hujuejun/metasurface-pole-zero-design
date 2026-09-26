"""Analytic axial PML profiles for complex-frequency 3D FDFD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pzid.grid3d import UniformGrid3D


@dataclass(frozen=True)
class AxialPML3D:
    """Polynomial complex-coordinate stretch at both axial boundaries."""

    cells: int
    target_round_trip_amplitude: float = 1e-4
    polynomial_order: int = 3

    def validate(self, grid: UniformGrid3D) -> None:
        if self.cells < 0 or 2 * self.cells >= grid.nz:
            raise ValueError("PML cells must leave at least one interior layer")
        if not 0.0 < self.target_round_trip_amplitude < 1.0:
            raise ValueError("target_round_trip_amplitude must lie in (0, 1)")
        if self.polynomial_order < 1:
            raise ValueError("polynomial_order must be positive")

    def conductivity(
        self,
        grid: UniformGrid3D,
        *,
        speed_of_light: float = 1.0,
        stagger: Literal["node", "half"] = "half",
    ) -> NDArray[np.float64]:
        """Return the nonnegative PML conductivity sampled on one Yee grid."""

        self.validate(grid)
        if speed_of_light <= 0.0:
            raise ValueError("speed_of_light must be positive")
        sigma = np.zeros(grid.nz, dtype=float)
        if self.cells == 0:
            return sigma

        thickness = self.cells * grid.dz
        sigma_max = -(
            (self.polynomial_order + 1)
            * speed_of_light
            * np.log(self.target_round_trip_amplitude)
            / (2.0 * thickness)
        )
        if stagger == "half":
            depth = (np.arange(self.cells, dtype=float) + 0.5) / self.cells
            edge_to_interface = sigma_max * depth[::-1] ** self.polynomial_order
            sigma[: self.cells] = edge_to_interface
            sigma[-self.cells :] = edge_to_interface[::-1]
        elif stagger == "node":
            # E_x/E_y live at z=k*dz.  The lower array includes the outer
            # boundary node while the upper ghost node lies just outside the
            # stored array, so the two samples are shifted by one Yee step.
            lower_depth = np.arange(
                self.cells, 0, -1, dtype=float
            ) / self.cells
            upper_depth = np.arange(0, self.cells, dtype=float) / self.cells
            sigma[: self.cells] = sigma_max * lower_depth**self.polynomial_order
            sigma[-self.cells :] = sigma_max * upper_depth**self.polynomial_order
        else:
            raise ValueError("stagger must be 'node' or 'half'")
        return sigma

    def stretches(
        self,
        omega: complex,
        grid: UniformGrid3D,
        *,
        speed_of_light: float = 1.0,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Return node- and half-grid stretches ``1+i sigma/omega``.

        The rational dependence on ``omega`` is analytic away from zero and
        therefore supports evaluation in both complex-frequency half-planes.
        The PML attenuation must exceed the exponential growth of any target
        lower-half-plane outgoing wave.  More aggressive target attenuation
        is not automatically better on a fixed grid: it steepens the PML and
        must be accompanied by a convergence increase in ``cells``.
        """

        if complex(omega) == 0.0:
            raise ValueError("PML stretch is undefined at zero frequency")
        sigma_node = self.conductivity(
            grid, speed_of_light=speed_of_light, stagger="node"
        )
        sigma_half = self.conductivity(
            grid, speed_of_light=speed_of_light, stagger="half"
        )
        return (
            1.0 + 1j * sigma_node / complex(omega),
            1.0 + 1j * sigma_half / complex(omega),
        )
