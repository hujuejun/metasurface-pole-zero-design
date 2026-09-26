"""Global electromagnetic and coordinate conventions.

Keeping these choices in one importable object prevents sign, axis, and
polarization conventions from drifting between the forward solver, ports,
adjoint equations, and manuscript figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi


@dataclass(frozen=True)
class SimulationConvention:
    """Immutable conventions for the first two-dimensional model."""

    time_dependence: str = "exp(-i omega t)"
    invariant_axis: str = "y"
    periodic_axis: str = "x"
    propagation_axis: str = "z"
    polarization: str = "TM (H_y scalar; E_x and E_z in plane)"
    incident_port: int = 1
    transmitted_port: int = 2

    def passive_pole_is_valid(self, omega: complex) -> bool:
        """Return whether a frequency is in the passive pole half-plane."""

        return omega.imag < 0.0

    @staticmethod
    def first_diffraction_angular_frequency(
        period: float,
        port_index: float,
        speed_of_light: float = 1.0,
    ) -> float:
        """First normal-incidence diffraction threshold in a uniform port.

        The formula assumes a positive real refractive index and uses angular
        frequency. Normalized simulations use ``speed_of_light=1``.
        """

        if period <= 0.0:
            raise ValueError("period must be positive")
        if port_index <= 0.0:
            raise ValueError("port_index must be positive")
        return 2.0 * pi * speed_of_light / (period * port_index)

    def is_subwavelength(
        self,
        period: float,
        omega_max: float,
        port_indices: tuple[float, ...],
        speed_of_light: float = 1.0,
    ) -> bool:
        """Check that only the specular order propagates in every port."""

        if omega_max <= 0.0:
            raise ValueError("omega_max must be positive")
        if not port_indices:
            raise ValueError("at least one port index is required")
        thresholds = (
            self.first_diffraction_angular_frequency(
                period, index, speed_of_light
            )
            for index in port_indices
        )
        return omega_max < min(thresholds)


DEFAULT_CONVENTION = SimulationConvention()

