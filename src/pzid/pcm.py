"""Small, explicit effective-medium models for phase-change pilot studies.

The Lorentz--Lorenz rule is a model of a spatially homogeneous intermediate
state, not a measured Sb2Se3 material law.  Endpoint optical constants must
be supplied by the caller at the frequency being evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from pzid.hcg import (
    PairedHCGGeometryMapping,
    PairedTwoTierHCGGeometryMapping,
    SymmetricPairedHCGGeometryParameterization,
    SymmetricPairedTwoTierHCGGeometryParameterization,
)


def lorentz_lorenz_permittivity(
    amorphous_index: complex,
    crystalline_index: complex,
    crystalline_fraction: float | np.ndarray,
) -> complex | np.ndarray:
    """Mix endpoint dielectric functions at crystallization fraction 0..1.

    The complex refractive-index convention is ``n + 1j*k`` for a passive
    medium with the project's ``exp(-1j*omega*t)`` phasor convention.
    """

    fraction = np.asarray(crystalline_fraction, dtype=float)
    if not np.all(np.isfinite(fraction)) or np.any((fraction < 0.0) | (fraction > 1.0)):
        raise ValueError("crystalline_fraction must be finite and within [0, 1]")
    amorphous = complex(amorphous_index) ** 2
    crystalline = complex(crystalline_index) ** 2
    if amorphous.imag < -1e-14 or crystalline.imag < -1e-14:
        raise ValueError("endpoint dielectric functions must be passive")
    polarizability_amorphous = (amorphous - 1.0) / (amorphous + 2.0)
    polarizability_crystalline = (crystalline - 1.0) / (crystalline + 2.0)
    mixed = (
        (1.0 - fraction) * polarizability_amorphous
        + fraction * polarizability_crystalline
    )
    epsilon = (1.0 + 2.0 * mixed) / (1.0 - mixed)
    if np.ndim(epsilon) == 0:
        return complex(epsilon)
    return np.asarray(epsilon, dtype=np.complex128)


@dataclass(frozen=True)
class SymmetricPCMStateHCGParameterization:
    """One shared exact-interface geometry with a chosen HCG permittivity.

    The same state applies to both mirror-related HCGs.  The reference
    parameterization provides the physical contours and their analytic
    derivatives; only the material contrast is rescaled.  It therefore also
    supports a complex lossy permittivity without changing geometric masks.
    """

    reference: (
        SymmetricPairedHCGGeometryParameterization
        | SymmetricPairedTwoTierHCGGeometryParameterization
    )
    material_epsilon: complex

    def __post_init__(self) -> None:
        value = complex(self.material_epsilon)
        if not np.isfinite(value) or value.real <= 0.0 or value.imag < -1e-13:
            raise ValueError("material_epsilon must be finite and passive")

    @property
    def variable_count(self) -> int:
        return self.reference.variable_count

    @property
    def design_shape(self) -> tuple[int]:
        return self.reference.design_shape

    def map(
        self, raw: np.ndarray
    ) -> PairedHCGGeometryMapping | PairedTwoTierHCGGeometryMapping:
        baseline = self.reference.map(raw)
        port_epsilon = self.reference.port_index**2
        reference_contrast = self.reference.grating_index**2 - port_epsilon
        new_contrast = complex(self.material_epsilon) - port_epsilon
        factor = new_contrast / reference_contrast
        return replace(
            baseline,
            epsilon_r=np.asarray(
                port_epsilon + factor * (baseline.epsilon_r - port_epsilon),
                dtype=np.complex128,
            ),
            epsilon_derivatives=np.asarray(
                factor * baseline.epsilon_derivatives, dtype=np.complex128
            ),
        )

    def pullback(
        self,
        epsilon_gradient: np.ndarray,
        mapping: PairedHCGGeometryMapping | PairedTwoTierHCGGeometryMapping,
    ) -> np.ndarray:
        return self.reference.pullback(epsilon_gradient, mapping)
