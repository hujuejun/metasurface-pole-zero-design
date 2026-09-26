"""Chen time-reversal pole constraints for the 3D specular channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pzid.adjoint3d import scattering_matrix_with_jacobian_3d
from pzid.poles import PoleTarget
from pzid.ports3d import Polarization
from pzid.scattering3d import ScatteringSetup3D, polarization_channel_indices


ComplexArray: TypeAlias = NDArray[np.complex128]


@dataclass(frozen=True)
class PoleConstraintResult3D:
    """Full-channel time-reversal residual and cell-epsilon Jacobian."""

    residual: ComplexArray
    jacobian: ComplexArray
    scattering_matrix: ComplexArray
    evaluation_frequency: complex
    incoming_amplitudes: ComplexArray
    polarization: Polarization


def time_reversal_incoming_3d(
    coupling_ratio: complex,
    polarization: Polarization,
) -> ComplexArray:
    """Embed ``[1, sigma*]`` in the four-channel 3D port basis."""

    lower, upper = polarization_channel_indices(polarization)
    incoming = np.zeros(4, dtype=np.complex128)
    incoming[lower] = 1.0
    incoming[upper] = np.conj(coupling_ratio)
    return incoming


def time_reversal_residual_3d(
    scattering_matrix: ArrayLike,
    coupling_ratio: complex,
    polarization: Polarization,
) -> ComplexArray:
    r"""Return ``S(p*) a`` across all open specular channels.

    Including both polarizations prevents an apparent pole constraint from
    silently radiating into the orthogonal polarization.  For a symmetry-
    protected single-polarization design, the two cross-polarized residuals
    vanish identically and the expression reduces to the two-port Chen form.
    """

    matrix = np.asarray(scattering_matrix, dtype=np.complex128)
    if matrix.shape != (4, 4):
        raise ValueError("scattering_matrix must have shape (4, 4)")
    return np.asarray(
        matrix @ time_reversal_incoming_3d(coupling_ratio, polarization)
    )


def solve_pole_constraint_3d(
    target: PoleTarget,
    epsilon_r: ArrayLike,
    setup: ScatteringSetup3D,
    *,
    polarization: Polarization = "x",
) -> PoleConstraintResult3D:
    """Evaluate a prescribed passive-pole residual and discrete adjoint."""

    epsilon = np.asarray(epsilon_r, dtype=np.complex128)
    if np.max(np.abs(epsilon.imag)) > 1e-13:
        raise ValueError("time-reversal pole enforcement requires lossless epsilon")
    evaluation_frequency = complex(np.conj(target.frequency))
    solution = scattering_matrix_with_jacobian_3d(
        evaluation_frequency, epsilon, setup
    )
    incoming = time_reversal_incoming_3d(
        target.coupling_ratio, polarization
    )
    residual = np.asarray(solution.matrix @ incoming)
    jacobian = np.einsum("abcde,b->acde", solution.jacobian, incoming)
    return PoleConstraintResult3D(
        residual=residual,
        jacobian=np.asarray(jacobian),
        scattering_matrix=solution.matrix,
        evaluation_frequency=evaluation_frequency,
        incoming_amplitudes=incoming,
        polarization=polarization,
    )
