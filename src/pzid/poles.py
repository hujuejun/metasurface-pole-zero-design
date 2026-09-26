"""Time-reversal residuals for prescribed passive poles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pzid.adjoint import scattering_matrix_with_jacobian
from pzid.twoport import TwoPortScatteringSetup


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class PoleTarget:
    """Desired passive pole and upper-to-lower port coupling ratio."""

    frequency: complex
    coupling_ratio: complex

    def __post_init__(self) -> None:
        if complex(self.frequency).imag >= 0.0:
            raise ValueError("a passive target pole must lie below the real axis")
        values = np.asarray((self.frequency, self.coupling_ratio))
        if not np.all(np.isfinite(values)):
            raise ValueError("pole target values must be finite")


@dataclass(frozen=True)
class PoleConstraintResult:
    """Chen time-reversal residual and complex permittivity Jacobian."""

    residual: ComplexArray
    jacobian: ComplexArray
    scattering_matrix: ComplexArray
    evaluation_frequency: complex
    incoming_amplitudes: ComplexArray


def time_reversal_residual(
    scattering_matrix: ArrayLike,
    coupling_ratio: complex,
) -> ComplexArray:
    r"""Evaluate ``S @ [1, sigma.conjugate()]``.

    The supplied matrix must be evaluated at the complex conjugate of the
    target passive-pole frequency. A zero vector enforces both the pole
    frequency and its specified port-coupling ratio in a lossless reciprocal
    system. This full-matrix zero is an enforcement surrogate, not a
    transmission- or reflection-channel zero.
    """

    matrix = np.asarray(scattering_matrix, dtype=np.complex128)
    if matrix.shape != (2, 2):
        raise ValueError("scattering_matrix must have shape (2, 2)")
    incoming = np.asarray((1.0, np.conj(coupling_ratio)), dtype=np.complex128)
    return np.asarray(matrix @ incoming)


def solve_pole_constraint(
    target: PoleTarget,
    epsilon_r: ArrayLike,
    setup: TwoPortScatteringSetup,
) -> PoleConstraintResult:
    """Evaluate a prescribed-pole residual and its discrete adjoint Jacobian."""

    epsilon = np.asarray(epsilon_r, dtype=np.complex128)
    if np.max(np.abs(epsilon.imag)) > 1e-13:
        raise ValueError("time-reversal pole enforcement requires lossless epsilon")
    evaluation_frequency = complex(np.conj(target.frequency))
    solution = scattering_matrix_with_jacobian(
        evaluation_frequency, epsilon, setup
    )
    incoming = np.asarray(
        (1.0, np.conj(target.coupling_ratio)), dtype=np.complex128
    )
    residual = np.asarray(solution.matrix @ incoming)
    jacobian = np.einsum("abij,b->aij", solution.jacobian, incoming)
    return PoleConstraintResult(
        residual=residual,
        jacobian=np.asarray(jacobian),
        scattering_matrix=solution.matrix,
        evaluation_frequency=evaluation_frequency,
        incoming_amplitudes=incoming,
    )
