"""Complex-analytic discrete adjoints for TM permittivity variables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import splu

from pzid.grid import Grid2D
from pzid.twoport import (
    TwoPortLinearSystem,
    TwoPortScatteringSetup,
    assemble_two_port_system,
)


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ScatteringJacobianSolution:
    """Two-port matrix and its complex Jacobian with respect to real epsilon."""

    matrix: ComplexArray
    jacobian: ComplexArray
    fields: ComplexArray
    adjoint_fields: ComplexArray
    system: TwoPortLinearSystem


def _harmonic_derivatives(left: complex, right: complex) -> tuple[complex, complex]:
    denominator = left + right
    if denominator == 0.0:
        raise ValueError("adjacent flux coefficients have zero harmonic sum")
    return (
        2.0 * right**2 / denominator**2,
        2.0 * left**2 / denominator**2,
    )


def _axial_conductance_derivatives(
    lower: complex,
    upper: complex,
    lower_width: float,
    upper_width: float,
) -> tuple[complex, complex]:
    """Derivatives of the unequal-cell axial face conductance."""

    denominator = lower_width * upper + upper_width * lower
    if denominator == 0.0:
        raise ValueError("adjacent axial coefficients have zero weighted sum")
    return (
        2.0 * lower_width * upper**2 / denominator**2,
        2.0 * upper_width * lower**2 / denominator**2,
    )


def tm_bilinear_epsilon_gradient(
    epsilon_r: ArrayLike,
    grid: Grid2D,
    left_field: ArrayLike,
    right_field: ArrayLike,
    *,
    stretch_z: ArrayLike = 1.0,
) -> ComplexArray:
    r"""Return ``d(left.T @ A @ right) / d epsilon_r``.

    ``A`` is the zero-Bloch-wavevector TM operator assembled by
    :func:`pzid.fdfd.assemble_tm_operator`. The derivative is bilinear: the
    left field is *not* conjugated. This is the complex-analytic transpose
    adjoint required for complex residuals. The exact radiation-boundary
    update is independent of design permittivity and therefore adds no term.
    """

    epsilon = np.asarray(epsilon_r, dtype=np.complex128)
    if epsilon.shape != grid.shape:
        raise ValueError(f"epsilon_r must have shape {grid.shape}")
    left = np.asarray(left_field, dtype=np.complex128)
    right = np.asarray(right_field, dtype=np.complex128)
    if left.size != grid.size or right.size != grid.size:
        raise ValueError("left_field and right_field must match the grid size")
    left = left.reshape(grid.shape)
    right = right.reshape(grid.shape)

    stretch = np.asarray(stretch_z, dtype=np.complex128)
    if stretch.ndim == 0:
        stretch = np.full(grid.nz, stretch, dtype=np.complex128)
    if stretch.shape != (grid.nz,):
        raise ValueError(f"stretch_z must be scalar or have shape {(grid.nz,)}")
    if np.any(stretch == 0.0) or np.any(epsilon == 0.0):
        raise ValueError("stretch_z and epsilon_r must be nonzero")

    stretch_2d = stretch[:, np.newaxis]
    coefficient_x = stretch_2d / epsilon
    coefficient_z = 1.0 / (stretch_2d * epsilon)
    derivative_x = -coefficient_x / epsilon
    derivative_z = -coefficient_z / epsilon
    gradient = np.zeros(grid.shape, dtype=np.complex128)

    def add_face(
        first: tuple[int, int],
        second: tuple[int, int],
        coefficient: ComplexArray,
        coefficient_derivative: ComplexArray,
        face_weight: float,
        face_derivatives: tuple[complex, complex] | None = None,
    ) -> None:
        first_coefficient = coefficient[first]
        second_coefficient = coefficient[second]
        if face_derivatives is None:
            derivative_first, derivative_second = _harmonic_derivatives(
                first_coefficient, second_coefficient
            )
        else:
            derivative_first, derivative_second = face_derivatives
        field_product = (
            (left[first] - left[second])
            * (right[first] - right[second])
            * face_weight
        )
        gradient[first] += (
            field_product * derivative_first * coefficient_derivative[first]
        )
        gradient[second] += (
            field_product * derivative_second * coefficient_derivative[second]
        )

    for iz in range(grid.nz):
        for ix in range(grid.nx):
            add_face(
                (iz, ix),
                (iz, (ix + 1) % grid.nx),
                coefficient_x,
                derivative_x,
                grid.dz_cells[iz] / grid.reference_dz / grid.dx**2,
            )
            if iz < grid.nz - 1:
                add_face(
                    (iz, ix),
                    (iz + 1, ix),
                    coefficient_z,
                    derivative_z,
                    1.0 / grid.reference_dz,
                    _axial_conductance_derivatives(
                        coefficient_z[iz, ix],
                        coefficient_z[iz + 1, ix],
                        float(grid.dz_cells[iz]),
                        float(grid.dz_cells[iz + 1]),
                    ),
                )

    # The base operator has one Dirichlet ghost contribution at each axial
    # edge. Radiation boundaries replace the ghost value but retain this
    # diagonal term. Port-edge epsilon is fixed in valid scattering designs.
    gradient[0] += (
        left[0]
        * right[0]
        * derivative_z[0]
        / (grid.boundary_dz("lower") * grid.reference_dz)
    )
    gradient[-1] += (
        left[-1]
        * right[-1]
        * derivative_z[-1]
        / (grid.boundary_dz("upper") * grid.reference_dz)
    )
    return gradient


def scattering_matrix_with_jacobian(
    omega: complex,
    epsilon_r: ArrayLike,
    setup: TwoPortScatteringSetup,
) -> ScatteringJacobianSolution:
    r"""Solve ``S`` and ``dS/d epsilon`` using two transpose adjoints.

    The Jacobian has shape ``(2, 2, nz, nx)``. Its first two axes are the
    outgoing and incoming port indices. Derivatives are with respect to real
    perturbations of the relative permittivity; material filtering and
    parameterization chain rules are applied at later optimization layers.
    """

    system = assemble_two_port_system(omega, epsilon_r, setup)
    factorization = splu(system.operator.tocsc())
    fields = np.asarray(factorization.solve(system.sources))
    matrix = system.output_projector @ fields + system.output_offset
    adjoint_fields = np.asarray(
        factorization.solve(system.output_projector.T, trans="T")
    )

    jacobian = np.empty((2, 2, *setup.grid.shape), dtype=np.complex128)
    for outgoing in range(2):
        for incoming in range(2):
            jacobian[outgoing, incoming] = -tm_bilinear_epsilon_gradient(
                system.epsilon_r,
                setup.grid,
                adjoint_fields[:, outgoing],
                fields[:, incoming],
            )
    return ScatteringJacobianSolution(
        matrix=np.asarray(matrix),
        jacobian=jacobian,
        fields=fields,
        adjoint_fields=adjoint_fields,
        system=system,
    )
