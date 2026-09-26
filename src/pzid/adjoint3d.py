"""Complex-analytic transpose adjoints for the 3D Maxwell solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import splu

from pzid.fdfd3d import MaxwellOperator3D
from pzid.scattering3d import (
    ScatteringLinearSystem3D,
    ScatteringSetup3D,
    assemble_scattering_system_3d,
)


ComplexArray: TypeAlias = NDArray[np.complex128]


@dataclass(frozen=True)
class ScatteringJacobianSolution3D:
    """Four-channel scattering matrix and cell-permittivity Jacobian."""

    matrix: ComplexArray
    jacobian: ComplexArray
    fields: ComplexArray
    adjoint_fields: ComplexArray
    system: ScatteringLinearSystem3D


def maxwell_bilinear_epsilon_gradient_3d(
    system: MaxwellOperator3D,
    left_field: ArrayLike,
    right_field: ArrayLike,
) -> ComplexArray:
    r"""Return ``d(left.T @ A @ right) / d epsilon_cell``.

    The fields are not conjugated.  Cell permittivity enters the Yee operator
    through fixed arithmetic interpolation maps, so their transposes give the
    exact cell-centered pullback.
    """

    left = np.asarray(left_field, dtype=np.complex128).reshape(-1)
    right = np.asarray(right_field, dtype=np.complex128).reshape(-1)
    if left.size != system.grid.field_size or right.size != system.grid.field_size:
        raise ValueError("left_field and right_field must match grid.field_size")
    wave_number_squared = (system.omega / system.speed_of_light) ** 2
    gradient = np.zeros(system.grid.cell_count, dtype=np.complex128)
    for component, interpolation in enumerate(system.interpolation):
        component_slice = system.grid.component_slice(component)
        edge_product = (
            system.epsilon_scale[component]
            * left[component_slice]
            * right[component_slice]
        )
        gradient -= wave_number_squared * np.asarray(
            interpolation.T @ edge_product
        ).reshape(-1)
    return gradient.reshape(system.grid.shape)


def scattering_matrix_with_jacobian_3d(
    omega: complex,
    epsilon_r: ArrayLike,
    setup: ScatteringSetup3D,
) -> ScatteringJacobianSolution3D:
    r"""Solve ``S`` and ``dS/d epsilon_cell`` using four transpose adjoints.

    The Jacobian shape is ``(4, 4, nz, ny, nx)`` in the documented channel
    order.  It is the complex-analytic derivative with respect to real-valued
    cell permittivity perturbations.
    """

    system = assemble_scattering_system_3d(omega, epsilon_r, setup)
    factorization = splu(system.maxwell.matrix.tocsc())
    fields = np.asarray(factorization.solve(system.sources))
    matrix = system.output_projector @ fields + system.output_offset
    adjoint_fields = np.asarray(
        factorization.solve(system.output_projector.T, trans="T")
    )

    jacobian = np.empty((4, 4, *setup.grid.shape), dtype=np.complex128)
    for outgoing in range(4):
        for incoming in range(4):
            jacobian[outgoing, incoming] = (
                -maxwell_bilinear_epsilon_gradient_3d(
                    system.maxwell,
                    adjoint_fields[:, outgoing],
                    fields[:, incoming],
                )
            )
    return ScatteringJacobianSolution3D(
        matrix=np.asarray(matrix),
        jacobian=jacobian,
        fields=fields,
        adjoint_fields=adjoint_fields,
        system=system,
    )
