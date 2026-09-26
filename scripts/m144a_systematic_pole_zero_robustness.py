from __future__ import annotations

from typing import Any

import numpy as np

from pzid.adjoint import scattering_matrix_with_jacobian

from pzid.roots import refine_scalar_root_complex_secant

from pzid.scattering import solve_symmetric_scattering

from pzid.twoport import solve_two_port_scattering

from scripts.m142e_optimize_patterned_q_poles import build_fixed_grid, physical_epsilon_and_derivatives

PARAMETERS = ("mirror_scale", "cavity_scale", "ridge_fill", "ridge_height_um", "high_index")

def _encode(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}

def _epsilon(
    values: np.ndarray, geometry: dict[str, Any], material: dict[str, Any], scalar: Any
) -> tuple[np.ndarray, np.ndarray]:
    shifted_geometry = dict(geometry)
    shifted_geometry["ridge_fill_factor"] = float(values[2])
    shifted_geometry["ridge_height_um"] = float(values[3])
    epsilon, thickness_derivatives = physical_epsilon_and_derivatives(
        values[:2], shifted_geometry, material, scalar
    )
    reference_high = float(material["high_index"])
    low = float(material["low_index"])
    contrast = reference_high**2 - low**2
    high = float(values[4])
    fraction = (epsilon - low**2) / contrast
    remapped = low**2 + (high**2 - low**2) * fraction
    scaled_derivatives = thickness_derivatives * ((high**2 - low**2) / contrast)
    return remapped, scaled_derivatives

def _channel(matrix: np.ndarray, parity: int) -> complex:
    return complex(matrix[0, 0] + parity * matrix[0, 1])

def _adjoint_pole_gradient(
    values: np.ndarray,
    pole: complex,
    parity: int,
    geometry: dict[str, Any],
    material: dict[str, Any],
    scalar: Any,
    twoport: Any,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, complex, np.ndarray]:
    epsilon, thickness_derivatives = _epsilon(values, geometry, material, scalar)
    depth = abs(pole.imag)
    mapping_derivatives = np.empty((len(PARAMETERS), *epsilon.shape), dtype=float)
    mapping_derivatives[:2] = thickness_derivatives
    for index, step in ((2, float(protocol["fill_derivative_step"])), (3, float(protocol["height_derivative_step_um"]))):
        displacement = np.zeros(len(PARAMETERS))
        displacement[index] = step
        plus, _ = _epsilon(values + displacement, geometry, material, scalar)
        minus, _ = _epsilon(values - displacement, geometry, material, scalar)
        mapping_derivatives[index] = (plus - minus) / (2.0 * step)
    low = float(material["low_index"])
    high = float(values[4])
    mapping_derivatives[4] = 2.0 * high * (epsilon - low**2) / (high**2 - low**2)

    upper = np.conj(pole)
    evaluation = scattering_matrix_with_jacobian(upper, epsilon, twoport)
    residual = _channel(evaluation.matrix, parity)
    channel_jacobian = evaluation.jacobian[0, 0] + parity * evaluation.jacobian[0, 1]
    parameter_gradient = np.einsum("zx,pzx->p", channel_jacobian, mapping_derivatives)
    step_frequency = float(protocol["frequency_derivative_step_in_pole_depths"]) * depth
    plus_matrix = solve_two_port_scattering(upper + step_frequency, epsilon, twoport).matrix
    minus_matrix = solve_two_port_scattering(upper - step_frequency, epsilon, twoport).matrix
    frequency_gradient = (_channel(plus_matrix, parity) - _channel(minus_matrix, parity)) / (2.0 * step_frequency)
    return np.conj(-parameter_gradient / frequency_gradient), residual, epsilon

def _root(
    coefficient: str,
    epsilon: np.ndarray,
    scalar: Any,
    start: complex,
    depth: float,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    def response(omega: complex) -> complex:
        coefficients = solve_symmetric_scattering(omega, epsilon, scalar).coefficients
        return complex(getattr(coefficients, coefficient))

    guess = start + (0.08 + 0.05j) * depth
    root = refine_scalar_root_complex_secant(
        response,
        start,
        guess,
        kind="pole" if coefficient == "transmission" else "zero",
        tolerance=float(protocol["root_tolerance"]),
        max_evaluations=int(protocol["root_max_evaluations"]),
        search_radius=max(0.05, 10.0 * depth),
    )
    verified = bool(np.isfinite(root.frequency) and root.residual_norm < 1e-6 and (coefficient != "transmission" or root.frequency.imag < 0))
    return {
        "verified": verified,
        "frequency": _encode(root.frequency),
        "root_success_flag": bool(root.success),
        "root_residual_norm": float(root.residual_norm),
        "root_evaluations": int(root.evaluations),
    }
