"""Shared numerical definitions required by the published studies."""
from __future__ import annotations
import numpy as np
from pzid.adjoint import scattering_matrix_with_jacobian
from pzid.fabrication import periodic_bilinear_resample, periodic_bilinear_resample_adjoint
from pzid.feature_projection2d import project_dual_phase_minimum_feature
from pzid.measured_dispersion import PassiveLorentzModel
from pzid.twoport import solve_two_port_scattering
from scripts.m210a_100nm_seed_projection import geometry_audit

def contour_fraction(mask: np.ndarray, coarse_shape: tuple[int, int]) -> np.ndarray:
    nz, nx = coarse_shape
    if mask.shape[0] % nz or mask.shape[1] % nx:
        raise ValueError('binary contour does not integer-refine the FDFD grid')
    fz, fx = (mask.shape[0] // nz, mask.shape[1] // nx)
    return mask.reshape(nz, fz, nx, fx).mean(axis=(1, 3))

def physical_contour(density: np.ndarray, setup: object, config: dict, fine_cell_um: float) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    factor = int(config['contour_refinement'])
    raw = periodic_bilinear_resample(density, (density.shape[0] * factor, density.shape[1] * factor)) >= 0.5
    contour, projection = project_dual_phase_minimum_feature(raw, cell_um=fine_cell_um, diameter_um=float(config['minimum_solid_and_air_feature_um']), design_rows=(24 * factor, 168 * factor))
    audit = geometry_audit(contour, fine_cell_um=fine_cell_um, config=config)
    return (contour, contour_fraction(contour, setup.grid.shape), projection, audit)

def material_response(model: PassiveLorentzModel, omega_native: complex, scale: float, *, time_reversed: bool) -> complex:
    omega_physical = omega_native / scale
    return model.reversed_epsilon(omega_physical) if time_reversed else complex(model.epsilon(omega_physical))

def objective(fraction: np.ndarray, setup: object, model: PassiveLorentzModel, scale: float, target_pole: complex, offsets: np.ndarray, reflection_weight: float, absorptance_weight: float, *, with_jacobian: bool) -> tuple[np.ndarray, np.ndarray | None, dict]:
    evaluate = scattering_matrix_with_jacobian if with_jacobian else solve_two_port_scattering
    values: list[complex] = []
    derivatives: list[np.ndarray] = []
    background: list[dict] = []

    def solve(omega: complex, reversed_medium: bool):
        material_epsilon = material_response(model, omega, scale, time_reversed=reversed_medium)
        epsilon = 1.0 + fraction * (material_epsilon - 1.0)
        result = evaluate(omega, epsilon, setup)
        jac = result.jacobian * (material_epsilon - 1.0) if with_jacobian else None
        return (result.matrix, jac)

    def append(value: complex, derivative: np.ndarray | None, weight: float=1.0) -> None:
        values.append(weight * value)
        if with_jacobian:
            assert derivative is not None
            derivatives.append(weight * derivative)
    upper, upper_jac = solve(target_pole.conjugate(), True)
    for vector in (np.array([1.0, -1.0]), np.array([1.0, 1.0])):
        output = upper @ vector
        jac_output = np.einsum('abij,b->aij', upper_jac, vector) if with_jacobian else None
        for port in range(2):
            append(output[port], jac_output[port] if jac_output is not None else None)
    pole_end = len(values)
    sample_frequencies = target_pole.real + np.r_[0.0, offsets] * -target_pole.imag
    for sample_index, omega in enumerate(sample_frequencies):
        matrix, jac = solve(float(omega), False)
        t, r = (matrix[1, 0], matrix[0, 0])
        T, R = (float(abs(t) ** 2), float(abs(r) ** 2))
        A = 1.0 - T - R
        background.append({'omega_native': float(omega), 'wavelength_um': float(2 * np.pi * scale / omega), 'T': T, 'R': R, 'A': A})
        r_jac = jac[0, 0] if with_jacobian else None
        t_jac = jac[1, 0] if with_jacobian else None
        append(r, r_jac, 1.0 if sample_index == 0 else reflection_weight)
        absorption_jac = -2.0 * np.real(np.conj(t) * t_jac + np.conj(r) * r_jac) if with_jacobian else None
        append(A, absorption_jac, absorptance_weight)
    residual = np.asarray(values, dtype=complex)
    jacobian = np.asarray(derivatives, dtype=complex) if with_jacobian else None
    metrics = {'cost': float(np.vdot(residual, residual).real), 'time_reversed_pole_residual': float(np.linalg.norm(residual[:pole_end])), 'center': background[0], 'real_frequency_controls': background, 'maximum_background_reflectance': float(max((point['R'] for point in background))), 'maximum_control_absorptance': float(max((point['A'] for point in background)))}
    return (residual, jacobian, metrics)

def soft_pullback(residual: np.ndarray, jacobian: np.ndarray, density: np.ndarray, subpixels: int, slope: float) -> np.ndarray:
    fine = periodic_bilinear_resample(density, (density.shape[0] * subpixels, density.shape[1] * subpixels))
    logistic = 1.0 / (1.0 + np.exp(-np.clip(slope * (fine - 0.5), -35, 35)))
    local = slope * logistic * (1 - logistic) / subpixels ** 2
    fraction_gradient = 2 * np.real(np.einsum('k,kij->ij', residual.conj(), jacobian))
    fine_gradient = np.repeat(np.repeat(fraction_gradient, subpixels, axis=0), subpixels, axis=1)
    return periodic_bilinear_resample_adjoint(local * fine_gradient, density.shape)
