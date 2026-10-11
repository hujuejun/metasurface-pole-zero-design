"""Shared numerical definitions required by the published studies."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from pzid.constraints import channel_zero_density_constraint, pole_density_constraint
from pzid.hcg import SymmetricPairedTwoTierHCGGeometryParameterization
from pzid.pcm import SymmetricPCMStateHCGParameterization, lorentz_lorenz_permittivity
from pzid.poles import PoleTarget
from pzid.twoport import solve_two_port_scattering

def _reference_fraction(amorphous: complex, crystalline: complex, target_index: float) -> float:

    def polarizability(epsilon: complex) -> complex:
        return (epsilon - 1.0) / (epsilon + 2.0)
    fraction = (polarizability(target_index ** 2) - polarizability(amorphous ** 2)) / (polarizability(crystalline ** 2) - polarizability(amorphous ** 2))
    if abs(fraction.imag) > 1e-10 or not 0.0 <= fraction.real <= 1.0:
        raise ValueError('the reference index is not inside the lossless PCM interval')
    return float(fraction.real)

def _constraint_system(raw: np.ndarray, reference: SymmetricPairedTwoTierHCGGeometryParameterization, setup: Any, target: PoleTarget, pole_weight: float, zero_fractions: list[float], amorphous: complex, crystalline: complex, omega: float) -> tuple[np.ndarray, np.ndarray]:
    pole = pole_density_constraint(raw, SymmetricPCMStateHCGParameterization(reference, reference.grating_index ** 2), target, setup)
    residual = [pole_weight * pole.residual]
    jacobian = [pole_weight * pole.jacobian]
    for fraction in zero_fractions:
        epsilon_material = complex(lorentz_lorenz_permittivity(amorphous, crystalline, fraction))
        state = SymmetricPCMStateHCGParameterization(reference, epsilon_material)
        reflection = channel_zero_density_constraint(raw, state, omega, 'reflection', setup)
        residual.append(reflection.residual)
        jacobian.append(reflection.jacobian)
    return (np.concatenate(residual), np.vstack(jacobian))

def _finite_difference_audit(raw: np.ndarray, evaluate: Any, step: float=1e-06) -> float:
    _, analytic = evaluate(raw)
    columns = []
    for index in range(raw.size):
        plus = raw.copy()
        minus = raw.copy()
        plus[index] += step
        minus[index] -= step
        columns.append((evaluate(plus)[0] - evaluate(minus)[0]) / (2.0 * step))
    finite = np.column_stack(columns)
    return float(np.linalg.norm(analytic - finite) / max(np.linalg.norm(finite), 1e-12))

def _held_out(raw: np.ndarray, reference: SymmetricPairedTwoTierHCGGeometryParameterization, setup: Any, amorphous: complex, crystalline: complex, omega: float, state_points: int) -> dict[str, Any]:
    alpha = np.linspace(0.0, 1.0, state_points)
    epsilon_states = lorentz_lorenz_permittivity(amorphous, crystalline, alpha)
    transmission = np.empty(state_points, dtype=np.complex128)
    reflection = np.empty_like(transmission)
    for index, epsilon_material in enumerate(epsilon_states):
        state = SymmetricPCMStateHCGParameterization(reference, complex(epsilon_material))
        matrix = solve_two_port_scattering(omega, state.map(raw).epsilon_r, setup).matrix
        transmission[index] = matrix[1, 0]
        reflection[index] = matrix[0, 0]
    T = np.abs(transmission) ** 2
    R = np.abs(reflection) ** 2
    phase = np.unwrap(np.angle(transmission))
    phase -= phase[0]
    candidate_mask = alpha <= 0.9 + 1e-12
    worst_index = int(np.argmax(R))
    return {'alpha': alpha, 't': transmission, 'r': reflection, 'T': T, 'R': R, 'phase_change_pi': phase / math.pi, 'minimum_transmittance': float(T.min()), 'maximum_reflectance': float(R.max()), 'phase_span_pi': float(abs(phase[-1]) / math.pi), 'worst_reflection_fraction': float(alpha[worst_index]), 'below_90pct_fraction_minimum_transmittance': float(np.min(T[candidate_mask])), 'below_90pct_fraction_phase_span_pi': float(abs(phase[candidate_mask][-1]) / math.pi), 'maximum_energy_error': float(np.max(np.abs(T + R - 1.0)))}

def _write_held_out(output: Path, name: str, held: dict[str, Any]) -> None:
    np.savez_compressed(output / f'{name}_held_out.npz', alpha=held['alpha'], transmission=held['t'], reflection=held['r'], transmittance=held['T'], reflectance=held['R'], phase_change_pi=held['phase_change_pi'])
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True, constrained_layout=True)
    axes[0].plot(held['alpha'], held['T'], label='T')
    axes[0].plot(held['alpha'], held['R'], label='R')
    axes[1].plot(held['alpha'], held['phase_change_pi'], color='tab:blue')
    axes[0].set(ylabel='Power', ylim=(-0.03, 1.03))
    axes[1].set(xlabel='Crystalline fraction', ylabel='Phase change / π')
    for axis in axes:
        axis.axvline(0.9, color='black', linestyle='--', linewidth=0.8)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.savefig(output / f'{name}_response.png', dpi=220)
    plt.close(figure)
