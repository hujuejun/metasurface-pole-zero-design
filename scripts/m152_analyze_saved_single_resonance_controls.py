from __future__ import annotations

from pathlib import Path

import numpy as np

def load_curve(path: Path, real_name: str, imag_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    omega = np.asarray(data["omega_rad_per_um"], dtype=float)
    transmission = np.asarray(data[real_name] + 1j * data[imag_name], dtype=complex)
    detuning = np.asarray(data[data.dtype.names[1]], dtype=float)
    if len(omega) < 9 or not np.all(np.diff(omega) > 0) or not np.all(np.isfinite(transmission)):
        raise ValueError(f"invalid saved spectrum: {path}")
    return omega, detuning, transmission

def sample_indices(detuning: np.ndarray, offsets: list[float]) -> np.ndarray:
    indices = np.asarray([int(np.argmin(abs(detuning - value))) for value in offsets], dtype=int)
    if len(set(indices.tolist())) != len(offsets) or not np.allclose(detuning[indices], offsets, atol=1e-7):
        raise ValueError("requested training detunings must be distinct saved data points")
    return indices

def pole_model(detuning: np.ndarray, transmission: np.ndarray, training: np.ndarray) -> np.ndarray:
    x = np.asarray(detuning, dtype=float)
    basis = np.column_stack((np.ones_like(x), x, 1.0 / (x + 1j)))
    coefficients = np.linalg.solve(basis[training], transmission[training])
    return basis @ coefficients

def direct_pade_2_1(detuning: np.ndarray, transmission: np.ndarray, training: np.ndarray) -> np.ndarray:
    """Fit four unconstrained complex coefficients from four complex samples."""

    x = np.asarray(detuning, dtype=float)
    t = np.asarray(transmission, dtype=complex)
    if len(training) != 4:
        raise ValueError("[2/1] fit requires exactly four samples")
    matrix = np.column_stack((np.ones(4), x[training], x[training] ** 2, -t[training] * x[training]))
    a0, a1, a2, b1 = np.linalg.solve(matrix, t[training])
    denominator = 1.0 + b1 * x
    if np.min(abs(denominator)) < 1e-10:
        raise ValueError("direct rational model has a pole on the held-out grid")
    return (a0 + a1 * x + a2 * x**2) / denominator

def errors(actual: np.ndarray, predicted: np.ndarray, heldout: np.ndarray, floor: float) -> dict[str, float | int]:
    if not np.any(heldout):
        raise ValueError("no held-out points")
    phase_mask = heldout & (abs(actual) ** 2 >= floor)
    return {
        "complex_t_rmse": float(np.sqrt(np.mean(abs(predicted[heldout] - actual[heldout]) ** 2))),
        "transmittance_rmse": float(np.sqrt(np.mean((abs(predicted[heldout]) ** 2 - abs(actual[heldout]) ** 2) ** 2))),
        "phase_mae_rad_above_floor": float(np.mean(abs(np.angle(predicted[phase_mask] / actual[phase_mask])))) if np.any(phase_mask) else float("nan"),
        "heldout_count": int(np.count_nonzero(heldout)),
        "phase_count": int(np.count_nonzero(phase_mask)),
    }
