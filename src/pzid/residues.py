"""Fixed-pole rational fits for scalar scattering-channel audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class FixedPoleFit:
    """Least-squares scalar response with prescribed simple-pole locations."""

    poles: ComplexArray
    residues: ComplexArray
    polynomial_coefficients: ComplexArray
    center: float
    scale: float
    fitted: ComplexArray
    normalized_rms_error: float


def fit_fixed_poles(
    frequency: Sequence[float] | NDArray[np.float64],
    response: Sequence[complex] | ComplexArray,
    poles: Sequence[complex] | ComplexArray,
    *,
    polynomial_degree: int = 2,
) -> FixedPoleFit:
    """Fit a smooth polynomial background plus prescribed simple poles.

    Scaling the real-frequency interval before solving keeps the polynomial
    and pole columns comparable.  Returned residues use the physical
    frequency coordinate, so ``response ~= residue / (omega - pole)``.
    """

    omega = np.asarray(frequency, dtype=float)
    values = np.asarray(response, dtype=np.complex128)
    pole_values = np.asarray(poles, dtype=np.complex128)
    if omega.ndim != 1 or values.shape != omega.shape:
        raise ValueError("frequency and response must be equal-length vectors")
    if omega.size < polynomial_degree + pole_values.size + 1:
        raise ValueError("too few samples for the requested rational fit")
    if polynomial_degree < 0 or pole_values.ndim != 1 or pole_values.size == 0:
        raise ValueError("the fit requires a nonnegative degree and at least one pole")
    if np.any(~np.isfinite(omega)) or np.any(~np.isfinite(values)):
        raise ValueError("fit samples must be finite")
    center = float(0.5 * (np.min(omega) + np.max(omega)))
    scale = float(0.5 * (np.max(omega) - np.min(omega)))
    if scale <= 0.0:
        raise ValueError("frequency samples must span a nonzero interval")
    x = (omega - center) / scale
    scaled_poles = (pole_values - center) / scale
    columns = [x**degree for degree in range(polynomial_degree + 1)]
    columns.extend(1.0 / (x - pole) for pole in scaled_poles)
    matrix = np.column_stack(columns)
    coefficients, _, _, _ = np.linalg.lstsq(matrix, values, rcond=None)
    fitted = matrix @ coefficients
    denominator = max(float(np.sqrt(np.mean(np.abs(values) ** 2))), 1e-15)
    error = float(np.sqrt(np.mean(np.abs(fitted - values) ** 2)) / denominator)
    background_count = polynomial_degree + 1
    # c/(x-xp) = c*scale/(omega-p).
    residues = coefficients[background_count:] * scale
    return FixedPoleFit(
        poles=pole_values,
        residues=residues,
        polynomial_coefficients=coefficients[:background_count],
        center=center,
        scale=scale,
        fitted=fitted,
        normalized_rms_error=error,
    )

