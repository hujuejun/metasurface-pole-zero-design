"""Measured optical constants and a passive, analytic Lorentz continuation.

Frequencies are angular wavenumbers in rad/um, with fields proportional to
exp(-i omega t).  Positive oscillator strength and damping give nonnegative
absorption on the positive real-frequency axis.  The fit is an approximation
to supplied data, not a claim that a finite-band measurement uniquely fixes
the complex-frequency material response.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class OpticalConstants:
    wavelength_um: np.ndarray
    n: np.ndarray
    k: np.ndarray


def load_nk_csv(path: str | Path) -> OpticalConstants:
    """Read numeric wavelength, n, and k columns; never evaluate file text."""
    wavelength: list[float] = []
    n: list[float] = []
    k: list[float] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"lambda_um", "n", "k"}.issubset(reader.fieldnames):
            raise ValueError("expected columns lambda_um,n,k")
        for row in reader:
            wavelength.append(float(row["lambda_um"]))
            n.append(float(row["n"]))
            k.append(float(row["k"]))
    lam = np.asarray(wavelength)
    real = np.asarray(n)
    imag = np.asarray(k)
    if (len(lam) < 10 or not np.all(np.isfinite(lam))
            or not np.all(np.diff(lam) > 0) or not np.all(np.isfinite(real))
            or not np.all(np.isfinite(imag)) or np.any(real <= 0) or np.any(imag < 0)):
        raise ValueError("wavelengths must increase and passive n,k must be finite")
    return OpticalConstants(lam, real, imag)


@dataclass(frozen=True)
class PassiveLorentzModel:
    epsilon_infinity: float
    resonance_omega: tuple[float, ...]
    damping_omega: tuple[float, ...]
    oscillator_strength: tuple[float, ...]

    def epsilon(self, omega: complex | np.ndarray) -> complex | np.ndarray:
        frequency = np.asarray(omega, dtype=complex)
        result = np.full(frequency.shape, self.epsilon_infinity, dtype=complex)
        for resonance, damping, strength in zip(
            self.resonance_omega, self.damping_omega, self.oscillator_strength, strict=True
        ):
            result += strength / (resonance**2 - frequency**2 - 1j * damping * frequency)
        return complex(result) if result.ndim == 0 else result

    def index(self, omega: complex | np.ndarray) -> complex | np.ndarray:
        result = np.sqrt(self.epsilon(omega))
        return complex(result) if np.ndim(result) == 0 else result

    def reversed_epsilon(self, upper_omega: complex) -> complex:
        """Gain-medium permittivity at omega* for a passive pole at omega.

        The lossy time-reversal zero must *not* use physical epsilon at the
        upper-half-plane frequency.  This quantity is conjugate to epsilon
        at the lower-half-plane partner.
        """
        if complex(upper_omega).imag <= 0:
            raise ValueError("time-reversed evaluation requires upper-half-plane omega")
        return complex(np.conj(self.epsilon(np.conj(upper_omega))))

    def as_dict(self) -> dict:
        return {
            "epsilon_infinity": self.epsilon_infinity,
            "resonance_omega_rad_per_um": list(self.resonance_omega),
            "damping_omega_rad_per_um": list(self.damping_omega),
            "oscillator_strength_rad2_per_um2": list(self.oscillator_strength),
            "time_convention": "exp(-i omega t)",
        }

    @classmethod
    def from_dict(cls, data: dict) -> PassiveLorentzModel:
        return cls(
            float(data["epsilon_infinity"]),
            tuple(map(float, data["resonance_omega_rad_per_um"])),
            tuple(map(float, data["damping_omega_rad_per_um"])),
            tuple(map(float, data["oscillator_strength_rad2_per_um2"])),
        )


def fit_ipdip_lorentz(data: OpticalConstants) -> tuple[PassiveLorentzModel, dict]:
    """Fit a small, positive-strength Lorentz basis around the 5-um design.

    The 4.4--5.65 um fit approaches the strong 5.75-um band and uses extra weight
    on 4.65--5.44 um, the intended optical assessment band.  Multiple fixed
    initial guesses limit local-fit dependence.  Bounds encode distinct
    short-, central-, and longer-wavelength material resonances.
    """
    fit = (data.wavelength_um >= 4.4) & (data.wavelength_um <= 5.65)
    lam = data.wavelength_um[fit]
    measured = data.n[fit] + 1j * data.k[fit]
    omega = 2 * np.pi / lam
    local = (lam >= 4.65) & (lam <= 5.44)
    weights = np.where(local, 8.0, 1.0)

    # Parameters: epsilon_inf, three strengths, three resonance wavelengths,
    # then three damping rates.  All strengths and damping rates are positive.
    lower = np.array([1.0, 0, 0, 0, 3.0, 5.60, 6.0, 0.001, 0.001, 0.001])
    upper = np.array([4.0, 5, 1, 5, 4.4, 5.90, 12.0, 2.0, 0.3, 2.0])

    def unpack(parameters: np.ndarray) -> PassiveLorentzModel:
        return PassiveLorentzModel(
            epsilon_infinity=float(parameters[0]),
            resonance_omega=tuple(2 * np.pi / parameters[4:7]),
            damping_omega=tuple(parameters[7:10]),
            oscillator_strength=tuple(parameters[1:4]),
        )

    def residual(parameters: np.ndarray) -> np.ndarray:
        predicted = unpack(parameters).index(omega)
        difference = predicted - measured
        return np.r_[weights * difference.real, 2 * weights * difference.imag]

    starts = [
        [2.28, 0.01, 0.015, 0.01, 3.5, 5.75, 7.0, 0.2, 0.025, 0.2],
        [2.28, 0.02, 0.020, 0.01, 4.0, 5.76, 8.0, 0.4, 0.04, 0.4],
        [2.26, 0.005, 0.018, 0.02, 3.2, 5.74, 6.5, 0.1, 0.02, 0.1],
    ]
    trials = [least_squares(residual, start, bounds=(lower, upper),
                            max_nfev=1200, ftol=1e-11, xtol=1e-11, gtol=1e-11)
              for start in starts]
    best = min(trials, key=lambda trial: np.linalg.norm(trial.fun))
    model = unpack(best.x)
    predicted = np.asarray(model.index(omega))
    difference = predicted - measured
    diagnostics = {
        "fit_wavelength_um": [float(lam[0]), float(lam[-1])],
        "assessment_wavelength_um": [4.65, 5.44],
        "fit_count": int(len(lam)),
        "assessment_count": int(np.count_nonzero(local)),
        "max_abs_n_error_assessment": float(np.max(np.abs(difference.real[local]))),
        "max_abs_k_error_assessment": float(np.max(np.abs(difference.imag[local]))),
        "rms_n_error_assessment": float(np.sqrt(np.mean(difference.real[local]**2))),
        "rms_k_error_assessment": float(np.sqrt(np.mean(difference.imag[local]**2))),
        "max_abs_n_error_full_fit": float(np.max(np.abs(difference.real))),
        "max_abs_k_error_full_fit": float(np.max(np.abs(difference.imag))),
        "fit_cost": float(best.cost),
        "fit_success": bool(best.success),
        "trial_costs": [float(trial.cost) for trial in trials],
    }
    return model, diagnostics
