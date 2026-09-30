# Required numerical helpers for the final GMR results; exploratory entry point removed.
"""Matched-start complex-pole versus three-real-frequency transmission control.

Both optimizers use the same two physical grating variables, Maxwell adjoints,
starts, bounds, least-squares algorithm and a 60-frequency-factorization
budget. They do *not* solve mathematically identical problems: the direct
objective matches three complex t samples, while the pole objective places
one complex transmission-channel pole via a time-reversal determinant.
Independent final pole roots quantify whether each method actually preserved
the intended resonance. No held-out spectra are used in either objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import least_squares

from pzid.adjoint import scattering_matrix_with_jacobian
from pzid.roots import refine_scalar_root
from pzid.twoport import solve_two_port_scattering

from scripts.m164_gmr_pole_adjoint import (
    build_setup, channel_transmission, physical_epsilon_and_derivatives, pole_system,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complex(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def direct_system(values: np.ndarray, frequencies: np.ndarray, targets: np.ndarray,
                  config: dict, setup) -> tuple[np.ndarray, np.ndarray]:
    epsilon, derivative = physical_epsilon_and_derivatives(values, config, setup)
    residual = np.empty(2 * len(frequencies), dtype=float)
    jacobian = np.empty((2 * len(frequencies), len(values)), dtype=float)
    for index, (frequency, target) in enumerate(zip(frequencies, targets, strict=True)):
        result = scattering_matrix_with_jacobian(float(frequency), epsilon, setup)
        error = result.matrix[1, 0] - target
        gradient = np.einsum("zx,pzx->p", result.jacobian[1, 0], derivative)
        residual[2 * index : 2 * index + 2] = (error.real, error.imag)
        jacobian[2 * index] = gradient.real
        jacobian[2 * index + 1] = gradient.imag
    return residual, jacobian


def _verify_pole(values: np.ndarray, target: complex, config: dict, setup) -> dict:
    epsilon, _ = physical_epsilon_and_derivatives(values, config, setup)
    response = lambda w: channel_transmission(w, epsilon, setup)
    result = refine_scalar_root(response, target, kind="pole", tolerance=1e-9,
                                max_evaluations=100, search_radius=max(0.1, 15 * abs(target.imag)))
    return {
        "pole": _complex(result.frequency),
        "root_residual": result.residual_norm,
        "error_in_target_depths": float(abs(result.frequency - target) / abs(target.imag)),
        "root_success": bool(result.residual_norm < 1e-6 and result.frequency.imag < 0),
    }
