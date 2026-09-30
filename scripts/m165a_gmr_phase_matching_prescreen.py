# Required numerical helpers for the final GMR results; exploratory entry point removed.
"""Fourier-modal feasibility seeds for common-frequency physical GMR Q targets.

This is *not* the claimed adjoint design step: it only chooses feasible period
and ridge-height starting points. M165B independently refines targets using a
Maxwell adjoint on the FDFD mesh with period fixed per case. All screen and
held-out roots remain transmission-channel poles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares, minimize_scalar

from pzid.rcwa2d import centered_binary_grating_layer, homogeneous_layer, solve_tm_stack
from pzid.roots import refine_scalar_root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _response(config: dict, period: float, height: float, harmonics: int):
    layers = (
        homogeneous_layer(float(config["high_index"]), float(config["slab_thickness_um"]), harmonics),
        centered_binary_grating_layer(float(config["port_index"]), float(config["high_index"]),
                                      float(config["ridge_fill_factor"]), height, harmonics),
    )
    return lambda omega: solve_tm_stack(
        omega, layers, period_um=period, port_index=float(config["port_index"]),
        harmonic_half_width=harmonics,
    ).transmission


def _complex(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}
