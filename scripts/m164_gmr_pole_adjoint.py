# Required numerical helpers for the final GMR results; exploratory entry point removed.
"""Adjoint inverse design of a Q ladder in one physical binary GMR geometry.

The complex determinant of the time-reversed two-port scattering matrix is
used as Chen-style pole constraint. The determinant is an optimization
surrogate only: acceptance independently solves 1/t=0 for the *transmission
channel* at each optimized geometry. Its gradients use the Maxwell transpose
adjoint and the exact area-overlap physical-interface shape chain rule.
"""

from __future__ import annotations

import argparse
import csv
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
from pzid.gmr2d import SingleToothGMR
from pzid.grid import UniformGrid2D
from pzid.roots import refine_scalar_root
from pzid.twoport import TwoPortScatteringSetup, solve_two_port_scattering


FS_PER_UM_OVER_C = 3.3356409519815205


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complex(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def build_setup(config: dict) -> TwoPortScatteringSetup:
    grid = UniformGrid2D(
        nx=int(config["lateral_cells"]), nz=int(config["axial_cells"]),
        dx=float(config["period_um"]) / int(config["lateral_cells"]),
        dz=float(config["axial_cell_size_um"]),
    )
    return TwoPortScatteringSetup(
        grid=grid,
        lower_source_plane=int(config["lower_source_plane"]),
        upper_source_plane=int(config["upper_source_plane"]),
        lower_monitor_planes=tuple(config["lower_monitor_planes"]),
        upper_monitor_planes=tuple(config["upper_monitor_planes"]),
        port_index=float(config["port_index"]),
    )


def physical_epsilon_and_derivatives(
    coordinates: np.ndarray, config: dict, setup: TwoPortScatteringSetup
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(coordinates, dtype=float)
    if values.shape != (2,):
        raise ValueError("coordinates must be [slab thickness, ridge height]")
    geometry = SingleToothGMR(
        period_um=float(config["period_um"]),
        slab_thickness_um=float(values[0]),
        ridge_height_um=float(values[1]),
        ridge_fill_factor=float(config["ridge_fill_factor"]),
        high_index=float(config["high_index"]),
        port_index=float(config["port_index"]),
    )
    return geometry.rasterize_with_derivatives(
        setup.grid, base_z_um=float(config["base_z_um"])
    )


def pole_system(
    coordinates: np.ndarray, target: complex, config: dict,
    setup: TwoPortScatteringSetup,
) -> tuple[np.ndarray, np.ndarray]:
    epsilon, physical_derivative = physical_epsilon_and_derivatives(coordinates, config, setup)
    evaluated = scattering_matrix_with_jacobian(np.conj(target), epsilon, setup)
    s = evaluated.matrix
    ds = np.einsum("abzx,pzx->abp", evaluated.jacobian, physical_derivative)
    determinant = s[0, 0] * s[1, 1] - s[0, 1] * s[1, 0]
    derivative = (
        ds[0, 0] * s[1, 1] + s[0, 0] * ds[1, 1]
        - ds[0, 1] * s[1, 0] - s[0, 1] * ds[1, 0]
    )
    residual = np.asarray([determinant.real, determinant.imag], dtype=float)
    jacobian = np.asarray([derivative.real, derivative.imag], dtype=float)
    return residual, jacobian


def check_adjoint_gradient(
    coordinates: np.ndarray, target: complex, config: dict,
    setup: TwoPortScatteringSetup,
) -> dict:
    step = float(config["gradient_test_step_um"])
    residual, jacobian = pole_system(coordinates, target, config, setup)
    errors = []
    for index in range(2):
        offset = np.eye(2)[index] * step
        plus = pole_system(coordinates + offset, target, config, setup)[0]
        minus = pole_system(coordinates - offset, target, config, setup)[0]
        finite_difference = (plus - minus) / (2 * step)
        error = np.linalg.norm(finite_difference - jacobian[:, index]) / max(
            np.linalg.norm(finite_difference), 1e-10
        )
        errors.append(float(error))
    return {"initial_residual": residual.tolist(), "relative_errors": errors}


def channel_transmission(
    frequency: complex, epsilon: np.ndarray, setup: TwoPortScatteringSetup
) -> complex:
    return complex(solve_two_port_scattering(frequency, epsilon, setup).matrix[1, 0])


def verify_channel_pole(
    coordinates: np.ndarray, target: complex, config: dict,
    setup: TwoPortScatteringSetup,
) -> tuple[dict, dict[str, np.ndarray], np.ndarray]:
    epsilon, _ = physical_epsilon_and_derivatives(coordinates, config, setup)
    response = lambda frequency: channel_transmission(frequency, epsilon, setup)
    root = refine_scalar_root(
        response, target, kind="pole", tolerance=1e-9,
        max_evaluations=60, search_radius=max(0.1, 15 * abs(target.imag)),
    )
    if root.residual_norm > 1e-6 or root.frequency.imag >= 0:
        raise RuntimeError(f"independent transmission-channel pole verification failed: {root}")
    # The transmission-channel zero is separate evidence, never inferred from
    # the S-determinant pole condition.
    zero = refine_scalar_root(
        response, complex(root.frequency.real), kind="zero", tolerance=1e-9,
        max_evaluations=60, search_radius=max(0.08, 8 * abs(target.imag)),
    )
    if zero.residual_norm > 1e-6:
        raise RuntimeError(f"independent transmission zero verification failed: {zero}")
    width = float(config["spectrum_half_widths_in_pole_depths"]) * abs(root.frequency.imag)
    omega = np.linspace(root.frequency.real - width, root.frequency.real + width,
                        int(config["spectrum_points"]))
    matrices = [solve_two_port_scattering(float(w), epsilon, setup).matrix for w in omega]
    t = np.asarray([s[1, 0] for s in matrices])
    r = np.asarray([s[0, 0] for s in matrices])
    step = 0.02 * abs(root.frequency.imag)
    delay = np.angle(response(root.frequency.real + step) / response(root.frequency.real - step))
    delay = float(delay / (2 * step) * FS_PER_UM_OVER_C)
    metrics = {
        "verified_pole": _complex(root.frequency),
        "verified_zero": _complex(zero.frequency),
        "verified_quality_factor": float(root.frequency.real / (-2 * root.frequency.imag)),
        "pole_root_residual": root.residual_norm,
        "zero_root_residual": zero.residual_norm,
        "pole_error_in_target_depths": float(abs(root.frequency - target) / abs(target.imag)),
        "transmittance_at_pole_real": float(abs(response(root.frequency.real)) ** 2),
        "minimum_local_transmittance": float(np.min(abs(t) ** 2)),
        "maximum_local_reflectance": float(np.max(abs(r) ** 2)),
        "maximum_local_energy_error": float(np.max(abs(abs(t) ** 2 + abs(r) ** 2 - 1))),
        "group_delay_fs_at_pole_real": delay,
        "pole_delay_scale_fs": float(FS_PER_UM_OVER_C / abs(root.frequency.imag)),
    }
    curves = {
        "omega_rad_per_um": omega,
        "detuning_in_pole_depths": (omega - root.frequency.real) / abs(root.frequency.imag),
        "t_real": t.real, "t_imag": t.imag,
        "transmittance": abs(t) ** 2, "reflectance": abs(r) ** 2,
        "phase_rad_unwrapped": np.unwrap(np.angle(t)),
    }
    return metrics, curves, epsilon


def _write_curve(path: Path, curves: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(curves.keys())
        writer.writerows(zip(*curves.values(), strict=True))
