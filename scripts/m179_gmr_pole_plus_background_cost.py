# Required numerical helpers for the final GMR results; exploratory entry point removed.
"""Supplemental GMR pole-plus-one-background matched-start adjoint arm.

This protocol is separately documented as M179 because it was conceived
after the M177 implementation smoke test. It reuses M177's frozen starts and
held-out gates but does not change or overwrite any M177 records.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares

from scripts.m164_gmr_pole_adjoint import (
    build_setup, channel_transmission, physical_epsilon_and_derivatives, pole_system,
)
from scripts.m169_gmr_matched_direct_control import direct_system
from scripts.m177_gmr_preregistered_cost_benchmark import (
    as_complex, sha256, verify_fdfd, verify_held_out,
)


def prepare(
    case: dict, reference: dict, base: dict, supplement: dict,
) -> tuple[dict, dict, object]:
    config = dict(base)
    config["period_um"] = float(case["reference_preparation"]["reference_geometry_period_um"])
    setup = build_setup(config)
    target_pole = as_complex(case["reference_preparation"]["reference_fdfd_pole"])
    frequency = (
        target_pole.real
        + float(supplement["background_sample_offset_in_fdfd_reference_pole_depths"]) * abs(target_pole.imag)
    )
    reference_coordinates = np.asarray(
        case["reference_preparation"]["reference_geometry_slab_ridge_um"], dtype=float
    )
    epsilon, _ = physical_epsilon_and_derivatives(reference_coordinates, config, setup)
    began = time.perf_counter()
    target_transmission = channel_transmission(frequency, epsilon, setup)
    nominal_start = np.asarray(case["shared_screen_start_coordinates_um"], dtype=float)
    pole_residual, pole_jacobian = pole_system(nominal_start, target_pole, config, setup)
    background_residual, background_jacobian = direct_system(
        nominal_start, np.asarray((frequency,)), np.asarray((target_transmission,)), config, setup
    )
    typical = np.asarray(supplement["typical_physical_coordinate_perturbation_um"], dtype=float)
    scale_pole = max(
        np.linalg.norm(pole_jacobian * typical[np.newaxis, :]),
        float(supplement["minimum_normalization_scale"]),
    )
    scale_background = max(
        np.linalg.norm(background_jacobian * typical[np.newaxis, :]),
        float(supplement["minimum_normalization_scale"]),
    )
    return {
        "reference_fdfd_pole": case["reference_preparation"]["reference_fdfd_pole"],
        "background_frequency_rad_per_um": float(frequency),
        "background_target_t": {"real": float(target_transmission.real), "imag": float(target_transmission.imag)},
        "normalization_scale_pole": float(scale_pole),
        "normalization_scale_background": float(scale_background),
        "nominal_pole_residual": pole_residual.tolist(),
        "nominal_background_residual": background_residual.tolist(),
        "preparation_fdfd_frequency_factorizations": 3,
        "preparation_wall_seconds": time.perf_counter() - began,
    }, config, setup


def run_trial(
    start: np.ndarray, preparation: dict, config: dict, setup,
    reference: dict, moderate: dict, fourier_config: dict,
    m177_protocol: dict, supplement: dict,
) -> dict:
    target_pole = as_complex(preparation["reference_fdfd_pole"])
    frequency = float(preparation["background_frequency_rad_per_um"])
    target_t = as_complex(preparation["background_target_t"])
    scale_pole = float(preparation["normalization_scale_pole"])
    scale_background = float(preparation["normalization_scale_background"])
    bounds = (
        np.asarray((config["slab_thickness_bounds_um"][0], config["ridge_height_bounds_um"][0])),
        np.asarray((config["slab_thickness_bounds_um"][1], config["ridge_height_bounds_um"][1])),
    )
    cache_x = None
    cache_result = None
    calls = 0

    def system(values: np.ndarray):
        nonlocal cache_x, cache_result, calls
        if cache_x is None or not np.array_equal(values, cache_x):
            pole_residual, pole_jacobian = pole_system(values, target_pole, config, setup)
            background_residual, background_jacobian = direct_system(
                values, np.asarray((frequency,)), np.asarray((target_t,)), config, setup
            )
            cache_result = (
                np.r_[pole_residual / scale_pole, background_residual / scale_background],
                np.vstack((pole_jacobian / scale_pole, background_jacobian / scale_background)),
            )
            cache_x = np.asarray(values).copy()
            calls += 1
        return cache_result

    started = time.perf_counter()
    optimized = least_squares(
        lambda x: system(x)[0], start, jac=lambda x: system(x)[1],
        bounds=bounds,
        max_nfev=int(supplement["maximum_objective_frequency_factorizations"]) // 2,
        xtol=1e-11, ftol=1e-11, gtol=1e-11,
    )
    optimized_seconds = time.perf_counter() - started
    fdfd = verify_fdfd(optimized.x, target_pole, config, setup, m177_protocol)
    held_out = (
        verify_held_out(optimized.x, reference, moderate, config, fourier_config, m177_protocol)
        if fdfd["pass"] else {"pass": False, "skipped_because_fdfd_gate_failed": True, "response_evaluations": 0}
    )
    return {
        "start_coordinates_um": start.tolist(),
        "final_coordinates_um": optimized.x.tolist(),
        "objective_residual": float(np.linalg.norm(optimized.fun)),
        "optimizer_success": bool(optimized.success),
        "optimizer_message": str(optimized.message),
        "objective_adjoint_evaluations": calls,
        "objective_fdfd_frequency_factorizations": 2 * calls,
        "objective_wall_seconds": optimized_seconds,
        "fdfd_root_verification_frequency_factorizations": int(fdfd.get("root_evaluations", 0)),
        "fourier_response_evaluations": int(held_out.get("response_evaluations", 0)),
        "fdfd_validation": fdfd,
        "held_out_validation": held_out,
        "qualified": bool(fdfd["pass"] and held_out["pass"]),
        "method_specific_wall_seconds": time.perf_counter() - started,
    }
