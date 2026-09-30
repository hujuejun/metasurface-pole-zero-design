# Required numerical helpers for the final GMR results; exploratory entry point removed.
"""Matched-start GMR pole-adjoint versus strong adaptive direct controls.

The full frozen protocol is docs/m177-gmr-preregistered-cost-benchmark.md.
This is reference-geometry recovery, not blind analytic-pole discovery.
All expensive method-specific calls and elapsed time are kept in the ledger.
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

import numpy as np
import scipy
from scipy.optimize import least_squares, minimize_scalar

from pzid.roots import refine_scalar_root
from pzid.twoport import solve_two_port_scattering
from scripts.m164_gmr_pole_adjoint import (
    FS_PER_UM_OVER_C,
    build_setup,
    channel_transmission,
    physical_epsilon_and_derivatives,
    pole_system,
)
from scripts.m166_gmr_high_fidelity_analysis import _solver
from scripts.m169_gmr_matched_direct_control import direct_system


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_complex(record: dict) -> complex:
    return complex(float(record["real"]), float(record["imag"]))


def complex_record(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def reference_preparation(
    quality: int, reference: dict, mesh_seed: dict, protocol: dict,
    base: dict, fourier_config: dict,
) -> tuple[dict, dict, object, np.ndarray]:
    config = dict(base)
    config["period_um"] = float(reference["final_period_um"])
    setup = build_setup(config)
    coordinates = np.asarray((reference["slab_thickness_um"], reference["ridge_height_um"]), dtype=float)
    epsilon, _ = physical_epsilon_and_derivatives(coordinates, config, setup)
    seed = as_complex(mesh_seed["meshes"][0]["transmission_channel_pole"])
    started = time.perf_counter()
    pole = refine_scalar_root(
        lambda w: channel_transmission(w, epsilon, setup), seed,
        kind="pole", tolerance=1e-10, max_evaluations=60,
        search_radius=max(0.1, 15 * abs(seed.imag)),
    )
    if pole.residual_norm > float(protocol["root_residual_tolerance"]) or pole.frequency.imag >= 0:
        raise RuntimeError(f"Q={quality} reference FDFD pole failed: {pole}")
    prep = {
        "reference_fdfd_pole": complex_record(pole.frequency),
        "reference_pole_root_evaluations": int(pole.evaluations),
        "reference_pole_preparation_wall_seconds": time.perf_counter() - started,
        "reference_geometry_period_um": float(config["period_um"]),
        "reference_geometry_slab_ridge_um": coordinates.tolist(),
        "reference_h81_pole": reference["channel_pole"],
        "reference_h101_pole": reference["held_out_pole"],
    }
    target_sets = {}
    gamma = abs(pole.frequency.imag)
    for method in ("direct3", "direct5"):
        offsets = np.asarray(protocol["methods"][method]["sample_offsets_in_reference_pole_depths"], dtype=float)
        frequencies = pole.frequency.real + offsets * gamma
        began = time.perf_counter()
        values = np.asarray([channel_transmission(float(w), epsilon, setup) for w in frequencies])
        target_sets[method] = {
            "offsets_in_reference_pole_depths": offsets.tolist(),
            "sample_frequencies": frequencies.tolist(),
            "sample_targets": [complex_record(value) for value in values],
            "preparation_fdfd_frequency_factorizations": len(frequencies),
            "preparation_wall_seconds": time.perf_counter() - began,
        }
    prep["direct_target_preparation"] = target_sets
    return prep, config, setup, coordinates


def verify_fdfd(
    values: np.ndarray, target: complex, config: dict, setup, protocol: dict,
) -> dict:
    epsilon, _ = physical_epsilon_and_derivatives(values, config, setup)
    started = time.perf_counter()
    calls = 0

    def response(frequency: complex) -> complex:
        nonlocal calls
        calls += 1
        return channel_transmission(frequency, epsilon, setup)

    try:
        result = refine_scalar_root(
            response, target,
            kind="pole", tolerance=1e-9, max_evaluations=80,
            search_radius=max(0.1, 15 * abs(target.imag)),
        )
        error = abs(result.frequency - target) / abs(target.imag)
        passed = (
            result.residual_norm <= float(protocol["root_residual_tolerance"])
            and result.frequency.imag < 0
            and error <= float(protocol["fdfd_channel_pole_gate_in_reference_pole_depths"])
        )
        return {
            "pole": complex_record(result.frequency),
            "root_residual": float(result.residual_norm),
            "error_in_reference_depths": float(error),
            "root_evaluations": calls,
            "wall_seconds": time.perf_counter() - started,
            "pass": bool(passed),
        }
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return {"pass": False, "error": repr(exc), "root_evaluations": calls,
                "wall_seconds": time.perf_counter() - started}


def verify_held_out(
    values: np.ndarray, reference: dict, moderate: dict,
    config: dict, fourier_config: dict, protocol: dict,
) -> dict:
    started = time.perf_counter()
    slab, ridge = map(float, values)
    period = float(config["period_um"])
    roots = {}
    count = 0
    try:
        for label, harmonics, target_record in (
            ("h81", 81, reference["channel_pole"]),
            ("h101", 101, reference["held_out_pole"]),
        ):
            target = as_complex(target_record)
            response = _solver(fourier_config, slab, ridge, period, harmonics)

            def counted_response(frequency: complex) -> complex:
                nonlocal count
                count += 1
                return response(frequency).transmission

            before = count
            found = refine_scalar_root(
                counted_response, target,
                kind="pole", tolerance=1e-10, max_evaluations=80,
                search_radius=max(0.05, 12 * abs(target.imag)),
            )
            roots[label] = {
                "pole": complex_record(found.frequency),
                "root_residual": float(found.residual_norm),
                "error_in_reference_depths": float(abs(found.frequency - target) / abs(target.imag)),
                "response_evaluations": count - before,
            }
        frequency = float(moderate["operating_frequency_rad_per_um"])
        reference_delay = float(moderate["held_out"]["delay_fs"])
        reference_gamma = abs(as_complex(reference["held_out_pole"]).imag)
        step = float(protocol["held_out_delay_step_in_reference_pole_depths"]) * reference_gamma
        response = _solver(fourier_config, slab, ridge, period, 101)
        center = response(frequency)
        plus = response(frequency + step)
        minus = response(frequency - step)
        count += 3
        if min(abs(plus.transmission), abs(minus.transmission)) < 1e-5:
            raise RuntimeError("held-out delay stencil crossed a transmission zero")
        delay = float(np.angle(plus.transmission / minus.transmission) / (2 * step) * FS_PER_UM_OVER_C)
        root_h81 = as_complex(roots["h81"]["pole"])
        root_h101 = as_complex(roots["h101"]["pole"])
        root_motion = abs(root_h81 - root_h101) / reference_gamma
        T = float(abs(center.transmission) ** 2)
        passed = bool(
            roots["h81"]["root_residual"] <= float(protocol["root_residual_tolerance"])
            and roots["h101"]["root_residual"] <= float(protocol["root_residual_tolerance"])
            and roots["h101"]["error_in_reference_depths"] <= float(protocol["held_out_h101_pole_gate_in_reference_pole_depths"])
            and root_motion <= float(protocol["h81_to_h101_candidate_pole_gate_in_reference_pole_depths"])
            and T >= float(protocol["moderate_transmission_gate"])
            and float(protocol["held_out_delay_minimum_fraction_of_reference"]) * reference_delay <= delay
            <= float(protocol["held_out_delay_maximum_fraction_of_reference"]) * reference_delay
        )
        return {
            "pass": passed,
            "roots": roots,
            "h81_h101_root_motion_in_reference_depths": float(root_motion),
            "T_at_fixed_operating_frequency": T,
            "delay_fs_at_fixed_operating_frequency": delay,
            "reference_delay_fs": reference_delay,
            "response_evaluations": count,
            "wall_seconds": time.perf_counter() - started,
        }
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return {"pass": False, "roots": roots, "response_evaluations": count,
                "wall_seconds": time.perf_counter() - started, "error": repr(exc)}


def recenter_from_real_transmission(
    values: np.ndarray, target: complex, config: dict, setup, protocol: dict,
) -> tuple[float, dict]:
    epsilon, _ = physical_epsilon_and_derivatives(values, config, setup)
    gamma = abs(target.imag)
    half_width = float(protocol["direct_real_frequency_recentering_half_widths_in_reference_pole_depths"]) * gamma
    calls = 0

    def transmittance(frequency: float) -> float:
        nonlocal calls
        calls += 1
        return float(abs(channel_transmission(float(frequency), epsilon, setup)) ** 2)

    started = time.perf_counter()
    found = minimize_scalar(
        transmittance,
        bounds=(target.real - half_width, target.real + half_width),
        method="bounded",
        options={
            "xatol": float(protocol["direct_real_frequency_recentering_tolerance_in_reference_pole_depths"]) * gamma,
            "maxiter": int(protocol["direct_real_frequency_recentering_max_evaluations"]),
        },
    )
    return float(found.x), {
        "frequency_rad_per_um": float(found.x),
        "minimum_real_transmittance": float(found.fun),
        "success": bool(found.success),
        "fdfd_frequency_factorizations": calls,
        "wall_seconds": time.perf_counter() - started,
    }


def optimize_pole(start: np.ndarray, target: complex, config: dict, setup,
                  bounds: tuple[np.ndarray, np.ndarray], protocol: dict) -> tuple[np.ndarray, list[dict]]:
    cache_x = None
    cache_result = None
    calls = 0

    def system(values: np.ndarray):
        nonlocal cache_x, cache_result, calls
        if cache_x is None or not np.array_equal(values, cache_x):
            cache_result = pole_system(values, target, config, setup)
            cache_x = np.asarray(values).copy()
            calls += 1
        return cache_result

    began = time.perf_counter()
    optimized = least_squares(
        lambda x: system(x)[0], start, jac=lambda x: system(x)[1],
        bounds=bounds,
        max_nfev=int(protocol["methods"]["pole"]["maximum_objective_frequency_factorizations"]),
        xtol=1e-11, ftol=1e-11, gtol=1e-11,
    )
    return optimized.x, [{
        "stage": 1,
        "objective_residual": float(np.linalg.norm(optimized.fun)),
        "objective_fdfd_frequency_factorizations": calls,
        "optimizer_success": bool(optimized.success),
        "optimizer_message": str(optimized.message),
        "optimization_wall_seconds": time.perf_counter() - began,
    }]


def optimize_direct_stage(
    start: np.ndarray, frequencies: np.ndarray, targets: np.ndarray,
    config: dict, setup, bounds: tuple[np.ndarray, np.ndarray],
    stage_factorization_budget: int,
) -> tuple[np.ndarray, dict]:
    cache_x = None
    cache_result = None
    calls = 0

    def system(values: np.ndarray):
        nonlocal cache_x, cache_result, calls
        if cache_x is None or not np.array_equal(values, cache_x):
            cache_result = direct_system(values, frequencies, targets, config, setup)
            cache_x = np.asarray(values).copy()
            calls += 1
        return cache_result

    began = time.perf_counter()
    optimized = least_squares(
        lambda x: system(x)[0], start, jac=lambda x: system(x)[1],
        bounds=bounds, max_nfev=stage_factorization_budget // len(frequencies),
        xtol=1e-11, ftol=1e-11, gtol=1e-11,
    )
    return optimized.x, {
        "objective_residual": float(np.linalg.norm(optimized.fun)),
        "objective_fdfd_frequency_factorizations": calls * len(frequencies),
        "optimizer_success": bool(optimized.success),
        "optimizer_message": str(optimized.message),
        "optimization_wall_seconds": time.perf_counter() - began,
    }


def run_method(
    method: str, start: np.ndarray, target: complex, reference: dict, moderate: dict,
    reference_coordinates: np.ndarray, preparation: dict,
    config: dict, setup, fourier_config: dict,
    bounds: tuple[np.ndarray, np.ndarray], protocol: dict,
) -> dict:
    began = time.perf_counter()
    stages = []
    values = start.copy()
    target_generation_calls = 0
    target_generation_wall = 0.0
    scan_calls = 0
    scan_wall = 0.0
    held_out_calls = 0
    fdfd_root_calls = 0
    if method == "pole":
        values, stages = optimize_pole(values, target, config, setup, bounds, protocol)
        fdfd = verify_fdfd(values, target, config, setup, protocol)
        fdfd_root_calls += int(fdfd.get("root_evaluations", 0))
        held_out = (
            verify_held_out(values, reference, moderate, config, fourier_config, protocol)
            if fdfd["pass"] else {"pass": False, "skipped_because_fdfd_gate_failed": True, "response_evaluations": 0}
        )
        held_out_calls += int(held_out.get("response_evaluations", 0))
        stages[0]["fdfd_validation"] = fdfd
        stages[0]["held_out_validation"] = held_out
    else:
        offsets = np.asarray(protocol["methods"][method]["sample_offsets_in_reference_pole_depths"], dtype=float)
        reference_epsilon, _ = physical_epsilon_and_derivatives(reference_coordinates, config, setup)
        center = target.real
        gamma = abs(target.imag)
        for stage_index, stage_budget in enumerate(protocol["methods"][method]["objective_frequency_factorization_stages"], start=1):
            frequencies = center + offsets * gamma
            if stage_index == 1:
                targets = np.asarray([
                    as_complex(value) for value in preparation["direct_target_preparation"][method]["sample_targets"]
                ])
                new_target_calls = 0
            else:
                sample_start = time.perf_counter()
                targets = np.asarray([channel_transmission(float(w), reference_epsilon, setup) for w in frequencies])
                target_generation_wall += time.perf_counter() - sample_start
                target_generation_calls += len(frequencies)
                new_target_calls = len(frequencies)
            values, stage = optimize_direct_stage(values, frequencies, targets, config, setup, bounds, stage_budget)
            stage.update({"stage": stage_index, "sample_center_rad_per_um": center,
                          "sample_frequencies": frequencies.tolist(),
                          "sample_target_fdfd_frequency_factorizations": new_target_calls})
            fdfd = verify_fdfd(values, target, config, setup, protocol)
            fdfd_root_calls += int(fdfd.get("root_evaluations", 0))
            held_out = (
                verify_held_out(values, reference, moderate, config, fourier_config, protocol)
                if fdfd["pass"] else {"pass": False, "skipped_because_fdfd_gate_failed": True, "response_evaluations": 0}
            )
            held_out_calls += int(held_out.get("response_evaluations", 0))
            stage["fdfd_validation"] = fdfd
            stage["held_out_validation"] = held_out
            stages.append(stage)
            if held_out["pass"] or stage_index == len(protocol["methods"][method]["objective_frequency_factorization_stages"]):
                break
            center, scan = recenter_from_real_transmission(values, target, config, setup, protocol)
            scan_calls += int(scan["fdfd_frequency_factorizations"])
            scan_wall += float(scan["wall_seconds"])
            stage["real_transmission_recentering"] = scan
    objective_calls = sum(int(stage["objective_fdfd_frequency_factorizations"]) for stage in stages)
    qualified = any(stage["fdfd_validation"]["pass"] and stage["held_out_validation"]["pass"] for stage in stages)
    return {
        "method": method,
        "start_coordinates_um": start.tolist(),
        "final_coordinates_um": values.tolist(),
        "stages": stages,
        "qualified": bool(qualified),
        "objective_fdfd_frequency_factorizations": objective_calls,
        "target_generation_fdfd_frequency_factorizations": target_generation_calls,
        "real_recentering_fdfd_frequency_factorizations": scan_calls,
        "fdfd_root_verification_frequency_factorizations": fdfd_root_calls,
        "fourier_response_evaluations": held_out_calls,
        "method_specific_wall_seconds": time.perf_counter() - began,
        "real_recentering_wall_seconds": scan_wall,
        "target_generation_wall_seconds": target_generation_wall,
    }
