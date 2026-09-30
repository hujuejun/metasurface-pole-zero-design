# Required numerical helpers for the final GMR results; exploratory entry point removed.
"""Direct controls pinned to the actual held-out physical operating frequency."""

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

from scripts.m164_gmr_pole_adjoint import (
    build_setup, channel_transmission, physical_epsilon_and_derivatives,
)
from scripts.m177_gmr_preregistered_cost_benchmark import (
    as_complex, optimize_direct_stage, recenter_from_real_transmission,
    sha256, verify_fdfd, verify_held_out,
)


def prepare_method(method: str, reference_epsilon: np.ndarray, setup,
                   target: complex, reference: dict, moderate: dict,
                   protocol: dict) -> dict:
    physical = float(moderate["operating_frequency_rad_per_um"])
    delta = (
        float(protocol["physical_delay_bracket_in_h101_pole_depths"])
        * abs(as_complex(reference["held_out_pole"]).imag)
    )
    pinned = (
        np.asarray((physical,), dtype=float) if method == "direct3_physical"
        else np.asarray((physical - delta, physical, physical + delta), dtype=float)
    )
    resonant = np.asarray((target.real, target.real + abs(target.imag)), dtype=float)
    frequencies = np.r_[pinned, resonant]
    began = time.perf_counter()
    targets = np.asarray([channel_transmission(float(w), reference_epsilon, setup) for w in frequencies])
    return {
        "physical_operating_frequency_rad_per_um": physical,
        "physical_delay_bracket_rad_per_um": float(delta),
        "pinned_physical_frequencies_rad_per_um": pinned.tolist(),
        "initial_resonance_frequencies_rad_per_um": resonant.tolist(),
        "sample_frequencies_rad_per_um": frequencies.tolist(),
        "sample_targets": [{"real": float(z.real), "imag": float(z.imag)} for z in targets],
        "target_generation_fdfd_frequency_factorizations": len(frequencies),
        "target_generation_wall_seconds": time.perf_counter() - began,
    }


def run_trial(
    method: str, start: np.ndarray, preparation: dict, target: complex,
    reference: dict, moderate: dict, reference_epsilon: np.ndarray,
    config: dict, setup, fourier_config: dict,
    bounds: tuple[np.ndarray, np.ndarray], protocol: dict,
) -> dict:
    began = time.perf_counter()
    values = start.copy()
    pinned = np.asarray(preparation["pinned_physical_frequencies_rad_per_um"], dtype=float)
    pinned_targets = np.asarray([as_complex(z) for z in preparation["sample_targets"][:len(pinned)]])
    center = target.real
    gamma = abs(target.imag)
    stages = []
    new_target_calls = 0
    new_target_wall = 0.0
    scan_calls = 0
    scan_wall = 0.0
    fdfd_root_calls = 0
    fourier_calls = 0
    for index, budget in enumerate(protocol["methods"][method]["objective_frequency_factorization_stages"], start=1):
        resonance_frequencies = np.asarray((center, center + gamma), dtype=float)
        frequencies = np.r_[pinned, resonance_frequencies]
        if index == 1:
            targets = np.asarray([as_complex(z) for z in preparation["sample_targets"]])
            stage_target_calls = 0
        else:
            target_started = time.perf_counter()
            resonance_targets = np.asarray([
                channel_transmission(float(w), reference_epsilon, setup) for w in resonance_frequencies
            ])
            new_target_wall += time.perf_counter() - target_started
            new_target_calls += 2
            stage_target_calls = 2
            targets = np.r_[pinned_targets, resonance_targets]
        values, stage = optimize_direct_stage(
            values, frequencies, targets, config, setup, bounds, int(budget),
        )
        stage.update({
            "stage": index,
            "pinned_physical_frequencies_rad_per_um": pinned.tolist(),
            "sample_center_rad_per_um": center,
            "sample_frequencies_rad_per_um": frequencies.tolist(),
            "recentered_reference_target_factorizations": stage_target_calls,
        })
        fdfd = verify_fdfd(values, target, config, setup, protocol)
        fdfd_root_calls += int(fdfd.get("root_evaluations", 0))
        held_out = (
            verify_held_out(values, reference, moderate, config, fourier_config, protocol)
            if fdfd["pass"] else {"pass": False, "skipped_because_fdfd_gate_failed": True, "response_evaluations": 0}
        )
        fourier_calls += int(held_out.get("response_evaluations", 0))
        stage["fdfd_validation"] = fdfd
        stage["held_out_validation"] = held_out
        stages.append(stage)
        if held_out["pass"] or index == len(protocol["methods"][method]["objective_frequency_factorization_stages"]):
            break
        center, scan = recenter_from_real_transmission(values, target, config, setup, protocol)
        scan_calls += int(scan["fdfd_frequency_factorizations"])
        scan_wall += float(scan["wall_seconds"])
        stage["real_transmission_recentering"] = scan
    qualified = any(stage["fdfd_validation"]["pass"] and stage["held_out_validation"]["pass"] for stage in stages)
    return {
        "method": method,
        "start_coordinates_um": start.tolist(),
        "final_coordinates_um": values.tolist(),
        "stages": stages,
        "qualified": bool(qualified),
        "objective_fdfd_frequency_factorizations": sum(int(stage["objective_fdfd_frequency_factorizations"]) for stage in stages),
        "target_generation_fdfd_frequency_factorizations": new_target_calls,
        "real_recentering_fdfd_frequency_factorizations": scan_calls,
        "fdfd_root_verification_frequency_factorizations": fdfd_root_calls,
        "fourier_response_evaluations": fourier_calls,
        "method_specific_wall_seconds": time.perf_counter() - began,
        "target_generation_wall_seconds": new_target_wall,
        "real_recentering_wall_seconds": scan_wall,
    }
