"""Correct FDFD adjoint target bias with independent Fourier channel roots.

Each correction *still* uses the fixed-complex-frequency FDFD Maxwell
adjoint to move two physical shape coordinates. The Fourier solver reports
held-out channel roots and supplies a secant-like target-offset calibration.
This stage is not counted as a pure one-shot adjoint result; all Fourier
evaluations are part of its total design cost and are kept in the record.
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

from scripts.m164_gmr_pole_adjoint import (
    _write_curve, build_setup, pole_system, verify_channel_pole,
)
from scripts.m165a_gmr_phase_matching_prescreen import _response


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complex(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def independent_pole(
    coordinates: np.ndarray, period: float, target: complex,
    config: dict, harmonics: int,
) -> tuple[complex, float, int]:
    response = _response({
        "high_index": config["high_index"],
        "port_index": config["port_index"],
        "slab_thickness_um": float(coordinates[0]),
        "ridge_fill_factor": config["ridge_fill_factor"],
    }, period, float(coordinates[1]), harmonics)
    half = 0.07
    omega = np.linspace(target.real - half, target.real + half, 141)
    intensity = np.asarray([abs(response(float(w))) ** 2 for w in omega])
    index = int(np.argmin(intensity))
    if index == 0 or index == omega.size - 1:
        raise RuntimeError("independent resonance left the near-target window")
    minimum = minimize_scalar(
        lambda w: abs(response(float(w))) ** 2,
        bounds=(float(omega[index - 1]), float(omega[index + 1])),
        method="bounded", options={"xatol": 1e-12},
    )
    seed = complex(float(minimum.x), target.imag)
    root = refine_scalar_root(
        response, seed, kind="pole", tolerance=1e-10,
        max_evaluations=100, search_radius=max(0.1, 12 * abs(target.imag)),
    )
    if root.residual_norm > 1e-6 or root.frequency.imag >= 0:
        raise RuntimeError(f"independent Fourier channel pole failed: {root}")
    return root.frequency, root.residual_norm, root.evaluations + len(omega) + minimum.nfev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m165c_gmr_cross_solver_pole_correction.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/m165c_gmr_cross_solver_pole_correction"))
    parser.add_argument("--start-from-summary", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    source_path = Path(protocol["source_adjoint_summary"])
    base_path = Path(protocol["base_adjoint_config"])
    source = json.loads(source_path.read_text(encoding="utf-8"))["cases"]
    prior_cases = None
    if args.start_from_summary is not None:
        prior_cases = json.loads(args.start_from_summary.read_text(encoding="utf-8"))["cases"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    revision = subprocess.run(("git", "rev-parse", "HEAD"), check=False, capture_output=True, text=True)
    (output / "metadata.json").write_text(json.dumps({
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "command": sys.argv,
        "input_sha256": {str(path): _sha256(path) for path in (
            args.config, source_path, base_path,
            *([args.start_from_summary] if args.start_from_summary is not None else []),
        )},
        "script_sha256": _sha256(Path(__file__)),
        "geometry_sha256": _sha256(Path("src/pzid/gmr2d.py")),
        "fdfd_adjoint_sha256": _sha256(Path("src/pzid/adjoint.py")),
        "fourier_solver_sha256": _sha256(Path("src/pzid/rcwa2d.py")),
    }, indent=2) + "\n", encoding="utf-8")
    cases = []
    for quality in protocol["target_quality_factors"]:
        source_case = next(item for item in source if item["target_quality_factor"] == quality)
        config = dict(base)
        period = float(source_case["fixed_period_um"])
        config["period_um"] = period
        setup = build_setup(config)
        desired = complex(**source_case["target_pole"])
        coordinates = np.asarray(source_case["optimized_slab_and_ridge_um"], dtype=float)
        corrected_target = desired
        if prior_cases is not None:
            prior = next(item for item in prior_cases if item["quality_factor"] == quality)
            coordinates = np.asarray(prior["final_coordinates_um"], dtype=float)
            corrected_target = complex(**prior["rounds"][-1]["fdfd_adjoint_target"])
        rounds = []
        for index in range(int(protocol["maximum_correction_rounds"]) + 1):
            independent, independent_residual, fourier_calls = independent_pole(
                coordinates, period, desired, config, int(protocol["fourier_harmonic_half_width"]),
            )
            error = float(abs(independent - desired) / abs(desired.imag))
            row = {
                "round": index,
                "slab_and_ridge_um": coordinates.tolist(),
                "fdfd_adjoint_target": _complex(corrected_target),
                "fourier_pole": _complex(independent),
                "fourier_pole_root_residual": independent_residual,
                "fourier_evaluations": fourier_calls,
                "fourier_error_in_desired_depths": error,
            }
            rounds.append(row)
            (output / f"q{int(quality):04d}_rounds.json").write_text(
                json.dumps(rounds, indent=2) + "\n", encoding="utf-8")
            print(f"Q={quality} round={index}: H{protocol['fourier_harmonic_half_width']} "
                  f"pole error={error:.4g} depths", flush=True)
            if error <= float(protocol["accepted_independent_pole_error_depths"]):
                break
            if index == int(protocol["maximum_correction_rounds"]):
                break
            # Multi-fidelity target-offset update; it does not differentiate
            # through the Fourier solver and is not an adjoint evaluation.
            corrected_target = corrected_target + (desired - independent)
            cache_x = None
            cache_result = None
            calls = 0

            def system(values: np.ndarray):
                nonlocal cache_x, cache_result, calls
                if cache_x is None or not np.array_equal(values, cache_x):
                    cache_result = pole_system(values, corrected_target, config, setup)
                    cache_x = np.asarray(values).copy()
                    calls += 1
                return cache_result

            bounds = (
                np.asarray((config["slab_thickness_bounds_um"][0], config["ridge_height_bounds_um"][0])),
                np.asarray((config["slab_thickness_bounds_um"][1], config["ridge_height_bounds_um"][1])),
            )
            began = time.perf_counter()
            optimized = least_squares(
                lambda x: system(x)[0], coordinates, jac=lambda x: system(x)[1],
                bounds=bounds, max_nfev=int(protocol["maximum_adjoint_evaluations_per_round"]),
                xtol=1e-11, ftol=1e-11, gtol=1e-11,
            )
            row["next_adjoint_target"] = _complex(corrected_target)
            row["next_adjoint_residual"] = float(np.linalg.norm(optimized.fun))
            row["next_adjoint_evaluations"] = calls
            row["next_adjoint_wall_seconds"] = time.perf_counter() - began
            coordinates = optimized.x.copy()
        final_fdfd, curves, epsilon = verify_channel_pole(coordinates, corrected_target, config, setup)
        suffix = f"q{int(quality):04d}"
        np.save(output / f"epsilon_{suffix}.npy", epsilon)
        _write_curve(output / f"spectrum_fdfd_{suffix}.csv", curves)
        case = {
            "quality_factor": quality,
            "desired_pole": _complex(desired),
            "period_um": period,
            "rounds": rounds,
            "final_coordinates_um": coordinates.tolist(),
            "final_fdfd": final_fdfd,
            "final_fourier_error_in_desired_depths": rounds[-1]["fourier_error_in_desired_depths"],
            "total_correction_adjoint_evaluations": sum(row.get("next_adjoint_evaluations", 0) for row in rounds),
            "total_fourier_evaluations": sum(row["fourier_evaluations"] for row in rounds),
        }
        cases.append(case)
        (output / "summary.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
