"""Adjoint-refine physical GMR Q targets from preregistered RCWA seeds.

M165A chooses each fixed phase-matching period as a *screen*. At that period,
this program adjusts only physical slab thickness and ridge etch height using
the FDFD Maxwell adjoint of a Chen-style time-reversal pole residual. The
transmission-channel pole and zero are then independently root-refined; the
Fourier-modal pole is an additional held-out cross-solver check.
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

from pzid.roots import refine_scalar_root

from scripts.m164_gmr_pole_adjoint import (
    _write_curve, build_setup, check_adjoint_gradient,
    physical_epsilon_and_derivatives, pole_system, verify_channel_pole,
)
from scripts.m165a_gmr_phase_matching_prescreen import _response


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complex(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m165b_gmr_adjoint_q_ladder.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/m165b_gmr_adjoint_q_ladder"))
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    screen_path = Path(protocol["source_phase_matching_screen"])
    base_path = Path(protocol["base_adjoint_config"])
    screen = json.loads(screen_path.read_text(encoding="utf-8"))["cases"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    revision = subprocess.run(("git", "rev-parse", "HEAD"), check=False, capture_output=True, text=True)
    (output / "metadata.json").write_text(json.dumps({
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__, "platform": platform.platform(),
        "command": sys.argv,
        "input_sha256": {str(p): _sha256(p) for p in (args.config, screen_path, base_path)},
        "script_sha256": _sha256(Path(__file__)),
        "adjoint_wrapper_sha256": _sha256(Path("scripts/m164_gmr_pole_adjoint.py")),
        "geometry_sha256": _sha256(Path("src/pzid/gmr2d.py")),
        "fdfd_adjoint_sha256": _sha256(Path("src/pzid/adjoint.py")),
        "independent_fourier_sha256": _sha256(Path("src/pzid/rcwa2d.py")),
    }, indent=2) + "\n", encoding="utf-8")
    output_summary: list[dict] = []
    for quality in protocol["target_quality_factors"]:
        source = next(item for item in screen if item["target_quality_factor"] == quality)
        config = dict(base)
        config["period_um"] = float(source["screened_period_um"])
        setup = build_setup(config)
        target = complex(**source["target_pole"])
        start = np.asarray((base["start_slab_thickness_um"], source["screened_ridge_height_um"]), dtype=float)
        preflight = check_adjoint_gradient(start, target, config, setup)
        if max(preflight["relative_errors"]) > float(base["gradient_relative_tolerance"]):
            raise RuntimeError(f"Q={quality} physical adjoint gradient failed: {preflight}")
        bounds = (
            np.asarray((base["slab_thickness_bounds_um"][0], base["ridge_height_bounds_um"][0])),
            np.asarray((base["slab_thickness_bounds_um"][1], base["ridge_height_bounds_um"][1])),
        )
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
            lambda x: system(x)[0], start, jac=lambda x: system(x)[1], bounds=bounds,
            max_nfev=int(protocol["maximum_function_evaluations"]),
            xtol=1e-11, ftol=1e-11, gtol=1e-11,
        )
        elapsed = time.perf_counter() - began
        verified, curves, epsilon = verify_channel_pole(optimized.x, target, config, setup)
        suffix = f"q{int(quality):04d}"
        _write_curve(output / f"spectrum_{suffix}.csv", curves)
        np.save(output / f"epsilon_{suffix}.npy", epsilon)
        high = int(protocol["independent_fourier_harmonics"])
        independent = _response({
            "high_index": config["high_index"], "port_index": config["port_index"],
            "slab_thickness_um": float(optimized.x[0]),
            "ridge_fill_factor": config["ridge_fill_factor"],
        }, float(config["period_um"]), float(optimized.x[1]), high)
        independent_root = refine_scalar_root(
            independent, target, kind="pole", tolerance=1e-10,
            max_evaluations=100, search_radius=max(0.2, 20 * abs(target.imag)),
        )
        case = {
            "target_quality_factor": quality,
            "target_pole": _complex(target),
            "fixed_period_um": config["period_um"],
            "start_slab_and_ridge_um": start.tolist(),
            "optimized_slab_and_ridge_um": optimized.x.tolist(),
            "preflight": preflight,
            "optimizer_success": bool(optimized.success),
            "optimizer_message": str(optimized.message),
            "time_reversal_determinant_residual": float(np.linalg.norm(optimized.fun)),
            "unique_adjoint_evaluations": calls,
            "equivalent_frequency_factorizations": calls,
            "optimization_wall_seconds": elapsed,
            "fdfd_verification": verified,
            "fourier_verification": {
                "harmonics": high,
                "pole": _complex(independent_root.frequency),
                "quality_factor": float(independent_root.frequency.real / (-2 * independent_root.frequency.imag)),
                "root_residual": independent_root.residual_norm,
                "fdfd_fourier_separation_in_target_depths": float(
                    abs(independent_root.frequency - complex(**verified["verified_pole"])) / abs(target.imag)
                ),
            },
        }
        output_summary.append(case)
        (output / "summary.json").write_text(json.dumps({
            "method": "physical slab/ridge FDFD pole adjoint at RCWA-screened fixed periods",
            "cases": output_summary,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Q={quality}: FDFD={verified['verified_quality_factor']:.2f}, "
              f"FDFD target error={verified['pole_error_in_target_depths']:.3f} depths, "
              f"H{high} cross-solver={case['fourier_verification']['fdfd_fourier_separation_in_target_depths']:.2f} depths, "
              f"adjoint solves={calls}", flush=True)


if __name__ == "__main__":
    main()
