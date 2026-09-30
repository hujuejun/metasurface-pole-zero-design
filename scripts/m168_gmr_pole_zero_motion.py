"""Predict local transmission pole/zero motion from physical GMR adjoints.

The pole sensitivity is implicit differentiation of the time-reversed
two-port determinant; the zero sensitivity is implicit differentiation of
the transmission-channel zero. Their physical-coordinate derivatives use
Maxwell transpose adjoints. Held-out perturbed roots are independently
refined in FDFD without computing any real-frequency spectra. The random
perturbations are synthetic local tests, not a manufacturing-yield model.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from pzid.adjoint import scattering_matrix_with_jacobian
from pzid.roots import refine_scalar_root
from pzid.twoport import solve_two_port_scattering

from scripts.m164_gmr_pole_adjoint import (
    build_setup, channel_transmission, physical_epsilon_and_derivatives,
    pole_system,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complex(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _determinant(frequency: complex, epsilon: np.ndarray, setup) -> complex:
    return complex(np.linalg.det(solve_two_port_scattering(frequency, epsilon, setup).matrix))


def _sensitivity(coordinates: np.ndarray, pole: complex, zero: complex,
                 config: dict, setup, step_fraction: float) -> tuple[np.ndarray, np.ndarray, dict]:
    epsilon, physical_derivatives = physical_epsilon_and_derivatives(coordinates, config, setup)
    pole_residual, pole_jacobian = pole_system(coordinates, pole, config, setup)
    f_x = pole_jacobian[0] + 1j * pole_jacobian[1]
    step = step_fraction * abs(pole.imag)
    f_plus = _determinant(np.conj(pole) + step, epsilon, setup)
    f_minus = _determinant(np.conj(pole) - step, epsilon, setup)
    f_omega = (f_plus - f_minus) / (2 * step)
    pole_gradient = np.conj(-f_x / f_omega)
    evaluated = scattering_matrix_with_jacobian(zero, epsilon, setup)
    t_x = np.einsum("zx,pzx->p", evaluated.jacobian[1, 0], physical_derivatives)
    t_plus = channel_transmission(zero + step, epsilon, setup)
    t_minus = channel_transmission(zero - step, epsilon, setup)
    t_omega = (t_plus - t_minus) / (2 * step)
    zero_gradient = -t_x / t_omega
    audit = {
        "time_reversed_pole_determinant_residual": float(np.linalg.norm(pole_residual)),
        "transmission_zero_residual": float(abs(evaluated.matrix[1, 0])),
        "complex_frequency_difference_step": step,
        "pole_gradient_rad_per_um_per_um": [_complex(x) for x in pole_gradient],
        "zero_gradient_rad_per_um_per_um": [_complex(x) for x in zero_gradient],
    }
    return pole_gradient, zero_gradient, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m168_gmr_pole_zero_motion.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/m168_gmr_pole_zero_motion"))
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    source_path = Path(protocol["source_corrected_summary"])
    high_path = Path(protocol["source_q3200_summary"])
    base_path = Path(protocol["base_adjoint_config"])
    source = json.loads(source_path.read_text(encoding="utf-8"))["cases"]
    high = json.loads(high_path.read_text(encoding="utf-8"))["cases"]
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
        "input_sha256": {str(p): _sha256(p) for p in (args.config, source_path, high_path, base_path)},
        "script_sha256": _sha256(Path(__file__)),
        "geometry_sha256": _sha256(Path("src/pzid/gmr2d.py")),
        "adjoint_sha256": _sha256(Path("src/pzid/adjoint.py")),
    }, indent=2) + "\n", encoding="utf-8")
    rng = np.random.default_rng(int(protocol["random_seed"]))
    rows = []
    for quality in protocol["quality_factors"]:
        source_case = next(x for x in (high if quality == 3200 else source) if x["quality_factor"] == quality)
        coordinates = np.asarray(source_case["final_coordinates_um"], dtype=float)
        config = dict(base)
        config["period_um"] = float(source_case["period_um"])
        setup = build_setup(config)
        base_pole = complex(**source_case["final_fdfd"]["verified_pole"])
        base_zero = complex(**source_case["final_fdfd"]["verified_zero"])
        pole_gradient, zero_gradient, audit = _sensitivity(
            coordinates, base_pole, base_zero, config, setup,
            float(protocol["complex_frequency_derivative_step_in_pole_depths"]),
        )
        trials = []
        for index in range(int(protocol["held_out_perturbations_per_design"])):
            delta = rng.normal(size=2) * np.asarray((protocol["slab_sigma_um"], protocol["ridge_sigma_um"]))
            trial_epsilon, _ = physical_epsilon_and_derivatives(coordinates + delta, config, setup)
            predicted_pole = base_pole + np.dot(pole_gradient, delta)
            predicted_zero = base_zero + np.dot(zero_gradient, delta)
            response = lambda frequency: channel_transmission(frequency, trial_epsilon, setup)
            exact_pole = refine_scalar_root(
                response, predicted_pole, kind="pole", tolerance=1e-9,
                max_evaluations=80, search_radius=max(0.05, 10 * abs(base_pole.imag)),
            )
            exact_zero = refine_scalar_root(
                response, predicted_zero, kind="zero", tolerance=1e-9,
                max_evaluations=80, search_radius=max(0.05, 10 * abs(base_pole.imag)),
            )
            if exact_pole.residual_norm > 1e-6 or exact_zero.residual_norm > 1e-6:
                raise RuntimeError(f"held-out root verification failed Q={quality}, trial={index}")
            trials.append({
                "index": index, "delta_slab_and_ridge_um": delta.tolist(),
                "predicted_pole": _complex(predicted_pole),
                "verified_pole": _complex(exact_pole.frequency),
                "predicted_zero": _complex(predicted_zero),
                "verified_zero": _complex(exact_zero.frequency),
                "pole_prediction_error_in_base_depths": float(abs(exact_pole.frequency - predicted_pole) / abs(base_pole.imag)),
                "zero_prediction_error_in_base_depths": float(abs(exact_zero.frequency - predicted_zero) / abs(base_pole.imag)),
                "pole_motion_in_base_depths": float(abs(exact_pole.frequency - base_pole) / abs(base_pole.imag)),
                "zero_motion_in_base_depths": float(abs(exact_zero.frequency - base_zero) / abs(base_pole.imag)),
            })
        pole_errors = np.asarray([x["pole_prediction_error_in_base_depths"] for x in trials])
        zero_errors = np.asarray([x["zero_prediction_error_in_base_depths"] for x in trials])
        row = {
            "quality_target": quality,
            "base_coordinates_um": coordinates.tolist(),
            "base_pole": _complex(base_pole), "base_zero": _complex(base_zero),
            "adjoint_audit": audit,
            "trials": trials,
            "median_pole_error_depths": float(np.median(pole_errors)),
            "maximum_pole_error_depths": float(np.max(pole_errors)),
            "median_zero_error_depths": float(np.median(zero_errors)),
            "maximum_zero_error_depths": float(np.max(zero_errors)),
            "passes_maximum_error_gate": bool(max(np.max(pole_errors), np.max(zero_errors))
                                               <= float(protocol["maximum_prediction_error_in_pole_depths"])),
        }
        rows.append(row)
        (output / "summary.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n", encoding="utf-8")
        print(f"Q={quality}: median pole/zero errors {row['median_pole_error_depths']:.3g}/"
              f"{row['median_zero_error_depths']:.3g} depths", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True)
    for row in rows:
        pole = complex(**row["base_pole"])
        for trial in row["trials"]:
            axes[0].scatter(trial["pole_motion_in_base_depths"],
                            trial["pole_prediction_error_in_base_depths"], s=16,
                            label=f"Q≈{row['quality_target']}" if trial["index"] == 0 else None)
            axes[1].scatter(trial["zero_motion_in_base_depths"],
                            trial["zero_prediction_error_in_base_depths"], s=16)
    for axis, title in zip(axes, ("Transmission pole", "Transmission zero"), strict=True):
        axis.set_yscale("log")
        axis.set_xlabel("Verified root motion / initial pole depth")
        axis.set_ylabel("First-order prediction error / initial pole depth")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output / "gmr_adjoint_root_motion.png", dpi=220)
    fig.savefig(output / "gmr_adjoint_root_motion.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
