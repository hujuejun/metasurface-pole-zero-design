"""Independent high-harmonic GMR channel roots, spectra and delay audit.

Physical slab/ridge coordinates come from M165 pole-adjoint continuation.
Period is polished only as an explicit, non-adjoint phase-matching coordinate;
this operation and all Fourier solves are recorded. No transmission or phase
samples were included in the FDFD adjoint objective.
"""

from __future__ import annotations

import argparse
import csv
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

from pzid.rcwa2d import centered_binary_grating_layer, homogeneous_layer, solve_tm_stack
from pzid.roots import refine_scalar_root

from scripts.m164_gmr_pole_adjoint import FS_PER_UM_OVER_C


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complex(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _solver(config: dict, slab: float, ridge: float, period: float, harmonics: int):
    layers = (
        homogeneous_layer(float(config["high_index"]), slab, harmonics),
        centered_binary_grating_layer(float(config["port_index"]), float(config["high_index"]),
                                      float(config["ridge_fill_factor"]), ridge, harmonics),
    )
    return lambda omega: solve_tm_stack(
        omega, layers, period_um=period, port_index=float(config["port_index"]),
        harmonic_half_width=harmonics,
    )


def _pole(config: dict, slab: float, ridge: float, period: float, harmonics: int,
          seed: complex) -> tuple[complex, float, int]:
    solve = _solver(config, slab, ridge, period, harmonics)
    found = refine_scalar_root(
        lambda omega: solve(omega).transmission, seed, kind="pole",
        tolerance=1e-10, max_evaluations=100, search_radius=max(0.05, 15 * abs(seed.imag)),
    )
    if found.residual_norm > 1e-6 or found.frequency.imag >= 0:
        raise RuntimeError(f"high-harmonic channel pole refinement failed: {found}")
    return found.frequency, found.residual_norm, found.evaluations


def _phase_delay(response, omega: float, step: float) -> float:
    plus = response(omega + step).transmission
    minus = response(omega - step).transmission
    return float(np.angle(plus / minus) / (2 * step) * FS_PER_UM_OVER_C)


def _compact_fit(omega: np.ndarray, t: np.ndarray, pole: complex,
                 zero: complex) -> dict[str, float]:
    gamma = abs(pole.imag)
    delta = (omega - pole.real) / gamma
    selected = np.asarray([-4.0, -1.0, 1.0, 4.0])
    selection = np.asarray([int(np.argmin(abs(delta - value))) for value in selected])
    input_delta = delta[selection]
    input_omega = omega[selection]
    input_t = t[selection]
    # Root-informed form: t=(b0+b1*delta)*(omega-zero)/(omega-pole).
    root_factor = (input_omega - zero) / (input_omega - pole)
    b = np.linalg.lstsq(np.column_stack((root_factor, root_factor * input_delta)),
                        input_t, rcond=None)[0]
    pred_pz = (b[0] + b[1] * delta) * (omega - zero) / (omega - pole)
    # Four-complex-sample direct [2/1] rational interpolation, no root input.
    equation = np.column_stack((np.ones(4), input_delta, input_delta**2,
                                -input_t * input_delta))
    direct = np.linalg.solve(equation, input_t)
    pred_direct = (direct[0] + direct[1] * delta + direct[2] * delta**2) / (
        1 + direct[3] * delta
    )
    heldout = np.ones(omega.size, dtype=bool)
    heldout[selection] = False
    rmse_pz = float(np.sqrt(np.mean(abs(pred_pz[heldout] - t[heldout]) ** 2)))
    rmse_direct = float(np.sqrt(np.mean(abs(pred_direct[heldout] - t[heldout]) ** 2)))
    return {
        "pole_zero_linear_background_heldout_complex_t_rmse": rmse_pz,
        "direct_four_sample_2_over_1_heldout_complex_t_rmse": rmse_direct,
        "number_of_real_axis_calibration_samples_each": 4,
        "pole_zero_model_also_uses_verified_complex_roots": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m166_gmr_high_fidelity_analysis.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/m166_gmr_high_fidelity_analysis"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    first_path = Path(config["source_corrected_summary"])
    high_path = Path(config["source_q3200_summary"])
    first = json.loads(first_path.read_text(encoding="utf-8"))["cases"]
    high = json.loads(high_path.read_text(encoding="utf-8"))["cases"]
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    revision = subprocess.run(("git", "rev-parse", "HEAD"), check=False, capture_output=True, text=True)
    (output / "metadata.json").write_text(json.dumps({
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__, "platform": platform.platform(),
        "command": sys.argv,
        "inputs_sha256": {str(p): _sha256(p) for p in (args.config, first_path, high_path)},
        "script_sha256": _sha256(Path(__file__)),
        "fourier_solver_sha256": _sha256(Path("src/pzid/rcwa2d.py")),
    }, indent=2) + "\n", encoding="utf-8")
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True, constrained_layout=True)
    for quality in config["quality_factors"]:
        source = next(x for x in (high if quality == 3200 else first) if x["quality_factor"] == quality)
        target = complex(**source["desired_pole"])
        slab, ridge = map(float, source["final_coordinates_um"])
        period = float(source["period_um"])
        seed = complex(**source["rounds"][-1]["fourier_pole"])
        history = []
        harmonics = int(config["period_polish_harmonics"])
        step = float(config["period_derivative_step_um"])
        for iteration in range(int(config["maximum_period_polish_steps"]) + 1):
            root, root_residual, calls = _pole(config, slab, ridge, period, harmonics, seed)
            error = abs(root.real - target.real) / abs(target.imag)
            history.append({
                "iteration": iteration, "period_um": period,
                "pole": _complex(root), "root_residual": root_residual,
                "root_evaluations": calls, "real_frequency_error_in_target_depths": error,
            })
            seed = root
            print(f"Q={quality} period polish {iteration}: omega error={error:.4g} depths", flush=True)
            if error <= float(config["reference_period_error_limit_in_pole_depths"]):
                break
            if iteration == int(config["maximum_period_polish_steps"]):
                break
            plus, _, _ = _pole(config, slab, ridge, period + step, harmonics, root)
            minus, _, _ = _pole(config, slab, ridge, period - step, harmonics, root)
            slope = (plus.real - minus.real) / (2 * step)
            if abs(slope) < 0.1:
                raise RuntimeError("period sensitivity collapsed")
            period += (target.real - root.real) / slope
            if not 0 < period < 2 * np.pi / target.real / float(config["port_index"]):
                raise RuntimeError("period polish crossed exterior diffraction threshold")
        root, _, _ = _pole(config, slab, ridge, period, harmonics, seed)
        heldout_h = int(config["held_out_root_harmonics"])
        heldout, heldout_residual, heldout_calls = _pole(config, slab, ridge, period, heldout_h, root)
        solver = _solver(config, slab, ridge, period, int(config["spectrum_harmonics"]))
        zero_result = refine_scalar_root(
            lambda w: solver(w).transmission, complex(root.real), kind="zero",
            tolerance=1e-10, max_evaluations=100, search_radius=max(0.08, 10 * abs(root.imag)),
        )
        if zero_result.residual_norm > 1e-6:
            raise RuntimeError(f"channel zero failed for Q={quality}: {zero_result}")
        zero = zero_result.frequency
        gamma = abs(root.imag)
        width = float(config["spectrum_half_widths_in_pole_depths"]) * gamma
        omega = np.linspace(root.real - width, root.real + width, int(config["spectrum_points"]))
        solved = [solver(float(w)) for w in omega]
        t = np.asarray([entry.transmission for entry in solved])
        r = np.asarray([entry.reflection for entry in solved])
        phase = np.unwrap(np.angle(t))
        delay_reference = _phase_delay(solver, root.real,
                                       float(config["reference_delay_step_in_pole_depths"]) * gamma)
        delay_adaptive = _phase_delay(solver, root.real,
                                      float(config["adaptive_delay_step_in_pole_depths"]) * gamma)
        delay_fixed = _phase_delay(solver, root.real,
                                   float(config["fixed_delay_step_rad_per_um"]))
        root_factor_delay = FS_PER_UM_OVER_C * np.imag(
            1 / (root.real - zero) - 1 / (root.real - root)
        )
        compact = _compact_fit(omega, t, root, zero)
        suffix = f"q{int(quality):04d}"
        with (output / f"spectrum_{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("omega_rad_per_um", "detuning_in_pole_depths", "t_real", "t_imag",
                             "r_real", "r_imag", "T", "R", "phase_unwrapped_rad"))
            writer.writerows(zip(omega, (omega - root.real) / gamma, t.real, t.imag,
                                 r.real, r.imag, abs(t)**2, abs(r)**2, phase, strict=True))
        center = int(np.argmin(abs(omega - root.real)))
        row = {
            "quality_target": quality, "target_pole": _complex(target),
            "slab_thickness_um": slab, "ridge_height_um": ridge,
            "initial_period_um": source["period_um"], "final_period_um": period,
            "period_polish_history": history,
            "channel_pole_harmonics": harmonics, "channel_pole": _complex(root),
            "channel_zero": _complex(zero),
            "channel_zero_residual": zero_result.residual_norm,
            "achieved_quality_factor": float(root.real / (2 * gamma)),
            "complex_pole_target_error_in_target_depths": float(abs(root - target) / abs(target.imag)),
            "held_out_harmonics": heldout_h, "held_out_pole": _complex(heldout),
            "held_out_pole_residual": heldout_residual, "held_out_pole_evaluations": heldout_calls,
            "held_out_root_motion_in_target_depths": float(abs(heldout - root) / abs(target.imag)),
            "spectrum_harmonics": int(config["spectrum_harmonics"]),
            "center_T": float(abs(t[center]) ** 2), "center_R": float(abs(r[center]) ** 2),
            "minimum_T": float(np.min(abs(t) ** 2)), "maximum_R": float(np.max(abs(r) ** 2)),
            "maximum_energy_error": float(np.max(abs(abs(t)**2 + abs(r)**2 - 1))),
            "delay_reference_fs": delay_reference,
            "delay_adaptive_fs": delay_adaptive, "delay_fixed_fs": delay_fixed,
            "pole_depth_delay_scale_fs": float(FS_PER_UM_OVER_C / gamma),
            "pole_zero_factor_delay_fs": float(root_factor_delay),
            "zero_and_background_delay_remainder_fs": float(delay_reference - root_factor_delay),
            "fixed_step_relative_delay_error": float(abs(delay_fixed - delay_reference) / max(abs(delay_reference), 1e-12)),
            "adaptive_step_relative_delay_error": float(abs(delay_adaptive - delay_reference) / max(abs(delay_reference), 1e-12)),
            "compact_model": compact,
            "numerically_converged_harmonics": bool(
                abs(heldout - root) / abs(target.imag)
                <= float(config["held_out_harmonic_root_motion_gate_depths"])
            ),
        }
        rows.append(row)
        (output / "summary.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n", encoding="utf-8")
        x = (omega - root.real) / gamma
        axes[0].plot(x, abs(t)**2, label=f"Q≈{root.real/(2*gamma):.0f}")
        axes[1].plot(x, (phase - phase[0]) / np.pi, label=f"Q≈{root.real/(2*gamma):.0f}")
        print(f"Q={quality}: achieved {row['achieved_quality_factor']:.1f}, "
              f"H81-to-H101 {row['held_out_root_motion_in_target_depths']:.3f} depths, "
              f"delay {delay_reference:.2f} fs", flush=True)
    axes[0].set_ylabel("Transmittance")
    axes[1].set_ylabel("Relative transmission phase / π")
    axes[1].set_xlabel("Frequency detuning / pole depth")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(output / "gmr_q_series_transmission_phase.png", dpi=220)
    fig.savefig(output / "gmr_q_series_transmission_phase.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    q = np.asarray([x["achieved_quality_factor"] for x in rows])
    axes[0].loglog(q, [x["delay_reference_fs"] for x in rows], "o-", label="Measured local delay")
    axes[0].loglog(q, [x["pole_depth_delay_scale_fs"] for x in rows], "s--", label="1 / pole depth")
    axes[0].set_xlabel("Verified channel-pole Q")
    axes[0].set_ylabel("Group-delay scale (fs)")
    axes[0].legend(frameon=False)
    axes[1].semilogx(q, [x["fixed_step_relative_delay_error"] for x in rows], "o-", label="Fixed Δω")
    axes[1].semilogx(q, [x["adaptive_step_relative_delay_error"] for x in rows], "s--", label="Adaptive 0.2γ")
    axes[1].set_xlabel("Verified channel-pole Q")
    axes[1].set_ylabel("Relative delay error")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(output / "gmr_group_delay_and_sampling.png", dpi=220)
    fig.savefig(output / "gmr_group_delay_and_sampling.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
