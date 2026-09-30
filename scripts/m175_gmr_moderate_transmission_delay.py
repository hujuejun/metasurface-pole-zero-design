"""Measure useful transmitted GMR delay at one common nonzero-throughput detuning.

This is a frozen-geometry diagnostic of M166, not an optimization or a new
design search. At each Q, the physical frequency is Re(p_H81)-2*|Im(p_H81)|.
H81 and held-out H101 evaluate the *same physical frequency*. Central phase
differences use local response solves, avoiding a derivative from the M166
0.2-depth spectrum grid. The detuning was chosen post hoc from M166 spectra.
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

from scripts.m166_gmr_high_fidelity_analysis import _solver
from scripts.m164_gmr_pole_adjoint import FS_PER_UM_OVER_C


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_delay(response, frequency: float, step: float) -> float:
    plus = response(frequency + step).transmission
    minus = response(frequency - step).transmission
    if min(abs(plus), abs(minus)) < 1e-5:
        raise RuntimeError("delay step entered a near-zero transmission region")
    return float(np.angle(plus / minus) / (2 * step) * FS_PER_UM_OVER_C)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m175_gmr_moderate_transmission_delay.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/m175_gmr_moderate_transmission_delay_v1"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_path = Path(config["source_summary"])
    source_config_path = Path(config["source_config"])
    cases = json.loads(source_path.read_text(encoding="utf-8"))["cases"]
    geometry_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if [case["quality_target"] for case in cases] != [50, 100, 200, 400, 800, 1600, 3200]:
        raise RuntimeError("source GMR series is incomplete")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=False)
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_base_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "command": sys.argv,
        "inputs_sha256": {str(path): sha256(path) for path in (args.config, source_path, source_config_path)},
        "script_sha256": sha256(Path(__file__)),
        "solver_sha256": sha256(Path("src/pzid/rcwa2d.py")),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    results = []
    for case in cases:
        pole = complex(**case["channel_pole"])
        gamma = abs(pole.imag)
        frequency = pole.real + float(config["operating_detuning_in_h81_pole_depths"]) * gamma
        period = float(case["final_period_um"])
        slab = float(case["slab_thickness_um"])
        ridge = float(case["ridge_height_um"])
        responses = {
            "primary": _solver(geometry_config, slab, ridge, period, int(config["primary_harmonics"])),
            "held_out": _solver(geometry_config, slab, ridge, period, int(config["held_out_harmonics"])),
        }
        step = float(config["primary_delay_step_in_h81_pole_depths"]) * gamma
        fine_step = float(config["step_convergence_delay_step_in_h81_pole_depths"]) * gamma
        evaluations = {}
        for name, response in responses.items():
            solved = response(frequency)
            delay = phase_delay(response, frequency, step)
            evaluations[name] = {
                "harmonics": int(config["primary_harmonics" if name == "primary" else "held_out_harmonics"]),
                "T": float(abs(solved.transmission) ** 2),
                "R": float(abs(solved.reflection) ** 2),
                "delay_fs": delay,
                "energy_error": float(abs(abs(solved.transmission) ** 2 + abs(solved.reflection) ** 2 - 1)),
            }
        fine_delay = phase_delay(responses["primary"], frequency, fine_step)
        primary = evaluations["primary"]
        held_out = evaluations["held_out"]
        row = {
            "target_Q": int(case["quality_target"]),
            "achieved_Q": float(case["achieved_quality_factor"]),
            "operating_frequency_rad_per_um": frequency,
            "detuning_in_h81_pole_depths": float(config["operating_detuning_in_h81_pole_depths"]),
            "pole_depth_delay_scale_fs": float(FS_PER_UM_OVER_C / gamma),
            "primary": primary,
            "held_out": held_out,
            "primary_fine_step_delay_fs": fine_delay,
            "primary_step_relative_difference": float(abs(fine_delay - primary["delay_fs"]) / abs(fine_delay)),
            "held_out_delay_relative_difference": float(abs(held_out["delay_fs"] - primary["delay_fs"]) / abs(held_out["delay_fs"])),
        }
        row["moderate_transmission_pass"] = bool(min(primary["T"], held_out["T"]) >= float(config["moderate_transmission_threshold"]))
        row["numerical_delay_pass"] = bool(
            min(primary["delay_fs"], held_out["delay_fs"]) > 0
            and row["primary_step_relative_difference"] <= float(config["delay_step_relative_tolerance"])
            and row["held_out_delay_relative_difference"] <= float(config["held_out_delay_relative_tolerance"])
            and max(primary["energy_error"], held_out["energy_error"]) < 1e-8
        )
        results.append(row)
        print(
            f"Q={row['achieved_Q']:.1f} T81={primary['T']:.4f} T101={held_out['T']:.4f} "
            f"delay81={primary['delay_fs']:.2f} fs delay101={held_out['delay_fs']:.2f} fs",
            flush=True,
        )
    with (output / "operating_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("target_Q", "achieved_Q", "operating_frequency_rad_per_um", "T_H81", "T_H101",
                         "delay_H81_fs", "delay_H101_fs", "pole_depth_delay_scale_fs", "step_error", "harmonic_delay_error"))
        for row in results:
            writer.writerow((row["target_Q"], row["achieved_Q"], row["operating_frequency_rad_per_um"],
                             row["primary"]["T"], row["held_out"]["T"], row["primary"]["delay_fs"],
                             row["held_out"]["delay_fs"], row["pole_depth_delay_scale_fs"],
                             row["primary_step_relative_difference"], row["held_out_delay_relative_difference"]))
    q = np.asarray([row["achieved_Q"] for row in results])
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True, constrained_layout=True)
    for solver, label, marker in (("primary", "H81", "o"), ("held_out", "H101", "s")):
        axes[0].plot(q, [row[solver]["T"] for row in results], marker=marker, label=label)
        axes[1].plot(q, [row[solver]["delay_fs"] for row in results], marker=marker, label=label)
    axes[0].axhline(float(config["moderate_transmission_threshold"]), ls="--", c="0.4", lw=1, label="T=0.5")
    axes[0].set_ylabel("Transmittance at $\\Delta\\omega=-2\\gamma$")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Verified channel-pole Q")
    axes[1].set_ylabel("Transmitted group delay (fs)")
    axes[1].legend()
    fig.savefig(output / "moderate_transmission_delay.png", dpi=200)
    fig.savefig(output / "moderate_transmission_delay.pdf")
    plt.close(fig)
    summary = {
        "method": "frozen M166 binary geometries, no optimization; common post-hoc -2-depth operating point",
        "cases": results,
        "all_moderate_transmission_pass": all(row["moderate_transmission_pass"] for row in results),
        "all_numerical_delay_pass": all(row["numerical_delay_pass"] for row in results),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not summary["all_moderate_transmission_pass"] or not summary["all_numerical_delay_pass"]:
        raise RuntimeError("one or more moderate-transmission or numerical-delay gates failed")


if __name__ == "__main__":
    main()
