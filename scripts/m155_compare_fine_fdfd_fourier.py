"""Compare saved fine-grid FDFD spectra and independent Fourier-modal data."""

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

from scripts.m153_fourier_modal_reference_check import physical_layers
from pzid.rcwa2d import solve_tm_stack


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> str | None:
    result = subprocess.run(("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def write_curve(path: Path, columns: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns.keys())
        writer.writerows(zip(*columns.values(), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m155_fine_fdfd_fourier_modal_comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/m155_fine_fdfd_fourier_modal_comparison"))
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    fdfd_dir = Path(protocol["source_fdfd_axial_run"])
    joint_dir = Path(protocol["source_fdfd_joint_run"])
    fourier_dir = Path(protocol["source_fourier_run"])
    fdfd_summary_path = fdfd_dir / "summary.json"
    joint_summary_path = joint_dir / "summary.json"
    fourier_summary_path = fourier_dir / "summary.json"
    design_summary_path = Path(protocol["source_design_run"]) / "summary.json"
    geometry_path = Path(protocol["source_geometry_config"])
    material_path = Path(protocol["source_material_config"])
    fdfd_summary = json.loads(fdfd_summary_path.read_text(encoding="utf-8"))
    joint_summary = json.loads(joint_summary_path.read_text(encoding="utf-8"))
    fourier_summary = json.loads(fourier_summary_path.read_text(encoding="utf-8"))
    design_summary = json.loads(design_summary_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    material = json.loads(material_path.read_text(encoding="utf-8"))
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    inputs = [args.config, fdfd_summary_path, joint_summary_path, fourier_summary_path, design_summary_path, geometry_path, material_path]
    cases = []
    figure, axes = plt.subplots(2, 3, figsize=(11.6, 6.3), constrained_layout=True)
    for column, quality in enumerate(protocol["target_quality_factors"]):
        fdfd_case = next(item for item in fdfd_summary["cases"] if item["target_quality_factor"] == quality)
        joint_case = next(item for item in joint_summary["cases"] if item["target_quality_factor"] == quality)
        fourier_case = next(item for item in fourier_summary["cases"] if item["target_quality_factor"] == quality and item["harmonic_half_width"] == protocol["fourier_harmonic_half_width"])
        design_case = next(item for item in design_summary["cases"] if item["target_quality_factor"] == quality)
        fdfd_path = fdfd_dir / f"spectrum_q{int(quality):03d}_dz002nm.csv"
        inputs.append(fdfd_path)
        fdfd_data = np.genfromtxt(fdfd_path, delimiter=",", names=True)
        omega = np.asarray(fdfd_data["omega_rad_per_um"])
        depth = 4.053667940115862 / (2 * float(quality))
        detuning = (omega - 4.053667940115862) / depth
        fdfd_t_full = np.asarray(fdfd_data["t_real"] + 1j * fdfd_data["t_imag"])
        fdfd_t = fdfd_t_full
        fdfd_r_power = np.asarray(fdfd_data["reflectance"])
        layers, length = physical_layers(design_case["optimized_coordinates"], geometry, material, int(protocol["fourier_harmonic_half_width"]))
        responses = [solve_tm_stack(value, layers, period_um=float(geometry["period_um"]), port_index=float(material["port_index"]), harmonic_half_width=int(protocol["fourier_harmonic_half_width"])) for value in omega]
        fourier_t = np.asarray([response.transmission for response in responses]) * np.exp(-1j * omega * float(material["port_index"]) * length)
        fourier_r_power = np.asarray([abs(response.reflection) ** 2 for response in responses])
        mid = len(omega) // 2
        fdfd_phase = np.unwrap(np.angle(fdfd_t / fdfd_t[mid]))
        fourier_phase = np.unwrap(np.angle(fourier_t / fourier_t[mid]))
        joint_pole = complex(**joint_case["refined_pole"])
        fourier_pole = complex(**fourier_case["rcwa_pole"])
        pole_separation = abs(joint_pole - fourier_pole) / depth
        maximum_t_difference = float(np.max(abs(abs(fdfd_t) ** 2 - abs(fourier_t) ** 2)))
        maximum_r_difference = float(np.max(abs(fdfd_r_power - fourier_r_power)))
        case = {
            "target_quality_factor": quality,
            "common_spectrum_points": int(len(omega)),
            "independent_fourier_forward_solves_at_exact_fdfd_frequencies": int(len(omega)),
            "fdfd_joint_nx32_dz2p5nm_pole": {"real": joint_pole.real, "imag": joint_pole.imag},
            "fourier_m17_pole": {"real": fourier_pole.real, "imag": fourier_pole.imag},
            "pole_separation_in_target_depths": float(pole_separation),
            "max_abs_transmittance_difference_fdfd_axial_vs_fourier": maximum_t_difference,
            "max_abs_reflectance_difference_fdfd_axial_vs_fourier": maximum_r_difference,
            "complex_t_rmse_fdfd_axial_vs_fourier": float(np.sqrt(np.mean(abs(fdfd_t - fourier_t) ** 2))),
            "relative_phase_mae_rad_fdfd_axial_vs_fourier": float(np.mean(abs(np.angle((fourier_t / fourier_t[mid]) / (fdfd_t / fdfd_t[mid]))))),
            "fdfd_axial_pole_shift_from_5nm_in_target_depths": fdfd_case["pole_shift_from_optimized_mesh_in_target_depths"],
            "fdfd_joint_pole_shift_from_5nm_in_target_depths": joint_case["pole_shift_from_source_in_target_depths"],
            "passes_preregistered_pole_and_t_gates": bool(pole_separation <= protocol["pole_agreement_gate_in_target_depths"] and maximum_t_difference <= protocol["max_transmittance_difference_gate"]),
        }
        cases.append(case)
        write_curve(output / f"comparison_q{int(quality):03d}.csv", {
            "omega_rad_per_um": omega,
            "detuning_in_target_depths": detuning,
            "fdfd_transmittance": abs(fdfd_t) ** 2,
            "fourier_transmittance": abs(fourier_t) ** 2,
            "fdfd_reflectance": fdfd_r_power,
            "fourier_reflectance": fourier_r_power,
            "fdfd_relative_phase_rad": fdfd_phase,
            "fourier_relative_phase_rad": fourier_phase,
        })
        axes[0, column].plot(detuning, abs(fdfd_t) ** 2, "k-", label="fine FDFD T")
        axes[0, column].plot(detuning, abs(fourier_t) ** 2, "C0--", label="Fourier-modal T")
        axes[0, column].plot(detuning, fdfd_r_power, "k:", label="fine FDFD R")
        axes[0, column].plot(detuning, fourier_r_power, "C1-.", label="Fourier-modal R")
        axes[0, column].set_title(f"Q={quality:.0f}")
        axes[1, column].plot(detuning, fdfd_phase / np.pi, "k-", label="fine FDFD")
        axes[1, column].plot(detuning, fourier_phase / np.pi, "C0--", label="Fourier-modal")
        axes[1, column].set_xlabel("Detuning / target pole depth")
        if column == 0:
            axes[0, column].set_ylabel("Power")
            axes[1, column].set_ylabel("Relative t phase / pi")
        for axis in (axes[0, column], axes[1, column]):
            axis.grid(True, alpha=0.3, linewidth=0.5)
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[1, 0].legend(frameon=False, fontsize=8)
    figure.savefig(output / "m155_fine_fdfd_fourier_rt_phase.png", dpi=200)
    plt.close(figure)
    summary = {"method": "saved-data fine FDFD vs independently converged Fourier-modal comparison", "cases": cases}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": {str(path): sha256(path) for path in inputs},
        "command": sys.argv,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for case in cases:
        print(
            f"Q={case['target_quality_factor']:.0f}: pole separation="
            f"{case['pole_separation_in_target_depths']:.6f} depths, "
            f"max |dT|={case['max_abs_transmittance_difference_fdfd_axial_vs_fourier']:.6f}, "
            f"phase MAE={case['relative_phase_mae_rad_fdfd_axial_vs_fourier']:.6f} rad, "
            f"gates={case['passes_preregistered_pole_and_t_gates']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
