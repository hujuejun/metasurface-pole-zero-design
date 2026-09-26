"""M158B: frozen-mesh and independent Fourier checks of fine Q400 design."""

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
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from pzid.rcwa2d import solve_tm_stack
from pzid.roots import refine_scalar_root_complex_secant
from pzid.scattering import solve_symmetric_scattering

from scripts.m142e_optimize_patterned_q_poles import build_fixed_grid, physical_epsilon_and_derivatives
from scripts.m153_fourier_modal_reference_check import physical_layers


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def fdfd_root(
    coordinates: np.ndarray,
    config: dict[str, Any],
    material: dict[str, Any],
    start: complex,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    grid, scalar, _ = build_fixed_grid(config, material)
    epsilon, _ = physical_epsilon_and_derivatives(coordinates, config, material, scalar)

    def transmission(omega: complex) -> complex:
        return solve_symmetric_scattering(omega, epsilon, scalar).coefficients.transmission

    root = refine_scalar_root_complex_secant(
        transmission,
        start,
        start + (0.08 + 0.05j) * abs(start.imag),
        kind="pole",
        tolerance=float(protocol["root_tolerance"]),
        max_evaluations=int(protocol["root_max_evaluations"]),
        search_radius=max(0.1, 10 * abs(start.imag)),
    )
    return {
        "lateral_cells": grid.nx,
        "axial_cells": grid.nz,
        "axial_cell_size_um": grid.dz,
        "physical_coordinates_unchanged": coordinates.tolist(),
        "pole": encode(root.frequency),
        "root_residual": root.residual_norm,
        "root_success_flag": bool(root.success),
        "verified_channel_pole": bool(np.isfinite(root.frequency) and root.frequency.imag < 0 and root.residual_norm <= float(protocol["root_accept_residual"])),
        "root_evaluations": root.evaluations,
    }


def fourier_root_and_curve(
    coordinates: np.ndarray,
    geometry: dict[str, Any],
    material: dict[str, Any],
    harmonics: int,
    start: complex,
    frequencies: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    layers, length = physical_layers(coordinates.tolist(), geometry, material, harmonics)

    def response(omega: complex):
        return solve_tm_stack(
            omega, layers, period_um=float(geometry["period_um"]),
            port_index=float(material["port_index"]), harmonic_half_width=harmonics,
        )

    root = refine_scalar_root_complex_secant(
        lambda omega: response(omega).transmission,
        start,
        start + (0.08 + 0.05j) * abs(start.imag),
        kind="pole",
        tolerance=float(protocol["root_tolerance"]),
        max_evaluations=int(protocol["root_max_evaluations"]),
        search_radius=max(0.1, 10 * abs(start.imag)),
    )
    solved = [response(value) for value in frequencies]
    t = np.asarray([item.transmission for item in solved]) * np.exp(-1j * frequencies * float(material["port_index"]) * length)
    r = np.asarray([item.reflection for item in solved])
    record = {
        "harmonic_half_width": harmonics,
        "total_harmonics": 2 * harmonics + 1,
        "physical_stack_length_um": length,
        "pole": encode(root.frequency),
        "root_residual": root.residual_norm,
        "root_success_flag": bool(root.success),
        "verified_channel_pole": bool(np.isfinite(root.frequency) and root.frequency.imag < 0 and root.residual_norm <= float(protocol["root_accept_residual"])),
        "root_evaluations": root.evaluations,
        "energy_balance_max_abs_error": float(np.max(abs(abs(t) ** 2 + abs(r) ** 2 - 1))),
    }
    return record, t, r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m158_q400_independent_gates.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/m158_q400_independent_gates"))
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    source_config_path = Path(protocol["source_config"])
    material_path = Path(protocol["source_material_config"])
    source_run = Path(protocol["source_run"])
    summary_path = source_run / "summary.json"
    spectrum_path = source_run / "spectrum_q400.csv"
    geometry = json.loads(source_config_path.read_text(encoding="utf-8"))
    material = json.loads(material_path.read_text(encoding="utf-8"))
    source = json.loads(summary_path.read_text(encoding="utf-8"))
    case = next(item for item in source["cases"] if item["target_quality_factor"] == protocol["target_quality_factor"])
    if case["verification"]["pole_error_in_target_depths"] > 0.02:
        raise RuntimeError("source Q400 adjoint design missed the prescribed pole gate")
    coordinates = np.asarray(case["optimized_coordinates"], dtype=float)
    baseline_pole = complex(**case["verification"]["verified_pole"])
    target_depth = abs(complex(**case["target_pole"]).imag)
    spectrum = np.genfromtxt(spectrum_path, names=True, delimiter=",")
    omega = np.asarray(spectrum["omega_rad_per_um"], dtype=float)
    fdfd_t = np.asarray(spectrum["t_real"] + 1j * spectrum["t_imag"])
    fdfd_r_power = np.asarray(spectrum["reflectance"])
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    revision = subprocess.run(("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=False)
    metadata = {
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "command": sys.argv,
        "input_sha256": {str(path): hash_file(path) for path in (args.config, source_config_path, material_path, summary_path, spectrum_path)},
        "code_sha256": {str(path): hash_file(path) for path in (Path(__file__), Path("src/pzid/rcwa2d.py"), Path("src/pzid/scattering.py"))},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    fdfd_cases = []
    for nx in protocol["frozen_lateral_cells_at_2p5nm"]:
        mesh_config = dict(geometry)
        mesh_config["lateral_cells"] = int(nx)
        result = fdfd_root(coordinates, mesh_config, material, baseline_pole, protocol)
        result["pole_motion_from_optimized_mesh_in_target_depths"] = float(abs(complex(**result["pole"]) - baseline_pole) / target_depth)
        fdfd_cases.append(result)
        print(f"frozen Nx{nx}: shift={result['pole_motion_from_optimized_mesh_in_target_depths']:.5f} depths, verified={result['verified_channel_pole']}", flush=True)
    fourier_cases = []
    t_final = r_final = None
    for order in protocol["fourier_harmonic_half_widths"]:
        result, t, r = fourier_root_and_curve(coordinates, geometry, material, int(order), baseline_pole, omega, protocol)
        result["pole_motion_from_optimized_fdfd_in_target_depths"] = float(abs(complex(**result["pole"]) - baseline_pole) / target_depth)
        fourier_cases.append(result)
        t_final, r_final = t, r
        print(f"Fourier M={order}: shift={result['pole_motion_from_optimized_fdfd_in_target_depths']:.5f} depths, verified={result['verified_channel_pole']}", flush=True)
    assert t_final is not None and r_final is not None
    t_difference = float(np.max(abs(abs(t_final) ** 2 - abs(fdfd_t) ** 2)))
    r_difference = float(np.max(abs(abs(r_final) ** 2 - fdfd_r_power)))
    relative_phase = np.unwrap(np.angle(t_final / t_final[len(t_final)//2]))
    fdfd_phase = np.unwrap(np.angle(fdfd_t / fdfd_t[len(fdfd_t)//2]))
    phase_mae = float(np.mean(abs(relative_phase - fdfd_phase)))
    harmonic_motion = float(abs(complex(**fourier_cases[-1]["pole"]) - complex(**fourier_cases[-2]["pole"])) / target_depth)
    gates = {
        "source_adjoint_pole": bool(case["verification"]["pole_error_in_target_depths"] <= 0.02),
        "fdfd_roots_verified": all(item["verified_channel_pole"] for item in fdfd_cases),
        "fourier_roots_verified": all(item["verified_channel_pole"] for item in fourier_cases),
        "fdfd_lateral_motion": bool(max(item["pole_motion_from_optimized_mesh_in_target_depths"] for item in fdfd_cases) <= float(protocol["fdfd_lateral_pole_motion_gate_in_target_depths"])),
        "fourier_order_motion": bool(harmonic_motion <= float(protocol["fourier_order_motion_gate_in_target_depths"])),
        "fdfd_fourier_pole_agreement": bool(fourier_cases[-1]["pole_motion_from_optimized_fdfd_in_target_depths"] <= float(protocol["fdfd_fourier_pole_gate_in_target_depths"])),
        "same_frequency_transmittance": bool(t_difference <= float(protocol["same_frequency_max_abs_transmittance_difference_gate"])),
    }
    result = {
        "physical_coordinates_unchanged": coordinates.tolist(),
        "target_quality_factor": 400,
        "target_depth": target_depth,
        "fdfd_cases": fdfd_cases,
        "fourier_cases": fourier_cases,
        "fourier_order_pole_motion_in_target_depths": harmonic_motion,
        "same_frequency_max_abs_transmittance_difference": t_difference,
        "same_frequency_max_abs_reflectance_difference": r_difference,
        "same_frequency_relative_phase_mae_rad": phase_mae,
        "gates": gates,
        "publication_gate_passed": all(gates.values()),
        "wall_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savetxt(
        output / "same_frequency_rt_phase.csv",
        np.column_stack((omega, (omega - baseline_pole.real) / target_depth, abs(fdfd_t) ** 2, fdfd_r_power, fdfd_phase, abs(t_final) ** 2, abs(r_final) ** 2, relative_phase)),
        delimiter=",",
        header="omega_rad_per_um,detuning_in_target_depths,fdfd_T,fdfd_R,fdfd_relative_phase_rad,fourier_T,fourier_R,fourier_relative_phase_rad",
        comments="",
    )
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.6), sharex=True, constrained_layout=True)
    x = (omega - baseline_pole.real) / target_depth
    axes[0].plot(x, abs(fdfd_t) ** 2, "k-", label="FDFD T, Nx32 / 2.5 nm")
    axes[0].plot(x, abs(t_final) ** 2, "C0--", label="Fourier-modal T, M17")
    axes[0].plot(x, fdfd_r_power, "k:", label="FDFD R")
    axes[0].plot(x, abs(r_final) ** 2, "C1--", label="Fourier-modal R")
    axes[0].set_ylabel("Power")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(x, fdfd_phase, "k-", label="FDFD")
    axes[1].plot(x, relative_phase, "C0--", label="Fourier-modal")
    axes[1].set_ylabel("Relative transmission phase (rad)")
    axes[1].set_xlabel("Detuning / target pole depth")
    for axis in axes:
        axis.grid(True, alpha=0.25, linewidth=0.5)
    fig.savefig(output / "m158_q400_same_frequency_cross_solver.png", dpi=240)
    fig.savefig(output / "m158_q400_same_frequency_cross_solver.pdf")
    plt.close(fig)
    print(f"M158B publication gate: {all(gates.values())}; max |delta T|={t_difference:.5f}; harmonic motion={harmonic_motion:.5f} depths", flush=True)


if __name__ == "__main__":
    main()
