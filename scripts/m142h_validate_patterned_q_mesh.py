"""M142H: independent fine-mesh validation of physical M142E Q designs."""

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
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from pzid.roots import refine_scalar_root_complex_secant
from pzid.scattering import solve_symmetric_scattering

from scripts.m142_single_transmission_q_screen import FS_PER_UM_OVER_C
from scripts.m142e_optimize_patterned_q_poles import build_fixed_grid, physical_epsilon_and_derivatives


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str | None:
    result = subprocess.run(("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _write_curve(path: Path, columns: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns.keys())
        writer.writerows(zip(*columns.values(), strict=True))


def evaluate_lift(
    source_case: dict[str, Any],
    dz_um: float,
    protocol: dict[str, Any],
    geometry_config: dict[str, Any],
    source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    mesh_config = dict(geometry_config)
    mesh_config["axial_cell_size_um"] = dz_um
    grid, scalar, _ = build_fixed_grid(mesh_config, source)
    coordinates = np.asarray(source_case["optimized_coordinates"], dtype=float)
    epsilon, _ = physical_epsilon_and_derivatives(
        coordinates, mesh_config, source, scalar
    )
    coarse_pole = complex(
        source_case["verification"]["verified_pole"]["real"],
        source_case["verification"]["verified_pole"]["imag"],
    )
    target = complex(source_case["target_pole"]["real"], source_case["target_pole"]["imag"])

    def coefficients(frequency: complex):
        return solve_symmetric_scattering(frequency, epsilon, scalar).coefficients

    root = refine_scalar_root_complex_secant(
        lambda frequency: coefficients(frequency).transmission,
        coarse_pole,
        coarse_pole + (0.08 + 0.05j) * abs(coarse_pole.imag),
        kind="pole",
        tolerance=float(protocol["root_tolerance"]),
        max_evaluations=int(protocol["root_max_evaluations"]),
        search_radius=max(0.1, 10.0 * abs(coarse_pole.imag)),
    )
    if not root.success or root.residual_norm > 1e-6 or root.frequency.imag >= 0.0:
        raise RuntimeError(f"finer-mesh channel root failed: Q={source_case['target_quality_factor']}, dz={dz_um}: {root}")
    pole = root.frequency
    gamma = abs(pole.imag)
    detuning = np.linspace(
        -float(protocol["local_spectrum_half_widths_in_refined_pole_depth"]),
        float(protocol["local_spectrum_half_widths_in_refined_pole_depth"]),
        int(protocol["local_spectrum_points"]),
    )
    omega = pole.real + gamma * detuning
    responses = [coefficients(frequency) for frequency in omega]
    t = np.asarray([response.transmission for response in responses])
    r = np.asarray([response.reflection for response in responses])
    phase = np.unwrap(np.angle(t))
    step = float(protocol["delay_finite_difference_step_in_refined_pole_depth"]) * gamma
    t_minus = coefficients(pole.real - step).transmission
    t_plus = coefficients(pole.real + step).transmission
    delay = np.angle(t_plus / t_minus) / (2.0 * step) * FS_PER_UM_OVER_C
    middle = len(omega) // 2
    metrics = {
        "target_quality_factor": source_case["target_quality_factor"],
        "physical_coordinates_unchanged": coordinates.tolist(),
        "dz_um": dz_um,
        "grid_nx": grid.nx,
        "grid_nz": grid.nz,
        "refined_pole": {"real": pole.real, "imag": pole.imag},
        "refined_quality_factor": pole.real / (2.0 * gamma),
        "pole_shift_from_optimized_mesh_in_target_depths": abs(pole - coarse_pole) / abs(target.imag),
        "pole_error_from_target_in_target_depths": abs(pole - target) / abs(target.imag),
        "root_residual_norm": root.residual_norm,
        "root_evaluations": root.evaluations,
        "transmittance_at_pole_real": float(abs(t[middle]) ** 2),
        "reflectance_at_pole_real": float(abs(r[middle]) ** 2),
        "local_peak_transmittance": float(np.max(abs(t) ** 2)),
        "energy_balance_max_abs_error": float(np.max(abs(abs(t) ** 2 + abs(r) ** 2 - 1.0))),
        "group_delay_fs_at_pole_real": float(delay),
        "delay_times_pole_depth_dimensionless": float(delay / FS_PER_UM_OVER_C * gamma),
    }
    curves = {
        "omega_rad_per_um": omega,
        "detuning_in_refined_pole_depths": detuning,
        "t_real": t.real,
        "t_imag": t.imag,
        "transmittance": abs(t) ** 2,
        "reflectance": abs(r) ** 2,
        "transmission_phase_rad_unwrapped": phase,
    }
    return metrics, curves, epsilon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m142h_patterned_q_mesh_lift.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/m142h_patterned_q_mesh_lift"))
    arguments = parser.parse_args()
    protocol = json.loads(arguments.config.read_text(encoding="utf-8"))
    geometry_path = Path(protocol["source_pole_adjoint_config"])
    geometry_config = json.loads(geometry_path.read_text(encoding="utf-8"))
    source_path = Path(geometry_config["source_analytic_config"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_run = Path(protocol["source_pole_adjoint_run"])
    source_summary_path = source_run / "summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    output = arguments.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "config_sha256": _sha256(arguments.config),
        "script_sha256": _sha256(Path(__file__)),
        "source_summary_sha256": _sha256(source_summary_path),
        "source_geometry_config_sha256": _sha256(geometry_path),
        "command": sys.argv,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    cases: list[dict[str, Any]] = []
    for target_quality in protocol["target_quality_factors"]:
        source_case = next(
            item for item in source_summary["cases"]
            if item["target_quality_factor"] == target_quality
        )
        for dz_um in protocol["axial_cell_sizes_um"]:
            metrics, curves, epsilon = evaluate_lift(
                source_case, float(dz_um), protocol, geometry_config, source
            )
            cases.append(metrics)
            suffix = f"q{int(target_quality):03d}_dz{int(round(1000 * dz_um)):03d}nm"
            _write_curve(output / f"spectrum_{suffix}.csv", curves)
            np.save(output / f"permittivity_{suffix}.npy", epsilon)
            (output / "summary.json").write_text(
                json.dumps({"method": "fixed physical geometry mesh lift, no optimization", "cases": cases}, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"Qtarget={target_quality:.0f} dz={dz_um:.3f} um: "
                f"Q={metrics['refined_quality_factor']:.2f}, "
                f"target pole error={metrics['pole_error_from_target_in_target_depths']:.3f} depths, "
                f"Tpeak={metrics['transmittance_at_pole_real']:.6f}",
                flush=True,
            )
    figure, axes = plt.subplots(1, 2, figsize=(9.1, 3.7), constrained_layout=True)
    for quality in protocol["target_quality_factors"]:
        source_case = next(item for item in source_summary["cases"] if item["target_quality_factor"] == quality)
        subset = [item for item in cases if item["target_quality_factor"] == quality]
        dx = [float(geometry_config["axial_cell_size_um"]), *[item["dz_um"] for item in subset]]
        qs = [source_case["verification"]["verified_quality_factor"], *[item["refined_quality_factor"] for item in subset]]
        errors = [source_case["verification"]["pole_error_in_target_depths"], *[item["pole_error_from_target_in_target_depths"] for item in subset]]
        axes[0].plot(dx, qs, "o-", label=f"target Q={quality:.0f}")
        axes[1].plot(dx, errors, "o-", label=f"target Q={quality:.0f}")
    axes[0].set_ylabel("Independently verified Q")
    axes[1].set_ylabel("Pole error / target depth")
    for axis in axes:
        axis.set_xlabel("Axial cell size (um)")
        axis.grid(True, alpha=0.3, linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(output / "m142h_fixed_geometry_mesh_trend.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
