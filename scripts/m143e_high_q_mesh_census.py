"""Frozen physical-geometry axial/lateral mesh census of high-Q M143 designs."""

from __future__ import annotations

import argparse
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


def evaluate_case(
    source_case: dict[str, Any],
    mesh_case: dict[str, Any],
    geometry: dict[str, Any],
    material: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    mesh = dict(geometry)
    mesh["lateral_cells"] = int(mesh_case["lateral_cells"])
    mesh["axial_cell_size_um"] = float(mesh_case["axial_cell_size_um"])
    grid, scalar, _ = build_fixed_grid(mesh, material)
    coordinates = np.asarray(source_case["optimized_coordinates"], dtype=float)
    epsilon, _ = physical_epsilon_and_derivatives(coordinates, mesh, material, scalar)
    reference_pole = complex(**source_case["verification"]["verified_pole"])
    target_pole = complex(**source_case["target_pole"])
    depth = abs(target_pole.imag)

    def response(frequency: complex):
        return solve_symmetric_scattering(frequency, epsilon, scalar).coefficients

    root = refine_scalar_root_complex_secant(
        lambda frequency: response(frequency).transmission,
        reference_pole,
        reference_pole + (0.08 + 0.05j) * depth,
        kind="pole",
        tolerance=float(protocol["root_tolerance"]),
        max_evaluations=int(protocol["root_max_evaluations"]),
        search_radius=max(0.05, 10.0 * depth),
    )
    valid = bool(root.residual_norm <= 1e-6 and root.frequency.imag < 0 and np.isfinite(root.frequency))
    result: dict[str, Any] = {
        "target_quality_factor": float(source_case["target_quality_factor"]),
        "mesh_label": str(mesh_case["label"]),
        "lateral_cells": grid.nx,
        "axial_cell_size_um": grid.dz,
        "grid_nz": grid.nz,
        "physical_coordinates_unchanged": coordinates.tolist(),
        "root_success_flag": bool(root.success),
        "root_residual_norm": float(root.residual_norm),
        "root_evaluations": int(root.evaluations),
        "verified_channel_root": valid,
    }
    if not valid:
        result["root_failure_message"] = str(root)
        return result
    pole = root.frequency
    gamma = abs(pole.imag)
    center = response(pole.real)
    step = float(protocol["delay_step_in_refined_pole_depth"]) * gamma
    transmission_minus = response(pole.real - step).transmission
    transmission_plus = response(pole.real + step).transmission
    delay = np.angle(transmission_plus / transmission_minus) / (2.0 * step) * FS_PER_UM_OVER_C
    pole_shift = abs(pole - reference_pole) / depth
    result.update(
        {
            "refined_pole": {"real": float(pole.real), "imag": float(pole.imag)},
            "refined_quality_factor": float(pole.real / (2.0 * gamma)),
            "pole_shift_from_source_in_target_depths": float(pole_shift),
            "pole_error_from_target_in_target_depths": float(abs(pole - target_pole) / depth),
            "passes_frozen_pole_shift_gate": bool(pole_shift <= float(protocol["pole_shift_gate_in_target_depths"])),
            "transmittance_at_pole_real": float(abs(center.transmission) ** 2),
            "reflectance_at_pole_real": float(abs(center.reflection) ** 2),
            "energy_balance_abs_error": float(abs(abs(center.transmission) ** 2 + abs(center.reflection) ** 2 - 1.0)),
            "group_delay_fs_at_pole_real": float(delay),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m143e_high_q_mesh_census.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/m143e_high_q_mesh_census"))
    arguments = parser.parse_args()
    protocol = json.loads(arguments.config.read_text(encoding="utf-8"))
    geometry_path = Path(protocol["source_fine_adjoint_config"])
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    material_path = Path(geometry["source_analytic_config"])
    material = json.loads(material_path.read_text(encoding="utf-8"))
    source_path = Path(protocol["source_fine_adjoint_run"]) / "summary.json"
    source_summary = json.loads(source_path.read_text(encoding="utf-8"))
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
        "source_summary_sha256": _sha256(source_path),
        "geometry_config_sha256": _sha256(geometry_path),
        "material_config_sha256": _sha256(material_path),
        "command": sys.argv,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    cases: list[dict[str, Any]] = []
    for mesh_case in protocol["mesh_cases"]:
        for quality in protocol["target_quality_factors"]:
            source_case = next(case for case in source_summary["cases"] if case["target_quality_factor"] == quality)
            case = evaluate_case(source_case, mesh_case, geometry, material, protocol)
            cases.append(case)
            (output / "summary.json").write_text(
                json.dumps({"method": "frozen physical geometry mesh root census", "cases": cases}, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"{mesh_case['label']} Qtarget={quality:.0f}: "
                f"Q={case.get('refined_quality_factor', float('nan')):.2f}, "
                f"pole shift={case.get('pole_shift_from_source_in_target_depths', float('nan')):.3f} depths, "
                f"root valid={case['verified_channel_root']}",
                flush=True,
            )
    figure, axis = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    mesh_labels = [item["label"] for item in protocol["mesh_cases"]]
    for quality in protocol["target_quality_factors"]:
        subset = [case for case in cases if case["target_quality_factor"] == quality]
        axis.plot(
            range(len(mesh_labels)),
            [case.get("pole_shift_from_source_in_target_depths", np.nan) for case in subset],
            "o-", label=f"target Q={quality:.0f}",
        )
    axis.axhline(float(protocol["pole_shift_gate_in_target_depths"]), linestyle="--", color="black", linewidth=0.8, label="0.02-depth gate")
    axis.set_xticks(range(len(mesh_labels)), mesh_labels)
    axis.set_ylabel("Frozen-geometry pole shift / target depth")
    axis.set_xlabel("Independent mesh")
    axis.grid(True, alpha=0.3, linewidth=0.5)
    axis.legend(frameon=False)
    figure.savefig(output / "m143e_high_q_mesh_shift.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
