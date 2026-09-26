"""M159B: fine-mesh direct adaptive controls at exact pole-run starts."""

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
from scipy.optimize import least_squares

from pzid.twoport import solve_two_port_scattering

from scripts.m142e_optimize_patterned_q_poles import _verify_final, build_fixed_grid
from scripts.m142f_compare_direct_transmission import check_direct_adjoint, direct_complex_transmission_system


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m159b_fine_mesh_matched_start_direct.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/m159b_fine_mesh_matched_start_direct"))
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    geometry_path = Path(protocol["source_pole_adjoint_config"])
    pole_run = Path(protocol["source_pole_adjoint_run"])
    pole_summary_path = pole_run / "summary.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    material_path = Path(geometry["source_analytic_config"])
    material = json.loads(material_path.read_text(encoding="utf-8"))
    pole_summary = json.loads(pole_summary_path.read_text(encoding="utf-8"))
    grid, scalar, twoport = build_fixed_grid(geometry, material)
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
        "grid_nx": grid.nx,
        "grid_nz": grid.nz,
        "input_sha256": {str(path): sha256(path) for path in (args.config, geometry_path, material_path, pole_summary_path)},
        "script_sha256": sha256(Path(__file__)),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    bounds = (
        np.asarray((geometry["mirror_thickness_scale_bounds"][0], geometry["cavity_thickness_scale_bounds"][0])),
        np.asarray((geometry["mirror_thickness_scale_bounds"][1], geometry["cavity_thickness_scale_bounds"][1])),
    )
    rows = []
    for quality in protocol["target_quality_factors"]:
        pole_case = next(item for item in pole_summary["cases"] if item["target_quality_factor"] == quality)
        start = np.asarray(pole_case[protocol["start_coordinate_field"]], dtype=float)
        target_pole = complex(**pole_case["target_pole"])
        frequencies = target_pole.real + abs(target_pole.imag) * np.asarray(protocol["real_frequency_offsets_in_target_depths"], dtype=float)
        epsilon = np.load(pole_run / f"permittivity_q{int(quality):03d}.npy")
        targets = np.asarray([solve_two_port_scattering(float(omega), epsilon, twoport).matrix[1, 0] for omega in frequencies])
        if not rows:
            audit = check_direct_adjoint(
                start, frequencies, targets, geometry, material, scalar, twoport,
                float(protocol["finite_difference_check_step"]),
            )
            (output / "preflight.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            if max(audit["relative_errors"]) > float(protocol["finite_difference_check_tolerance"]):
                raise RuntimeError(f"direct-adjoint Jacobian gate failed: {audit}")
            print(f"matched-start direct gradient preflight: {max(audit['relative_errors']):.2e}", flush=True)
        cached_x = None
        cached_result = None
        calls = 0

        def system(values: np.ndarray):
            nonlocal cached_x, cached_result, calls
            if cached_x is None or not np.array_equal(values, cached_x):
                cached_result = direct_complex_transmission_system(
                    values, frequencies, targets, geometry, material, scalar, twoport
                )
                cached_x = np.asarray(values).copy()
                calls += 1
            return cached_result

        started = time.perf_counter()
        result = least_squares(
            lambda values: system(values)[0], start,
            jac=lambda values: system(values)[1], bounds=bounds,
            method="trf", max_nfev=int(protocol["max_function_evaluations"]),
            xtol=1e-11, ftol=1e-11, gtol=1e-11,
        )
        wall = time.perf_counter() - started
        verification, curves, final_epsilon = _verify_final(result.x, target_pole, geometry, material, scalar)
        np.save(output / f"permittivity_q{int(quality):03d}.npy", final_epsilon)
        np.savetxt(
            output / f"spectrum_q{int(quality):03d}.csv",
            np.column_stack(list(curves.values())), delimiter=",",
            header=",".join(curves.keys()), comments="",
        )
        row = {
            "target_quality_factor": quality,
            "start_used": start.tolist(),
            "pole_run_start_used": pole_case["start_used"],
            "start_coordinates_exactly_match": bool(np.array_equal(start, np.asarray(pole_case["start_used"]))),
            "optimized_coordinates": result.x.tolist(),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "unique_adjoint_evaluations": calls,
            "frequency_factorizations": calls * len(frequencies),
            "optimization_wall_seconds": wall,
            "direct_sample_count": len(frequencies),
            "real_residual_count": 2 * len(frequencies),
            "final_direct_residual_norm": float(np.linalg.norm(result.fun)),
            "verification": verification,
            "reference_epsilon_sha256": sha256(pole_run / f"permittivity_q{int(quality):03d}.npy"),
        }
        rows.append(row)
        (output / "summary.json").write_text(json.dumps({"method": "matched-start fine-grid adaptive direct complex-t adjoint", "cases": rows}, indent=2) + "\n", encoding="utf-8")
        print(f"Q={quality:.0f}: direct pole error={verification['pole_error_in_target_depths']:.4g} depths, factorizations={row['frequency_factorizations']}, wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
