"""M142E: optimize prescribed transmission-channel pole Qs by 2D adjoints."""

from __future__ import annotations

import argparse
import csv
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
from scipy.optimize import least_squares

from pzid.adjoint import scattering_matrix_with_jacobian
from pzid.grid import UniformGrid2D
from pzid.optimize import complex_system_to_real
from pzid.roots import refine_scalar_root_complex_secant
from pzid.scattering import SymmetricScatteringSetup, solve_symmetric_scattering
from pzid.twoport import TwoPortScatteringSetup, solve_two_port_scattering

from scripts.m142_single_transmission_q_screen import FS_PER_UM_OVER_C
from scripts.m142b_validate_q_ladder_fdfd import make_problem, rasterize_layers
from scripts.m142c_screen_patterned_cavity import add_centered_cavity_ridge
from scripts.m142d_fixed_topology_analytic_targets import scaled_cavity_layers


def build_fixed_grid(
    config: dict[str, Any], source: dict[str, Any]
) -> tuple[UniformGrid2D, SymmetricScatteringSetup, TwoPortScatteringSetup]:
    """Keep one Maxwell mesh for every geometry and every Q target."""

    max_layers = scaled_cavity_layers(
        int(config["fixed_mirror_pairs_per_side"]),
        float(source["wavelength_um"]),
        float(source["high_index"]),
        float(source["low_index"]),
        float(config["mirror_thickness_scale_bounds"][1]),
        float(config["cavity_thickness_scale_bounds"][1]),
    )
    grid, _, scalar = make_problem(
        max_layers,
        period_um=float(config["period_um"]),
        lateral_cells=int(config["lateral_cells"]),
        dz_um=float(config["axial_cell_size_um"]),
        port_index=float(source["port_index"]),
        padding_um=float(config["port_padding_each_side_um"]),
    )
    twoport = TwoPortScatteringSetup(
        grid=grid,
        lower_source_plane=scalar.source_plane,
        upper_source_plane=grid.nz - 1 - scalar.source_plane,
        lower_monitor_planes=scalar.input_monitor_planes,
        upper_monitor_planes=scalar.output_monitor_planes,
        port_index=scalar.port_index,
    )
    return grid, scalar, twoport


def physical_epsilon_and_derivatives(
    coordinates: np.ndarray,
    config: dict[str, Any],
    source: dict[str, Any],
    scalar: SymmetricScatteringSetup,
) -> tuple[np.ndarray, np.ndarray]:
    """Area-averaged epsilon and exact coordinate derivatives off grid edges.

    A boundary at b contributes (epsilon_left-epsilon_right) db/dp / dz to
    the cell that contains it. The fixed ridge stays centered as the symmetric
    cavity expands or contracts, so its derivative with respect to either
    thickness coordinate is identically zero.
    """

    values = np.asarray(coordinates, dtype=float)
    if values.shape != (2,):
        raise ValueError("two physical thickness coordinates are required")
    grid = scalar.grid
    pairs = int(config["fixed_mirror_pairs_per_side"])
    reference = scaled_cavity_layers(
        pairs,
        float(source["wavelength_um"]),
        float(source["high_index"]),
        float(source["low_index"]),
        1.0,
        1.0,
    )
    layers = scaled_cavity_layers(
        pairs,
        float(source["wavelength_um"]),
        float(source["high_index"]),
        float(source["low_index"]),
        float(values[0]),
        float(values[1]),
    )
    thickness = np.asarray([d for _, d in layers], dtype=float)
    reference_thickness = np.asarray([d for _, d in reference], dtype=float)
    derivative_thickness = np.zeros((2, len(layers)), dtype=float)
    derivative_thickness[0] = reference_thickness
    derivative_thickness[0, 2 * pairs] = 0.0
    derivative_thickness[1, 2 * pairs] = reference_thickness[2 * pairs]
    total_length = float(np.sum(thickness))
    base = 0.5 * (grid.height - total_length)
    if base <= 0.45:
        raise ValueError("stack approaches a source or monitor plane")
    epsilon = rasterize_layers(
        layers, grid, float(source["port_index"]), base_z_um=base
    )
    epsilon = add_centered_cavity_ridge(
        epsilon,
        scalar,
        layers,
        pairs,
        float(source["high_index"]),
        float(source["low_index"]),
        fill_factor=float(config["ridge_fill_factor"]),
        height_um=float(config["ridge_height_um"]),
    )
    boundaries = base + np.concatenate(([0.0], np.cumsum(thickness)))
    boundary_velocities = (
        -0.5 * np.sum(derivative_thickness, axis=1)[:, None]
        + np.concatenate(
            (np.zeros((2, 1)), np.cumsum(derivative_thickness, axis=1)), axis=1
        )
    )
    derivatives = np.zeros((2, *grid.shape), dtype=float)
    n_port = float(source["port_index"])
    indices = [n_port, *[n for n, _ in layers], n_port]
    for boundary_index, location in enumerate(boundaries):
        row = int(np.searchsorted(grid.z_edges, location, side="right") - 1)
        if not 0 <= row < grid.nz:
            raise ValueError("material boundary left FDFD grid")
        contrast = indices[boundary_index] ** 2 - indices[boundary_index + 1] ** 2
        derivatives[:, row, :] += (
            contrast * boundary_velocities[:, boundary_index] / grid.dz
        )[:, None]
    return epsilon, derivatives


def select_pole_parity(
    target: complex,
    feasible_coordinates: np.ndarray,
    config: dict[str, Any],
    source: dict[str, Any],
    scalar: SymmetricScatteringSetup,
    twoport: TwoPortScatteringSetup,
) -> tuple[int, dict[str, float]]:
    epsilon, _ = physical_epsilon_and_derivatives(
        feasible_coordinates, config, source, scalar
    )
    matrix = solve_two_port_scattering(np.conj(target), epsilon, twoport).matrix
    even_norm = float(np.linalg.norm(matrix @ np.asarray((1.0, 1.0))))
    odd_norm = float(np.linalg.norm(matrix @ np.asarray((1.0, -1.0))))
    parity = 1 if even_norm < odd_norm else -1
    return parity, {"even_null_norm": even_norm, "odd_null_norm": odd_norm}


def pole_adjoint_system(
    coordinates: np.ndarray,
    target: complex,
    parity: int,
    config: dict[str, Any],
    source: dict[str, Any],
    scalar: SymmetricScatteringSetup,
    twoport: TwoPortScatteringSetup,
) -> tuple[np.ndarray, np.ndarray]:
    epsilon, d_epsilon = physical_epsilon_and_derivatives(
        coordinates, config, source, scalar
    )
    evaluation = scattering_matrix_with_jacobian(np.conj(target), epsilon, twoport)
    incoming = np.asarray((1.0, float(parity)), dtype=complex)
    residual = np.asarray(evaluation.matrix @ incoming)
    coordinate_jacobian = np.einsum(
        "abzx,pzx,b->ap", evaluation.jacobian, d_epsilon, incoming
    )
    return complex_system_to_real(residual, coordinate_jacobian)


def check_coordinate_jacobian(
    coordinates: np.ndarray,
    target: complex,
    parity: int,
    config: dict[str, Any],
    source: dict[str, Any],
    scalar: SymmetricScatteringSetup,
    twoport: TwoPortScatteringSetup,
) -> dict[str, float]:
    step = float(config["finite_difference_check_step"])
    epsilon, derivative = physical_epsilon_and_derivatives(coordinates, config, source, scalar)
    mapped_errors: list[float] = []
    for index in range(2):
        displacement = np.eye(2)[index] * step
        plus, _ = physical_epsilon_and_derivatives(coordinates + displacement, config, source, scalar)
        minus, _ = physical_epsilon_and_derivatives(coordinates - displacement, config, source, scalar)
        numeric = (plus - minus) / (2.0 * step)
        mapped_errors.append(float(np.linalg.norm(numeric - derivative[index]) / max(np.linalg.norm(numeric), 1.0)))
    residual, jacobian = pole_adjoint_system(
        coordinates, target, parity, config, source, scalar, twoport
    )
    system_errors: list[float] = []
    for index in range(2):
        displacement = np.eye(2)[index] * step
        plus, _ = pole_adjoint_system(
            coordinates + displacement, target, parity, config, source, scalar, twoport
        )
        minus, _ = pole_adjoint_system(
            coordinates - displacement, target, parity, config, source, scalar, twoport
        )
        numeric = (plus - minus) / (2.0 * step)
        system_errors.append(float(np.linalg.norm(numeric - jacobian[:, index]) / max(np.linalg.norm(numeric), 1.0)))
    return {
        "epsilon_mapping_relative_errors": mapped_errors,
        "pole_adjoint_relative_errors": system_errors,
        "initial_pole_residual_norm": float(np.linalg.norm(residual)),
        "epsilon_min": float(epsilon.min()),
        "epsilon_max": float(epsilon.max()),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str | None:
    result = subprocess.run(("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _verify_final(
    coordinates: np.ndarray,
    target: complex,
    config: dict[str, Any],
    source: dict[str, Any],
    scalar: SymmetricScatteringSetup,
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    epsilon, _ = physical_epsilon_and_derivatives(coordinates, config, source, scalar)
    transmission = lambda frequency: solve_symmetric_scattering(frequency, epsilon, scalar).coefficients.transmission
    root = refine_scalar_root_complex_secant(
        transmission, target, target + (0.08 + 0.05j) * abs(target.imag),
        kind="pole", tolerance=1e-8, max_evaluations=30,
        search_radius=max(0.1, 10.0 * abs(target.imag)),
    )
    if not root.success or root.residual_norm > 1e-6 or root.frequency.imag >= 0.0:
        raise RuntimeError(f"independent transmission-channel pole verification failed: {root}")
    half_width = float(config["local_spectrum_half_widths_in_target_depth"]) * abs(target.imag)
    omega = np.linspace(target.real - half_width, target.real + half_width, int(config["local_spectrum_points"]))
    solved = [solve_symmetric_scattering(value, epsilon, scalar).coefficients for value in omega]
    t = np.asarray([entry.transmission for entry in solved])
    r = np.asarray([entry.reflection for entry in solved])
    center = int(np.argmin(abs(omega - root.frequency.real)))
    step = 0.02 * abs(root.frequency.imag)
    t_minus = transmission(root.frequency.real - step)
    t_plus = transmission(root.frequency.real + step)
    delay_fs = np.angle(t_plus / t_minus) / (2.0 * step) * FS_PER_UM_OVER_C
    metrics = {
        "verified_pole": {"real": root.frequency.real, "imag": root.frequency.imag},
        "verified_quality_factor": root.frequency.real / (-2.0 * root.frequency.imag),
        "pole_error_abs_rad_per_um": abs(root.frequency - target),
        "pole_error_in_target_depths": abs(root.frequency - target) / abs(target.imag),
        "root_residual_norm": root.residual_norm,
        "root_evaluations": root.evaluations,
        "near_pole_real_transmittance": float(abs(t[center]) ** 2),
        "near_pole_real_reflectance": float(abs(r[center]) ** 2),
        "local_peak_transmittance": float(np.max(abs(t) ** 2)),
        "local_energy_balance_max_abs_error": float(np.max(abs(abs(t) ** 2 + abs(r) ** 2 - 1.0))),
        "group_delay_fs_at_pole_real": float(delay_fs),
        "delay_times_pole_depth_dimensionless": float(delay_fs / FS_PER_UM_OVER_C * abs(root.frequency.imag)),
    }
    curves = {
        "omega_rad_per_um": omega,
        "detuning_in_target_depths": (omega - target.real) / abs(target.imag),
        "t_real": t.real,
        "t_imag": t.imag,
        "transmittance": abs(t) ** 2,
        "reflectance": abs(r) ** 2,
        "transmission_phase_rad_unwrapped": np.unwrap(np.angle(t)),
    }
    return metrics, curves, epsilon


def _write_curve(path: Path, curves: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(curves.keys())
        writer.writerows(zip(*curves.values(), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m142e_patterned_q_pole_adjoint.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/m142e_patterned_q_pole_adjoint"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--start-from-summary", type=Path)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    source_path = Path(config["source_analytic_config"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    feasible_path = Path(config["source_feasible_targets"])
    feasible = json.loads(feasible_path.read_text(encoding="utf-8"))
    grid, scalar, twoport = build_fixed_grid(config, source)
    if grid.period >= float(source["wavelength_um"]) / float(source["port_index"]):
        raise ValueError("exterior ports must remain specular-only")
    output = arguments.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
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
        "source_analytic_config_sha256": _sha256(source_path),
        "source_feasible_targets_sha256": _sha256(feasible_path),
        "command": sys.argv,
        "grid_nx": grid.nx,
        "grid_nz": grid.nz,
        "grid_dx_um": grid.dx,
        "grid_dz_um": grid.dz,
    }
    prior_summary = None
    if arguments.start_from_summary is not None:
        prior_summary = json.loads(arguments.start_from_summary.read_text(encoding="utf-8"))
        metadata["start_from_summary_sha256"] = _sha256(arguments.start_from_summary)
        metadata["start_from_summary_path"] = str(arguments.start_from_summary)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    omega0 = 2.0 * np.pi / float(source["wavelength_um"])
    start = np.asarray(config["common_start"], dtype=float)
    bounds = (
        np.asarray((config["mirror_thickness_scale_bounds"][0], config["cavity_thickness_scale_bounds"][0]), dtype=float),
        np.asarray((config["mirror_thickness_scale_bounds"][1], config["cavity_thickness_scale_bounds"][1]), dtype=float),
    )
    preflight: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    parity_values: list[int] = []
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 7.0), constrained_layout=True)
    for quality in config["target_quality_factors"]:
        target = complex(omega0, -omega0 / (2.0 * float(quality)))
        case_start = start
        if prior_summary is not None:
            prior_case = next(
                item for item in prior_summary["cases"]
                if item["target_quality_factor"] == quality
            )
            case_start = np.asarray(prior_case["optimized_coordinates"], dtype=float)
        feasible_case = next(item for item in feasible["cases"] if item["target_quality_factor"] == quality)
        feasible_x = np.asarray((feasible_case["mirror_thickness_scale"], feasible_case["cavity_thickness_scale"]))
        parity, parity_audit = select_pole_parity(
            target, feasible_x, config, source, scalar, twoport
        )
        parity_values.append(parity)
        if quality == config["target_quality_factors"][0]:
            preflight = check_coordinate_jacobian(
                case_start, target, parity, config, source, scalar, twoport
            )
            (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
            worst_error = max(
                *preflight["epsilon_mapping_relative_errors"],
                *preflight["pole_adjoint_relative_errors"],
            )
            if worst_error > float(config["finite_difference_check_tolerance"]):
                raise RuntimeError(f"adjoint preflight gradient error {worst_error} exceeds gate")
            print(f"gradient preflight passed: max relative error {worst_error:.2e}", flush=True)
            if arguments.preflight_only:
                return
        cached_x: np.ndarray | None = None
        cached_result: tuple[np.ndarray, np.ndarray] | None = None
        calls = 0

        def system(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            nonlocal cached_x, cached_result, calls
            if cached_x is None or not np.array_equal(values, cached_x):
                cached_result = pole_adjoint_system(
                    values, target, parity, config, source, scalar, twoport
                )
                cached_x = np.asarray(values).copy()
                calls += 1
            assert cached_result is not None
            return cached_result

        started = time.perf_counter()
        result = least_squares(
            lambda values: system(values)[0],
            case_start,
            jac=lambda values: system(values)[1],
            bounds=bounds,
            method="trf",
            max_nfev=int(config["max_function_evaluations"]),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        wall_seconds = time.perf_counter() - started
        verified, curves, epsilon = _verify_final(
            result.x, target, config, source, scalar
        )
        case = {
            "target_quality_factor": quality,
            "target_pole": {"real": target.real, "imag": target.imag},
            "pole_parity_sign": parity,
            "parity_audit_at_analytic_feasible_geometry": parity_audit,
            "protocol_common_start": start.tolist(),
            "start_used": case_start.tolist(),
            "start_from_summary": str(arguments.start_from_summary) if prior_summary is not None else None,
            "optimized_coordinates": result.x.tolist(),
            "analytically_feasible_coordinates_not_used_as_start": feasible_x.tolist(),
            "optimization_success": bool(result.success),
            "optimizer_message": str(result.message),
            "optimizer_function_evaluations": int(result.nfev),
            "optimizer_jacobian_evaluations": int(result.njev) if result.njev is not None else None,
            "unique_adjoint_evaluations": calls,
            "equivalent_frequency_factorizations": calls,
            "optimization_wall_seconds": wall_seconds,
            "final_time_reversal_residual_norm": float(np.linalg.norm(result.fun)),
            "verification": verified,
        }
        cases.append(case)
        suffix = f"q{int(quality):03d}"
        _write_curve(output / f"spectrum_{suffix}.csv", curves)
        np.save(output / f"permittivity_{suffix}.npy", epsilon)
        (output / "summary.json").write_text(
            json.dumps({"method": "2D periodic pole adjoint", "preflight": preflight, "cases": cases}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"Q target {quality:.0f}: achieved {verified['verified_quality_factor']:.2f}, "
            f"pole error={verified['pole_error_in_target_depths']:.4f} depths, "
            f"T={verified['near_pole_real_transmittance']:.5f}, "
            f"adjoint solves={calls}",
            flush=True,
        )
        x = curves["detuning_in_target_depths"]
        axes[0].plot(x, curves["transmittance"], label=f"Q={quality:.0f}")
        axes[1].plot(x, curves["transmission_phase_rad_unwrapped"] / np.pi, label=f"Q={quality:.0f}")
    if len(set(parity_values)) != 1:
        raise RuntimeError("Q targets lie on different axial-parity branches")
    axes[0].set_ylabel("Transmittance")
    axes[0].set_ylim(-0.03, 1.04)
    axes[1].set_ylabel("Transmission phase (pi units)")
    axes[1].set_xlabel("Detuning from common target center / target pole depth")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(True, linewidth=0.5, alpha=0.3)
    figure.savefig(output / "m142e_adjoint_q_series_rt_phase.png", dpi=200)
    plt.close(figure)
    passed = all(
        case["verification"]["pole_error_in_target_depths"]
        <= float(config["max_pole_error_in_target_depths"])
        for case in cases
    )
    print(f"M142E pole gate passed: {passed}", flush=True)


if __name__ == "__main__":
    main()
