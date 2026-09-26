"""Simultaneous N=32/N=48 continuation and held-out N=64 audit of M27D."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pzid.constraints import (
    channel_value_density_constraint,
    combine_density_constraints,
    pole_density_constraint,
)
from pzid.density import (
    CrossGridPairedNestedReliefDensityParameterization,
    DensityParameterization,
)
from pzid.fabrication import binarity_index, gray_fraction
from pzid.grid import UniformGrid2D
from pzid.hybrid import (
    PeriodDoubledCoupledSurfaceReliefSlabsLayout,
    symmetric_slab_tm_frequencies,
)
from pzid.optimize import LMOptions, LMResult, solve_underdetermined_lm
from pzid.plotting import plot_refractive_index, save_figure
from pzid.poles import PoleTarget
from pzid.roots import refine_scalar_root
from pzid.twoport import solve_two_port_scattering

from scripts.m3_optimize_pole_zero import write_history
from scripts.m4b_analyze_eigenchannels import resonance_estimates
from scripts.m4b_optimize_kerker_bar import _channel_response, _parity_response
from scripts.m6_fabrication_continue import _scaled_port_setup
from scripts.m11_screen_fp_tm1_hybrid import _root_json, _sweep, _write_spectrum
from scripts.m20_discover_opposite_parity_dual_gmr import _pair_record
from scripts.m22_screen_coupled_single_mode_slabs import _diagnose_candidate
from scripts.m27b_screen_coupled_folded_qbic import _build as _build_parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def _frequency(root: dict[str, Any]) -> complex:
    return complex(*root["frequency"])


def _interval_fill(grid: UniformGrid2D, start: float, stop: float) -> np.ndarray:
    lower = np.arange(grid.nz, dtype=float) * grid.dz
    upper = lower + grid.dz
    overlap = np.maximum(0.0, np.minimum(upper, stop) - np.maximum(lower, start))
    return np.clip(overlap / grid.dz, 0.0, 1.0)


def _physical_system(
    config: dict[str, Any],
    parent_config: dict[str, Any],
    cells: int,
    beta: float,
) -> tuple[Any, CrossGridPairedNestedReliefDensityParameterization, Any]:
    geometry = config["geometry"]
    period = float(geometry["period_um"])
    spacing = period / cells
    nz_float = float(geometry["total_height_um"]) / spacing
    nz = int(round(nz_float))
    if not np.isclose(nz, nz_float, rtol=0.0, atol=1e-10):
        raise ValueError(f"N={cells} does not exactly tile the physical cell height")
    grid = UniformGrid2D(nx=cells, nz=nz, dx=spacing, dz=spacing)
    low = float(geometry["low_index"])
    high = float(geometry["high_index"])
    port = float(geometry["lower_port_thickness_um"])
    slab = float(geometry["slab_thickness_um"])
    inner_thickness = float(geometry["inner_relief_thickness_um"])
    outer_thickness = float(geometry["outer_relief_thickness_um"])
    slab_fill = _interval_fill(grid, port, port + slab)
    slab_fill += np.flip(slab_fill)
    background = np.full(grid.shape, low**2, dtype=float)
    background += slab_fill[:, np.newaxis] * (high**2 - low**2)
    inner_start = port + slab
    inner_fill = _interval_fill(grid, inner_start, inner_start + inner_thickness)
    outer_start = inner_start + inner_thickness
    outer_fill = _interval_fill(grid, outer_start, outer_start + outer_thickness)
    controls = config["parameterization"]
    common = {
        "grid": grid,
        "low_index": low,
        "high_index": high,
        "filter_radius": float(controls["filter_radius_um"]),
        "projection_beta": beta,
        "projection_threshold": float(controls["projection_threshold"]),
    }
    inner_filter = DensityParameterization(design_rows=(0, 1), **common)
    outer_filter = DensityParameterization(design_rows=(1, 2), **common)
    parameterization = CrossGridPairedNestedReliefDensityParameterization(
        master_nx=int(geometry["master_cells_per_period"]),
        inner=inner_filter,
        outer=outer_filter,
        lower_inner_fill=inner_fill,
        lower_outer_fill=outer_fill,
        fixed_background_epsilon=background,
    )
    _, _, reference_setup = _reference_setup(parent_config)
    setup = _scaled_port_setup(reference_setup, grid)
    lower_slab = np.flatnonzero(slab_fill > 0.0)
    lower_slab = lower_slab[lower_slab < grid.nz // 2]
    lower_relief = np.flatnonzero((inner_fill + outer_fill) > 0.0)
    lower_slab_rows = (int(lower_slab[0]), int(lower_slab[-1] + 1))
    lower_grating_rows = (int(lower_relief[0]), int(lower_relief[-1] + 1))
    upper_grating_rows = (
        grid.nz - lower_grating_rows[1], grid.nz - lower_grating_rows[0]
    )
    upper_slab_rows = (grid.nz - lower_slab_rows[1], grid.nz - lower_slab_rows[0])
    layout = PeriodDoubledCoupledSurfaceReliefSlabsLayout(
        lower_slab_rows=lower_slab_rows,
        lower_grating_rows=lower_grating_rows,
        spacer_rows=(lower_grating_rows[1], upper_grating_rows[0]),
        upper_grating_rows=upper_grating_rows,
        upper_slab_rows=upper_slab_rows,
        structure_rows=(lower_slab_rows[0], upper_slab_rows[1]),
        cavity_rows=(lower_slab_rows[0], upper_slab_rows[1]),
        left_grating_columns=(0, cells // 2),
        right_grating_columns=(cells // 2, cells),
        parent_half_period_cells=cells // 2,
    )
    return setup, parameterization, layout


def _reference_setup(parent_config: dict[str, Any]) -> tuple[Any, Any, Any]:
    grid, epsilon, _, setup = _build_parent(parent_config, 40, 1, 12, 2)
    return grid, epsilon, setup


def _refine_pair(
    epsilon: np.ndarray,
    setup: Any,
    guesses: Sequence[complex],
    config: dict[str, Any],
) -> tuple[Any, Any]:
    controls = config["verification"]
    roots = []
    for parity, guess in zip(("even", "odd"), guesses, strict=True):
        roots.append(
            refine_scalar_root(
                lambda value, selected=parity: _parity_response(
                    value, epsilon, setup, selected
                ),
                guess,
                kind="pole",
                tolerance=float(controls["root_tolerance"]),
                max_evaluations=int(controls["root_max_evaluations"]),
                search_radius=float(controls["root_search_radius_rad_per_um"]),
            )
        )
    residual_gate = float(controls["root_residual_gate"])
    if all(root.success and root.residual_norm <= residual_gate for root in roots):
        return roots[0], roots[1]

    center = float(np.mean([value.real for value in guesses]))
    radius = float(controls["root_search_radius_rad_per_um"])
    omega = np.linspace(center - radius, center + radius, 241)
    spectrum = _sweep(2.0 * np.pi / omega, epsilon, setup)
    discovered = []
    for parity, response in zip(
        ("even", "odd"), (spectrum["even"], spectrum["odd"]), strict=True
    ):
        estimates = resonance_estimates(
            2.0 * np.pi / omega, np.asarray(response), maximum_count=8
        )
        if not estimates:
            raise RuntimeError(f"no {parity} resonance found in fallback scan")
        estimate = min(
            estimates, key=lambda item: abs(float(item["omega_rad_per_um"]) - center)
        )
        guess = complex(
            float(estimate["omega_rad_per_um"]),
            -0.5 * max(float(estimate["linewidth_rad_per_um"]), 1e-4),
        )
        root = refine_scalar_root(
            lambda value, selected=parity: _parity_response(
                value, epsilon, setup, selected
            ),
            guess,
            kind="pole",
            tolerance=float(controls["root_tolerance"]),
            max_evaluations=int(controls["root_max_evaluations"]),
            search_radius=radius,
        )
        if not root.success or root.residual_norm > residual_gate:
            raise RuntimeError(
                f"{parity} fallback root residual {root.residual_norm:.3e} exceeds gate"
            )
        discovered.append(root)
    return discovered[0], discovered[1]


def _constraint_blocks(
    values: np.ndarray,
    systems: Sequence[tuple[int, Any, Any]],
    targets: dict[int, tuple[complex, complex]],
    config: dict[str, Any],
    reflection_targets: dict[int, complex] | None,
    raw_binarity_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    residuals = []
    jacobians = []
    norms = []
    scale = 1.0 / np.sqrt(len(systems))
    for cells, setup, parameterization in systems:
        evaluations = [
            pole_density_constraint(
                values, parameterization, PoleTarget(target, coupling), setup
            )
            for target, coupling in zip(targets[cells], (-1.0, 1.0), strict=True)
        ]
        weights = [float(config["optimization"]["pole_constraint_weight"])] * 8
        if reflection_targets is not None:
            evaluations.append(
                channel_value_density_constraint(
                    values,
                    parameterization,
                    0.5 * (targets[cells][0].real + targets[cells][1].real),
                    "reflection",
                    reflection_targets[cells],
                    setup,
                )
            )
            weights.extend(
                [float(config["optimization"]["reflection_constraint_weight"])] * 2
            )
        residual, jacobian = combine_density_constraints(
            *evaluations, weights=np.asarray(weights)
        )
        residuals.append(scale * residual)
        jacobians.append(scale * jacobian)
        norms.append(float(np.linalg.norm(residual)))
    combined_residual = np.concatenate(residuals)
    combined_jacobian = np.vstack(jacobians)
    if raw_binarity_weight > 0.0:
        raw = np.asarray(values, dtype=float).reshape(2, -1)
        count = raw.shape[1]
        binary_residual = np.mean(4.0 * raw * (1.0 - raw), axis=1)
        binary_jacobian = np.zeros((2, raw.size), dtype=float)
        binary_jacobian[0, :count] = 4.0 * (1.0 - 2.0 * raw[0]) / count
        binary_jacobian[1, count:] = 4.0 * (1.0 - 2.0 * raw[1]) / count
        combined_residual = np.concatenate(
            (combined_residual, raw_binarity_weight * binary_residual)
        )
        combined_jacobian = np.vstack(
            (combined_jacobian, raw_binarity_weight * binary_jacobian)
        )
    return combined_residual, combined_jacobian, norms


def _solve_stage(
    name: str,
    initial: np.ndarray,
    systems: Sequence[tuple[int, Any, Any]],
    targets: dict[int, tuple[complex, complex]],
    config: dict[str, Any],
    iterations: int,
    reflection_targets: dict[int, complex] | None = None,
    raw_binarity_weight: float = 0.0,
) -> tuple[LMResult, int]:
    evaluations = 0

    def system(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        residual, jacobian, norms = _constraint_blocks(
            values,
            systems,
            targets,
            config,
            reflection_targets,
            raw_binarity_weight,
        )
        print(
            f"{name} evaluation {evaluations}: combined={np.linalg.norm(residual):.6e} "
            + " ".join(
                f"N={cells}:{norm:.3e}"
                for (cells, _, _), norm in zip(systems, norms, strict=True)
            ),
            flush=True,
        )
        return residual, jacobian

    controls = config["optimization"]
    result = solve_underdetermined_lm(
        system,
        initial,
        options=LMOptions(
            max_iterations=iterations,
            initial_damping=float(controls["initial_damping"]),
            residual_tolerance=1e-7,
            step_tolerance=float(controls["step_tolerance"]),
            active_set_bounds=bool(controls["active_set_bounds"]),
            active_set_max_iterations=int(controls["active_set_max_iterations"]),
        ),
    )
    return result, evaluations


def _mask_metrics(mapping: Any) -> dict[str, float]:
    controls = np.stack(
        (mapping.projected_inner, mapping.projected_outer_control)
    )
    material = np.stack(
        (mapping.projected_inner, mapping.projected_outer_material)
    )
    return {
        "control_binarity_index": binarity_index(controls),
        "control_gray_fraction": gray_fraction(controls),
        "material_binarity_index": binarity_index(material),
        "material_gray_fraction": gray_fraction(material),
    }


def _hard_lateral_fills(
    raw: np.ndarray, target_nx: int, threshold: float, subpixels: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw, dtype=float).reshape(2, -1)
    master_nx = values.shape[1]
    coordinate = (
        (np.arange(target_nx * subpixels, dtype=float) + 0.5)
        / subpixels
        * master_nx
        / target_nx
        - 0.5
    )
    left = np.floor(coordinate).astype(int)
    fraction = coordinate - left
    left %= master_nx
    right = (left + 1) % master_nx
    samples = (
        (1.0 - fraction)[np.newaxis, :] * values[:, left]
        + fraction[np.newaxis, :] * values[:, right]
    )
    samples = samples.reshape(2, target_nx, subpixels)
    inner = samples[0] >= threshold
    outer = inner & (samples[1] >= threshold)
    return np.mean(inner, axis=1), np.mean(outer, axis=1)


def _hard_binary_epsilon(
    raw: np.ndarray,
    parameterization: CrossGridPairedNestedReliefDensityParameterization,
    threshold: float,
    subpixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    inner_lateral, outer_lateral = _hard_lateral_fills(
        raw, parameterization.grid.nx, threshold, subpixels
    )
    inner_axial, outer_axial = parameterization.axial_fill_profiles
    epsilon = parameterization._background()
    contrast = parameterization.high_index**2 - parameterization.low_index**2
    epsilon += contrast * inner_axial[:, np.newaxis] * inner_lateral[np.newaxis, :]
    epsilon += contrast * outer_axial[:, np.newaxis] * outer_lateral[np.newaxis, :]
    if not np.allclose(epsilon, np.flipud(epsilon), rtol=0.0, atol=1e-12):
        raise RuntimeError("hard rasterization broke axial symmetry")
    return epsilon, {
        "inner_lateral_fill": inner_lateral,
        "outer_lateral_fill": outer_lateral,
        "inner_is_connected_to_slab": True,
        "outer_is_subset_of_inner": bool(np.all(outer_lateral <= inner_lateral + 1e-14)),
    }


def _diagnose(
    epsilon: np.ndarray,
    setup: Any,
    layout: Any,
    guesses: Sequence[complex],
    isolated_frequency: float,
    parent_config: dict[str, Any],
    config: dict[str, Any],
    label: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    roots = _refine_pair(epsilon, setup, guesses, config)
    modes = {}
    fields = {}
    for parity, root in zip(("even", "odd"), roots, strict=True):
        mode, field = _diagnose_candidate(
            root,
            parity,
            isolated_frequency,
            epsilon,
            setup.grid,
            layout,
            setup,
            parent_config,
        )
        modes[parity] = mode
        fields[parity] = np.asarray(field)
    pair = _pair_record(modes["even"], modes["odd"])
    return {
        "identifier": label,
        "even_mode": modes["even"],
        "odd_mode": modes["odd"],
        "pair": pair,
        **pair,
        "both_modes_pass_assignment": bool(
            modes["even"]["passes_assignment_gates"]
            and modes["odd"]["passes_assignment_gates"]
        ),
    }, fields


def _channel_roots(
    epsilon: np.ndarray,
    setup: Any,
    pole_center: complex,
    config: dict[str, Any],
) -> dict[str, Any]:
    controls = config["verification"]
    common = {
        "tolerance": float(controls["root_tolerance"]),
        "max_evaluations": int(controls["root_max_evaluations"]),
        "search_radius": float(controls["root_search_radius_rad_per_um"]),
    }
    reflection = refine_scalar_root(
        lambda value: _channel_response(value, epsilon, setup, "reflection"),
        complex(pole_center.real, 0.0),
        kind="zero",
        **common,
    )
    transmission = refine_scalar_root(
        lambda value: _channel_response(value, epsilon, setup, "transmission"),
        np.conj(pole_center),
        kind="zero",
        **common,
    )
    return {
        "reflection_zero": _root_json(reflection),
        "transmission_zero": _root_json(transmission),
    }


def _response(
    epsilon: np.ndarray,
    setup: Any,
    target: complex,
    config: dict[str, Any],
    *,
    points: int | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    controls = config["verification"]
    gamma = -target.imag
    omega = np.linspace(
        target.real - float(controls["spectrum_half_width_in_gamma"]) * gamma,
        target.real + float(controls["spectrum_half_width_in_gamma"]) * gamma,
        int(controls["spectrum_points"] if points is None else points),
    )
    spectrum = _sweep(2.0 * np.pi / omega, epsilon, setup)
    matrix = np.asarray(spectrum["matrices"])
    transmission = matrix[:, 1, 0]
    reflection = matrix[:, 0, 0]
    performance = np.abs(omega - target.real) <= (
        float(controls["performance_half_width_in_gamma"]) * gamma
    )
    phase = np.unwrap(np.angle(transmission))
    metrics = {
        "minimum_transmittance_in_performance_window": float(
            np.min(np.abs(transmission[performance]) ** 2)
        ),
        "maximum_reflectance_in_performance_window": float(
            np.max(np.abs(reflection[performance]) ** 2)
        ),
        "full_plot_phase_span_pi": float(np.ptp(phase) / np.pi),
        "performance_window_phase_span_pi": float(np.ptp(phase[performance]) / np.pi),
        "center_reflectance": float(
            abs(solve_two_port_scattering(target.real, epsilon, setup).matrix[0, 0]) ** 2
        ),
        "energy_balance_max_abs_error": float(spectrum["energy_balance_max_abs_error"]),
    }
    return metrics, {"omega": omega, "spectrum": spectrum}


def _plot_continuation(records: list[dict[str, Any]]) -> plt.Figure:
    x = np.arange(1, len(records) + 1)
    residual = np.asarray([record["residual_norm"] for record in records])
    binary = np.asarray([record["n48_material_binarity_index"] for record in records])
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 6.3), sharex=True, constrained_layout=True)
    axes[0].semilogy(x, np.maximum(residual, 1e-14), marker="o", markersize=3)
    axes[0].set_ylabel("Combined residual")
    axes[0].set_title("M27E simultaneous N=32/N=48 continuation")
    axes[1].plot(x, binary, marker="s", markersize=3)
    axes[1].set_ylabel("N=48 material binarity index")
    axes[1].set_xlabel("Accepted continuation stage")
    for axis in axes:
        axis.grid(True, linewidth=0.4, alpha=0.35)
    return figure


def _plot_structures(records: list[dict[str, Any]]) -> plt.Figure:
    figure, axes = plt.subplots(
        1, len(records), figsize=(3.6 * len(records), 5.0), sharex=True, sharey=True,
        constrained_layout=True,
    )
    image = None
    for axis, record in zip(np.atleast_1d(axes), records, strict=True):
        image = axis.imshow(
            np.sqrt(record["epsilon"]),
            origin="lower",
            extent=(0.0, record["period"], 0.0, record["height"]),
            interpolation="nearest",
            aspect="equal",
            vmin=1.0,
            vmax=3.42,
        )
        held = " (held out)" if record["held_out"] else ""
        axis.set_title(f"N={record['cells']}{held}")
        axis.set_xlabel("x (µm)")
    np.atleast_1d(axes)[0].set_ylabel("z (µm)")
    assert image is not None
    figure.colorbar(image, ax=axes, label="Refractive index, n", shrink=0.82)
    figure.suptitle("One fixed hard-binary contour on three meshes")
    return figure


def _plot_response(records: list[dict[str, Any]], target: complex) -> plt.Figure:
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 6.5), sharex=True, constrained_layout=True)
    for record in records:
        omega = record["response"]["omega"]
        matrix = np.asarray(record["response"]["spectrum"]["matrices"])
        label = f"N={record['cells']}" + (" held out" if record["held_out"] else "")
        axes[0].plot(omega, np.abs(matrix[:, 1, 0]) ** 2, label=f"T, {label}")
        axes[0].plot(omega, np.abs(matrix[:, 0, 0]) ** 2, linestyle="--", label=f"R, {label}")
        phase = np.unwrap(np.angle(matrix[:, 1, 0]))
        phase -= phase[0]
        axes[1].plot(omega, phase / np.pi, label=label)
    for axis in axes:
        axis.axvline(target.real, color="black", linestyle=":", linewidth=0.8)
        axis.grid(True, linewidth=0.4, alpha=0.35)
        axis.legend(frameon=False, fontsize=8, ncol=2)
    axes[0].set_ylabel("Power")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("Hard-binary real-axis mesh comparison")
    axes[1].set_xlabel("Real frequency (rad/µm)")
    axes[1].set_ylabel("Transmission phase change / π")
    return figure


def _plot_roots(records: list[dict[str, Any]], target: complex) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    colors = ("tab:blue", "tab:orange", "tab:green")
    for record, color in zip(records, colors, strict=True):
        cells = record["cells"]
        even = _frequency(record["diagnostics"]["even_mode"]["root"])
        odd = _frequency(record["diagnostics"]["odd_mode"]["root"])
        rz = _frequency(record["channel_roots"]["reflection_zero"])
        tz = _frequency(record["channel_roots"]["transmission_zero"])
        label = f"N={cells}" + (" held out" if record["held_out"] else "")
        axes[0].scatter(even.real, even.imag, marker="o", color=color, label=f"{label}, even pole")
        axes[0].scatter(odd.real, odd.imag, marker="s", facecolors="none", edgecolors=color, label=f"{label}, odd pole")
        axes[1].scatter(tz.real, tz.imag, marker="^", color=color, label=f"{label}, t zero")
        axes[1].scatter(rz.real, rz.imag, marker="D", facecolors="none", edgecolors=color, label=f"{label}, r zero")
    axes[0].scatter(target.real, target.imag, marker="x", s=95, color="black", linewidths=2, label="common pole target")
    axes[1].scatter(target.real, -target.imag, marker="x", s=95, color="black", linewidths=2, label="conjugate target")
    for axis, title in zip(axes, ("Parity poles", "Channel zeros"), strict=True):
        axis.axhline(0.0, color="0.45", linewidth=0.7)
        axis.set_xlabel("Re frequency (rad/µm)")
        axis.set_ylabel("Im frequency (rad/µm)")
        axis.set_title(title)
        axis.grid(True, linewidth=0.4, alpha=0.35)
        axis.legend(frameon=False, fontsize=7)
    return figure


def _plot_held_out_fields(record: dict[str, Any]) -> plt.Figure:
    epsilon = record["epsilon"]
    fields = record["fields"]
    extent = (0.0, record["period"], 0.0, record["height"])
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 5.0), sharex=True, sharey=True, constrained_layout=True)
    axes[0].imshow(np.sqrt(epsilon), origin="lower", extent=extent, aspect="equal", interpolation="nearest", vmin=1.0, vmax=3.42)
    axes[0].set_title("N=64 hard-binary structure")
    for axis, parity in zip(axes[1:], ("even", "odd"), strict=True):
        field = np.real(fields[parity])
        limit = max(float(np.max(np.abs(field))), np.finfo(float).tiny)
        axis.imshow(field, origin="lower", extent=extent, aspect="equal", cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.contour(np.sqrt(epsilon), levels=[2.0], origin="lower", extent=extent, colors="black", linewidths=0.45)
        axis.set_title(f"Re Hy, {parity} pole")
    for axis in axes:
        axis.set_xlabel("x (µm)")
    axes[0].set_ylabel("z (µm)")
    figure.suptitle("Held-out N=64 mode verification")
    return figure


def _write_response(path: Path, response: dict[str, Any]) -> None:
    omega = np.asarray(response["omega"])
    matrix = np.asarray(response["spectrum"]["matrices"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("omega_rad_per_um", "t_real", "t_imag", "r_real", "r_imag", "T", "R"))
        for frequency, scattering in zip(omega, matrix, strict=True):
            t = scattering[1, 0]
            r = scattering[0, 0]
            writer.writerow((frequency, t.real, t.imag, r.real, r.imag, abs(t) ** 2, abs(r) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m27e_fan_1d_multimesh_binary_tm.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/m27e_fan_1d_multimesh_binary"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    parent_config = json.loads(Path(config["parent_config"]).read_text(encoding="utf-8"))
    output = arguments.output
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    current = np.loadtxt(Path(config["parent_run"]) / "optimized_raw_density.csv", delimiter=",").ravel()
    initial_beta = float(config["parameterization"]["initial_projection_beta"])
    optimization_cells = tuple(int(value) for value in config["optimization"]["optimization_cells_per_period"])
    initial_systems = []
    for cells in optimization_cells:
        setup, parameterization, layout = _physical_system(config, parent_config, cells, initial_beta)
        initial_systems.append((cells, setup, parameterization, layout))
    parent_epsilon = np.loadtxt(Path(config["parent_run"]) / "optimized_permittivity.csv", delimiter=",")
    n32_epsilon = initial_systems[0][2].map(current).epsilon_r
    n32_reproduction_error = float(np.max(np.abs(parent_epsilon - n32_epsilon)))
    parent_summary = json.loads((Path(config["parent_run"]) / "summary.json").read_text(encoding="utf-8"))
    guesses = tuple(
        _frequency(parent_summary["final_diagnostics"][f"{parity}_mode"]["root"])
        for parity in ("even", "odd")
    )
    initial_roots: dict[int, tuple[complex, complex]] = {}
    initial_root_json = {}
    for cells, setup, parameterization, _ in initial_systems:
        roots = _refine_pair(parameterization.map(current).epsilon_r, setup, guesses, config)
        values = tuple(complex(root.frequency) for root in roots)
        initial_roots[cells] = values
        initial_root_json[str(cells)] = [_root_json(root) for root in roots]
    common_target = sum((value for roots in initial_roots.values() for value in roots), 0.0j) / (2 * len(initial_roots))
    endpoint_targets = {cells: (common_target, common_target) for cells in optimization_cells}
    endpoint_systems = [(cells, setup, parameterization) for cells, setup, parameterization, _ in initial_systems]
    endpoint_residual, endpoint_jacobian, endpoint_norms = _constraint_blocks(
        current,
        endpoint_systems,
        endpoint_targets,
        config,
        {cells: 0.0j for cells in optimization_cells},
    )
    singular_values = np.linalg.svd(endpoint_jacobian, compute_uv=False)
    tolerance = max(endpoint_jacobian.shape) * np.finfo(float).eps * singular_values[0]
    audit = {
        "n32_parent_reproduction_max_abs_epsilon_error": n32_reproduction_error,
        "shared_variable_count": int(current.size),
        "combined_constraint_rows": int(endpoint_jacobian.shape[0]),
        "combined_jacobian_rank": int(np.count_nonzero(singular_values > tolerance)),
        "combined_jacobian_singular_values": [float(value) for value in singular_values],
        "endpoint_residual_norm": float(np.linalg.norm(endpoint_residual)),
        "per_mesh_endpoint_residual_norm": {
            str(cells): norm for cells, norm in zip(optimization_cells, endpoint_norms, strict=True)
        },
        "initial_roots": initial_root_json,
        "common_target": _pair(common_target),
        "n64_excluded_from_objective": True,
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savetxt(output / "initial_endpoint_jacobian.csv", endpoint_jacobian, delimiter=",")
    metadata = {
        "config_sha256": _sha256(arguments.config),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "density_source_sha256": _sha256(Path("src/pzid/density.py")),
        "parent_density_sha256": _sha256(Path(config["parent_run"]) / "optimized_raw_density.csv"),
        "parent_summary_sha256": _sha256(Path(config["parent_run"]) / "summary.json"),
        "numpy_version": np.__version__,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": audit}, indent=2), flush=True)
    if arguments.audit_only:
        return

    records_path = output / "continuation_records.json"
    stage_records: list[dict[str, Any]] = []
    if (arguments.resume or arguments.postprocess_only) and records_path.exists():
        stage_records = json.loads(records_path.read_text(encoding="utf-8"))
        if stage_records:
            current = np.loadtxt(output / stage_records[-1]["raw_density_file"], delimiter=",").ravel()
    stage_specs = []
    for stage in config["optimization"]["pole_homotopy"]:
        fraction = float(stage["fraction"])
        stage_specs.append({
            "group": "pole_homotopy",
            "value": fraction,
            "beta": initial_beta,
            "iterations": int(stage["max_iterations"]),
        })
    for stage in config["optimization"]["reflection_homotopy"]:
        stage_specs.append({
            "group": "reflection_homotopy",
            "value": float(stage["fraction"]),
            "beta": initial_beta,
            "iterations": int(stage["max_iterations"]),
        })
    for stage in config["optimization"]["projection_continuation"]:
        stage_specs.append({
            "group": "projection_continuation",
            "value": float(stage["beta"]),
            "beta": float(stage["beta"]),
            "iterations": int(stage["max_iterations"]),
        })
    for stage in config["optimization"].get("binarity_continuation", []):
        stage_specs.append({
            "group": "binarity_continuation",
            "value": float(stage["weight"]),
            "beta": float(config["optimization"]["projection_continuation"][-1]["beta"]),
            "iterations": int(stage["max_iterations"]),
        })
    reflection_start: dict[int, complex] | None = None
    if stage_records:
        reflection_records = [record for record in stage_records if record["group"] == "reflection_homotopy"]
        if reflection_records:
            reflection_start = {
                int(key): complex(*value)
                for key, value in reflection_records[0]["reflection_start"].items()
            }
    if not arguments.postprocess_only:
        for index, spec in enumerate(stage_specs, start=1):
            if index <= len(stage_records):
                continue
            beta = float(spec["beta"])
            systems = []
            parameterizations = {}
            for cells in optimization_cells:
                setup, parameterization, _ = _physical_system(config, parent_config, cells, beta)
                systems.append((cells, setup, parameterization))
                parameterizations[cells] = parameterization
            if spec["group"] == "pole_homotopy":
                fraction = float(spec["value"])
                targets = {
                    cells: tuple(start + fraction * (common_target - start) for start in initial_roots[cells])
                    for cells in optimization_cells
                }
                reflection_targets = None
            else:
                targets = endpoint_targets
                if reflection_start is None:
                    reflection_start = {
                        cells: complex(solve_two_port_scattering(common_target.real, parameterizations[cells].map(current).epsilon_r, setup).matrix[0, 0])
                        for cells, setup, _ in systems
                    }
                fraction = float(spec["value"]) if spec["group"] == "reflection_homotopy" else 1.0
                reflection_targets = {
                    cells: (1.0 - fraction) * reflection_start[cells]
                    for cells in optimization_cells
                }
            name = f"stage_{index:02d}_{spec['group']}_{spec['value']:g}"
            result, evaluations = _solve_stage(
                name,
                current,
                systems,
                targets,
                config,
                int(spec["iterations"]),
                reflection_targets,
                float(spec["value"]) if spec["group"] == "binarity_continuation" else 0.0,
            )
            residual = float(np.linalg.norm(result.residual))
            accepted = bool(
                residual <= float(config["optimization"]["stage_acceptance_residual"])
                and (not result.history or result.history[-1].residual_norm <= np.linalg.norm(result.residual) + 1e-14)
            )
            raw_file = f"{name}_raw_density.csv"
            np.savetxt(output / raw_file, result.x.reshape(2, -1), delimiter=",")
            write_history(output / f"{name}_history.csv", result)
            metrics = {
                cells: _mask_metrics(parameterizations[cells].map(result.x))
                for cells in optimization_cells
            }
            record = {
                "index": index,
                "name": name,
                "group": spec["group"],
                "value": float(spec["value"]),
                "projection_beta": beta,
                "optimization_cells_per_period": list(optimization_cells),
                "residual_norm": residual,
                "function_evaluations": evaluations,
                "accepted_iterations": len(result.history),
                "solver_converged": result.converged,
                "solver_reason": result.reason,
                "accepted": accepted,
                "raw_density_file": raw_file,
                "n32_material_binarity_index": metrics[32]["material_binarity_index"],
                "n48_material_binarity_index": metrics[48]["material_binarity_index"],
                "mask_metrics": {str(key): value for key, value in metrics.items()},
            }
            if reflection_start is not None:
                record["reflection_start"] = {
                    str(key): _pair(value) for key, value in reflection_start.items()
                }
            stage_records.append(record)
            records_path.write_text(json.dumps(stage_records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            if not accepted:
                raise RuntimeError(f"{name} residual {residual:.3e} exceeds acceptance")
            current = result.x
    if len(stage_records) != len(stage_specs):
        raise RuntimeError("postprocessing requires every continuation stage")
    final_beta = float(stage_specs[-1]["beta"])
    np.savetxt(output / "optimized_master_raw_density.csv", current.reshape(2, -1), delimiter=",")

    isolated_frequency = float(
        symmetric_slab_tm_frequencies(
            period_um=float(config["geometry"]["period_um"]),
            slab_thickness_um=float(config["geometry"]["slab_thickness_um"]),
            core_index=float(config["geometry"]["high_index"]),
            cladding_index=float(config["geometry"]["low_index"]),
        )["even_hy"][0]
    )
    threshold = float(config["parameterization"]["projection_threshold"])
    subpixels = int(config["verification"]["binary_rasterization_subpixels"])
    validation = []
    plot_records = []
    root_guesses = (common_target, common_target)
    held_out_cells = int(config["verification"]["held_out_cells_per_period"])
    for cells_value in config["verification"]["validation_cells_per_period"]:
        cells = int(cells_value)
        setup, parameterization, layout = _physical_system(config, parent_config, cells, final_beta)
        projected_epsilon = parameterization.map(current).epsilon_r
        hard_epsilon, raster = _hard_binary_epsilon(current, parameterization, threshold, subpixels)
        print(f"validating hard-binary N={cells}", flush=True)
        diagnostics, fields = _diagnose(
            hard_epsilon,
            setup,
            layout,
            root_guesses,
            isolated_frequency,
            parent_config,
            config,
            f"hard_binary_n{cells}",
        )
        pole_center = 0.5 * (
            _frequency(diagnostics["even_mode"]["root"])
            + _frequency(diagnostics["odd_mode"]["root"])
        )
        channel = _channel_roots(hard_epsilon, setup, pole_center, config)
        metrics, response = _response(hard_epsilon, setup, common_target, config)
        projected_roots = _refine_pair(projected_epsilon, setup, root_guesses, config)
        projected_metrics, _ = _response(
            projected_epsilon, setup, common_target, config, points=101
        )
        np.savetxt(output / f"hard_binary_permittivity_n{cells}.csv", hard_epsilon, delimiter=",")
        np.savetxt(output / f"sharp_projected_permittivity_n{cells}.csv", projected_epsilon, delimiter=",")
        np.savetxt(output / f"hard_binary_even_field_real_n{cells}.csv", np.real(fields["even"]), delimiter=",")
        np.savetxt(output / f"hard_binary_even_field_imag_n{cells}.csv", np.imag(fields["even"]), delimiter=",")
        np.savetxt(output / f"hard_binary_odd_field_real_n{cells}.csv", np.real(fields["odd"]), delimiter=",")
        np.savetxt(output / f"hard_binary_odd_field_imag_n{cells}.csv", np.imag(fields["odd"]), delimiter=",")
        _write_response(output / f"hard_binary_response_n{cells}.csv", response)
        record = {
            "cells_per_period": cells,
            "held_out": cells == held_out_cells,
            "grid_shape": list(setup.grid.shape),
            "dx_um": setup.grid.dx,
            "fixed_physical_geometry": {
                "period_um": setup.grid.period,
                "height_um": setup.grid.height,
                "slab_thickness_um": float(config["geometry"]["slab_thickness_um"]),
                "inner_relief_thickness_um": float(config["geometry"]["inner_relief_thickness_um"]),
                "outer_relief_thickness_um": float(config["geometry"]["outer_relief_thickness_um"]),
            },
            "layout": asdict(layout),
            "hard_binary_raster": {
                "lateral_subpixels": subpixels,
                "outer_is_subset_of_inner": raster["outer_is_subset_of_inner"],
                "inner_lateral_fill": [float(value) for value in raster["inner_lateral_fill"]],
                "outer_lateral_fill": [float(value) for value in raster["outer_lateral_fill"]],
            },
            "sharp_projected_roots": [_root_json(root) for root in projected_roots],
            "sharp_projected_metrics": projected_metrics,
            "hard_binary_diagnostics": diagnostics,
            "hard_binary_channel_roots": channel,
            "hard_binary_metrics": metrics,
        }
        validation.append(record)
        plot_records.append({
            "cells": cells,
            "held_out": cells == held_out_cells,
            "epsilon": hard_epsilon,
            "period": setup.grid.period,
            "height": setup.grid.height,
            "diagnostics": diagnostics,
            "channel_roots": channel,
            "response": response,
            "fields": fields,
        })
        root_guesses = (
            _frequency(diagnostics["even_mode"]["root"]),
            _frequency(diagnostics["odd_mode"]["root"]),
        )

    centers = []
    linewidths = []
    mesh_gates = []
    for record in validation:
        diagnostics = record["hard_binary_diagnostics"]
        even = _frequency(diagnostics["even_mode"]["root"])
        odd = _frequency(diagnostics["odd_mode"]["root"])
        center = 0.5 * (even + odd)
        linewidth = -even.imag - odd.imag
        centers.append(center)
        linewidths.append(linewidth)
        metrics = record["hard_binary_metrics"]
        mesh_gates.append(bool(
            diagnostics["center_mismatch_over_mean_linewidth"]
            <= float(config["verification"]["maximum_pole_separation_over_mean_linewidth"])
            and metrics["maximum_reflectance_in_performance_window"]
            <= float(config["verification"]["maximum_reflectance"])
            and metrics["minimum_transmittance_in_performance_window"]
            >= float(config["verification"]["minimum_transmittance"])
            and metrics["full_plot_phase_span_pi"]
            >= float(config["verification"]["minimum_phase_span_pi"])
        ))
    mean_linewidth = float(np.mean(linewidths))
    center_spread = float(max(abs(value - np.mean(centers)) for value in centers) / mean_linewidth)
    convergence_gate = center_spread <= float(config["verification"]["maximum_mesh_pole_center_spread_over_mean_linewidth"])
    summary = {
        "stage": "M27E simultaneous N=32/N=48 binary continuation and N=64 held-out verification",
        "audit": audit,
        "common_pole_target": _pair(common_target),
        "continuation_stages": stage_records,
        "final_projection_beta": final_beta,
        "hard_binary_threshold": threshold,
        "hard_binary_lateral_subpixels": subpixels,
        "validation": validation,
        "mesh_performance_gates": mesh_gates,
        "mesh_pole_center_spread_over_mean_linewidth": center_spread,
        "mesh_convergence_gate": convergence_gate,
        "all_gates_pass": bool(all(mesh_gates) and convergence_gate),
        "provenance": config["provenance"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    figure = _plot_continuation(stage_records)
    save_figure(figure, figures / "m27e_continuation.png", dpi=230)
    plt.close(figure)
    figure = _plot_structures(plot_records)
    save_figure(figure, figures / "m27e_hard_binary_structures.png", dpi=230)
    plt.close(figure)
    figure = _plot_response(plot_records, common_target)
    save_figure(figure, figures / "m27e_hard_binary_mesh_response.png", dpi=230)
    plt.close(figure)
    figure = _plot_roots(plot_records, common_target)
    save_figure(figure, figures / "m27e_hard_binary_complex_roots.png", dpi=230)
    plt.close(figure)
    held_record = next(record for record in plot_records if record["held_out"])
    figure = _plot_held_out_fields(held_record)
    save_figure(figure, figures / "m27e_held_out_n64_fields.png", dpi=230)
    plt.close(figure)
    metadata["summary_sha256"] = _sha256(output / "summary.json")
    metadata["optimized_master_density_sha256"] = _sha256(output / "optimized_master_raw_density.csv")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "common_pole_target": _pair(common_target),
        "mesh_performance_gates": mesh_gates,
        "mesh_pole_center_spread_over_mean_linewidth": center_spread,
        "mesh_convergence_gate": convergence_gate,
        "all_gates_pass": summary["all_gates_pass"],
        "validation": [
            {
                "cells": record["cells_per_period"],
                "held_out": record["held_out"],
                "pair": record["hard_binary_diagnostics"]["pair"],
                "metrics": record["hard_binary_metrics"],
            }
            for record in validation
        ],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
