"""Multi-mesh, binary-aware continuation of the M6 pole-zero design."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    AxiallySymmetricDensityParameterization,
    CrossGridSymmetricDensityParameterization,
    DensityParameterization,
)
from pzid.fabrication import (
    binary_permittivity,
    binarity_index,
    gray_fraction,
    periodic_bilinear_resample,
    projected_density_from_permittivity,
    subpixel_binary_permittivity,
)
from pzid.optimize import LMOptions, LMResult, solve_underdetermined_lm
from pzid.plotting import save_figure
from pzid.poles import PoleTarget
from pzid.twoport import TwoPortScatteringSetup

from scripts.m3_optimize_pole_zero import write_history
from scripts.m3_single_resonance import write_metadata
from scripts.m4b_optimize_kerker_bar import _frequency, write_spectrum
from scripts.m5_optimize_coupled_bars import build_optimization_problem
from scripts.m6_fabrication_continue import (
    _band_wavelengths,
    _root_metrics,
    _scaled_port_setup,
    _spectrum_metrics,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _common_pole(config: dict[str, Any]) -> complex:
    target = config["target"]
    return complex(
        float(target["common_pole_real_rad_per_um"]),
        float(target["common_pole_imag_rad_per_um"]),
    )


def _target_parameterization(
    config: dict[str, Any],
    master_setup: Any,
    master_design_rows: tuple[int, int],
    cells_per_period: int,
    beta: float,
) -> tuple[Any, CrossGridSymmetricDensityParameterization]:
    _, _, candidate_setup, _, _ = build_optimization_problem(
        config, cells_per_period=cells_per_period
    )
    setup = _scaled_port_setup(master_setup, candidate_setup.grid)
    scale = setup.grid.nz / master_setup.grid.nz
    start = int(round(master_design_rows[0] * scale))
    stop = setup.grid.nz - start
    base = DensityParameterization(
        grid=setup.grid,
        design_rows=(start, stop),
        low_index=float(config["benchmark"]["background_index"]),
        high_index=float(config["benchmark"]["refractive_index"]),
        filter_radius=float(config["parameterization"]["filter_radius_um"]),
        projection_beta=beta,
        projection_threshold=float(
            config["parameterization"]["projection_threshold"]
        ),
    )
    parameterization = CrossGridSymmetricDensityParameterization(
        master_grid=master_setup.grid,
        master_design_rows=master_design_rows,
        target=base,
    )
    reserved = (
        setup.lower_source_plane,
        setup.upper_source_plane,
        *setup.lower_monitor_planes,
        *setup.upper_monitor_planes,
    )
    if any(start <= row < stop for row in reserved):
        raise RuntimeError("scaled design interval overlaps a calibrated port row")
    return setup, parameterization


def _master_parameterization(
    config: dict[str, Any],
) -> tuple[Any, AxiallySymmetricDensityParameterization]:
    """Build the master grid while preserving one physical layout reference."""

    controls = config["continuation"]
    master_cells = int(controls["master_cells_per_period"])
    layout_cells = int(
        controls.get("layout_reference_cells_per_period", master_cells)
    )
    _, _, layout_setup, layout_reference, _ = build_optimization_problem(
        config, cells_per_period=layout_cells
    )
    _, _, candidate_setup, _, _ = build_optimization_problem(
        config, cells_per_period=master_cells
    )
    setup = _scaled_port_setup(layout_setup, candidate_setup.grid)
    scale = setup.grid.nz / layout_setup.grid.nz
    start = int(round(layout_reference.design_rows[0] * scale))
    stop = setup.grid.nz - start
    base = DensityParameterization(
        grid=setup.grid,
        design_rows=(start, stop),
        low_index=float(config["benchmark"]["background_index"]),
        high_index=float(config["benchmark"]["refractive_index"]),
        filter_radius=float(config["parameterization"]["filter_radius_um"]),
        projection_beta=float(config["parameterization"]["projection_beta"]),
        projection_threshold=float(
            config["parameterization"]["projection_threshold"]
        ),
    )
    reference = AxiallySymmetricDensityParameterization(base)
    reserved = (
        setup.lower_source_plane,
        setup.upper_source_plane,
        *setup.lower_monitor_planes,
        *setup.upper_monitor_planes,
    )
    if any(start <= row < stop for row in reserved):
        raise RuntimeError("master design interval overlaps a calibrated port row")
    return setup, reference


def _homogeneous_monitor_setup(
    epsilon: np.ndarray, setup: TwoPortScatteringSetup
) -> tuple[TwoPortScatteringSetup, list[dict[str, int]]]:
    """Move a monitor outward if a fixed contour clips its original row."""

    shifts: list[dict[str, int]] = []
    lower_monitors = []
    for original in setup.lower_monitor_planes:
        row = int(original)
        while not np.allclose(epsilon[row], epsilon[row, 0], rtol=0.0, atol=1e-12):
            row -= 1
            if row <= setup.lower_source_plane:
                raise RuntimeError("no homogeneous monitor row remains below the design")
        lower_monitors.append(row)
        if row != original:
            shifts.append({"original_row": int(original), "used_row": row})
    if not shifts:
        return setup, shifts
    lower_tuple = tuple(lower_monitors)
    return (
        TwoPortScatteringSetup(
            grid=setup.grid,
            lower_source_plane=setup.lower_source_plane,
            upper_source_plane=setup.upper_source_plane,
            lower_monitor_planes=lower_tuple,
            upper_monitor_planes=tuple(
                setup.grid.nz - 1 - row for row in reversed(lower_tuple)
            ),
            port_index=setup.port_index,
        ),
        shifts,
    )


def _initial_master_density(
    config: dict[str, Any], master_reference: Any
) -> np.ndarray:
    controls = config["continuation"]
    seed_run = controls.get("seed_run")
    if seed_run is not None:
        seed_run_path = Path(seed_run)
        seed = np.loadtxt(
            seed_run_path / "optimized_master_raw_density.csv", delimiter=","
        ).ravel()
        if seed.size == master_reference.variable_count:
            return np.asarray(seed)
        seed_config = json.loads(
            (seed_run_path / "config.json").read_text(encoding="utf-8")
        )
        _, seed_reference = _master_parameterization(seed_config)
        if seed.size != seed_reference.variable_count:
            raise ValueError("seed master density has an incompatible size")
        seed_full = np.zeros(seed_reference.grid.shape, dtype=float)
        seed_start, seed_stop = seed_reference.design_rows
        seed_full[seed_start:seed_stop] = seed_reference.expand(seed)
        master_full = periodic_bilinear_resample(
            seed_full, master_reference.grid.shape
        )
        start, stop = master_reference.design_rows
        local = master_full[start:stop]
        local = 0.5 * (local + np.flipud(local))
        return master_reference.reduce(local).ravel()
    parent_cells = int(controls["parent_cells_per_period"])
    _, _, parent_setup, parent_reference, _ = build_optimization_problem(
        config, cells_per_period=parent_cells
    )
    raw = np.loadtxt(
        Path(controls["parent_run"]) / "optimized_raw_density.csv",
        delimiter=",",
    ).reshape(parent_reference.design_shape)
    parent_full = np.zeros(parent_setup.grid.shape, dtype=float)
    parent_start, parent_stop = parent_reference.design_rows
    parent_full[parent_start:parent_stop] = parent_reference.expand(raw)
    master_full = periodic_bilinear_resample(
        parent_full, master_reference.grid.shape
    )
    start, stop = master_reference.design_rows
    local = master_full[start:stop]
    local = 0.5 * (local + np.flipud(local))
    return master_reference.reduce(local).ravel()


def _run_multimesh_stage(
    name: str,
    initial: np.ndarray,
    systems: Sequence[tuple[int, Any, Any]],
    pole_targets: Sequence[PoleTarget],
    transmission_zero: complex | None,
    reflection_targets: Sequence[tuple[float, complex]],
    config: dict[str, Any],
    *,
    iterations: int,
    residual_tolerance: float,
    lower_bound: np.ndarray | float = 0.0,
    upper_bound: np.ndarray | float = 1.0,
) -> tuple[LMResult, int]:
    evaluations = 0
    mesh_scale = 1.0 / np.sqrt(len(systems))
    pole_weight = float(config["optimization"]["pole_constraint_weight"])
    zero_weight = float(
        config["optimization"]["transmission_zero_constraint_weight"]
    )

    def system(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        residual_blocks = []
        jacobian_blocks = []
        norms = []
        for cells, setup, parameterization in systems:
            constraints = [
                pole_density_constraint(values, parameterization, target, setup)
                for target in pole_targets
            ]
            if transmission_zero is not None:
                constraints.append(
                    channel_value_density_constraint(
                        values,
                        parameterization,
                        transmission_zero,
                        "transmission",
                        0.0j,
                        setup,
                    )
                )
            constraints.extend(
                channel_value_density_constraint(
                    values,
                    parameterization,
                    frequency,
                    "reflection",
                    target_value,
                    setup,
                )
                for frequency, target_value in reflection_targets
            )
            weights = np.concatenate(
                (
                    np.full(4 * len(pole_targets), pole_weight),
                    np.full(2 if transmission_zero is not None else 0, zero_weight),
                    np.ones(2 * len(reflection_targets)),
                )
            )
            residual, jacobian = combine_density_constraints(
                *constraints, weights=weights
            )
            residual_blocks.append(mesh_scale * residual)
            jacobian_blocks.append(mesh_scale * jacobian)
            norms.append(f"N={cells}:{np.linalg.norm(residual):.3e}")
        combined_residual = np.concatenate(residual_blocks)
        combined_jacobian = np.vstack(jacobian_blocks)
        print(
            f"{name} evaluation {evaluations}: "
            f"residual={np.linalg.norm(combined_residual):.6e} "
            + " ".join(norms),
            flush=True,
        )
        return combined_residual, combined_jacobian

    controls = config["optimization"]
    result = solve_underdetermined_lm(
        system,
        initial,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        options=LMOptions(
            max_iterations=iterations,
            initial_damping=float(controls["initial_damping"]),
            residual_tolerance=residual_tolerance,
            step_tolerance=float(controls["step_tolerance"]),
            active_set_bounds=bool(controls.get("active_set_bounds", False)),
            active_set_max_iterations=int(
                controls.get("active_set_max_iterations", 20)
            ),
        ),
    )
    return result, evaluations


def _serialize_root(record: dict[str, Any]) -> complex:
    return complex(
        record["omega_real_rad_per_um"], record["omega_imag_rad_per_um"]
    )


def _plot_continuation(records: list[dict[str, Any]]) -> plt.Figure:
    stage = np.asarray([record["index"] for record in records])
    residual = np.asarray([record["residual_norm"] for record in records])
    binary = np.asarray([record["master_binarity"] for record in records])
    figure, axes = plt.subplots(
        2, 1, figsize=(7.0, 6.0), sharex=True, constrained_layout=True
    )
    axes[0].semilogy(stage, np.maximum(residual, 1e-14), marker="o")
    axes[0].set_ylabel("Combined constraint residual")
    axes[0].set_title("M7 multi-mesh binary continuation")
    axes[1].plot(stage, binary, marker="s")
    axes[1].set_xlabel("Continuation stage")
    axes[1].set_ylabel("Master-density binarity index")
    for axis in axes:
        axis.grid(True, linewidth=0.4, alpha=0.35)
    return figure


def _plot_structures(records: list[dict[str, Any]]) -> plt.Figure:
    figure, axes = plt.subplots(
        1, len(records), figsize=(3.2 * len(records), 4.2),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    image = None
    for axis, record in zip(axes, records, strict=True):
        epsilon = np.asarray(record["binary_epsilon"])
        setup = record["setup"]
        image = axis.imshow(
            np.sqrt(epsilon),
            origin="lower",
            extent=(0.0, setup.grid.period, 0.0, setup.grid.height),
            aspect="equal",
            interpolation="nearest",
            vmin=1.45,
            vmax=3.7,
        )
        axis.set_title(f"Binary, {record['cells_per_period']} cells/period")
        axis.set_xlabel("x (µm)")
    axes[0].set_ylabel("z (µm)")
    assert image is not None
    figure.colorbar(image, ax=axes, label="Refractive index, n", shrink=0.82)
    return figure


def _plot_mesh_response(
    records: list[dict[str, Any]], common_pole: complex
) -> plt.Figure:
    figure, axes = plt.subplots(
        2, 1, figsize=(7.3, 6.6), sharex=True, constrained_layout=True
    )
    for record in records:
        omega = np.asarray(record["omega_rad_per_um"])
        order = np.argsort(omega)
        reflection = np.asarray(record["reflectance"])[order]
        deembedded = np.asarray(record["deembedded"])[order]
        phase = np.unwrap(np.angle(deembedded))
        phase -= phase[0]
        label = f"{record['cells_per_period']} cells/period"
        axes[0].semilogy(
            omega[order], np.maximum(reflection, 1e-12), label=label
        )
        axes[1].plot(omega[order], phase / np.pi, label=label)
    gamma = -common_pole.imag
    for axis in axes:
        axis.axvspan(
            common_pole.real - 3.0 * gamma,
            common_pole.real + 3.0 * gamma,
            color="0.5",
            alpha=0.08,
        )
        axis.grid(True, linewidth=0.4, alpha=0.35)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Reflectance")
    axes[0].set_title("Hard-binary response after multi-mesh reoptimization")
    axes[1].set_xlabel("ω (rad/µm)")
    axes[1].set_ylabel("De-embedded phase change / π")
    return figure


def _plot_roots(
    records: list[dict[str, Any]], common_pole: complex
) -> plt.Figure:
    figure, axes = plt.subplots(
        1, 3, figsize=(14.3, 4.5), constrained_layout=True
    )
    markers = ("o", "s", "D", "^")
    for record, marker in zip(records, markers, strict=False):
        label = f"{record['cells_per_period']} cells/period"
        parity = record["parity_roots"]
        axes[0].scatter(
            [_serialize_root(parity["even_parity_pole"]).real,
             _serialize_root(parity["odd_parity_pole"]).real],
            [_serialize_root(parity["even_parity_pole"]).imag,
             _serialize_root(parity["odd_parity_pole"]).imag],
            marker=marker,
            s=55,
            label=label,
        )
        channel = record["channel_roots"]
        transmission_zero = _serialize_root(channel["transmission_zero"])
        reflection_zero = _serialize_root(channel["reflection_zero"])
        axes[1].scatter(
            transmission_zero.real,
            transmission_zero.imag,
            marker=marker,
            s=55,
            label=label,
        )
        axes[2].scatter(
            reflection_zero.real,
            reflection_zero.imag,
            marker=marker,
            s=55,
            label=label,
        )
    axes[0].scatter(
        common_pole.real, common_pole.imag, marker="x", s=95,
        color="black", linewidths=1.8, label="Target pole"
    )
    axes[1].scatter(
        common_pole.real, -common_pole.imag, marker="x", s=95,
        color="black", linewidths=1.8, label="Target t zero"
    )
    axes[2].scatter(
        common_pole.real, 0.0, marker="+", s=110,
        color="black", linewidths=1.8, label="Nominal real-axis r zero"
    )
    axes[0].set_title("Verified hard-binary parity poles")
    axes[1].set_title("Transmission zeros")
    axes[2].set_title("Reflection zeros (real-axis zoom)")
    for axis in axes:
        axis.axhline(0.0, color="0.55", linewidth=0.7)
        axis.set_xlabel("Re(ω) (rad/µm)")
        axis.set_ylabel("Im(ω) (rad/µm)")
        axis.grid(True, linewidth=0.4, alpha=0.3)
        axis.legend(frameon=False, fontsize=7)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/m7_binary_mesh_reopt_tm.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/m7_binary_mesh_reopt_v2")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    arguments = parser.parse_args()

    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    output = arguments.output
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_metadata(
        output / "metadata.json",
        arguments.config,
        Path("data/literature/yu_2015_huygens_metrics.csv"),
    )
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    parent = Path(config["continuation"]["parent_run"])
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    metadata["parent_summary_sha256"] = _sha256(parent / "summary.json") if (parent / "summary.json").exists() else None
    metadata["parent_density_sha256"] = _sha256(parent / "optimized_raw_density.csv") if (parent / "optimized_raw_density.csv").exists() else None
    seed_run = config["continuation"].get("seed_run")
    if seed_run is not None:
        metadata["seed_master_density_sha256"] = _sha256(
            Path(seed_run) / "optimized_master_raw_density.csv"
        )
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    master_cells = int(config["continuation"]["master_cells_per_period"])
    master_setup, master_reference = _master_parameterization(config)
    current = _initial_master_density(config, master_reference)
    records_path = output / "continuation_records.json"
    stage_records: list[dict[str, Any]] = []
    if (arguments.resume or arguments.postprocess_only) and records_path.exists():
        stage_records = json.loads(records_path.read_text(encoding="utf-8"))
        if stage_records:
            current = np.loadtxt(
                output / stage_records[-1]["raw_density_file"], delimiter=","
            ).ravel()
    if arguments.postprocess_only and not stage_records:
        raise FileNotFoundError("postprocess-only requires completed stage records")

    common_pole = _common_pole(config)
    gamma = -common_pole.imag
    pole_targets = (
        PoleTarget(common_pole, -1.0),
        PoleTarget(common_pole, 1.0),
    )
    reflection_targets = tuple(
        (common_pole.real + float(offset) * gamma, 0.0j)
        for offset in config["target"]["reflection_sample_offsets_in_gamma"]
    )
    stages = list(config["continuation"]["stages"])
    if not arguments.postprocess_only:
        for index, stage in enumerate(stages, start=1):
            if index <= len(stage_records):
                continue
            beta = float(stage["projection_beta"])
            systems = []
            for cells_value in stage["mesh_cells_per_period"]:
                cells = int(cells_value)
                setup, parameterization = _target_parameterization(
                    config,
                    master_setup,
                    master_reference.design_rows,
                    cells,
                    beta,
                )
                systems.append((cells, setup, parameterization))
            identifier = f"stage-{index:02d}-beta{beta:g}"
            result, evaluations = _run_multimesh_stage(
                identifier,
                current,
                systems,
                pole_targets,
                np.conj(common_pole),
                reflection_targets,
                config,
                iterations=int(stage["max_iterations"]),
                residual_tolerance=float(stage["residual_tolerance"]),
            )
            residual = float(np.linalg.norm(result.residual))
            accepted = residual <= float(
                config["optimization"]["stage_acceptance_residual"]
            )
            raw_file = f"raw_density_{identifier}.csv"
            np.savetxt(
                output / raw_file,
                result.x.reshape(master_reference.design_shape),
                delimiter=",",
            )
            write_history(output / f"history_{identifier}.csv", result)
            master_mapping = systems[-1][2].map(result.x)
            record = {
                "index": index,
                "identifier": identifier,
                "projection_beta": beta,
                "mesh_cells_per_period": [item[0] for item in systems],
                "function_evaluations": evaluations,
                "accepted_iterations": len(result.history),
                "solver_converged": result.converged,
                "reason": result.reason,
                "residual_norm": residual,
                "accepted": accepted,
                "master_binarity": binarity_index(master_mapping.projected),
                "master_gray_fraction": gray_fraction(master_mapping.projected),
                "raw_density_file": raw_file,
            }
            stage_records.append(record)
            records_path.write_text(
                json.dumps(stage_records, indent=2) + "\n", encoding="utf-8"
            )
            if not accepted:
                raise RuntimeError(
                    f"{identifier} residual {residual:.3e} exceeds acceptance"
                )
            current = result.x

    np.savetxt(
        output / "optimized_master_raw_density.csv",
        current.reshape(master_reference.design_shape),
        delimiter=",",
    )
    final_beta = float(stages[len(stage_records) - 1]["projection_beta"])
    wavelengths = _band_wavelengths(
        common_pole, int(config["continuation"]["band_samples"])
    )
    threshold = float(config["continuation"]["hard_threshold"])
    raster_subpixels = int(
        config["continuation"].get("binary_rasterization_subpixels", 1)
    )
    _, physical_parameterization = _target_parameterization(
        config,
        master_setup,
        master_reference.design_rows,
        master_cells,
        final_beta,
    )
    physical_master_density = projected_density_from_permittivity(
        physical_parameterization.map(current).epsilon_r,
        physical_parameterization.low_index,
        physical_parameterization.high_index,
    )
    validation_records = []
    plot_records = []
    for cells_value in config["continuation"]["validation_cells_per_period"]:
        cells = int(cells_value)
        candidate_setup, parameterization = _target_parameterization(
            config,
            master_setup,
            master_reference.design_rows,
            cells,
            final_beta,
        )
        setup = candidate_setup
        mapping = parameterization.map(current)
        projected_full = projected_density_from_permittivity(
            mapping.epsilon_r,
            parameterization.low_index,
            parameterization.high_index,
        )
        if raster_subpixels > 1:
            binary_epsilon, binary_fill_fraction = subpixel_binary_permittivity(
                physical_master_density,
                setup.grid.shape,
                threshold,
                parameterization.low_index,
                parameterization.high_index,
                subpixels=raster_subpixels,
            )
        else:
            binary_epsilon = binary_permittivity(
                projected_full,
                threshold,
                parameterization.low_index,
                parameterization.high_index,
            )
            binary_fill_fraction = (
                projected_full >= threshold
            ).astype(float)
        setup, monitor_shifts = _homogeneous_monitor_setup(binary_epsilon, setup)
        continuous_metrics, continuous_spectrum = _spectrum_metrics(
            mapping.epsilon_r, setup, wavelengths, common_pole, config
        )
        binary_metrics, binary_spectrum = _spectrum_metrics(
            binary_epsilon, setup, wavelengths, common_pole, config
        )
        parity, channel, root_errors = _root_metrics(
            binary_epsilon, setup, common_pole, config
        )
        write_spectrum(
            output / f"continuous_spectrum_n{cells}.csv",
            wavelengths,
            *continuous_spectrum,
        )
        write_spectrum(
            output / f"binary_spectrum_n{cells}.csv",
            wavelengths,
            *binary_spectrum,
        )
        np.savetxt(
            output / f"projected_density_n{cells}.csv",
            projected_full,
            delimiter=",",
        )
        np.savetxt(
            output / f"binary_permittivity_n{cells}.csv",
            binary_epsilon,
            delimiter=",",
        )
        np.savetxt(
            output / f"binary_fill_fraction_n{cells}.csv",
            binary_fill_fraction,
            delimiter=",",
        )
        transmission, reflection, empty = binary_spectrum
        deembedded = transmission / empty
        record = {
            "cells_per_period": cells,
            "grid_shape": list(setup.grid.shape),
            "continuous_binarity_index": binarity_index(projected_full),
            "continuous_gray_fraction": gray_fraction(projected_full),
            "binary_geometry_representation": (
                "fixed master-grid contour with subpixel permittivity area averaging"
                if raster_subpixels > 1
                else "cell-center threshold"
            ),
            "binary_rasterization_subpixels_per_axis": raster_subpixels,
            "lower_monitor_row_shifts": monitor_shifts,
            "continuous_metrics": continuous_metrics,
            "binary_metrics": binary_metrics,
            "parity_roots": parity,
            "channel_roots": channel,
            "root_errors": root_errors,
        }
        validation_records.append(record)
        plot_records.append(
            {
                **record,
                "setup": setup,
                "binary_epsilon": binary_epsilon,
                "omega_rad_per_um": 2.0 * np.pi / wavelengths,
                "reflectance": np.abs(reflection) ** 2,
                "deembedded": deembedded,
            }
        )

    robustness = []
    setup, parameterization = _target_parameterization(
        config,
        master_setup,
        master_reference.design_rows,
        master_cells,
        final_beta,
    )
    projected = physical_master_density
    for value in config["continuation"]["threshold_robustness_values"]:
        if raster_subpixels > 1:
            candidate, _ = subpixel_binary_permittivity(
                projected,
                setup.grid.shape,
                float(value),
                parameterization.low_index,
                parameterization.high_index,
                subpixels=raster_subpixels,
            )
        else:
            candidate = binary_permittivity(
                projected,
                float(value),
                parameterization.low_index,
                parameterization.high_index,
            )
        metrics, _ = _spectrum_metrics(
            candidate, setup, wavelengths, common_pole, config
        )
        robustness.append({"threshold": float(value), "metrics": metrics})

    limits = config["verification"]
    gates = []
    for record in validation_records:
        metrics = record["binary_metrics"]
        gates.append(
            bool(
                metrics["band_maximum_reflectance"]
                <= float(limits["maximum_band_reflectance"])
                and metrics["band_minimum_transmittance"]
                >= float(limits["minimum_band_transmittance"])
                and metrics["phase_window_span_pi"]
                >= float(limits["minimum_phase_span_pi"])
                and max(record["root_errors"].values())
                <= float(limits["maximum_root_error_over_linewidth"])
            )
        )

    figure = _plot_continuation(stage_records)
    save_figure(figure, figures / "m7_continuation.png", dpi=220)
    plt.close(figure)
    figure = _plot_structures(plot_records)
    save_figure(figure, figures / "m7_binary_structures.png", dpi=220)
    plt.close(figure)
    figure = _plot_mesh_response(plot_records, common_pole)
    save_figure(figure, figures / "m7_binary_mesh_response.png", dpi=220)
    plt.close(figure)
    figure = _plot_roots(validation_records, common_pole)
    save_figure(figure, figures / "m7_binary_complex_roots.png", dpi=220)
    plt.close(figure)

    summary = {
        "stage": "M7 multi-mesh binary-aware reoptimization",
        "parent_run": str(parent),
        "seed_run": seed_run,
        "master_cells_per_period": master_cells,
        "master_design_rows": list(master_reference.design_rows),
        "master_variable_count": int(current.size),
        "hard_threshold": threshold,
        "binary_rasterization_subpixels_per_axis": raster_subpixels,
        "target_common_pole": {
            "real": common_pole.real,
            "imag": common_pole.imag,
        },
        "continuation_stages": stage_records,
        "validation": validation_records,
        "threshold_robustness": robustness,
        "mesh_gates": gates,
        "all_meshes_gate_pass": all(gates),
        "interpretation": (
            f"N is cells per period. One N={master_cells} master design was "
            "constrained on each mesh listed in the continuation records "
            "through an exact resampling adjoint; validation meshes are listed "
            "separately in this summary."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
