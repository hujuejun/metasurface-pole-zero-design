"""Connected, binary-aware, multi-mesh reoptimization of the M10 design."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.ndimage import binary_dilation

from pzid.fabrication import (
    binarity_index,
    gray_fraction,
    periodic_bilinear_resample,
    periodic_component_labels,
    periodic_component_sizes,
    projected_density_from_permittivity,
    subpixel_binary_permittivity,
)
from pzid.plotting import save_figure
from pzid.poles import PoleTarget

from scripts.m3_optimize_pole_zero import write_history
from scripts.m3_single_resonance import write_metadata
from scripts.m4b_optimize_kerker_bar import _frequency, write_spectrum
from scripts.m6_fabrication_continue import (
    _band_wavelengths,
    _root_metrics,
    _spectrum_metrics,
)
from scripts.m5_optimize_coupled_bars import (
    _refine_channel_roots,
    _refine_parity_poles,
)
from scripts.m7_binary_mesh_reopt import (
    _homogeneous_monitor_setup,
    _initial_master_density,
    _master_parameterization,
    _run_multimesh_stage,
    _target_parameterization,
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


def _periodic_dilation(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    coordinate = np.arange(-radius, radius + 1)
    zz, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    footprint = zz**2 + xx**2 <= radius**2
    tiled = np.tile(np.asarray(mask, dtype=bool), (1, 3))
    grown = binary_dilation(tiled, structure=footprint)
    nx = mask.shape[1]
    return np.asarray(grown[:, nx : 2 * nx])


def _shortest_periodic_path(
    density: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    void_cost: float,
) -> np.ndarray:
    """Return a minimum-cost four-neighbor path on an axial-periodic grid."""

    values = np.asarray(density, dtype=float)
    if values.shape != source.shape or values.shape != target.shape:
        raise ValueError("density and path masks must have matching shapes")
    if not np.any(source) or not np.any(target):
        raise ValueError("source and target masks must be nonempty")
    nz, nx = values.shape
    costs = 1.0 + float(void_cost) * (1.0 - np.clip(values, 0.0, 1.0)) ** 2
    distance = np.full(values.shape, np.inf)
    previous_z = np.full(values.shape, -1, dtype=int)
    previous_x = np.full(values.shape, -1, dtype=int)
    queue: list[tuple[float, int, int]] = []
    for z_index, x_index in np.argwhere(source):
        distance[z_index, x_index] = 0.0
        heapq.heappush(queue, (0.0, int(z_index), int(x_index)))
    endpoint: tuple[int, int] | None = None
    while queue:
        current_distance, z_index, x_index = heapq.heappop(queue)
        if current_distance != distance[z_index, x_index]:
            continue
        if target[z_index, x_index]:
            endpoint = (z_index, x_index)
            break
        for neighbor_z, neighbor_x in (
            (z_index - 1, x_index),
            (z_index + 1, x_index),
            (z_index, (x_index - 1) % nx),
            (z_index, (x_index + 1) % nx),
        ):
            if not 0 <= neighbor_z < nz:
                continue
            trial = current_distance + 0.5 * (
                costs[z_index, x_index] + costs[neighbor_z, neighbor_x]
            )
            if trial < distance[neighbor_z, neighbor_x]:
                distance[neighbor_z, neighbor_x] = trial
                previous_z[neighbor_z, neighbor_x] = z_index
                previous_x[neighbor_z, neighbor_x] = x_index
                heapq.heappush(queue, (trial, neighbor_z, neighbor_x))
    if endpoint is None:
        raise RuntimeError("no periodic bridge path was found")
    path = np.zeros(values.shape, dtype=bool)
    z_index, x_index = endpoint
    while not source[z_index, x_index]:
        path[z_index, x_index] = True
        next_z = previous_z[z_index, x_index]
        next_x = previous_x[z_index, x_index]
        if next_z < 0:
            raise RuntimeError("periodic bridge predecessor chain is incomplete")
        z_index, x_index = int(next_z), int(next_x)
    path[z_index, x_index] = True
    return path


def _connectivity_plan(
    density: np.ndarray,
    *,
    threshold: float,
    minimum_component_cells: int,
    bridge_radius_cells: int,
    void_cost: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Connect retained components and suppress undersized isolated islands."""

    values = np.asarray(density, dtype=float)
    mask = values >= threshold
    labels, count = periodic_component_labels(mask)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    retained = [
        label
        for label in range(1, count + 1)
        if sizes[label] >= minimum_component_cells
    ]
    if not retained and count:
        retained = [int(np.argmax(sizes[1:]) + 1)]
    if not retained:
        raise RuntimeError("connectivity planning requires nonempty material")
    middle = (values.shape[0] - 1) / 2.0
    anchor = max(
        retained,
        key=lambda label: (
            -abs(float(np.mean(np.argwhere(labels == label)[:, 0])) - middle),
            int(sizes[label]),
        ),
    )
    connected = labels == anchor
    bridge = np.zeros(mask.shape, dtype=bool)
    ordering = sorted(
        (label for label in retained if label != anchor),
        key=lambda label: int(sizes[label]),
        reverse=True,
    )
    paths = []
    for label in ordering:
        target = labels == label
        path = _shortest_periodic_path(values, connected, target, void_cost)
        bridge |= path
        connected |= target | path
        paths.append(int(np.count_nonzero(path)))
    forced = _periodic_dilation(bridge, bridge_radius_cells)
    forced |= np.flipud(forced)
    suppressed = np.isin(
        labels,
        [label for label in range(1, count + 1) if label not in retained],
    )
    suppressed |= np.flipud(suppressed)
    suppressed &= ~forced
    repaired = (mask & ~suppressed) | forced
    repaired |= np.flipud(repaired)
    record = {
        "components_before": count,
        "component_sizes_before": [int(value) for value in sorted(sizes[1:], reverse=True)],
        "retained_component_labels": retained,
        "anchor_component_label": anchor,
        "shortest_path_cells_before_dilation": paths,
        "forced_material_cells": int(np.count_nonzero(forced)),
        "suppressed_material_cells": int(np.count_nonzero(suppressed)),
        "components_after_plan": len(periodic_component_sizes(repaired)),
        "component_sizes_after_plan": list(periodic_component_sizes(repaired)),
    }
    return forced, suppressed, record


def _independent_mask(full_mask: np.ndarray, rows: int) -> np.ndarray:
    symmetric = np.asarray(full_mask, dtype=bool) | np.flipud(full_mask)
    return np.asarray(symmetric[:rows])


def _density_on_master(
    config: dict[str, Any],
    master_setup: Any,
    master_rows: tuple[int, int],
    current: np.ndarray,
    beta: float,
    filter_radius: float | None = None,
) -> tuple[np.ndarray, Any]:
    local_config = json.loads(json.dumps(config))
    if filter_radius is not None:
        local_config["parameterization"]["filter_radius_um"] = float(filter_radius)
    setup, parameterization = _target_parameterization(
        local_config,
        master_setup,
        master_rows,
        int(config["continuation"]["master_cells_per_period"]),
        beta,
    )
    mapping = parameterization.map(current)
    start, stop = parameterization.target.design_rows
    full = projected_density_from_permittivity(
        mapping.epsilon_r,
        parameterization.low_index,
        parameterization.high_index,
    )
    return np.asarray(full[start:stop]), setup


def _root_residuals(parity: dict[str, Any], channel: dict[str, Any]) -> dict[str, float]:
    return {
        "even": float(parity["even_parity_pole"]["residual_norm"]),
        "odd": float(parity["odd_parity_pole"]["residual_norm"]),
        "transmission_pole": float(channel["transmission_pole"]["residual_norm"]),
        "transmission_zero": float(channel["transmission_zero"]["residual_norm"]),
        "reflection_zero": float(channel["reflection_zero"]["residual_norm"]),
    }


def _plot_structure(
    density: np.ndarray,
    mask: np.ndarray,
    config: dict[str, Any],
) -> plt.Figure:
    period = float(config["benchmark"]["period_um"])
    master = int(config["continuation"]["master_cells_per_period"])
    dz = period / master
    extent = (0.0, period, 0.0, density.shape[0] * dz)
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 6.0), constrained_layout=True)
    first = axes[0].imshow(
        density,
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        cmap="gray_r",
        interpolation="nearest",
    )
    axes[0].set_title("Projected master density")
    axes[1].imshow(
        mask,
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        cmap="gray_r",
        interpolation="nearest",
    )
    axes[1].set_title("Connected hard-binary contour")
    for axis in axes:
        axis.set_xlabel("x (µm)")
        axis.set_ylabel("local z (µm)")
    figure.colorbar(first, ax=axes[0], label="Material density", shrink=0.75)
    return figure


def _plot_response(records: Sequence[dict[str, Any]], common: complex) -> plt.Figure:
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True, constrained_layout=True)
    for record in records:
        omega = np.asarray(record["omega_rad_per_um"])
        order = np.argsort(omega)
        axes[0].semilogy(
            (omega[order] - common.real) / (-common.imag),
            np.maximum(np.asarray(record["reflectance"])[order], 1e-12),
            label=f"N={record['cells_per_period']}",
        )
        phase = np.unwrap(np.angle(np.asarray(record["deembedded"])[order]))
        phase -= phase[0]
        axes[1].plot(
            (omega[order] - common.real) / (-common.imag),
            phase / np.pi,
            label=f"N={record['cells_per_period']}",
        )
    axes[0].axhline(0.05, color="tab:red", linestyle=":", label="R gate")
    axes[1].axhline(1.8, color="tab:red", linestyle=":", label="phase gate")
    axes[0].set_ylabel("Reflectance")
    axes[1].set_ylabel("Phase change / π")
    axes[1].set_xlabel("Normalized detuning $(\\omega-\\mathrm{Re}\\,\\omega_p)/\\gamma$")
    for axis in axes:
        axis.grid(True, linewidth=0.4, alpha=0.35)
        axis.legend(frameon=False)
    figure.suptitle("Connected hard-binary multi-mesh validation")
    return figure


def _plot_roots(records: Sequence[dict[str, Any]], common: complex) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 5.4), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    mesh_handles = []
    for color, record in zip(colors, records, strict=False):
        cells = record["cells_per_period"]
        parity = record["parity_roots"]
        channel = record["channel_roots"]
        even = _frequency(parity["even_parity_pole"])
        odd = _frequency(parity["odd_parity_pole"])
        transmission_pole = _frequency(channel["transmission_pole"])
        transmission_zero = _frequency(channel["transmission_zero"])
        reflection_zero = _frequency(channel["reflection_zero"])
        axes[0].scatter(even.real, even.imag, marker="o", color=color)
        axes[0].scatter(odd.real, odd.imag, marker="s", color=color)
        axes[0].scatter(
            transmission_pole.real,
            transmission_pole.imag,
            marker="x",
            color=color,
            s=55,
        )
        axes[0].scatter(
            transmission_zero.real,
            transmission_zero.imag,
            marker="v",
            color=color,
        )
        axes[1].scatter(
            reflection_zero.real,
            reflection_zero.imag,
            marker="D",
            color=color,
            s=42,
        )
        mesh_handles.append(
            Line2D([0], [0], color=color, marker="o", linestyle="none", label=f"N={cells}")
        )
    axes[0].scatter(common.real, common.imag, marker="*", s=160, color="k")
    axes[0].scatter(common.real, -common.imag, marker="+", s=140, color="k")
    axes[1].axhline(0.0, color="k", linestyle=":", linewidth=1.0)
    axes[0].set_title("Pole / transmission-zero plane")
    axes[1].set_title("Reflection zeros")
    axes[0].set_ylabel("Im(ω) (rad/µm)")
    for axis in axes:
        axis.axhline(0.0, color="0.5", linewidth=0.8)
        axis.set_xlabel("Re(ω) (rad/µm)")
        axis.grid(True, linewidth=0.4, alpha=0.35)
    marker_handles = [
        Line2D([0], [0], color="0.2", marker="o", linestyle="none", label="even-parity pole"),
        Line2D([0], [0], color="0.2", marker="s", linestyle="none", label="odd-parity pole"),
        Line2D([0], [0], color="0.2", marker="x", linestyle="none", label="transmission pole"),
        Line2D([0], [0], color="0.2", marker="v", linestyle="none", label="transmission zero"),
        Line2D([0], [0], color="0.2", marker="D", linestyle="none", label="reflection zero"),
        Line2D([0], [0], color="k", marker="*", linestyle="none", markersize=11, label="target pole"),
        Line2D([0], [0], color="k", marker="+", linestyle="none", markersize=10, label="target transmission zero"),
    ]
    first_legend = axes[0].legend(handles=mesh_handles, frameon=False, fontsize=8, loc="upper left")
    axes[0].add_artist(first_legend)
    axes[0].legend(handles=marker_handles, frameon=False, fontsize=7, loc="upper right")
    axes[1].legend(handles=mesh_handles, frameon=False, fontsize=8)
    figure.suptitle("Independently refined connected-design singularities")
    return figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/m10b_connected_binary_multimesh_tm.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/m10b_connected_binary_multimesh")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    arguments = parser.parse_args()

    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    if not isinstance(config["provenance"], dict):
        raise ValueError("The low-index continuation requires a provenance record")
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
    metadata["parent_density_sha256"] = _sha256(parent / "optimized_raw_density.csv")
    metadata["connectivity_source_sha256"] = _sha256(Path(__file__))
    metadata["channel_root_source_sha256"] = _sha256(
        Path("scripts/m5_optimize_coupled_bars.py")
    )
    metadata["validation_source_sha256"] = _sha256(
        Path("scripts/m6_fabrication_continue.py")
    )
    metadata["fabrication_source_sha256"] = _sha256(Path("src/pzid/fabrication.py"))
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    controls = config["continuation"]
    common = _common_pole(config)
    master_cells = int(controls["master_cells_per_period"])
    master_setup, master_reference = _master_parameterization(config)
    current = _initial_master_density(config, master_reference)
    stage_path = output / "continuation_records.json"
    bounds_path = output / "connectivity_bounds.npz"
    stage_records: list[dict[str, Any]] = []
    force_independent = np.zeros(master_reference.design_shape, dtype=bool)
    suppress_independent = np.zeros(master_reference.design_shape, dtype=bool)
    if (arguments.resume or arguments.postprocess_only) and stage_path.exists():
        stage_records = json.loads(stage_path.read_text(encoding="utf-8"))
        if stage_records:
            current = np.loadtxt(output / stage_records[-1]["raw_density_file"], delimiter=",").ravel()
            saved_bounds = np.load(bounds_path)
            force_independent = saved_bounds["forced_material"].astype(bool)
            suppress_independent = saved_bounds["suppressed_material"].astype(bool)
    if arguments.postprocess_only and not stage_records:
        raise FileNotFoundError("postprocess-only requires completed stage records")

    dx = float(config["benchmark"]["period_um"]) / master_cells
    minimum_cells = max(
        1,
        int(np.ceil(float(controls["minimum_retained_component_area_um2"]) / dx**2)),
    )
    bridge_radius = max(
        1,
        int(np.ceil(0.5 * float(controls["minimum_bridge_width_um"]) / dx)),
    )
    connectivity_history: list[dict[str, Any]] = []
    bridge_lower_bound = float(controls["bridge_density_lower_bound"])
    if not 0.5 < bridge_lower_bound <= 1.0:
        raise ValueError("bridge_density_lower_bound must lie in (0.5, 1]")
    if not stage_records:
        initial_density, _ = _density_on_master(
            config, master_setup, master_reference.design_rows, current, 2.0, 0.0
        )
        forced, suppressed, plan = _connectivity_plan(
            initial_density,
            threshold=float(controls["hard_threshold"]),
            minimum_component_cells=minimum_cells,
            bridge_radius_cells=bridge_radius,
            void_cost=float(controls["bridge_void_cost"]),
        )
        force_independent |= _independent_mask(forced, master_reference.design_shape[0])
        suppress_independent |= _independent_mask(suppressed, master_reference.design_shape[0])
        suppress_independent &= ~force_independent
        plan["stage"] = "initial_backbone"
        connectivity_history.append(plan)

    lower = np.zeros(master_reference.design_shape, dtype=float)
    upper = np.ones(master_reference.design_shape, dtype=float)
    np.savetxt(
        output / "prepared_optical_seed_raw_density.csv",
        current.reshape(master_reference.design_shape),
        delimiter=",",
    )
    np.savez(
        bounds_path,
        forced_material=force_independent,
        suppressed_material=suppress_independent,
    )
    (output / "connectivity_history.json").write_text(
        json.dumps(connectivity_history, indent=2) + "\n", encoding="utf-8"
    )

    precondition = controls["preconditioning"]
    seed_beta = float(precondition["projection_beta"])
    seed_cells = int(precondition["mesh_cells_per_period"][0])
    seed_config = json.loads(json.dumps(config))
    seed_config["parameterization"]["filter_radius_um"] = float(
        precondition["filter_radius_um"]
    )
    seed_setup, seed_parameterization = _target_parameterization(
        seed_config,
        master_setup,
        master_reference.design_rows,
        seed_cells,
        seed_beta,
    )
    seed_epsilon = seed_parameterization.map(current).epsilon_r
    initial_even = complex(
        float(config["target"]["initial_even_pole_real_rad_per_um"]),
        float(config["target"]["initial_even_pole_imag_rad_per_um"]),
    )
    initial_odd = complex(
        float(config["target"]["initial_odd_pole_real_rad_per_um"]),
        float(config["target"]["initial_odd_pole_imag_rad_per_um"]),
    )
    filtered_seed_parity = _refine_parity_poles(
        seed_epsilon, seed_setup, initial_even, initial_odd, config
    )
    filtered_seed_channels = _refine_channel_roots(
        seed_epsilon, seed_setup, common, config
    )
    filtered_seed = {
        "parity_roots": filtered_seed_parity,
        "channel_roots": filtered_seed_channels,
        "periodic_component_sizes": list(
            periodic_component_sizes(
                _density_on_master(
                    config,
                    master_setup,
                    master_reference.design_rows,
                    current,
                    seed_beta,
                    float(precondition["filter_radius_um"]),
                )[0]
                >= float(controls["hard_threshold"])
            )
        ),
    }
    (output / "prepared_filtered_optical_seed.json").write_text(
        json.dumps(filtered_seed, indent=2) + "\n", encoding="utf-8"
    )
    if arguments.prepare_only:
        print(json.dumps(filtered_seed, indent=2), flush=True)
        return

    precondition_path = output / "preconditioning_records.json"
    precondition_records: list[dict[str, Any]] = []
    if arguments.resume and precondition_path.exists():
        precondition_records = json.loads(precondition_path.read_text(encoding="utf-8"))
        if not stage_records and precondition_records:
            current = np.loadtxt(
                output / precondition_records[-1]["raw_density_file"], delimiter=","
            ).ravel()
    if not stage_records and not arguments.postprocess_only:
        precondition_bridge_bound = float(
            precondition["bridge_density_lower_bound"]
        )
        upper.fill(1.0)
        lower[force_independent] = precondition_bridge_bound
        current = np.clip(
            current.reshape(master_reference.design_shape), lower, upper
        ).ravel()
        systems = [(seed_cells, seed_setup, seed_parameterization)]
        seed_even = _frequency(filtered_seed_parity["even_parity_pole"])
        seed_odd = _frequency(filtered_seed_parity["odd_parity_pole"])
        for index, fraction_value in enumerate(precondition["pole_homotopy_fractions"]):
            fraction = float(fraction_value)
            even_target = seed_even + fraction * (common - seed_even)
            odd_target = seed_odd + fraction * (common - seed_odd)
            identifier = f"pre-poles-{index:02d}-fraction{fraction:g}"
            result, evaluations = _run_multimesh_stage(
                identifier,
                current,
                systems,
                (PoleTarget(even_target, -1.0), PoleTarget(odd_target, 1.0)),
                None,
                (),
                config,
                iterations=int(precondition["max_iterations"]),
                residual_tolerance=float(
                    precondition["pole_waypoint_residual_tolerance"]
                ),
                lower_bound=lower.ravel(),
                upper_bound=upper.ravel(),
            )
            residual = float(np.linalg.norm(result.residual))
            accepted = residual <= float(
                precondition["pole_waypoint_acceptance_residual"]
            )
            raw_file = f"raw_density_{identifier}.csv"
            np.savetxt(output / raw_file, result.x.reshape(master_reference.design_shape), delimiter=",")
            write_history(output / f"history_{identifier}.csv", result)
            record = {
                "identifier": identifier,
                "group": "connected-seed pole homotopy",
                "fraction": fraction,
                "residual_norm": residual,
                "accepted": accepted,
                "function_evaluations": evaluations,
                "raw_density_file": raw_file,
            }
            precondition_records.append(record)
            precondition_path.write_text(
                json.dumps(precondition_records, indent=2) + "\n", encoding="utf-8"
            )
            if not accepted:
                raise RuntimeError(f"{identifier} failed at residual {residual:.3e}")
            current = result.x

        current_epsilon = seed_parameterization.map(current).epsilon_r
        current_channels = _refine_channel_roots(
            current_epsilon, seed_setup, common, config
        )
        seed_zero = _frequency(current_channels["transmission_zero"])
        for index, fraction_value in enumerate(
            precondition["transmission_zero_homotopy_fractions"]
        ):
            fraction = float(fraction_value)
            zero_target = seed_zero + fraction * (np.conj(common) - seed_zero)
            identifier = f"pre-zero-{index:02d}-fraction{fraction:g}"
            result, evaluations = _run_multimesh_stage(
                identifier,
                current,
                systems,
                (PoleTarget(common, -1.0), PoleTarget(common, 1.0)),
                zero_target,
                (),
                config,
                iterations=int(precondition["max_iterations"]),
                residual_tolerance=float(precondition["residual_tolerance"]),
                lower_bound=lower.ravel(),
                upper_bound=upper.ravel(),
            )
            residual = float(np.linalg.norm(result.residual))
            accepted = residual <= float(precondition["acceptance_residual"])
            raw_file = f"raw_density_{identifier}.csv"
            np.savetxt(output / raw_file, result.x.reshape(master_reference.design_shape), delimiter=",")
            write_history(output / f"history_{identifier}.csv", result)
            record = {
                "identifier": identifier,
                "group": "transmission-zero homotopy",
                "fraction": fraction,
                "target_frequency": [zero_target.real, zero_target.imag],
                "residual_norm": residual,
                "accepted": accepted,
                "function_evaluations": evaluations,
                "raw_density_file": raw_file,
            }
            precondition_records.append(record)
            precondition_path.write_text(
                json.dumps(precondition_records, indent=2) + "\n", encoding="utf-8"
            )
            if not accepted:
                raise RuntimeError(f"{identifier} failed at residual {residual:.3e}")
            current = result.x

        offsets = [float(value) for value in config["target"]["reflection_sample_offsets_in_gamma"]]
        for index, maximum_value in enumerate(precondition["reflection_band_groups_in_gamma"]):
            maximum = float(maximum_value)
            reflection = tuple(
                (common.real + offset * (-common.imag), 0.0j)
                for offset in offsets
                if abs(offset) <= maximum + 1e-12
            )
            identifier = f"pre-reflection-{index:02d}-{maximum:g}gamma"
            result, evaluations = _run_multimesh_stage(
                identifier,
                current,
                systems,
                (PoleTarget(common, -1.0), PoleTarget(common, 1.0)),
                np.conj(common),
                reflection,
                config,
                iterations=int(precondition["max_iterations"]),
                residual_tolerance=float(precondition["residual_tolerance"]),
                lower_bound=lower.ravel(),
                upper_bound=upper.ravel(),
            )
            residual = float(np.linalg.norm(result.residual))
            accepted = residual <= float(precondition["acceptance_residual"])
            raw_file = f"raw_density_{identifier}.csv"
            np.savetxt(output / raw_file, result.x.reshape(master_reference.design_shape), delimiter=",")
            write_history(output / f"history_{identifier}.csv", result)
            record = {
                "identifier": identifier,
                "group": "reflection-band homotopy",
                "maximum_offset_in_gamma": maximum,
                "residual_norm": residual,
                "accepted": accepted,
                "function_evaluations": evaluations,
                "raw_density_file": raw_file,
            }
            precondition_records.append(record)
            precondition_path.write_text(
                json.dumps(precondition_records, indent=2) + "\n", encoding="utf-8"
            )
            if not accepted:
                raise RuntimeError(f"{identifier} failed at residual {residual:.3e}")
            current = result.x

    stages = list(controls["stages"])
    if not arguments.postprocess_only:
        for stage_index, stage in enumerate(stages, start=1):
            if stage_index <= len(stage_records):
                continue
            beta = float(stage["projection_beta"])
            filter_radius = float(stage["filter_radius_um"])
            stage_bridge_bound = float(stage["bridge_density_lower_bound"])
            suppress_islands = stage_bridge_bound > 0.0
            require_connectivity = bool(stage["require_connectivity"])
            lower = np.zeros(master_reference.design_shape, dtype=float)
            upper = np.ones(master_reference.design_shape, dtype=float)
            lower[force_independent] = stage_bridge_bound
            if suppress_islands:
                upper[suppress_independent] = 0.0
            current = np.clip(current.reshape(master_reference.design_shape), lower, upper).ravel()
            systems = []
            system_labels = []
            stage_config = json.loads(json.dumps(config))
            stage_config["parameterization"]["filter_radius_um"] = filter_radius
            for threshold in stage["projection_thresholds"]:
                threshold_config = json.loads(json.dumps(stage_config))
                threshold_config["parameterization"]["projection_threshold"] = float(threshold)
                for cells_value in stage["mesh_cells_per_period"]:
                    setup, parameterization = _target_parameterization(
                        threshold_config,
                        master_setup,
                        master_reference.design_rows,
                        int(cells_value),
                        beta,
                    )
                    systems.append((int(cells_value), setup, parameterization))
                    system_labels.append(
                        {"cells_per_period": int(cells_value), "projection_threshold": float(threshold)}
                    )
            reflection = tuple(
                (common.real + float(offset) * (-common.imag), 0.0j)
                for offset in stage["reflection_offsets_in_gamma"]
            )
            repairs = []
            result = None
            evaluations_total = 0
            for repair_round in range(int(controls["maximum_connectivity_repairs_per_stage"]) + 1):
                identifier = f"stage-{stage_index:02d}-beta{beta:g}-repair{repair_round}"
                result, evaluations = _run_multimesh_stage(
                    identifier,
                    current,
                    systems,
                    (PoleTarget(common, -1.0), PoleTarget(common, 1.0)),
                    np.conj(common),
                    reflection,
                    config,
                    iterations=int(stage["max_iterations"]),
                    residual_tolerance=float(stage["residual_tolerance"]),
                    lower_bound=lower.ravel(),
                    upper_bound=upper.ravel(),
                )
                evaluations_total += evaluations
                current = result.x
                density, _ = _density_on_master(
                    stage_config, master_setup, master_reference.design_rows, current, beta
                )
                component_sizes = periodic_component_sizes(
                    density >= float(controls["hard_threshold"])
                )
                if len(component_sizes) == 1 or not require_connectivity:
                    break
                forced, suppressed, plan = _connectivity_plan(
                    density,
                    threshold=float(controls["hard_threshold"]),
                    minimum_component_cells=minimum_cells,
                    bridge_radius_cells=bridge_radius,
                    void_cost=float(controls["bridge_void_cost"]),
                )
                new_force = _independent_mask(forced, master_reference.design_shape[0])
                new_suppress = _independent_mask(suppressed, master_reference.design_shape[0])
                force_independent |= new_force
                suppress_independent |= new_suppress
                suppress_independent &= ~force_independent
                lower[force_independent] = stage_bridge_bound
                upper[suppress_independent] = 0.0
                current = np.clip(current.reshape(master_reference.design_shape), lower, upper).ravel()
                plan["stage"] = stage_index
                plan["repair_round"] = repair_round + 1
                repairs.append(plan)
                connectivity_history.append(plan)
            assert result is not None
            residual = float(np.linalg.norm(result.residual))
            density, _ = _density_on_master(
                    stage_config, master_setup, master_reference.design_rows, current, beta
            )
            final_sizes = periodic_component_sizes(density >= float(controls["hard_threshold"]))
            accepted = bool(
                residual <= float(config["optimization"]["stage_acceptance_residual"])
                and (len(final_sizes) == 1 or not require_connectivity)
            )
            raw_file = f"raw_density_stage-{stage_index:02d}-beta{beta:g}.csv"
            np.savetxt(output / raw_file, current.reshape(master_reference.design_shape), delimiter=",")
            write_history(output / f"history_stage-{stage_index:02d}-beta{beta:g}.csv", result)
            record = {
                "index": stage_index,
                "projection_beta": beta,
                "filter_radius_um": filter_radius,
                "bridge_density_lower_bound": stage_bridge_bound,
                "island_suppression_active": suppress_islands,
                "connectivity_required": require_connectivity,
                "systems": system_labels,
                "reflection_offsets_in_gamma": stage["reflection_offsets_in_gamma"],
                "function_evaluations": evaluations_total,
                "accepted_iterations": len(result.history),
                "reason": result.reason,
                "residual_norm": residual,
                "accepted": accepted,
                "binarity_index": binarity_index(density),
                "gray_fraction": gray_fraction(density),
                "periodic_component_count": len(final_sizes),
                "periodic_component_sizes": list(final_sizes),
                "forced_independent_variables": int(np.count_nonzero(force_independent)),
                "suppressed_independent_variables": int(np.count_nonzero(suppress_independent)),
                "connectivity_repairs": repairs,
                "raw_density_file": raw_file,
            }
            stage_records.append(record)
            stage_path.write_text(json.dumps(stage_records, indent=2) + "\n", encoding="utf-8")
            np.savez(
                bounds_path,
                forced_material=force_independent,
                suppressed_material=suppress_independent,
            )
            (output / "connectivity_history.json").write_text(
                json.dumps(connectivity_history, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(record, indent=2), flush=True)
            if not accepted:
                raise RuntimeError(
                    f"stage {stage_index} failed: residual={residual:.3e}, components={len(final_sizes)}"
                )

    np.savetxt(
        output / "optimized_master_raw_density.csv",
        current.reshape(master_reference.design_shape),
        delimiter=",",
    )
    final_beta = float(stages[len(stage_records) - 1]["projection_beta"])
    final_filter_radius = float(stages[len(stage_records) - 1]["filter_radius_um"])
    final_config = json.loads(json.dumps(config))
    final_config["parameterization"]["filter_radius_um"] = final_filter_radius
    master_density_local, _ = _density_on_master(
        final_config, master_setup, master_reference.design_rows, current, final_beta
    )
    master_mask = master_density_local >= float(controls["hard_threshold"])
    master_full_density = np.zeros(master_setup.grid.shape, dtype=float)
    start, stop = master_reference.design_rows
    master_full_density[start:stop] = master_density_local
    wavelengths = _band_wavelengths(common, int(controls["band_samples"]))
    validation = []
    plot_records = []
    gates = []
    for cells_value in controls["validation_cells_per_period"]:
        cells = int(cells_value)
        setup, parameterization = _target_parameterization(
            final_config,
            master_setup,
            master_reference.design_rows,
            cells,
            final_beta,
        )
        binary_epsilon, fill = subpixel_binary_permittivity(
            master_full_density,
            setup.grid.shape,
            float(controls["hard_threshold"]),
            parameterization.low_index,
            parameterization.high_index,
            subpixels=int(controls["binary_rasterization_subpixels"]),
        )
        setup, monitor_shifts = _homogeneous_monitor_setup(binary_epsilon, setup)
        metrics, spectrum = _spectrum_metrics(
            binary_epsilon, setup, wavelengths, common, config
        )
        sampled_omega = 2.0 * np.pi / wavelengths
        sampled_reflectance = np.abs(spectrum[1]) ** 2
        minimum_order = np.argsort(sampled_reflectance)
        reflection_zero_guesses = tuple(
            complex(sampled_omega[index]) for index in minimum_order[:5]
        )
        parity, channel, root_errors = _root_metrics(
            binary_epsilon,
            setup,
            common,
            config,
            reflection_zero_guesses=reflection_zero_guesses,
        )
        root_residuals = _root_residuals(parity, channel)
        contour = periodic_bilinear_resample(
            master_full_density, setup.grid.shape
        ) >= float(controls["hard_threshold"])
        component_sizes = periodic_component_sizes(contour)
        limits = config["verification"]
        gate = bool(
            len(component_sizes) == 1
            and metrics["band_maximum_reflectance"] <= float(limits["maximum_band_reflectance"])
            and metrics["band_minimum_transmittance"] >= float(limits["minimum_band_transmittance"])
            and metrics["phase_window_span_pi"] >= float(limits["minimum_phase_span_pi"])
            and max(root_errors.values()) <= float(limits["maximum_root_error_over_linewidth"])
            and max(root_residuals.values()) <= float(limits["maximum_root_residual"])
        )
        write_spectrum(output / f"binary_spectrum_n{cells}.csv", wavelengths, *spectrum)
        np.savetxt(output / f"binary_permittivity_n{cells}.csv", binary_epsilon, delimiter=",")
        np.savetxt(output / f"binary_fill_fraction_n{cells}.csv", fill, delimiter=",")
        transmission, reflection, empty = spectrum
        record = {
            "cells_per_period": cells,
            "grid_shape": list(setup.grid.shape),
            "binary_metrics": metrics,
            "parity_roots": parity,
            "channel_roots": channel,
            "root_errors": root_errors,
            "root_residuals": root_residuals,
            "periodic_component_count": len(component_sizes),
            "periodic_component_sizes": list(component_sizes),
            "monitor_shifts": monitor_shifts,
            "gate": gate,
        }
        validation.append(record)
        gates.append(gate)
        plot_records.append(
            {
                **record,
                "omega_rad_per_um": 2.0 * np.pi / wavelengths,
                "reflectance": np.abs(reflection) ** 2,
                "deembedded": transmission / empty,
            }
        )

    summary = {
        "stage": "M10B connected binary multi-mesh reoptimization",
        "parent_run": str(parent),
        "target_common_pole": [common.real, common.imag],
        "master_cells_per_period": master_cells,
        "master_grid_shape": list(master_setup.grid.shape),
        "master_design_rows": list(master_reference.design_rows),
        "master_variable_count": int(current.size),
        "filter_radius_um": final_filter_radius,
        "minimum_retained_component_cells": minimum_cells,
        "requested_minimum_bridge_width_um": float(controls["minimum_bridge_width_um"]),
        "bridge_density_lower_bound": bridge_lower_bound,
        "implemented_bridge_radius_cells": bridge_radius,
        "implemented_bridge_width_um": (2 * bridge_radius + 1) * dx,
        "hard_threshold": float(controls["hard_threshold"]),
        "master_binarity_index": binarity_index(master_density_local),
        "master_gray_fraction": gray_fraction(master_density_local),
        "master_periodic_component_count": len(periodic_component_sizes(master_mask)),
        "master_periodic_component_sizes": list(periodic_component_sizes(master_mask)),
        "forced_independent_variables": int(np.count_nonzero(force_independent)),
        "suppressed_independent_variables": int(np.count_nonzero(suppress_independent)),
        "continuation_stages": stage_records,
        "validation": validation,
        "mesh_gates": gates,
        "all_meshes_gate_pass": all(gates),
        "provenance": config["provenance"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    figure = _plot_structure(master_density_local, master_mask, config)
    save_figure(figure, figures / "m10b_connected_binary_structure.png", dpi=220)
    plt.close(figure)
    figure = _plot_response(plot_records, common)
    save_figure(figure, figures / "m10b_connected_binary_response.png", dpi=220)
    plt.close(figure)
    figure = _plot_roots(validation, common)
    save_figure(figure, figures / "m10b_connected_binary_roots.png", dpi=220)
    plt.close(figure)
    print(json.dumps({
        "all_meshes_gate_pass": summary["all_meshes_gate_pass"],
        "components": summary["master_periodic_component_count"],
        "master_binarity_index": summary["master_binarity_index"],
        "mesh_metrics": [record["binary_metrics"] for record in validation],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
