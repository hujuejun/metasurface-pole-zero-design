"""Adjoint pole homotopy for the M5 coupled-bar spacer geometry."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pzid.density import (
    AxiallySymmetricDensityParameterization,
    DensityParameterization,
)
from pzid.plotting import plot_complex_frequency_channels, save_figure
from pzid.poles import PoleTarget
from pzid.roots import refine_scalar_root, refine_scalar_root_multistart

from scripts.m3_optimize_pole_zero import write_history
from scripts.m3_single_resonance import write_metadata
from scripts.m4b_optimize_kerker_bar import (
    _channel_response,
    _frequency,
    _inverse_density,
    _parity_response,
    _root_record,
    _run_stage,
    make_complex_map,
    plot_history,
    plot_real_axis,
    plot_structures,
    sweep_two_port,
    write_spectrum,
)
from scripts.m5_screen_coupled_bars import build_problem


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configured_pole(config: dict[str, Any], prefix: str) -> complex:
    target = config["target"]
    return complex(
        float(target[f"{prefix}_real_rad_per_um"]),
        float(target[f"{prefix}_imag_rad_per_um"]),
    )


def build_optimization_problem(
    config: dict[str, Any], *, cells_per_period: int
) -> tuple[
    Any,
    np.ndarray,
    Any,
    AxiallySymmetricDensityParameterization,
    np.ndarray,
]:
    benchmark = config["benchmark"]
    geometry, grid, epsilon, setup = build_problem(
        config,
        float(benchmark["seed_width_um"]),
        float(benchmark["seed_bar_height_um"]),
        float(benchmark["seed_central_gap_um"]),
        cells_per_period=cells_per_period,
    )
    occupied = np.flatnonzero(
        np.any(
            epsilon > float(benchmark["background_index"]) ** 2 + 1e-12,
            axis=1,
        )
    )
    symmetric_edge = min(int(occupied[0]), grid.nz - 1 - int(occupied[-1]))
    margin = int(config["parameterization"]["axial_margin_cells"])
    lower_reserved = max(setup.lower_source_plane, *setup.lower_monitor_planes)
    start = max(1, symmetric_edge - margin, lower_reserved + 1)
    stop = grid.nz - start
    if start > int(occupied[0]):
        raise RuntimeError("the port-safe design interval excludes seed material")
    controls = config["parameterization"]
    base = DensityParameterization(
        grid=grid,
        design_rows=(start, stop),
        low_index=float(benchmark["background_index"]),
        high_index=float(benchmark["refractive_index"]),
        filter_radius=float(controls["filter_radius_um"]),
        projection_beta=float(controls["projection_beta"]),
        projection_threshold=float(controls["projection_threshold"]),
    )
    parameterization = AxiallySymmetricDensityParameterization(base)
    initial_raw = parameterization.reduce(_inverse_density(epsilon, base))
    reserved = (
        setup.lower_source_plane,
        setup.upper_source_plane,
        *setup.lower_monitor_planes,
        *setup.upper_monitor_planes,
    )
    if any(start <= row < stop for row in reserved):
        raise RuntimeError("the design interval overlaps a calibrated port row")
    return geometry, epsilon, setup, parameterization, initial_raw


def _refine_parity_poles(
    epsilon: np.ndarray,
    setup: Any,
    even_guess: complex,
    odd_guess: complex,
    config: dict[str, Any],
) -> dict[str, Any]:
    controls = config["verification"]
    options = {
        "tolerance": float(controls["root_tolerance"]),
        "max_evaluations": int(controls["root_max_evaluations"]),
    }
    records = {}
    for parity, guess in (("even", even_guess), ("odd", odd_guess)):
        result = refine_scalar_root(
            lambda omega, selected=parity: _parity_response(
                omega, epsilon, setup, selected
            ),
            guess,
            kind="pole",
            search_radius=max(0.35, -1.5 * guess.imag),
            **options,
        )
        records[f"{parity}_parity_pole"] = _root_record(result)
    return records


def _refine_channel_roots(
    epsilon: np.ndarray,
    setup: Any,
    common_pole: complex,
    config: dict[str, Any],
    reflection_zero_guesses: Sequence[complex] | None = None,
) -> dict[str, Any]:
    controls = config["verification"]
    options = {
        "tolerance": float(controls["root_tolerance"]),
        "max_evaluations": int(controls["root_max_evaluations"]),
    }
    radius = max(0.35, -1.5 * common_pole.imag)
    transmission_pole = refine_scalar_root(
        lambda omega: _channel_response(omega, epsilon, setup, "transmission"),
        common_pole,
        kind="pole",
        search_radius=radius,
        **options,
    )
    transmission_zero = refine_scalar_root(
        lambda omega: _channel_response(omega, epsilon, setup, "transmission"),
        np.conj(common_pole),
        kind="zero",
        search_radius=radius,
        **options,
    )
    gamma = -common_pole.imag
    standard_reflection_guesses = tuple(
        complex(common_pole.real + offset * gamma)
        for offset in (-2.0, -1.0, 0.0, 1.0, 2.0)
    )
    supplied_reflection_guesses = tuple(
        complex(value) for value in (reflection_zero_guesses or ())
    )
    reflection_zero = refine_scalar_root_multistart(
        lambda omega: _channel_response(omega, epsilon, setup, "reflection"),
        supplied_reflection_guesses + standard_reflection_guesses,
        kind="zero",
        search_radius=radius,
        prefer_real_axis=True,
        **options,
    )
    return {
        "transmission_pole": _root_record(transmission_pole),
        "transmission_zero": _root_record(transmission_zero),
        "reflection_zero": _root_record(reflection_zero),
    }


def _plot_pole_homotopy(
    initial_roots: dict[str, Any],
    final_roots: dict[str, Any],
    final_even_target: complex,
    final_odd_target: complex,
    common_target: complex,
    channel_roots: dict[str, Any] | None,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
    for parity, marker in (("even", "o"), ("odd", "s")):
        initial = _frequency(initial_roots[f"{parity}_parity_pole"])
        verified = _frequency(final_roots[f"{parity}_parity_pole"])
        target = final_even_target if parity == "even" else final_odd_target
        axis.scatter(
            initial.real,
            initial.imag,
            marker=marker,
            facecolors="none",
            edgecolors="0.2",
            linewidths=1.3,
            s=70,
            label=f"Initial {parity} pole",
        )
        axis.scatter(
            target.real,
            target.imag,
            marker=marker,
            s=70,
            label=f"Reached {parity} target",
        )
        axis.scatter(
            verified.real,
            verified.imag,
            marker="x",
            s=65,
            label=f"Verified {parity} pole",
        )
    axis.scatter(
        common_target.real,
        common_target.imag,
        marker="*",
        s=100,
        label="Requested common pole",
    )
    if channel_roots is not None:
        t_zero = _frequency(channel_roots["transmission_zero"])
        r_zero = _frequency(channel_roots["reflection_zero"])
        axis.scatter(
            t_zero.real,
            t_zero.imag,
            marker="v",
            s=70,
            label="Verified transmission zero",
        )
        axis.scatter(
            r_zero.real,
            r_zero.imag,
            marker="D",
            s=55,
            label="Verified reflection zero",
        )
    axis.axhline(0.0, color="0.45", linewidth=0.8)
    axis.set_xlabel("Re(ω) (rad/µm)")
    axis.set_ylabel("Im(ω) (rad/µm)")
    axis.set_title("Coupled-bar pole homotopy and verified roots")
    axis.grid(True, linewidth=0.4, alpha=0.35)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    return figure


def _targeted_metrics(
    wavelengths: np.ndarray,
    transmission: np.ndarray,
    reflection: np.ndarray,
    empty_transmission: np.ndarray,
    target_pole: complex,
    config: dict[str, Any],
) -> dict[str, float]:
    order = np.argsort(2.0 * np.pi / wavelengths)
    omega = 2.0 * np.pi / wavelengths[order]
    t_value = transmission[order]
    r_value = reflection[order]
    phase = np.unwrap(np.angle(t_value / empty_transmission[order]))
    gamma = -target_pole.imag
    controls = config["verification"]
    reflection_halfwidth = float(
        controls["reflection_band_halfwidth_in_gamma"]
    )
    phase_halfwidth = float(controls["phase_window_halfwidth_in_gamma"])
    reflection_band = (
        np.abs(omega - target_pole.real) <= reflection_halfwidth * gamma
    )
    phase_window = np.abs(omega - target_pole.real) <= phase_halfwidth * gamma
    return {
        "reflection_band_halfwidth_in_gamma": reflection_halfwidth,
        "reflection_band_omega_min_rad_per_um": (
            target_pole.real - reflection_halfwidth * gamma
        ),
        "reflection_band_omega_max_rad_per_um": (
            target_pole.real + reflection_halfwidth * gamma
        ),
        "band_maximum_reflectance": float(
            np.max(np.abs(r_value[reflection_band]) ** 2)
        ),
        "band_minimum_transmittance": float(
            np.min(np.abs(t_value[reflection_band]) ** 2)
        ),
        "phase_window_halfwidth_in_gamma": phase_halfwidth,
        "phase_window_span_pi": float(np.ptp(phase[phase_window]) / np.pi),
        "full_sampled_phase_span_pi": float(np.ptp(phase) / np.pi),
        "energy_balance_max_abs_error": float(
            np.max(np.abs(np.abs(t_value) ** 2 + np.abs(r_value) ** 2 - 1.0))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/m5_coupled_bar_optimize_tm.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/m5_coupled_bar_optimize_n32")
    )
    parser.add_argument("--skip-map", action="store_true")
    arguments = parser.parse_args()

    config = _read_json(arguments.config)
    nx = int(config["mesh"]["cells_per_period"])
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
    geometry, initial_epsilon, setup, parameterization, initial_raw = (
        build_optimization_problem(config, cells_per_period=nx)
    )
    initial_even = _configured_pole(config, "initial_even_pole")
    initial_odd = _configured_pole(config, "initial_odd_pole")
    common_target = _configured_pole(config, "common_pole")
    initial_roots = _refine_parity_poles(
        initial_epsilon, setup, initial_even, initial_odd, config
    )
    initial_even = _frequency(initial_roots["even_parity_pole"])
    initial_odd = _frequency(initial_roots["odd_parity_pole"])
    np.savetxt(output / "initial_permittivity.csv", initial_epsilon, delimiter=",")
    np.savetxt(output / "initial_raw_density.csv", initial_raw, delimiter=",")
    (output / "initial_roots.json").write_text(
        json.dumps(initial_roots, indent=2) + "\n", encoding="utf-8"
    )

    spectrum = config["spectrum"]
    step = float(spectrum["verification_step_um"])
    wavelengths = np.arange(
        float(spectrum["start_um"]),
        float(spectrum["stop_um"]) + 0.5 * step,
        step,
    )
    initial_t, initial_r, initial_empty = sweep_two_port(
        wavelengths, initial_epsilon, setup
    )
    write_spectrum(
        output / "initial_spectrum.csv",
        wavelengths,
        initial_t,
        initial_r,
        initial_empty,
    )

    current = initial_raw.ravel()
    last_fraction = 0.0
    stage_records: list[tuple[str, str, Any, int, bool]] = []
    acceptance = float(config["optimization"]["stage_acceptance_residual"])
    for index, fraction_value in enumerate(
        config["target"]["pole_continuation_fractions"], start=1
    ):
        fraction = float(fraction_value)
        even_target = initial_even + fraction * (common_target - initial_even)
        odd_target = initial_odd + fraction * (common_target - initial_odd)
        identifier = f"pole-homotopy-{index:02d}-fraction-{fraction:g}"
        result, evaluations = _run_stage(
            identifier,
            current,
            parameterization,
            setup,
            (PoleTarget(even_target, -1.0), PoleTarget(odd_target, 1.0)),
            None,
            (),
            config,
            int(config["optimization"]["continuation_iterations"]),
        )
        residual = float(np.linalg.norm(result.residual))
        accepted = residual <= acceptance
        stage_records.append(
            ("Pole homotopy", identifier, result, evaluations, accepted)
        )
        write_history(output / f"optimization_history_{identifier}.csv", result)
        np.savetxt(
            output / f"raw_density_{identifier}.csv",
            result.x.reshape(parameterization.design_shape),
            delimiter=",",
        )
        if not accepted:
            break
        current = result.x
        last_fraction = fraction

    full_common_target = np.isclose(last_fraction, 1.0)
    gamma = -common_target.imag
    offsets = tuple(
        float(value)
        for value in config["target"]["reflection_sample_offsets_in_gamma"]
    )
    reflection_frequencies = tuple(
        common_target.real + offset * gamma for offset in offsets
    )
    if full_common_target:
        pole_targets = (
            PoleTarget(common_target, -1.0),
            PoleTarget(common_target, 1.0),
        )
        magnitudes = sorted({abs(offset) for offset in offsets})
        for group_index, maximum_offset in enumerate(magnitudes, start=1):
            active = tuple(
                (frequency, 0.0j)
                for offset, frequency in zip(
                    offsets, reflection_frequencies, strict=True
                )
                if abs(offset) <= maximum_offset + 1e-12
            )
            identifier = (
                f"reflection-band-{group_index}-offset-{maximum_offset:g}gamma"
            )
            result, evaluations = _run_stage(
                identifier,
                current,
                parameterization,
                setup,
                pole_targets,
                None,
                active,
                config,
                int(config["optimization"]["band_iterations"]),
            )
            residual = float(np.linalg.norm(result.residual))
            accepted = residual <= float(
                config["optimization"].get(
                    "band_stage_acceptance_residual", acceptance
                )
            )
            stage_records.append(
                ("Reflection band", identifier, result, evaluations, accepted)
            )
            write_history(
                output / f"optimization_history_{identifier}.csv", result
            )
            np.savetxt(
                output / f"raw_density_{identifier}.csv",
                result.x.reshape(parameterization.design_shape),
                delimiter=",",
            )
            if not accepted:
                break
            current = result.x

    final_even_target = initial_even + last_fraction * (
        common_target - initial_even
    )
    final_odd_target = initial_odd + last_fraction * (
        common_target - initial_odd
    )
    final_epsilon = parameterization.map(current).epsilon_r
    final_roots = _refine_parity_poles(
        final_epsilon, setup, final_even_target, final_odd_target, config
    )
    channel_roots = None
    if full_common_target:
        channel_roots = _refine_channel_roots(
            final_epsilon, setup, common_target, config
        )
    np.savetxt(
        output / "optimized_raw_density.csv",
        current.reshape(parameterization.design_shape),
        delimiter=",",
    )
    np.savetxt(output / "optimized_permittivity.csv", final_epsilon, delimiter=",")
    (output / "verified_roots.json").write_text(
        json.dumps(
            {"parity_roots": final_roots, "channel_roots": channel_roots},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    final_t, final_r, final_empty = sweep_two_port(
        wavelengths, final_epsilon, setup
    )
    write_spectrum(
        output / "optimized_spectrum.csv",
        wavelengths,
        final_t,
        final_r,
        final_empty,
    )

    figure = plot_structures(
        initial_epsilon,
        final_epsilon,
        setup,
        parameterization.design_rows,
        parameterization.low_index,
        parameterization.high_index,
        titles=("Initial coupled-bar seed", "Optimized coupled-spacer topology"),
    )
    save_figure(figure, figures / "m5_initial_optimized_structures.png", dpi=220)
    plt.close(figure)
    figure = _plot_pole_homotopy(
        initial_roots,
        final_roots,
        final_even_target,
        final_odd_target,
        common_target,
        channel_roots,
    )
    save_figure(figure, figures / "m5_verified_roots.png", dpi=220)
    plt.close(figure)
    figure = plot_real_axis(
        wavelengths,
        initial_t,
        initial_r,
        initial_empty,
        final_t,
        final_r,
        final_empty,
        common_target,
        title="M5 coupled-spacer phase-control response",
        band_halfwidth_in_gamma=float(
            config["verification"]["reflection_band_halfwidth_in_gamma"]
        ),
    )
    save_figure(figure, figures / "m5_real_axis_response.png", dpi=220)
    plt.close(figure)
    figure = plot_history(
        tuple((group, result) for group, _, result, _, _ in stage_records),
        title="Staged M5 coupled-spacer inverse design",
    )
    save_figure(figure, figures / "m5_optimization_history.png", dpi=220)
    plt.close(figure)

    if not arguments.skip_map:
        verification = config["verification"]
        real_axis, imag_axis, t_map, r_map = make_complex_map(
            final_epsilon,
            setup,
            common_target,
            int(verification["complex_map_real_points"]),
            int(verification["complex_map_imag_points"]),
        )
        np.savez_compressed(
            output / "complex_frequency_map.npz",
            real_omega=real_axis,
            imag_omega=imag_axis,
            transmission=t_map,
            reflection=r_map,
        )
        poles = tuple(
            _frequency(final_roots[f"{parity}_parity_pole"])
            for parity in ("even", "odd")
        )
        t_zeros = ()
        r_zeros = ()
        if channel_roots is not None:
            t_zeros = (_frequency(channel_roots["transmission_zero"]),)
            r_zeros = (_frequency(channel_roots["reflection_zero"]),)
        figure, _ = plot_complex_frequency_channels(
            real_axis,
            imag_axis,
            t_map,
            r_map,
            frequency_unit="rad/µm",
            db_floor=-50.0,
            vmin=-45.0,
            vmax=30.0,
            poles=poles,
            transmission_zeros=t_zeros,
            reflection_zeros=r_zeros,
        )
        save_figure(figure, figures / "m5_complex_frequency_channels.png", dpi=220)
        plt.close(figure)

    metrics = _targeted_metrics(
        wavelengths, final_t, final_r, final_empty, common_target, config
    )
    reached_targets = {
        "even": {"real": final_even_target.real, "imag": final_even_target.imag},
        "odd": {"real": final_odd_target.real, "imag": final_odd_target.imag},
    }
    root_errors = {
        parity: abs(
            _frequency(final_roots[f"{parity}_parity_pole"])
            - (final_even_target if parity == "even" else final_odd_target)
        )
        / max(
            -2.0
            * (final_even_target if parity == "even" else final_odd_target).imag,
            1e-12,
        )
        for parity in ("even", "odd")
    }
    verification = config["verification"]
    root_gate = all(
        error <= float(verification["maximum_root_error_over_linewidth"])
        for error in root_errors.values()
    )
    performance_gate = bool(
        metrics["band_maximum_reflectance"]
        <= float(verification["maximum_band_reflectance"])
        and metrics["band_minimum_transmittance"]
        >= float(verification["minimum_band_transmittance"])
        and metrics["phase_window_span_pi"]
        >= float(verification["minimum_phase_span_pi"])
    )
    summary = {
        "stage": "M5 coupled-bar spacer pole-zero inverse design",
        "cells_per_period": nx,
        "geometry": asdict(geometry),
        "design_rows": parameterization.design_rows,
        "variable_count": parameterization.variable_count,
        "requested_common_pole": {
            "real": common_target.real,
            "imag": common_target.imag,
        },
        "last_accepted_pole_fraction": last_fraction,
        "full_common_target_reached": bool(full_common_target),
        "reached_pole_targets": reached_targets,
        "stages": [
            {
                "group": group,
                "name": identifier,
                "function_evaluations": evaluations,
                "accepted_iterations": len(result.history),
                "solver_converged": result.converged,
                "accepted_by_residual": accepted,
                "reason": result.reason,
                "final_residual_norm": float(np.linalg.norm(result.residual)),
            }
            for group, identifier, result, evaluations, accepted in stage_records
        ],
        "initial_metrics": _targeted_metrics(
            wavelengths,
            initial_t,
            initial_r,
            initial_empty,
            common_target,
            config,
        ),
        "final_metrics": metrics,
        "root_errors_over_reached_target_linewidth": root_errors,
        "channel_roots": channel_roots,
        "maximum_density_change": float(
            np.max(np.abs(current - initial_raw.ravel()))
        ),
        "rms_density_change": float(
            np.sqrt(np.mean((current - initial_raw.ravel()) ** 2))
        ),
        "axial_permittivity_symmetry_error": float(
            np.max(np.abs(final_epsilon - np.flipud(final_epsilon)))
        ),
        "root_gate_pass": bool(root_gate),
        "performance_gate_pass": performance_gate,
        "m5_gate_pass": bool(full_common_target and root_gate and performance_gate),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
