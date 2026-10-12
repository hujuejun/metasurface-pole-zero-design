"""Adjoint pole-zero coalescence of the M27C folded-qBIC pair.

The design is the one-dimensional counterpart of the dual-qBIC Huygens
mechanism of Fan et al. (Science Advances 2025).  One unrestricted lateral
mask is copied to the two facing relief rows, preserving exact axial parity
while retaining the period-doubling perturbation.  Chen time-reversal
constraints first coalesce the even and odd poles.  A second continuation
then places a reflection zero on their common real frequency.
"""

from __future__ import annotations

import argparse
import csv
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
    DensityParameterization,
    PairedNestedPlanarReliefDensityParameterization,
    PlanarLayerDensityParameterization,
)
from pzid.hybrid import symmetric_slab_tm_frequencies
from pzid.optimize import LMOptions, solve_underdetermined_lm
from pzid.plotting import plot_refractive_index, save_figure
from pzid.poles import PoleTarget
from pzid.roots import refine_scalar_root
from pzid.twoport import solve_two_port_scattering

from scripts.m3_optimize_pole_zero import write_history
from scripts.m4b_optimize_kerker_bar import (
    _channel_response,
    _inverse_density,
    _parity_response,
)
from scripts.m11_screen_fp_tm1_hybrid import _root_json, _sweep, _write_spectrum
from scripts.m20_discover_opposite_parity_dual_gmr import _pair_record
from scripts.m22_screen_coupled_single_mode_slabs import _diagnose_candidate
from scripts.m27b_screen_coupled_folded_qbic import _build


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def _build_problem(
    config: dict[str, Any],
) -> tuple[Any, np.ndarray, Any, Any, PairedNestedPlanarReliefDensityParameterization, np.ndarray, dict[str, Any]]:
    parent_run = Path(config["parent_run"])
    parent_summary = json.loads((parent_run / "summary.json").read_text(encoding="utf-8"))
    parent_config = json.loads((parent_run / "config.json").read_text(encoding="utf-8"))
    seed = next(
        record for record in parent_summary["records"]
        if record["identifier"] == config["seed_identifier"]
    )
    grid, seed_epsilon, layout, setup = _build(
        parent_config,
        int(seed["spacer_rows"]),
        int(seed["inner_grating_rib_rows"]),
        int(seed["average_site_width_cells"]),
        int(seed["site_width_asymmetry_cells"]),
    )
    controls = config["parameterization"]
    common = {
        "grid": grid,
        "low_index": float(controls["low_index"]),
        "high_index": float(controls["high_index"]),
        "filter_radius": float(controls["filter_radius_um"]),
        "projection_beta": float(controls["projection_beta"]),
        "projection_threshold": float(controls["projection_threshold"]),
    }
    inner_base = DensityParameterization(
        design_rows=layout.lower_grating_rows, **common
    )
    outer_rows = (
        layout.lower_grating_rows[1],
        layout.lower_grating_rows[1] + 1,
    )
    if outer_rows[1] > layout.spacer_rows[1]:
        raise RuntimeError("the spacer cannot accommodate a nested outer relief tier")
    outer_base = DensityParameterization(design_rows=outer_rows, **common)
    inner = PlanarLayerDensityParameterization(inner_base)
    outer = PlanarLayerDensityParameterization(outer_base)
    parameterization = PairedNestedPlanarReliefDensityParameterization(
        inner, outer, seed_epsilon
    )
    initial_inner = inner.reduce(_inverse_density(seed_epsilon, inner_base))
    initial_outer = outer.reduce(_inverse_density(seed_epsilon, outer_base))
    initial_raw = np.stack((initial_inner.ravel(), initial_outer.ravel()))
    initial_epsilon = parameterization.map(initial_raw).epsilon_r
    if not np.allclose(initial_epsilon, seed_epsilon, rtol=0.0, atol=2e-13):
        raise RuntimeError("density inversion did not reproduce the M27C seed")
    if not np.array_equal(initial_epsilon, np.flipud(initial_epsilon)):
        raise RuntimeError("initial parameterization broke axial symmetry")
    fixed = np.ones(grid.nz, dtype=bool)
    for rows in parameterization.tier_rows:
        fixed[slice(*rows)] = False
    if not np.array_equal(initial_epsilon[fixed], seed_epsilon[fixed]):
        raise RuntimeError("parameterization modified a fixed slab, spacer, or port row")
    return (
        grid, initial_epsilon, layout, setup, parameterization,
        initial_raw, parent_config,
    )


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
                lambda value, chosen=parity: _parity_response(
                    value, epsilon, setup, chosen
                ),
                complex(guess),
                kind="pole",
                tolerance=float(controls["root_tolerance"]),
                max_evaluations=int(controls["root_max_evaluations"]),
                search_radius=float(controls["root_search_radius_rad_per_um"]),
            )
        )
    return roots[0], roots[1]


def _diagnose_with_controls(
    epsilon: np.ndarray,
    grid: Any,
    layout: Any,
    setup: Any,
    guesses: Sequence[complex],
    isolated_frequency: float,
    parent_config: dict[str, Any],
    config: dict[str, Any],
    label: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    roots = _refine_pair(epsilon, setup, guesses, config)
    modes: dict[str, dict[str, Any]] = {}
    fields: dict[str, np.ndarray] = {}
    for parity, root in zip(("even", "odd"), roots, strict=True):
        mode, field = _diagnose_candidate(
            root, parity, isolated_frequency, epsilon, grid, layout, setup,
            parent_config,
        )
        modes[parity] = mode
        fields[parity] = np.asarray(field)
    pair_metrics = _pair_record(modes["even"], modes["odd"])
    record = {
        "identifier": label,
        "even_mode": modes["even"],
        "odd_mode": modes["odd"],
        "pair": pair_metrics,
        **pair_metrics,
        "both_modes_pass_assignment": bool(
            modes["even"]["passes_assignment_gates"]
            and modes["odd"]["passes_assignment_gates"]
        ),
    }
    return record, fields


def _constraint_system(
    values: np.ndarray,
    parameterization: Any,
    setup: Any,
    pole_targets: Sequence[complex],
    config: dict[str, Any],
    reflection_target: complex | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    evaluations = [
        pole_density_constraint(
            values,
            parameterization,
            PoleTarget(frequency, coupling),
            setup,
        )
        for frequency, coupling in zip(pole_targets, (-1.0, 1.0), strict=True)
    ]
    weights = [float(config["optimization"]["pole_constraint_weight"])] * 8
    if reflection_target is not None:
        evaluations.append(
            channel_value_density_constraint(
                values,
                parameterization,
                complex(np.mean([value.real for value in pole_targets])),
                "reflection",
                reflection_target,
                setup,
            )
        )
        weights.extend(
            [float(config["optimization"]["reflection_constraint_weight"])] * 2
        )
    return combine_density_constraints(*evaluations, weights=np.asarray(weights))


def _solve(
    name: str,
    initial: np.ndarray,
    parameterization: Any,
    setup: Any,
    targets: Sequence[complex],
    config: dict[str, Any],
    iterations: int,
    reflection_target: complex | None = None,
) -> tuple[Any, int]:
    evaluations = 0

    def system(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        residual, jacobian = _constraint_system(
            values, parameterization, setup, targets, config, reflection_target
        )
        print(
            f"{name} evaluation {evaluations}: residual={np.linalg.norm(residual):.6e}",
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
            residual_tolerance=float(controls["residual_tolerance"]),
            step_tolerance=float(controls["step_tolerance"]),
            active_set_bounds=bool(controls["active_set_bounds"]),
            active_set_max_iterations=int(controls["active_set_max_iterations"]),
        ),
    )
    return result, evaluations


def _save_state(
    directory: Path,
    raw: np.ndarray,
    epsilon: np.ndarray,
    record: dict[str, Any],
    fields: dict[str, np.ndarray],
    extra: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(directory / "raw_density.csv", raw, delimiter=",")
    np.savetxt(directory / "permittivity.csv", epsilon, delimiter=",")
    for parity in ("even", "odd"):
        np.savetxt(
            directory / f"{parity}_field_real.csv", np.real(fields[parity]), delimiter=","
        )
        np.savetxt(
            directory / f"{parity}_field_imag.csv", np.imag(fields[parity]), delimiter=","
        )
    (directory / "state.json").write_text(
        json.dumps({**extra, "diagnostics": record}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _plot_initial_final(
    initial: np.ndarray, final: np.ndarray, grid: Any, design_rows: tuple[int, int]
) -> plt.Figure:
    figure, axes = plt.subplots(
        1, 2, figsize=(9.4, 4.6), sharex=True, sharey=True, constrained_layout=True
    )
    image = None
    for axis, epsilon, title in zip(
        axes,
        (initial, final),
        ("M27C starting structure", "M27D adjoint structure"),
        strict=True,
    ):
        _, _, image = plot_refractive_index(
            epsilon, grid, ax=axis, length_unit="µm", colorbar=False,
            vmin=1.0, vmax=3.42, design_rows=design_rows,
        )
        axis.set_title(title)
    assert image is not None
    figure.colorbar(image, ax=axes, label="Refractive index, n", shrink=0.84)
    return figure


def _plot_poles_zeros(
    initial: dict[str, Any], final: dict[str, Any], common: complex,
    reflection_zero: Any, transmission_zero: Any,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    initial_even = complex(*initial["even_mode"]["root"]["frequency"])
    initial_odd = complex(*initial["odd_mode"]["root"]["frequency"])
    final_even = complex(*final["even_mode"]["root"]["frequency"])
    final_odd = complex(*final["odd_mode"]["root"]["frequency"])
    axis.scatter(initial_even.real, initial_even.imag, marker="o", s=85,
                 facecolors="none", edgecolors="tab:blue", label="Initial even pole")
    axis.scatter(initial_odd.real, initial_odd.imag, marker="s", s=85,
                 facecolors="none", edgecolors="tab:orange", label="Initial odd pole")
    axis.scatter(final_even.real, final_even.imag, marker="o", s=65,
                 color="tab:blue", label="Final even pole")
    axis.scatter(final_odd.real, final_odd.imag, marker="s", s=65,
                 color="tab:orange", label="Final odd pole")
    axis.scatter(common.real, common.imag, marker="x", s=100, linewidths=2,
                 color="black", label="Common pole target")
    if reflection_zero.success:
        value = complex(reflection_zero.frequency)
        axis.scatter(value.real, value.imag, marker="D", s=72,
                     color="tab:green", label="Final reflection zero")
    if transmission_zero.success:
        value = complex(transmission_zero.frequency)
        axis.scatter(value.real, value.imag, marker="^", s=80,
                     color="tab:red", label="Final transmission zero")
    axis.axhline(0.0, color="0.2", linewidth=0.8)
    axis.set_xlabel("Re frequency (rad/µm)")
    axis.set_ylabel("Im frequency (rad/µm)")
    axis.set_title("M27D channel singularities and prescribed pole")
    axis.grid(True, linewidth=0.4, alpha=0.35)
    axis.legend(frameon=False, fontsize=8)
    return figure


def _plot_response_comparison(
    omega: np.ndarray,
    initial_spectrum: dict[str, Any],
    final_spectrum: dict[str, Any],
    common: complex,
) -> plt.Figure:
    figure, axes = plt.subplots(
        2, 2, figsize=(10.6, 7.2), sharex=True, constrained_layout=True
    )
    for column, (spectrum, title) in enumerate(
        ((initial_spectrum, "Initial"), (final_spectrum, "Adjoint result"))
    ):
        matrix = np.asarray(spectrum["matrices"])
        transmission = matrix[:, 1, 0]
        reflection = matrix[:, 0, 0]
        axes[0, column].plot(omega, np.abs(transmission) ** 2, label="T")
        axes[0, column].plot(omega, np.abs(reflection) ** 2, label="R")
        axes[0, column].set_title(title)
        axes[0, column].set_ylabel("Power")
        axes[0, column].set_ylim(-0.03, 1.03)
        axes[0, column].legend(frameon=False)
        axes[1, column].plot(omega, np.unwrap(np.angle(transmission)) / np.pi)
        axes[1, column].set_ylabel("Transmission phase / π")
        axes[1, column].set_xlabel("Real frequency (rad/µm)")
        for axis in axes[:, column]:
            axis.axvline(common.real, color="black", linestyle=":", linewidth=0.8)
            axis.grid(True, linewidth=0.4, alpha=0.35)
    figure.suptitle("M27D real-axis response around the common-pole target")
    return figure


def _write_stage_table(path: Path, stages: list[dict[str, Any]]) -> None:
    fields = (
        "group", "name", "fraction", "accepted", "residual_norm",
        "even_target_error_over_gamma", "odd_target_error_over_gamma",
        "pole_separation_over_mean_linewidth", "linewidth_ratio_even_over_odd",
        "center_reflectance", "function_evaluations",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for stage in stages:
            writer.writerow({field: stage.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/m27d_fan_1d_pole_zero_adjoint_tm.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/m27d_fan_1d_pole_zero_adjoint_n32"),
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--resume-reflection",
        action="store_true",
        help="Resume from the saved full pole-homotopy state in the output directory.",
    )
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    if not isinstance(config["provenance"], dict):
        raise ValueError("The qBIC continuation requires a provenance record")
    output = arguments.output
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (
        grid, initial_epsilon, layout, setup, parameterization,
        initial_raw, parent_config,
    ) = _build_problem(config)
    parent_summary = json.loads(
        (Path(config["parent_run"]) / "summary.json").read_text(encoding="utf-8")
    )
    seed_record = next(
        record for record in parent_summary["records"]
        if record["identifier"] == config["seed_identifier"]
    )
    seed_guesses = tuple(
        complex(*seed_record[f"{parity}_mode"]["root"]["frequency"])
        for parity in ("even", "odd")
    )
    common = 0.5 * (seed_guesses[0] + seed_guesses[1])
    gamma = -common.imag
    spacing = grid.dx
    isolated = symmetric_slab_tm_frequencies(
        period_um=grid.period,
        slab_thickness_um=int(parent_config["benchmark"]["single_slab_rows"]) * spacing,
        core_index=float(parent_config["benchmark"]["slab_index"]),
        cladding_index=float(parent_config["benchmark"]["port_index"]),
    )
    isolated_frequency = float(isolated["even_hy"][0])
    initial_record, initial_fields = _diagnose_with_controls(
        initial_epsilon, grid, layout, setup, seed_guesses, isolated_frequency,
        parent_config, config, "initial",
    )
    pole_residual, pole_jacobian = _constraint_system(
        initial_raw.ravel(), parameterization, setup, (common, common), config
    )
    combined_residual, combined_jacobian = _constraint_system(
        initial_raw.ravel(), parameterization, setup, (common, common), config, 0.0j
    )
    singular_values = np.linalg.svd(combined_jacobian, compute_uv=False)
    tolerance = max(combined_jacobian.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    linear_step = np.linalg.lstsq(
        combined_jacobian, -combined_residual, rcond=tolerance / singular_values[0]
    )[0]
    uncontrollable = combined_residual + combined_jacobian @ linear_step
    uncontrollable_fraction = float(
        np.linalg.norm(uncontrollable)
        / max(np.linalg.norm(combined_residual), np.finfo(float).tiny)
    )
    audit = {
        "variable_count": parameterization.variable_count,
        "pole_constraint_rows": int(pole_jacobian.shape[0]),
        "combined_constraint_rows": int(combined_jacobian.shape[0]),
        "combined_jacobian_rank": rank,
        "combined_jacobian_singular_values": [float(value) for value in singular_values],
        "pole_endpoint_residual_norm": float(np.linalg.norm(pole_residual)),
        "combined_endpoint_residual_norm": float(np.linalg.norm(combined_residual)),
        "linearized_uncontrollable_fraction": uncontrollable_fraction,
        "common_pole_target": _pair(common),
        "passes": bool(
            rank >= int(config["verification"]["minimum_endpoint_rank"])
            and uncontrollable_fraction <= float(
                config["verification"]["maximum_linearized_uncontrollable_fraction"]
            )
        ),
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    np.savetxt(output / "endpoint_jacobian.csv", combined_jacobian, delimiter=",")
    _save_state(
        output / "initial_state", initial_raw, initial_epsilon, initial_record,
        initial_fields, {"accepted": True},
    )
    metadata = {
        "config_sha256": _sha256(arguments.config),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "density_source_sha256": _sha256(Path("src/pzid/density.py")),
        "parent_summary_sha256": _sha256(Path(config["parent_run"]) / "summary.json"),
        "numpy_version": np.__version__,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"audit": audit}, indent=2), flush=True)
    if arguments.audit_only or not audit["passes"]:
        summary = {
            "stage": "M27D folded-qBIC pole-zero adjoint endpoint audit",
            "optimization_started": False,
            "audit": audit,
            "provenance": config["provenance"],
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return

    current = initial_raw.ravel()
    current_roots = seed_guesses
    stages: list[dict[str, Any]] = []
    full_pole_target = False
    if arguments.resume_reflection:
        final_pole_name = "pole_20_fraction_1"
        final_pole_directory = output / final_pole_name
        if not (final_pole_directory / "state.json").exists():
            raise FileNotFoundError(
                f"cannot resume: {final_pole_directory / 'state.json'} is missing"
            )
        current = np.loadtxt(
            final_pole_directory / "raw_density.csv", delimiter=","
        ).ravel()
        saved_state = json.loads(
            (final_pole_directory / "state.json").read_text(encoding="utf-8")
        )
        current_roots = tuple(
            complex(*saved_state["diagnostics"][f"{parity}_mode"]["root"]["frequency"])
            for parity in ("even", "odd")
        )
        existing_table = output / "stage_table.csv"
        if existing_table.exists():
            with existing_table.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    if row["group"] != "pole homotopy":
                        continue
                    parsed: dict[str, Any] = {
                        "group": row["group"],
                        "name": row["name"],
                        "accepted": row["accepted"].lower() == "true",
                    }
                    for key, value in row.items():
                        if key in {"group", "name", "accepted"} or value == "":
                            continue
                        parsed[key] = int(value) if key == "function_evaluations" else float(value)
                    stages.append(parsed)
        full_pole_target = True
    acceptance = float(config["optimization"]["stage_acceptance_residual"])
    maximum_error = float(config["optimization"]["maximum_target_error_over_gamma"])
    pole_fractions = (
        [] if arguments.resume_reflection
        else config["target"]["pole_continuation_fractions"]
    )
    for index, fraction_value in enumerate(pole_fractions, start=1):
        fraction = float(fraction_value)
        targets = tuple(
            seed + fraction * (common - seed) for seed in seed_guesses
        )
        name = f"pole_{index:02d}_fraction_{fraction:g}"
        result, evaluations = _solve(
            name, current, parameterization, setup, targets, config,
            int(config["optimization"]["continuation_iterations"]),
        )
        epsilon = parameterization.map(result.x).epsilon_r
        record, fields = _diagnose_with_controls(
            epsilon, grid, layout, setup, targets, isolated_frequency,
            parent_config, config, name,
        )
        verified = tuple(
            complex(*record[f"{parity}_mode"]["root"]["frequency"])
            for parity in ("even", "odd")
        )
        errors = [float(abs(root - target) / gamma) for root, target in zip(verified, targets, strict=True)]
        residual_norm = float(np.linalg.norm(result.residual))
        accepted = bool(
            residual_norm <= acceptance
            and max(errors) <= maximum_error
            and record["both_modes_pass_assignment"]
        )
        center_solution = solve_two_port_scattering(
            float(np.mean([root.real for root in verified])), epsilon, setup
        )
        stage = {
            "group": "pole homotopy", "name": name, "fraction": fraction,
            "accepted": accepted, "residual_norm": residual_norm,
            "even_target_error_over_gamma": errors[0],
            "odd_target_error_over_gamma": errors[1],
            "pole_separation_over_mean_linewidth": record["center_mismatch_over_mean_linewidth"],
            "linewidth_ratio_even_over_odd": record["linewidth_ratio_even_over_odd"],
            "center_reflectance": float(abs(center_solution.matrix[0, 0]) ** 2),
            "function_evaluations": evaluations,
        }
        stages.append(stage)
        _save_state(
            output / name, result.x.reshape(parameterization.design_shape), epsilon,
            record, fields, stage,
        )
        write_history(output / name / "optimization_history.csv", result)
        if not accepted:
            break
        current = result.x
        current_roots = verified
        full_pole_target = bool(np.isclose(fraction, 1.0))

    reflection_zero_reached = False
    if full_pole_target:
        pole_epsilon = parameterization.map(current).epsilon_r
        initial_reflection = complex(
            solve_two_port_scattering(common.real, pole_epsilon, setup).matrix[0, 0]
        )
        for index, fraction_value in enumerate(
            config["target"]["reflection_continuation_fractions"], start=1
        ):
            fraction = float(fraction_value)
            reflection_target = (1.0 - fraction) * initial_reflection
            name = f"reflection_{index:02d}_fraction_{fraction:g}"
            result, evaluations = _solve(
                name, current, parameterization, setup, (common, common), config,
                int(config["optimization"]["reflection_iterations"]),
                reflection_target,
            )
            epsilon = parameterization.map(result.x).epsilon_r
            record, fields = _diagnose_with_controls(
                epsilon, grid, layout, setup, (common, common), isolated_frequency,
                parent_config, config, name,
            )
            verified = tuple(
                complex(*record[f"{parity}_mode"]["root"]["frequency"])
                for parity in ("even", "odd")
            )
            errors = [float(abs(root - common) / gamma) for root in verified]
            solution = solve_two_port_scattering(common.real, epsilon, setup)
            reflection_error = abs(solution.matrix[0, 0] - reflection_target)
            residual_norm = float(np.linalg.norm(result.residual))
            accepted = bool(
                residual_norm <= acceptance
                and max(errors) <= maximum_error
                and reflection_error <= float(
                    config["optimization"]["maximum_reflection_target_error"]
                )
                and record["both_modes_pass_assignment"]
            )
            stage = {
                "group": "reflection homotopy", "name": name, "fraction": fraction,
                "accepted": accepted, "residual_norm": residual_norm,
                "even_target_error_over_gamma": errors[0],
                "odd_target_error_over_gamma": errors[1],
                "pole_separation_over_mean_linewidth": record["center_mismatch_over_mean_linewidth"],
                "linewidth_ratio_even_over_odd": record["linewidth_ratio_even_over_odd"],
                "center_reflectance": float(abs(solution.matrix[0, 0]) ** 2),
                "function_evaluations": evaluations,
            }
            stages.append(stage)
            _save_state(
                output / name, result.x.reshape(parameterization.design_shape), epsilon,
                record, fields,
                {**stage, "reflection_target": _pair(reflection_target)},
            )
            write_history(output / name / "optimization_history.csv", result)
            if not accepted:
                break
            current = result.x
            current_roots = verified
            reflection_zero_reached = bool(np.isclose(fraction, 1.0))

    final_epsilon = parameterization.map(current).epsilon_r
    final_record, final_fields = _diagnose_with_controls(
        final_epsilon, grid, layout, setup, current_roots, isolated_frequency,
        parent_config, config, "final",
    )
    final_poles = tuple(
        complex(*final_record[f"{parity}_mode"]["root"]["frequency"])
        for parity in ("even", "odd")
    )
    root_controls = config["verification"]
    final_reflection_zero = refine_scalar_root(
        lambda value: _channel_response(value, final_epsilon, setup, "reflection"),
        complex(common.real), kind="zero",
        tolerance=float(root_controls["root_tolerance"]),
        max_evaluations=int(root_controls["root_max_evaluations"]),
        search_radius=float(root_controls["root_search_radius_rad_per_um"]),
    )
    final_transmission_zero = refine_scalar_root(
        lambda value: _channel_response(value, final_epsilon, setup, "transmission"),
        np.conj(common), kind="zero",
        tolerance=float(root_controls["root_tolerance"]),
        max_evaluations=int(root_controls["root_max_evaluations"]),
        search_radius=float(root_controls["root_search_radius_rad_per_um"]),
    )
    half_width = float(root_controls["spectrum_half_width_in_gamma"]) * gamma
    omega = np.linspace(
        common.real - half_width, common.real + half_width,
        int(root_controls["spectrum_points"]),
    )
    initial_spectrum = _sweep(2.0 * np.pi / omega, initial_epsilon, setup)
    final_spectrum = _sweep(2.0 * np.pi / omega, final_epsilon, setup)
    _write_spectrum(output / "initial_spectrum.csv", 2.0 * np.pi / omega, initial_spectrum)
    _write_spectrum(output / "optimized_spectrum.csv", 2.0 * np.pi / omega, final_spectrum)
    final_matrix = np.asarray(final_spectrum["matrices"])
    final_t = final_matrix[:, 1, 0]
    final_r = final_matrix[:, 0, 0]
    performance_window = (
        np.abs(omega - common.real)
        <= float(root_controls["performance_half_width_in_gamma"]) * gamma
    )
    phase = np.unwrap(np.angle(final_t))
    metrics = {
        "minimum_transmittance_in_performance_window": float(
            np.min(np.abs(final_t[performance_window]) ** 2)
        ),
        "maximum_reflectance_in_performance_window": float(
            np.max(np.abs(final_r[performance_window]) ** 2)
        ),
        "full_plot_phase_span_pi": float(np.ptp(phase) / np.pi),
        "performance_window_phase_span_pi": float(
            np.ptp(phase[performance_window]) / np.pi
        ),
        "center_reflectance": float(
            abs(solve_two_port_scattering(common.real, final_epsilon, setup).matrix[0, 0]) ** 2
        ),
        "energy_balance_max_abs_error": float(final_spectrum["energy_balance_max_abs_error"]),
    }
    performance_pass = bool(
        metrics["minimum_transmittance_in_performance_window"]
        >= float(root_controls["minimum_transmittance"])
        and metrics["maximum_reflectance_in_performance_window"]
        <= float(root_controls["maximum_reflectance"])
        and metrics["full_plot_phase_span_pi"]
        >= float(root_controls["minimum_phase_span_pi"])
    )
    np.savetxt(
        output / "optimized_raw_density.csv",
        current.reshape(parameterization.design_shape), delimiter=",",
    )
    np.savetxt(output / "optimized_permittivity.csv", final_epsilon, delimiter=",")
    _write_stage_table(output / "stage_table.csv", stages)
    summary = {
        "stage": "M27D folded-qBIC pole-zero adjoint continuation",
        "audit": audit,
        "common_pole_target": _pair(common),
        "full_common_pole_reached": full_pole_target,
        "reflection_zero_reached": reflection_zero_reached,
        "stages": stages,
        "initial_diagnostics": initial_record,
        "final_diagnostics": final_record,
        "final_reflection_zero": _root_json(final_reflection_zero),
        "final_transmission_zero": _root_json(final_transmission_zero),
        "metrics": metrics,
        "passes_performance_gate": performance_pass,
        "provenance": config["provenance"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    figure = _plot_initial_final(
        initial_epsilon, final_epsilon, grid, parameterization.design_rows
    )
    save_figure(figure, figures / "m27d_initial_final_structure.png", dpi=220)
    plt.close(figure)
    figure = _plot_poles_zeros(
        initial_record, final_record, common,
        final_reflection_zero, final_transmission_zero,
    )
    save_figure(figure, figures / "m27d_complex_frequency_markers.png", dpi=220)
    plt.close(figure)
    figure = _plot_response_comparison(
        omega, initial_spectrum, final_spectrum, common
    )
    save_figure(figure, figures / "m27d_real_axis_comparison.png", dpi=220)
    plt.close(figure)
    print(json.dumps({
        "full_common_pole_reached": full_pole_target,
        "reflection_zero_reached": reflection_zero_reached,
        "final_poles": [_pair(value) for value in final_poles],
        "final_reflection_zero": _root_json(final_reflection_zero)["frequency"],
        "final_transmission_zero": _root_json(final_transmission_zero)["frequency"],
        "metrics": metrics,
        "passes_performance_gate": performance_pass,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
