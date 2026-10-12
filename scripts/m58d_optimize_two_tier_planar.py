"""Escape the M58A three-coordinate stationary point in a two-etch planar space."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from pzid.grid import UniformGrid2D
from pzid.hcg import SymmetricPairedTwoTierHCGGeometryParameterization
from pzid.plotting import plot_refractive_index
from pzid.twoport import TwoPortScatteringSetup

from scripts.m50_optimize_paired_hcg_geometry_pole_zero import (
    _build as _build_m50,
    _constraint_system,
    _interpolate_targets,
    _load_config,
    _refine_poles,
    _refine_zeros,
    _root_json,
    _spectrum,
    _write_spectrum,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()


def _complex(values: Sequence[float]) -> complex:
    return complex(float(values[0]), float(values[1]))


def _build(config: dict[str, Any]) -> tuple[Any, ...]:
    geometry = config["geometry"]
    cells = int(geometry["cells_per_period"])
    spacing = float(geometry["period_um"]) / cells
    grid = UniformGrid2D(
        nx=cells, nz=int(geometry["domain_rows"]), dx=spacing, dz=spacing
    )
    source = int(geometry["lower_source_row"])
    monitors = tuple(int(value) for value in geometry["lower_monitor_rows"])
    setup = TwoPortScatteringSetup(
        grid=grid,
        lower_source_plane=source,
        upper_source_plane=grid.nz - 1 - source,
        lower_monitor_planes=monitors,
        upper_monitor_planes=tuple(grid.nz - 1 - value for value in reversed(monitors)),
        port_index=float(geometry["port_index"]),
    )
    bounds = config["two_tier_geometry"]["bounds_um"]
    parameterization = SymmetricPairedTwoTierHCGGeometryParameterization(
        grid=grid,
        port_index=float(geometry["port_index"]),
        grating_index=float(geometry["grating_index"]),
        inner_height_bounds_um=tuple(float(value) for value in bounds["inner_height"]),
        outer_height_bounds_um=tuple(float(value) for value in bounds["outer_height"]),
        cavity_bounds_um=tuple(float(value) for value in bounds["cavity"]),
        inner_width_bounds_um=tuple(float(value) for value in bounds["inner_width"]),
        outer_width_bounds_um=tuple(float(value) for value in bounds["outer_width"]),
    )
    seed = config["two_tier_geometry"]["seed"]
    raw = parameterization.encode(
        float(seed["inner_tier_height_um"]),
        float(seed["outer_tier_height_um"]),
        float(seed["cavity_thickness_um"]),
        float(seed["inner_tier_width_um"]),
        float(seed["outer_tier_width_um"]),
    )
    return grid, setup, parameterization, raw


def _targets(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[complex]]:
    poles = [
        {"parity": str(item["parity"]), "frequency": _complex(item["frequency"])}
        for item in config["target_root_guesses"]["poles"]
    ]
    zeros = [_complex(item) for item in config["target_root_guesses"]["reflection_zeros"]]
    return poles, zeros


def _finite_difference_jacobian(
    raw: np.ndarray,
    parameterization: Any,
    setup: Any,
    poles: Sequence[dict[str, Any]],
    zeros: Sequence[complex],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual, analytic = _constraint_system(
        raw, parameterization, setup, poles, zeros, config
    )
    step = float(config["optimization"]["gradient_check_step"])
    columns = []
    for index in range(raw.size):
        plus = raw.copy()
        minus = raw.copy()
        plus[index] += step
        minus[index] -= step
        plus_residual, _ = _constraint_system(
            plus, parameterization, setup, poles, zeros, config
        )
        minus_residual, _ = _constraint_system(
            minus, parameterization, setup, poles, zeros, config
        )
        columns.append((plus_residual - minus_residual) / (2.0 * step))
    return residual, analytic, np.column_stack(columns)


def _optimize_stage(
    raw: np.ndarray,
    parameterization: Any,
    setup: Any,
    poles: Sequence[dict[str, Any]],
    zeros: Sequence[complex],
    config: dict[str, Any],
    stage: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    cached_raw: np.ndarray | None = None
    cached_system: tuple[np.ndarray, np.ndarray] | None = None
    evaluations = 0

    def evaluate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal cached_raw, cached_system, evaluations
        if cached_raw is None or not np.array_equal(values, cached_raw):
            cached_system = _constraint_system(
                values, parameterization, setup, poles, zeros, config
            )
            cached_raw = values.copy()
            evaluations += 1
            print(
                f"stage {stage} evaluation {evaluations}: "
                f"residual={np.linalg.norm(cached_system[0]):.6e}",
                flush=True,
            )
        assert cached_system is not None
        return cached_system

    initial_residual = float(np.linalg.norm(evaluate(raw)[0]))
    started = time.perf_counter()
    controls = config["optimization"]
    result = least_squares(
        lambda values: evaluate(values)[0],
        raw,
        jac=lambda values: evaluate(values)[1],
        bounds=(0.0, 1.0),
        x_scale=1.0,
        method="trf",
        tr_solver="exact",
        ftol=float(controls["residual_tolerance"]),
        xtol=float(controls["step_tolerance"]),
        gtol=float(controls["gradient_tolerance"]),
        max_nfev=int(controls["maximum_function_evaluations_per_stage"]),
    )
    final_residual = float(np.linalg.norm(evaluate(result.x)[0]))
    return np.asarray(result.x), {
        "initial_residual_norm": initial_residual,
        "final_residual_norm": final_residual,
        "function_evaluations": int(result.nfev),
        "unique_system_evaluations": evaluations,
        "elapsed_seconds": float(time.perf_counter() - started),
        "success": bool(result.success),
        "message": str(result.message),
    }


def _plot(
    output: Path,
    grid: Any,
    initial: np.ndarray,
    final: np.ndarray,
    reference: np.ndarray,
    spectra: dict[str, dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(9.3, 5.1), constrained_layout=True)
    for axis, epsilon, title in zip(
        axes,
        (initial, final, reference),
        ("M58A one-tier start", "M58D two-tier adjoint", "Held-out M50"),
        strict=True,
    ):
        plot_refractive_index(
            epsilon, grid, ax=axis, length_unit="um", colorbar=False, vmin=1.0, vmax=3.48
        )
        axis.set_title(title)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"two_tier_planar_structures.{suffix}", dpi=300)
    plt.close(figure)

    styles = {"initial": ("0.55", ":"), "adjoint": ("tab:blue", "-"), "reference": ("black", "--")}
    figure, axes = plt.subplots(2, 1, figsize=(7.4, 6.8), sharex=True, constrained_layout=True)
    for name, spectrum in spectra.items():
        color, linestyle = styles[name]
        omega = spectrum["omega"]
        axes[0].plot(omega, np.abs(spectrum["transmission"]) ** 2,
                     color=color, linestyle=linestyle, label=f"{name} T")
        axes[0].plot(omega, np.abs(spectrum["reflection"]) ** 2,
                     color=color, linestyle=linestyle, alpha=0.65, label=f"{name} R")
        axes[1].plot(omega, spectrum["phase"] / np.pi,
                     color=color, linestyle=linestyle, label=name)
    axes[0].set(ylabel="Power", ylim=(-0.03, 1.03))
    axes[1].set(xlabel="Real frequency (rad/um)", ylabel="Transmission phase / pi")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(frameon=False, ncol=2, fontsize=8)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"two_tier_planar_response.{suffix}", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config, config_record = _load_config(args.config)
    if not isinstance(config["registration"], dict) or not isinstance(config["provenance"], dict):
        raise ValueError("The HCG continuation requires registration and provenance records")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    grid, setup, parameterization, initial_raw = _build(config)
    target_poles, target_zeros = _targets(config)
    initial_epsilon = parameterization.map(initial_raw).epsilon_r
    initial_specifications = config["initial_root_guesses"]["poles"]
    initial_zero_guesses = [
        _complex(item) for item in config["initial_root_guesses"]["reflection_zeros"]
    ]
    initial_pole_roots = _refine_poles(
        initial_epsilon, setup, initial_specifications, config
    )
    initial_zero_roots = _refine_zeros(
        initial_epsilon, setup, initial_zero_guesses, config
    )
    initial_poles = [
        {"parity": item["parity"], "frequency": complex(item["root"].frequency)}
        for item in initial_pole_roots
    ]
    initial_zeros = [complex(item.frequency) for item in initial_zero_roots]
    residual, analytic, finite = _finite_difference_jacobian(
        initial_raw, parameterization, setup, target_poles, target_zeros, config
    )
    audit = {
        "milestone": config["registration"]["milestone"],
        "target_registration_git_revision": _git_revision(),
        "config": config_record,
        "initial_raw": initial_raw.tolist(),
        "initial_geometry": list(parameterization.decode(initial_raw)),
        "initial_target_residual_norm": float(np.linalg.norm(residual)),
        "jacobian_shape": list(analytic.shape),
        "jacobian_singular_values": np.linalg.svd(analytic, compute_uv=False).tolist(),
        "complete_constraint_jacobian_finite_difference_relative_error": float(
            np.linalg.norm(analytic - finite) / np.linalg.norm(finite)
        ),
        "initial_poles": [
            {"parity": item["parity"], "root": _root_json(item["root"])}
            for item in initial_pole_roots
        ],
        "initial_reflection_zeros": [_root_json(item) for item in initial_zero_roots],
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    if args.audit_only:
        return

    current = initial_raw.copy()
    history = []
    fractions = [float(value) for value in config["optimization"]["continuation_fractions"]]
    for stage, fraction in enumerate(fractions, start=1):
        poles, zeros = _interpolate_targets(
            initial_poles, target_poles, initial_zeros, target_zeros, fraction
        )
        proposed, record = _optimize_stage(
            current, parameterization, setup, poles, zeros, config, stage
        )
        accepted = record["final_residual_norm"] <= float(
            config["optimization"]["maximum_stage_residual_norm"]
        )
        record.update(
            {
                "stage": stage,
                "fraction": fraction,
                "accepted": bool(accepted),
                "start_geometry": list(parameterization.decode(current)),
                "proposed_geometry": list(parameterization.decode(proposed)),
            }
        )
        history.append(record)
        (output / f"stage_{stage:02d}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        if not accepted:
            break
        current = proposed

    final_epsilon = parameterization.map(current).epsilon_r
    specifications = config["target_root_guesses"]["poles"]
    final_poles = _refine_poles(final_epsilon, setup, specifications, config)
    final_zeros = _refine_zeros(final_epsilon, setup, target_zeros, config)
    target_pole_values = [complex(item["frequency"]) for item in target_poles]
    final_pole_values = [complex(item["root"].frequency) for item in final_poles]
    final_zero_values = [complex(item.frequency) for item in final_zeros]
    mean_linewidth = float(np.mean([-2.0 * value.imag for value in target_pole_values]))
    pole_error = max(
        abs(actual - target)
        for actual, target in zip(final_pole_values, target_pole_values, strict=True)
    ) / mean_linewidth
    zero_error = max(
        abs(actual - target)
        for actual, target in zip(final_zero_values, target_zeros, strict=True)
    ) / mean_linewidth

    m50_config, m50_record = _load_config(Path(config["held_out_m50_config"]))
    reference_grid, _, reference_parameterization, _, reference_raw = _build_m50(m50_config)
    if reference_grid != grid:
        raise RuntimeError("held-out M50 grid differs from M58D grid")
    reference_epsilon = reference_parameterization.map(reference_raw).epsilon_r
    spectra = {
        "initial": _spectrum(initial_epsilon, setup, config),
        "adjoint": _spectrum(final_epsilon, setup, config),
        "reference": _spectrum(reference_epsilon, setup, config),
    }
    for name, spectrum in spectra.items():
        _write_spectrum(output / f"{name}_spectrum.csv", spectrum)
    np.savetxt(output / "final_raw.csv", current, delimiter=",")
    np.savetxt(output / "final_permittivity.csv", final_epsilon, delimiter=",")
    segment = spectra["adjoint"]["high_transmission_segment"]
    gates = config["gates"]
    summary = {
        "audit": audit,
        "history": history,
        "final_raw": current.tolist(),
        "final_geometry": list(parameterization.decode(current)),
        "held_out_m50_config": m50_record,
        "final_poles": [
            {"parity": item["parity"], "root": _root_json(item["root"])}
            for item in final_poles
        ],
        "final_reflection_zeros": [_root_json(item) for item in final_zeros],
        "maximum_pole_error_in_mean_linewidths": float(pole_error),
        "maximum_zero_error_in_mean_linewidths": float(zero_error),
        "high_transmission_segment": segment,
        "maximum_energy_balance_error": spectra["adjoint"]["energy_balance_max_abs_error"],
        "passes_pole_gate": bool(pole_error <= float(gates["maximum_pole_error_in_mean_linewidths"])),
        "passes_zero_gate": bool(zero_error <= float(gates["maximum_zero_error_in_mean_linewidths"])),
        "passes_response_gate": bool(
            segment["phase_span_pi"] >= float(gates["minimum_phase_span_pi"])
            and segment["minimum_transmittance"] >= float(gates["minimum_transmittance"])
            and spectra["adjoint"]["energy_balance_max_abs_error"]
            <= float(gates["maximum_energy_balance_error"])
        ),
        "provenance": config["provenance"],
    }
    summary["passes_m58d_gate"] = bool(
        len(history) == len(fractions)
        and all(item["accepted"] for item in history)
        and summary["passes_pole_gate"]
        and summary["passes_zero_gate"]
        and summary["passes_response_gate"]
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _plot(output, grid, initial_epsilon, final_epsilon, reference_epsilon, spectra)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
