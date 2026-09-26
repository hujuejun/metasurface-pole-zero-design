from __future__ import annotations

import csv

from pathlib import Path

from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt

import numpy as np

from pzid.constraints import (
    channel_value_density_constraint,
    combine_density_constraints,
    pole_density_constraint,
)

from pzid.density import (
    AxiallySymmetricDensityParameterization,
    DensityParameterization,
)

from pzid.optimize import LMOptions, LMResult, solve_underdetermined_lm

from pzid.poles import PoleTarget

from pzid.roots import ScalarRootResult, refine_scalar_root

from pzid.twoport import TwoPortScatteringSetup, solve_two_port_scattering

def _inverse_density(
    epsilon: np.ndarray, parameterization: DensityParameterization
) -> np.ndarray:
    start, stop = parameterization.design_rows
    desired_index = np.sqrt(epsilon[start:stop])
    projected = (desired_index - parameterization.low_index) / (
        parameterization.high_index - parameterization.low_index
    )
    beta = parameterization.projection_beta
    threshold = parameterization.projection_threshold
    denominator = np.tanh(beta * threshold) + np.tanh(beta * (1.0 - threshold))
    argument = projected * denominator - np.tanh(beta * threshold)
    argument = np.clip(
        argument,
        -1.0 + np.finfo(float).eps,
        1.0 - np.finfo(float).eps,
    )
    return np.clip(threshold + np.arctanh(argument) / beta, 0.0, 1.0)

def _parity_response(
    omega: complex,
    epsilon: np.ndarray,
    setup: TwoPortScatteringSetup,
    parity: str,
) -> complex:
    solution = solve_two_port_scattering(omega, epsilon, setup)
    matrix = solution.matrix
    empty_t = solution.system.reference_scattering[1, 0]
    if parity == "even":
        return complex((matrix[0, 0] - matrix[1, 0]) / (-empty_t))
    if parity == "odd":
        return complex((matrix[0, 0] + matrix[1, 0]) / empty_t)
    raise ValueError("parity must be 'even' or 'odd'")

def _channel_response(
    omega: complex,
    epsilon: np.ndarray,
    setup: TwoPortScatteringSetup,
    channel: str,
) -> complex:
    matrix = solve_two_port_scattering(omega, epsilon, setup).matrix
    if channel == "transmission":
        return complex(matrix[1, 0])
    if channel == "reflection":
        return complex(matrix[0, 0])
    raise ValueError("unknown channel")

def _root_record(result: ScalarRootResult) -> dict[str, Any]:
    omega = result.frequency
    record: dict[str, Any] = {
        "kind": result.kind,
        "success": result.success,
        "message": result.message,
        "evaluations": result.evaluations,
        "residual_norm": result.residual_norm,
        "omega_real_rad_per_um": omega.real,
        "omega_imag_rad_per_um": omega.imag,
        "wavelength_real_um": (2.0 * np.pi / omega).real,
        "wavelength_imag_um": (2.0 * np.pi / omega).imag,
    }
    if result.kind == "pole":
        record["quality_factor"] = omega.real / (-2.0 * omega.imag)
    return record

def _frequency(record: dict[str, Any]) -> complex:
    return complex(
        record["omega_real_rad_per_um"], record["omega_imag_rad_per_um"]
    )

def sweep_two_port(
    wavelengths: np.ndarray,
    epsilon: np.ndarray,
    setup: TwoPortScatteringSetup,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transmission = np.empty(wavelengths.size, dtype=np.complex128)
    reflection = np.empty_like(transmission)
    empty_transmission = np.empty_like(transmission)
    for index, wavelength in enumerate(wavelengths):
        solution = solve_two_port_scattering(2.0 * np.pi / wavelength, epsilon, setup)
        transmission[index] = solution.matrix[1, 0]
        reflection[index] = solution.matrix[0, 0]
        empty_transmission[index] = solution.system.reference_scattering[1, 0]
        if index % 25 == 0 or index + 1 == wavelengths.size:
            print(f"spectrum {index + 1}/{wavelengths.size}", flush=True)
    return transmission, reflection, empty_transmission

def write_spectrum(
    path: Path,
    wavelengths: np.ndarray,
    transmission: np.ndarray,
    reflection: np.ndarray,
    empty_transmission: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "wavelength_um",
                "omega_rad_per_um",
                "transmission_real",
                "transmission_imag",
                "reflection_real",
                "reflection_imag",
                "empty_transmission_real",
                "empty_transmission_imag",
                "deembedded_transmission_real",
                "deembedded_transmission_imag",
                "transmittance",
                "reflectance",
            )
        )
        deembedded = transmission / empty_transmission
        for wavelength, t_value, r_value, empty_value, deembedded_value in zip(
            wavelengths,
            transmission,
            reflection,
            empty_transmission,
            deembedded,
            strict=True,
        ):
            writer.writerow(
                (
                    wavelength,
                    2.0 * np.pi / wavelength,
                    t_value.real,
                    t_value.imag,
                    r_value.real,
                    r_value.imag,
                    empty_value.real,
                    empty_value.imag,
                    deembedded_value.real,
                    deembedded_value.imag,
                    abs(t_value) ** 2,
                    abs(r_value) ** 2,
                )
            )

def _run_stage(
    name: str,
    initial: np.ndarray,
    parameterization: AxiallySymmetricDensityParameterization,
    setup: TwoPortScatteringSetup,
    pole_targets: Sequence[PoleTarget],
    transmission_zero: complex | None,
    reflection_targets: Sequence[tuple[float, complex]],
    config: dict[str, Any],
    iterations: int,
) -> tuple[LMResult, int]:
    evaluations = 0

    def system(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
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
                complex(frequency),
                "reflection",
                target_value,
                setup,
            )
            for frequency, target_value in reflection_targets
        )
        pole_weight = float(config["optimization"]["pole_constraint_weight"])
        zero_weight = float(
            config["optimization"]["transmission_zero_constraint_weight"]
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
        print(
            f"{name} evaluation {evaluations}: "
            f"residual={np.linalg.norm(residual):.6e}",
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
            active_set_bounds=bool(controls.get("active_set_bounds", False)),
            active_set_max_iterations=int(
                controls.get("active_set_max_iterations", 20)
            ),
        ),
    )
    return result, evaluations

def _mode_field(
    omega: complex,
    ratio: float,
    epsilon: np.ndarray,
    setup: TwoPortScatteringSetup,
) -> tuple[np.ndarray, float, float]:
    solution = solve_two_port_scattering(omega.real, epsilon, setup)
    field = (solution.fields @ np.asarray((1.0, ratio))).reshape(setup.grid.shape)
    norm = max(float(np.linalg.norm(field)), np.finfo(float).tiny)
    even_error = float(np.linalg.norm(field - np.flipud(field)) / norm)
    odd_error = float(np.linalg.norm(field + np.flipud(field)) / norm)
    index = np.unravel_index(np.argmax(np.abs(field)), field.shape)
    aligned = field * np.exp(-1j * np.angle(field[index]))
    return np.real(aligned), even_error, odd_error

def plot_structures(
    initial_epsilon: np.ndarray,
    final_epsilon: np.ndarray,
    setup: TwoPortScatteringSetup,
    design_rows: tuple[int, int],
    low_index: float,
    high_index: float,
    titles: tuple[str, str] = (
        "Initial independent-mode bar",
        "Pole-zero optimized cell",
    ),
) -> plt.Figure:
    figure, axes = plt.subplots(
        1, 2, figsize=(8.4, 4.0), sharex=True, sharey=True, constrained_layout=True
    )
    for axis, epsilon, title in zip(
        axes,
        (initial_epsilon, final_epsilon),
        titles,
        strict=True,
    ):
        image = axis.imshow(
            np.sqrt(epsilon),
            origin="lower",
            extent=(0.0, setup.grid.period, 0.0, setup.grid.height),
            interpolation="nearest",
            aspect="equal",
            cmap="viridis",
            vmin=low_index,
            vmax=high_index,
        )
        for row in design_rows:
            axis.axhline(row * setup.grid.dz, color="white", linestyle="--", linewidth=0.7)
        axis.set_xlabel("x (um)")
        axis.set_title(title)
    axes[0].set_ylabel("z (um)")
    figure.colorbar(image, ax=axes, label="Refractive index, n", shrink=0.86)
    return figure

def plot_modes(
    roots: dict[str, Any],
    epsilon: np.ndarray,
    setup: TwoPortScatteringSetup,
) -> tuple[plt.Figure, list[dict[str, Any]]]:
    figure, axes = plt.subplots(
        1, 2, figsize=(8.4, 4.0), sharex=True, sharey=True, constrained_layout=True
    )
    diagnostics: list[dict[str, Any]] = []
    for axis, parity, ratio in zip(
        axes, ("even", "odd"), (-1.0, 1.0), strict=True
    ):
        pole = _frequency(roots[f"{parity}_parity_pole"])
        field, even_error, odd_error = _mode_field(pole, ratio, epsilon, setup)
        scale = max(float(np.max(np.abs(field))), np.finfo(float).tiny)
        image = axis.imshow(
            field / scale,
            origin="lower",
            extent=(0.0, setup.grid.period, 0.0, setup.grid.height),
            aspect="equal",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
        )
        axis.contour(
            setup.grid.x,
            setup.grid.z,
            np.sqrt(epsilon),
            levels=((np.sqrt(np.min(epsilon)) + np.sqrt(np.max(epsilon))) / 2.0,),
            colors="black",
            linewidths=0.6,
        )
        error = even_error if parity == "even" else odd_error
        diagnostics.append(
            {
                "parity": parity,
                "even_error": even_error,
                "odd_error": odd_error,
                "assigned_parity_error": error,
            }
        )
        axis.set_xlabel("x (um)")
        axis.set_title(f"{parity.capitalize()} $H_y$ mode\nparity error = {error:.2e}")
    axes[0].set_ylabel("z (um)")
    figure.colorbar(image, ax=axes, label="Phase-aligned Re($H_y$), normalized", shrink=0.86)
    return figure, diagnostics

def plot_real_axis(
    wavelengths: np.ndarray,
    initial_t: np.ndarray,
    initial_r: np.ndarray,
    initial_empty: np.ndarray,
    final_t: np.ndarray,
    final_r: np.ndarray,
    final_empty: np.ndarray,
    target_pole: complex,
    *,
    title: str = "M4B reflectionless phase-control response",
    band_halfwidth_in_gamma: float = 1.0,
) -> plt.Figure:
    order = np.argsort(2.0 * np.pi / wavelengths)
    omega = 2.0 * np.pi / wavelengths[order]
    initial_phase = np.unwrap(np.angle(initial_t[order] / initial_empty[order]))
    final_phase = np.unwrap(np.angle(final_t[order] / final_empty[order]))
    initial_phase -= initial_phase[0]
    final_phase -= final_phase[0]
    gamma = -target_pole.imag
    figure, axes = plt.subplots(
        2, 1, figsize=(7.2, 6.0), sharex=True, constrained_layout=True
    )
    axes[0].plot(omega, np.abs(initial_r[order]) ** 2, color="0.55", label="Initial R")
    axes[0].plot(omega, np.abs(final_r[order]) ** 2, label="Optimized R")
    axes[0].plot(omega, np.abs(final_t[order]) ** 2, linestyle="--", label="Optimized T")
    axes[0].set_ylabel("Power coefficient")
    axes[0].set_title(title)
    axes[0].legend(loc="best", frameon=False)
    axes[1].plot(omega, initial_phase / np.pi, color="0.55", label="Initial")
    axes[1].plot(omega, final_phase / np.pi, label="Optimized")
    axes[1].set_xlabel("omega (rad/um)")
    axes[1].set_ylabel("De-embedded transmission phase / pi")
    axes[1].legend(loc="best", frameon=False)
    for axis in axes:
        axis.axvspan(
            target_pole.real - band_halfwidth_in_gamma * gamma,
            target_pole.real + band_halfwidth_in_gamma * gamma,
            color="0.5",
            alpha=0.10,
        )
        axis.grid(True, linewidth=0.4, alpha=0.35)
    return figure

def plot_history(
    results: Sequence[tuple[str, LMResult]],
    *,
    title: str = "Staged M4B inverse design",
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    offset = 0
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    groups: dict[str, str] = {}
    for name, result in results:
        if name not in groups:
            groups[name] = colors[len(groups) % len(colors)]
        x = np.arange(1, len(result.history) + 1) + offset
        y = [entry.residual_norm for entry in result.history]
        if len(x):
            axis.semilogy(
                x,
                y,
                marker="o",
                markersize=3,
                color=groups[name],
                label=name if name not in axis.get_legend_handles_labels()[1] else "_nolegend_",
            )
            offset = int(x[-1])
    axis.set_xlabel("Accepted iteration")
    axis.set_ylabel("Stage constraint residual norm")
    axis.set_title(title)
    axis.grid(True, which="both", linewidth=0.4, alpha=0.35)
    axis.legend(frameon=False)
    return figure

def make_complex_map(
    epsilon: np.ndarray,
    setup: TwoPortScatteringSetup,
    target_pole: complex,
    real_points: int,
    imag_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gamma = -target_pole.imag
    real_axis = np.linspace(target_pole.real - 2.0 * gamma, target_pole.real + 2.0 * gamma, real_points)
    imag_axis = np.linspace(-1.3 * gamma, 1.3 * gamma, imag_points)
    transmission = np.empty((imag_points, real_points), dtype=np.complex128)
    reflection = np.empty_like(transmission)
    for row, imaginary in enumerate(imag_axis):
        for column, real in enumerate(real_axis):
            matrix = solve_two_port_scattering(complex(real, imaginary), epsilon, setup).matrix
            reflection[row, column] = matrix[0, 0]
            transmission[row, column] = matrix[1, 0]
        print(f"complex map {row + 1}/{imag_points}", flush=True)
    return real_axis, imag_axis, transmission, reflection
