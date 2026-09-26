from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pzid.twoport import TwoPortScatteringSetup

from scripts.m4b_optimize_kerker_bar import (
    _frequency,
    _run_stage,
    plot_modes,
    sweep_two_port,
    write_spectrum,
)

from scripts.m5_optimize_coupled_bars import (
    _refine_channel_roots,
    _refine_parity_poles,
    _targeted_metrics,
    build_optimization_problem,
)

def _band_wavelengths(common_pole: complex, count: int) -> np.ndarray:
    gamma = -common_pole.imag
    omega = np.linspace(
        common_pole.real - 3.0 * gamma,
        common_pole.real + 3.0 * gamma,
        count,
    )
    return 2.0 * np.pi / omega

def _scaled_port_setup(
    reference: TwoPortScatteringSetup, target_grid: Any
) -> TwoPortScatteringSetup:
    """Preserve the reference cell-center port locations on a finer grid."""

    scale = target_grid.nx / reference.grid.nx

    def scaled(row: int) -> int:
        return int(round((row + 0.5) * scale - 0.5))

    lower_source = scaled(reference.lower_source_plane)
    lower_monitors = tuple(
        scaled(row) for row in reference.lower_monitor_planes
    )
    return TwoPortScatteringSetup(
        grid=target_grid,
        lower_source_plane=lower_source,
        upper_source_plane=target_grid.nz - 1 - lower_source,
        lower_monitor_planes=lower_monitors,
        upper_monitor_planes=tuple(
            target_grid.nz - 1 - row for row in reversed(lower_monitors)
        ),
        port_index=reference.port_index,
    )

def _spectrum_metrics(
    epsilon: np.ndarray,
    setup: Any,
    wavelengths: np.ndarray,
    common_pole: complex,
    config: dict[str, Any],
) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    transmission, reflection, empty = sweep_two_port(
        wavelengths, epsilon, setup
    )
    metrics = _targeted_metrics(
        wavelengths,
        transmission,
        reflection,
        empty,
        common_pole,
        config,
    )
    return metrics, (transmission, reflection, empty)

def _root_metrics(
    epsilon: np.ndarray,
    setup: Any,
    common_pole: complex,
    config: dict[str, Any],
    reflection_zero_guesses: Sequence[complex] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    parity = _refine_parity_poles(
        epsilon, setup, common_pole, common_pole, config
    )
    channel = _refine_channel_roots(
        epsilon,
        setup,
        common_pole,
        config,
        reflection_zero_guesses=reflection_zero_guesses,
    )
    linewidth = -2.0 * common_pole.imag
    errors = {
        "even_pole_error_over_linewidth": abs(
            _frequency(parity["even_parity_pole"]) - common_pole
        )
        / linewidth,
        "odd_pole_error_over_linewidth": abs(
            _frequency(parity["odd_parity_pole"]) - common_pole
        )
        / linewidth,
        "transmission_pole_error_over_linewidth": abs(
            _frequency(channel["transmission_pole"]) - common_pole
        )
        / linewidth,
        "transmission_zero_error_over_linewidth": abs(
            _frequency(channel["transmission_zero"])
            - np.conj(common_pole)
        )
        / linewidth,
        "reflection_zero_imag_over_linewidth": abs(
            _frequency(channel["reflection_zero"]).imag
        )
        / linewidth,
    }
    return parity, channel, {key: float(value) for key, value in errors.items()}
