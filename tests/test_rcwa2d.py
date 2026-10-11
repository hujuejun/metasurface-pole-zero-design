"""Independent small Fourier-modal controls (no FDFD data used)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from pzid import rcwa2d_design_legacy as rcwa2d_legacy_m166
from pzid.multilayer import normal_incidence_multilayer
from pzid.rcwa2d import _modes, centered_binary_grating_layer, homogeneous_layer, solve_tm_stack


def _lamellar_bloch_trace(
    beta: float, *, omega: float, period: float, fill: float, low_index: float, high_index: float
) -> float:
    """Exact x-period transfer-matrix trace for a TM grating-layer mode."""

    def segment(index: float, width: float) -> np.ndarray:
        epsilon = index**2
        k = np.lib.scimath.sqrt(epsilon * omega**2 - beta**2)
        cosine = np.cos(k * width)
        sine_over_k = np.sinc(k * width / np.pi) * width
        return np.array(
            [[cosine, epsilon * sine_over_k],
             [-k**2 * sine_over_k / epsilon, cosine]],
            dtype=complex,
        )

    transfer = segment(low_index, period * (1 - fill)) @ segment(high_index, period * fill)
    assert abs(transfer.trace().imag) < 1e-12
    return float(transfer.trace().real - 2)


def test_tm_zero_harmonic_has_correct_normal_field_effective_permittivity() -> None:
    """The TM field is normal to the vertical ridge wall in the static limit."""

    low_index, high_index, fill, omega = 1.45, 2.4, 0.37, 4.0
    layer = centered_binary_grating_layer(low_index, high_index, fill, 0.2, 0)
    beta = _modes(omega, layer, np.array([0.0])).beta[0]
    effective_epsilon = 1 / (fill / high_index**2 + (1 - fill) / low_index**2)
    assert abs(beta - omega * np.sqrt(effective_epsilon)) < 1e-12


def test_tm_grating_layer_mode_matches_exact_interface_transfer() -> None:
    """Compare the Fourier modal eigenproblem with an independent 1D interface solution."""

    low_index, high_index, fill, omega, period = 1.45, 2.4, 0.37, 4.0, 0.25
    trace = lambda beta: _lamellar_bloch_trace(
        beta, omega=omega, period=period, fill=fill,
        low_index=low_index, high_index=high_index,
    )
    exact_beta = brentq(trace, omega * low_index, omega * high_index, xtol=1e-13)
    half_width = 20
    layer = centered_binary_grating_layer(low_index, high_index, fill, 0.2, half_width)
    transverse_k = 2 * np.pi * np.arange(-half_width, half_width + 1) / period
    modal_beta = _modes(omega, layer, transverse_k).beta
    assert np.min(abs(modal_beta - exact_beta)) < 2e-4


def test_tm_subwavelength_grating_scattering_matches_harmonic_effective_film() -> None:
    low_index, high_index, fill, omega = 1.45, 2.4, 0.37, 4.0
    thickness = 0.17
    half_width = 7
    layer = centered_binary_grating_layer(low_index, high_index, fill, thickness, half_width)
    effective_index = np.sqrt(1 / (fill / high_index**2 + (1 - fill) / low_index**2))
    film = normal_incidence_multilayer(
        omega, ((effective_index, thickness),),
        incident_index=low_index, substrate_index=low_index,
    )
    errors = []
    for period in (0.02, 0.01, 0.005):
        response = solve_tm_stack(
            omega, (layer,), period_um=period, port_index=low_index,
            harmonic_half_width=half_width,
        )
        errors.append(max(
            abs(response.transmission - film.transmission),
            abs(response.reflection + film.reflection),
        ))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 5e-4


def test_legacy_tm_factorization_requires_explicit_opt_in() -> None:
    low_index, high_index, fill, omega = 1.45, 2.4, 0.37, 4.0
    layer = centered_binary_grating_layer(low_index, high_index, fill, 0.2, 0)
    legacy_beta = _modes(omega, layer, np.array([0.0]), factorization="legacy").beta[0]
    arithmetic_epsilon = fill * high_index**2 + (1 - fill) * low_index**2
    assert abs(legacy_beta - omega * np.sqrt(arithmetic_epsilon)) < 1e-12


def test_explicit_legacy_branch_reproduces_archived_solver() -> None:
    omega, half_width = 4.05 - 0.02j, 5
    layers = (centered_binary_grating_layer(1.45, 2.4, 0.37, 0.17, half_width),)
    archived_layers = (
        rcwa2d_legacy_m166.centered_binary_grating_layer(1.45, 2.4, 0.37, 0.17, half_width),
    )
    current = solve_tm_stack(
        omega, layers, period_um=0.5, port_index=1.45,
        harmonic_half_width=half_width, factorization="legacy",
    )
    archived = rcwa2d_legacy_m166.solve_tm_stack(
        omega, archived_layers, period_um=0.5, port_index=1.45,
        harmonic_half_width=half_width,
    )
    assert abs(current.transmission - archived.transmission) < 1e-12
    assert abs(current.reflection - archived.reflection) < 1e-12


def test_uniform_film_matches_analytic_multilayer() -> None:
    omega = 4.1
    thickness = 0.21
    for harmonics in (0, 2, 4):
        response = solve_tm_stack(
            omega,
            (homogeneous_layer(2.4, thickness, harmonics),),
            period_um=0.5,
            port_index=1.45,
            harmonic_half_width=harmonics,
        )
        analytic = normal_incidence_multilayer(omega, ((2.4, thickness),), incident_index=1.45, substrate_index=1.45)
        assert abs(response.transmission - analytic.transmission) < 1e-11
        assert abs(response.reflection + analytic.reflection) < 1e-11
        assert abs(abs(response.transmission) ** 2 + abs(response.reflection) ** 2 - 1) < 1e-11


def test_centered_grating_matches_uniform_endpoints_and_conserves_power() -> None:
    for fill in (0.0, 1.0):
        grating = centered_binary_grating_layer(1.45, 2.4, fill, 0.06, 3)
        response = solve_tm_stack(4.05, (grating,), period_um=0.5, port_index=1.45, harmonic_half_width=3)
        index = 1.45 if fill == 0 else 2.4
        analytic = normal_incidence_multilayer(4.05, ((index, 0.06),), incident_index=1.45, substrate_index=1.45)
        assert abs(response.transmission - analytic.transmission) < 1e-11
    grating = centered_binary_grating_layer(1.45, 2.4, 0.25, 0.06, 5)
    response = solve_tm_stack(4.05, (grating,), period_um=0.5, port_index=1.45, harmonic_half_width=5)
    assert abs(abs(response.transmission) ** 2 + abs(response.reflection) ** 2 - 1) < 1e-10
    assert np.max(abs(response.transmitted_orders[np.arange(11) != 5])) > 0


def test_multilayer_real_and_complex_frequency_against_analytic() -> None:
    material_layers = ((2.4, 0.13), (1.45, 0.19), (2.1, 0.07), (1.45, 0.12))
    for omega in (4.05, 4.05 - 0.02j):
        for harmonics in (0, 3):
            layers = tuple(homogeneous_layer(index, thickness, harmonics) for index, thickness in material_layers)
            response = solve_tm_stack(omega, layers, period_um=0.5, port_index=1.45, harmonic_half_width=harmonics)
            analytic = normal_incidence_multilayer(omega, material_layers, incident_index=1.45, substrate_index=1.45)
            assert abs(response.transmission - analytic.transmission) < 1e-10
            assert abs(response.reflection + analytic.reflection) < 1e-10
