"""Independent small Fourier-modal controls (no FDFD data used)."""

from __future__ import annotations

import numpy as np

from pzid.multilayer import normal_incidence_multilayer
from pzid.rcwa2d import centered_binary_grating_layer, homogeneous_layer, solve_tm_stack


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
