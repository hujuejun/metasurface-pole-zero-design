from __future__ import annotations

import numpy as np

from pzid.multilayer import normal_incidence_multilayer, refine_multilayer_pole

FS_PER_UM_OVER_C = 1e9 / 299792458.0

def symmetric_cavity_layers(
    pair_count: int,
    wavelength_um: float,
    high_index: float,
    low_index: float,
    cavity_half_waves: int = 1,
) -> tuple[tuple[float, float], ...]:
    """Return [H,L]^N + low-index cavity + reverse([H,L]^N)."""

    if pair_count < 1 or cavity_half_waves < 1 or wavelength_um <= 0.0:
        raise ValueError("pair count, half waves, and wavelength must be positive")
    if high_index <= low_index or low_index <= 0.0:
        raise ValueError("indices must satisfy 0 < low < high")
    mirror = (
        (high_index, wavelength_um / (4.0 * high_index)),
        (low_index, wavelength_um / (4.0 * low_index)),
    ) * pair_count
    cavity = ((low_index, cavity_half_waves * wavelength_um / (2.0 * low_index)),)
    return mirror + cavity + tuple(reversed(mirror))

def deembedded_response(
    omega: complex,
    layers: tuple[tuple[float, float], ...],
    port_index: float,
) -> tuple[complex, complex, float]:
    """Return t relative to equal-length port medium, r, and delay in fs.

    The transfer-matrix transmission contains propagation through the stack.
    FDFD reports transmission relative to an empty cell; the exponential
    removes exactly that empty-cell phase from this analytic control.
    """

    result = normal_incidence_multilayer(
        omega, layers, incident_index=port_index, substrate_index=port_index
    )
    length_um = sum(thickness for _, thickness in layers)
    t = result.transmission * np.exp(-1j * omega * port_index * length_um)
    delay_um_over_c = (-result.denominator_derivative / result.denominator).imag
    delay_um_over_c -= port_index * length_um
    return complex(t), result.reflection, float(delay_um_over_c * FS_PER_UM_OVER_C)
