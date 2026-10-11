"""Shared numerical definitions required by the published studies."""
from __future__ import annotations
import numpy as np
from pzid.twoport import assemble_two_port_system
CENTER = 5.0

def reference_factor(wavelength: np.ndarray, setup: object, scale: float) -> np.ndarray:
    native_omega = 2 * np.pi * scale / wavelength
    dz = setup.port_dz
    distance = setup.resolved_upper_reference_z - setup.resolved_lower_reference_z
    k_discrete = 2 * np.arcsin(0.5 * dz * native_omega) / dz
    factor = np.exp(1j * k_discrete * distance)
    empty = assemble_two_port_system(2 * np.pi * scale / CENTER, np.ones(setup.grid.shape, dtype=float), setup).reference_scattering[1, 0]
    center_factor = np.exp(1j * 2 * np.arcsin(0.5 * dz * 2 * np.pi * scale / CENTER) / dz * distance)
    if abs(empty - center_factor) > 1e-08:
        raise RuntimeError('empty-cell FDFD reference-plane phase is not verified')
    return factor
