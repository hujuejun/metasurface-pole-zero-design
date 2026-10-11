"""Shared numerical definitions required by the published studies."""
from __future__ import annotations
from pzid.rcwa2d_design_legacy import centered_binary_grating_layer, homogeneous_layer, solve_tm_stack

def _response(config: dict, period: float, height: float, harmonics: int):
    layers = (homogeneous_layer(float(config['high_index']), float(config['slab_thickness_um']), harmonics), centered_binary_grating_layer(float(config['port_index']), float(config['high_index']), float(config['ridge_fill_factor']), height, harmonics))
    return lambda omega: solve_tm_stack(omega, layers, period_um=period, port_index=float(config['port_index']), harmonic_half_width=harmonics).transmission
