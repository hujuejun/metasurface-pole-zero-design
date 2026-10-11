"""Shared numerical definitions required by the published studies."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import numpy as np
from scripts.m58d_optimize_two_tier_planar import _build, _load_config
ROOT = Path(__file__).resolve().parents[1]

def _complex_index(values: list[float]) -> complex:
    if len(values) != 2:
        raise ValueError('index must be [n, k]')
    return complex(float(values[0]), float(values[1]))

def _build_scaled_baseline(config: dict[str, Any], baseline: dict[str, Any], cells_per_period: int | None=None) -> tuple[Any, Any, np.ndarray, float, list[float]]:
    optical = config['optical_model']
    target_wavelength = float(optical['target_wavelength_um'])
    center_omega = float(optical['baseline_high_transmission_center_omega_rad_per_um'])
    scale = target_wavelength / (2.0 * math.pi / center_omega) if optical['rescale_all_lengths_to_target_wavelength'] else 1.0
    m58d_config, _ = _load_config(ROOT / config['baseline_config'])
    if cells_per_period is not None:
        original_cells = int(m58d_config['geometry']['cells_per_period'])
        ratio = cells_per_period / original_cells
        geometry_config = m58d_config['geometry']
        geometry_config['cells_per_period'] = cells_per_period
        geometry_config['domain_rows'] = round(int(geometry_config['domain_rows']) * ratio)
        geometry_config['lower_source_row'] = round(int(geometry_config['lower_source_row']) * ratio)
        geometry_config['lower_monitor_rows'] = [round(int(row) * ratio) for row in geometry_config['lower_monitor_rows']]
    m58d_config['geometry']['period_um'] *= scale
    for bounds in m58d_config['two_tier_geometry']['bounds_um'].values():
        bounds[0] *= scale
        bounds[1] *= scale
    for name in ('inner_tier_height_um', 'outer_tier_height_um', 'cavity_thickness_um', 'inner_tier_width_um', 'outer_tier_width_um'):
        m58d_config['two_tier_geometry']['seed'][name] *= scale
    grid, setup, parameterization, _ = _build(m58d_config)
    geometry = [float(value) * scale for value in baseline['final_geometry']]
    if len(geometry) != 5:
        raise ValueError('M58D final geometry must contain five coordinates')
    mapping = parameterization.map(parameterization.encode(*geometry))
    port_epsilon = float(m58d_config['geometry']['port_index']) ** 2
    seed_epsilon = float(m58d_config['geometry']['grating_index']) ** 2
    fill_fraction = (mapping.epsilon_r - port_epsilon) / (seed_epsilon - port_epsilon)
    if np.min(fill_fraction) < -1e-12 or np.max(fill_fraction) > 1.0 + 1e-12:
        raise RuntimeError('M58D physical-interface fill fraction left [0, 1]')
    if not np.allclose(fill_fraction, np.flipud(fill_fraction), atol=1e-12, rtol=0.0):
        raise RuntimeError('the inherited M58D geometry lost axial symmetry')
    return (setup, fill_fraction, grid, scale, geometry)
