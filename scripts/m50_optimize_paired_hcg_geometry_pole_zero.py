"""Shared numerical definitions required by the published studies."""
from __future__ import annotations
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from pzid.constraints import channel_zero_density_constraint, combine_density_constraints, pole_density_constraint
from pzid.grid import UniformGrid2D
from pzid.hcg import SymmetricPairedHCGGeometryParameterization
from pzid.poles import PoleTarget
from pzid.roots import refine_scalar_root
from pzid.twoport import TwoPortScatteringSetup, solve_two_port_scattering
from scripts.m4b_optimize_kerker_bar import _channel_response, _parity_response
from scripts.m45_screen_paired_hcg import _best_high_transmission_segment

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == 'extends':
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    declaration = json.loads(path.read_text(encoding='utf-8'))
    parent = declaration.get('extends')
    record: dict[str, Any] = {'declaration': str(path), 'sha256': _sha256(path)}
    if parent is None:
        return (declaration, record)
    parent_path = (path.parent / str(parent)).resolve()
    base, parent_record = _load_config(parent_path)
    record['extends'] = parent_record
    return (_deep_merge(base, declaration), record)

def _complex(value: Sequence[float]) -> complex:
    return complex(float(value[0]), float(value[1]))

def _pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]

def _root_json(root: Any) -> dict[str, Any]:
    record = asdict(root)
    record['frequency'] = _pair(root.frequency)
    if root.frequency.imag < 0.0:
        record['quality_factor'] = float(root.frequency.real / (-2.0 * root.frequency.imag))
        record['linewidth_rad_per_um'] = float(-2.0 * root.frequency.imag)
    return record

def _build(config: dict[str, Any]) -> tuple[Any, ...]:
    geometry = config['geometry']
    cells = int(geometry['cells_per_period'])
    spacing = float(geometry['period_um']) / cells
    grid = UniformGrid2D(nx=cells, nz=int(geometry['domain_rows']), dx=spacing, dz=spacing)
    source = int(geometry['lower_source_row'])
    monitors = tuple((int(value) for value in geometry['lower_monitor_rows']))
    setup = TwoPortScatteringSetup(grid=grid, lower_source_plane=source, upper_source_plane=grid.nz - 1 - source, lower_monitor_planes=monitors, upper_monitor_planes=tuple((grid.nz - 1 - value for value in reversed(monitors))), port_index=float(geometry['port_index']))
    parameterization = SymmetricPairedHCGGeometryParameterization(grid=grid, port_index=float(geometry['port_index']), grating_index=float(geometry['grating_index']), thickness_bounds_um=tuple((float(value) for value in geometry['thickness_bounds_um'])), cavity_bounds_um=tuple((float(value) for value in geometry['cavity_bounds_um'])), width_bounds_um=tuple((float(value) for value in geometry['width_bounds_um'])))

    def encode(section: dict[str, Any]) -> np.ndarray:
        return parameterization.encode(float(section['hcg_thickness_um']), float(section['cavity_thickness_um']), float(section['grating_width_um']))
    return (grid, setup, parameterization, encode(config['initial_geometry']), encode(config['target_geometry_for_held_out_reference_only']))

def _refine_poles(epsilon: np.ndarray, setup: Any, specifications: Sequence[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    controls = config['root_refinement']
    roots = []
    for specification in specifications:
        parity = str(specification['parity'])
        root = refine_scalar_root(lambda value, chosen=parity: _parity_response(value, epsilon, setup, chosen), _complex(specification['frequency']), kind='pole', tolerance=float(controls['tolerance']), max_evaluations=int(controls['maximum_evaluations']), search_radius=float(controls['search_radius_rad_per_um']))
        roots.append({'parity': parity, 'root': root})
    return roots

def _refine_zeros(epsilon: np.ndarray, setup: Any, guesses: Sequence[complex], config: dict[str, Any]) -> list[Any]:
    controls = config['root_refinement']
    return [refine_scalar_root(lambda value: _channel_response(value, epsilon, setup, 'reflection'), guess, kind='zero', tolerance=float(controls['tolerance']), max_evaluations=int(controls['maximum_evaluations']), search_radius=float(controls['search_radius_rad_per_um'])) for guess in guesses]

def _constraint_system(raw: np.ndarray, parameterization: Any, setup: Any, pole_targets: Sequence[dict[str, Any]], zero_targets: Sequence[complex], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    evaluations = []
    weights = []
    pole_weight = float(config['optimization']['pole_constraint_weight'])
    zero_weight = float(config['optimization']['zero_constraint_weight'])
    for target in pole_targets:
        coupling = 1.0 if target['parity'] == 'odd' else -1.0
        evaluations.append(pole_density_constraint(raw, parameterization, PoleTarget(complex(target['frequency']), coupling), setup))
        weights.extend([pole_weight] * 4)
    for frequency in zero_targets:
        evaluations.append(channel_zero_density_constraint(raw, parameterization, complex(frequency), 'reflection', setup))
        weights.extend([zero_weight] * 2)
    return combine_density_constraints(*evaluations, weights=np.asarray(weights))

def _spectrum(epsilon: np.ndarray, setup: Any, config: dict[str, Any]) -> dict[str, Any]:
    controls = config['spectrum']
    omega = np.linspace(float(controls['omega_min_rad_per_um']), float(controls['omega_max_rad_per_um']), int(controls['points']))
    transmission = np.empty(omega.size, dtype=np.complex128)
    reflection = np.empty(omega.size, dtype=np.complex128)
    empty = np.empty(omega.size, dtype=np.complex128)
    energy_error = 0.0
    for index, frequency in enumerate(omega):
        result = solve_two_port_scattering(float(frequency), epsilon, setup)
        transmission[index] = result.matrix[1, 0]
        reflection[index] = result.matrix[0, 0]
        empty[index] = result.system.reference_scattering[1, 0]
        energy_error = max(energy_error, abs(abs(transmission[index]) ** 2 + abs(reflection[index]) ** 2 - 1.0))
    deembedded = transmission / empty
    phase = np.unwrap(np.angle(deembedded))
    segment = _best_high_transmission_segment(omega, deembedded, np.abs(transmission) ** 2, float(controls['minimum_transmittance_for_phase']), int(controls['minimum_segment_points']))
    return {'omega': omega, 'transmission': transmission, 'reflection': reflection, 'empty': empty, 'phase': phase, 'high_transmission_segment': segment, 'energy_balance_max_abs_error': float(energy_error)}

def _write_spectrum(path: Path, spectrum: dict[str, Any]) -> None:
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('omega_rad_per_um', 'transmission_real', 'transmission_imag', 'reflection_real', 'reflection_imag', 'transmittance', 'reflectance', 'deembedded_phase_pi'))
        for omega, transmission, reflection, phase in zip(spectrum['omega'], spectrum['transmission'], spectrum['reflection'], spectrum['phase'] / np.pi, strict=True):
            writer.writerow((omega, transmission.real, transmission.imag, reflection.real, reflection.imag, abs(transmission) ** 2, abs(reflection) ** 2, phase))

def _interpolate_targets(initial_poles: Sequence[dict[str, Any]], target_poles: Sequence[dict[str, Any]], initial_zeros: Sequence[complex], target_zeros: Sequence[complex], fraction: float) -> tuple[list[dict[str, Any]], list[complex]]:
    poles = [{'parity': initial['parity'], 'frequency': complex(initial['frequency']) + fraction * (complex(target['frequency']) - complex(initial['frequency']))} for initial, target in zip(initial_poles, target_poles, strict=True)]
    zeros = [initial + fraction * (target - initial) for initial, target in zip(initial_zeros, target_zeros, strict=True)]
    return (poles, zeros)
