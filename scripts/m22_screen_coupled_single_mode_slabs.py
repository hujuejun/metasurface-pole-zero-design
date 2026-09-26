from __future__ import annotations

from typing import Any

import numpy as np

from pzid.modes import (
    parity_errors,
    phase_align,
    resonant_residue_field,
    scattering_pole_residue,
    transverse_harmonic_fractions,
    two_port_residue_diagnostics,
)

from pzid.ports import discrete_axial_wavenumber

from pzid.twoport import TwoPortScatteringSetup, solve_two_port_scattering

from scripts.m11_screen_fp_tm1_hybrid import _json_radiation, _root_json, _sweep, _write_spectrum

def _pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]

def _overlap(first: np.ndarray, second: np.ndarray, rows: tuple[int, int]) -> float:
    a = np.asarray(first[slice(*rows)]).ravel()
    b = np.asarray(second[slice(*rows)]).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= np.finfo(float).tiny:
        return 0.0
    return float(abs(np.vdot(a, b)) / denominator)

def _isolated_slab_reference_fields(
    grid: Any,
    layout: Any,
    omega: float,
    parity: str,
    core_index: float,
    cladding_index: float,
) -> tuple[np.ndarray, np.ndarray]:
    beta = 2.0 * np.pi / grid.period
    h = np.sqrt(max(core_index**2 * omega**2 - beta**2, 0.0))
    q = np.sqrt(max(beta**2 - cladding_index**2 * omega**2, 0.0))

    def local_profile(rows: tuple[int, int]) -> np.ndarray:
        start, stop = rows
        center = 0.5 * (start + stop) * grid.dz
        half = 0.5 * (stop - start) * grid.dz
        relative = grid.z - center
        inside = np.abs(relative) <= half + 1e-12
        profile = np.empty(grid.nz, dtype=float)
        profile[inside] = np.cos(h * relative[inside])
        boundary = np.cos(h * half)
        outside = np.maximum(np.abs(relative[~inside]) - half, 0.0)
        profile[~inside] = boundary * np.exp(-q * outside)
        return profile

    lower = local_profile(layout.lower_slab_rows)
    upper = local_profile(layout.upper_slab_rows)
    if parity == "even":
        axial = lower + upper
    elif parity == "odd":
        axial = lower - upper
    else:
        raise ValueError(f"unsupported parity {parity!r}")
    phase = beta * (grid.x - 0.5 * grid.period)
    cosine = axial[:, None] * np.cos(phase)[None, :]
    sine = axial[:, None] * np.sin(phase)[None, :]
    return cosine.astype(np.complex128), sine.astype(np.complex128)

def _additional_orders(pole_real: float, setup: TwoPortScatteringSetup) -> int:
    maximum = int(
        np.floor(
            abs(pole_real)
            * setup.port_index
            * setup.grid.period
            / (2.0 * np.pi)
        )
    )
    return max(0, 2 * maximum)

def _diagnose_candidate(
    root: Any,
    parity: str,
    isolated_frequency: float,
    epsilon: np.ndarray,
    grid: Any,
    layout: Any,
    setup: TwoPortScatteringSetup,
    config: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    pole = complex(root.frequency)
    coupling_ratio = -1.0 if parity == "even" else 1.0
    field = phase_align(resonant_residue_field(pole, coupling_ratio, epsilon, setup))
    reference_cos, reference_sin = _isolated_slab_reference_fields(
        grid,
        layout,
        isolated_frequency,
        parity,
        float(config["benchmark"]["slab_index"]),
        float(config["benchmark"]["port_index"]),
    )
    analytic_overlap = max(
        _overlap(field, reference_cos, layout.structure_rows),
        _overlap(field, reference_sin, layout.structure_rows),
    )
    even_error, odd_error = parity_errors(field)
    harmonics = transverse_harmonic_fractions(field, layout.structure_rows)
    total_energy = max(float(np.sum(np.abs(field) ** 2)), np.finfo(float).tiny)
    lower_rows = layout.lower_component_rows
    upper_rows = layout.upper_component_rows
    lower_energy = float(np.sum(np.abs(field[slice(*lower_rows)]) ** 2))
    upper_energy = float(np.sum(np.abs(field[slice(*upper_rows)]) ** 2))
    structure_energy = (lower_energy + upper_energy) / total_energy
    balance = min(lower_energy, upper_energy) / max(
        max(lower_energy, upper_energy), np.finfo(float).tiny
    )
    residue_matrix = scattering_pole_residue(pole, epsilon, setup)
    radiation = two_port_residue_diagnostics(residue_matrix)
    reference_offset = (
        float(layout.structure_rows[0]) * grid.dz - setup.resolved_lower_reference_z
    )
    port_wavenumber = discrete_axial_wavenumber(
        pole, refractive_index=setup.port_index, dz=grid.dz
    )
    reflection_residue = complex(residue_matrix[0, 0])
    boundary_residue = reflection_residue * np.exp(
        -2j * port_wavenumber * reference_offset
    )
    assignment = config["assignment"]
    parity_error = even_error if parity == "even" else odd_error
    reasons: list[str] = []
    if root.residual_norm > float(config["root_refinement"]["residual_gate"]):
        reasons.append("root_residual")
    if parity_error > float(assignment["maximum_field_parity_error"]):
        reasons.append("field_parity")
    if harmonics["first_harmonic_pair_fraction"] < float(
        assignment["minimum_first_harmonic_pair_fraction"]
    ):
        reasons.append("first_harmonic")
    if analytic_overlap < float(assignment["minimum_isolated_mode_reference_overlap"]):
        reasons.append("isolated_mode_overlap")
    if structure_energy < float(
        assignment["minimum_bilayer_structure_energy_fraction"]
    ):
        reasons.append("bilayer_energy")
    additional_orders = _additional_orders(pole.real, setup)
    if additional_orders > int(assignment["maximum_additional_open_port_orders"]):
        reasons.append("extra_diffraction_ports")
    score = float(
        analytic_overlap
        * harmonics["first_harmonic_pair_fraction"]
        * np.sqrt(max(structure_energy, 0.0))
        * np.sqrt(max(balance, 0.0))
        / (1.0 + abs(pole.real - isolated_frequency))
    )
    return (
        {
            "parity": parity,
            "root": _root_json(root),
            "quality_factor": float(pole.real / (-2.0 * pole.imag)),
            "full_linewidth_rad_per_um": float(-2.0 * pole.imag),
            "isolated_slab_frequency_rad_per_um": float(isolated_frequency),
            "isolated_slab_frequency_offset_rad_per_um": float(
                pole.real - isolated_frequency
            ),
            "isolated_mode_reference_overlap": analytic_overlap,
            "field_parity": {"even_error": even_error, "odd_error": odd_error},
            "transverse_harmonics": harmonics,
            "bilayer_structure_energy_fraction": float(structure_energy),
            "lower_upper_energy_balance": float(balance),
            "radiation": _json_radiation(radiation),
            "reflection_residue": _pair(reflection_residue),
            "boundary_reflection_residue": _pair(boundary_residue),
            "boundary_reflection_residue_magnitude": float(abs(boundary_residue)),
            "boundary_reflection_residue_phase_pi": float(
                np.angle(boundary_residue) / np.pi
            ),
            "reference_offset_um": float(reference_offset),
            "additional_open_port_orders": additional_orders,
            "assignment_score": score,
            "assignment_rejection_reasons": reasons,
            "passes_assignment_gates": not reasons,
        },
        field,
    )
