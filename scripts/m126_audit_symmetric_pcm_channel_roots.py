"""M126: verify actual transmission poles and reflection zeros of PCM HCGs.

Pole/zero constraints used in the optimization are not treated as proof of
actual scalar-channel roots.  The physical structure is reconstructed from
five coordinates on each mesh and the complex channel is solved afresh.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.m117_symmetric_pcm_hcg_feasibility import ROOT, _build_scaled_baseline, _complex_index, _load_config
from scripts.m120_symmetric_pcm_pole_zero_correction import _build_reference_parameterization, _reference_fraction
from scripts.m121_verify_symmetric_pcm_adjoint_mesh import _digest
from scripts.m4b_optimize_kerker_bar import _parity_response
from pzid.constraints import pole_density_constraint
from pzid.pcm import SymmetricPCMStateHCGParameterization, lorentz_lorenz_permittivity
from pzid.poles import PoleTarget
from pzid.roots import refine_scalar_root_complex_secant
from pzid.twoport import solve_two_port_scattering


def _pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def _record(root: object) -> dict[str, object]:
    return {
        "frequency_rad_per_um": _pair(root.frequency),
        "success": bool(root.success),
        "residual_norm": float(root.residual_norm),
        "evaluations": int(root.evaluations),
        "message": str(root.message),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    optimization_path = ROOT / config["m123_optimization_config"]
    optimization = json.loads(optimization_path.read_text(encoding="utf-8"))
    material_path = ROOT / optimization["m117_config"]
    material_config = json.loads(material_path.read_text(encoding="utf-8"))
    baseline_path = ROOT / material_config["baseline_summary"]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    m58d, _ = _load_config(ROOT / material_config["baseline_config"])
    target_spec = m58d["target_root_guesses"]["poles"][
        int(optimization["pole_anchor"]["target_pole_index_in_m58d_config"])
    ]
    amorphous = _complex_index(material_config["source_material"]["amorphous_index"])
    crystalline = _complex_index(material_config["source_material"]["crystalline_index"])
    alpha_reference = _reference_fraction(
        amorphous, crystalline,
        float(optimization["pole_anchor"]["reference_grating_index"]),
    )
    omega_real = 2.0 * math.pi / float(optimization["target_wavelength_um"])
    radius = float(config["root_search_radius_rad_per_um"])
    stage_paths = [ROOT / value for value in config["stage_summaries"]]
    records = []
    for cells in config["cells_per_period"]:
        cells = int(cells)
        setup, _, grid, scale, _ = _build_scaled_baseline(
            material_config, baseline, cells_per_period=cells
        )
        reference, _ = _build_reference_parameterization(material_config, grid, scale)
        pole_target_frequency = complex(*target_spec["frequency"]) / scale
        coupling = -1.0 if target_spec["parity"] == "even" else 1.0
        target = PoleTarget(pole_target_frequency, coupling)
        for stage_path in stage_paths:
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            geometry = [float(value) for value in stage["geometry_um"]]
            raw = reference.encode(*geometry)
            epsilon_reference = complex(lorentz_lorenz_permittivity(
                amorphous, crystalline, alpha_reference
            ))
            state_reference = SymmetricPCMStateHCGParameterization(reference, epsilon_reference)
            mapped = state_reference.map(raw)
            time_reversal = pole_density_constraint(raw, state_reference, target, setup)

            def transmission(frequency: complex) -> complex:
                return complex(solve_two_port_scattering(
                    frequency, mapped.epsilon_r, setup
                ).matrix[1, 0])

            parity_root = refine_scalar_root_complex_secant(
                lambda frequency: _parity_response(
                    frequency, mapped.epsilon_r, setup, target_spec["parity"]
                ),
                pole_target_frequency - 0.001,
                pole_target_frequency + 0.001,
                kind="pole", tolerance=1e-9, max_evaluations=35,
                search_radius=radius,
            )
            pole_root = refine_scalar_root_complex_secant(
                transmission,
                parity_root.frequency - 0.0001,
                parity_root.frequency + 0.0001,
                kind="pole", tolerance=1e-9, max_evaluations=35,
                search_radius=min(radius, 0.03),
            )
            sample_offset = 0.0001
            pole_sample = solve_two_port_scattering(
                pole_root.frequency + sample_offset, mapped.epsilon_r, setup
            ).matrix
            residue = complex(sample_offset * pole_sample[1, 0])
            reflection_residue = complex(sample_offset * pole_sample[0, 0])
            residue_ratio = reflection_residue / residue
            inferred_parity = (
                "even" if abs(residue_ratio + 1.0) < abs(residue_ratio - 1.0)
                else "odd"
            )
            zero_records = []
            for fraction in config["reflection_zero_state_fractions"]:
                fraction = float(fraction)
                epsilon = complex(lorentz_lorenz_permittivity(
                    amorphous, crystalline, fraction
                ))
                mapped_state = SymmetricPCMStateHCGParameterization(reference, epsilon).map(raw)

                def reflection(frequency: complex) -> complex:
                    return complex(solve_two_port_scattering(
                        frequency, mapped_state.epsilon_r, setup
                    ).matrix[0, 0])

                real_axis_amplitude = reflection(complex(omega_real))
                zero = refine_scalar_root_complex_secant(
                    reflection, complex(omega_real - 0.001),
                    complex(omega_real + 0.001), kind="zero",
                    tolerance=1e-9, max_evaluations=35, search_radius=radius,
                )
                zero_records.append({
                    "crystalline_fraction": fraction,
                    "real_axis_reflection_amplitude": _pair(real_axis_amplitude),
                    "real_axis_reflectance": float(abs(real_axis_amplitude) ** 2),
                    "reflection_channel_zero": _record(zero),
                })
            record = {
                "stage": stage["name"],
                "cells_per_period": cells,
                "grid_shape": list(grid.shape),
                "geometry_um": geometry,
                "pole_reference_crystalline_fraction": alpha_reference,
                "prescribed_pole_frequency_rad_per_um": _pair(pole_target_frequency),
                "chen_time_reversal_residual_norm": float(np.linalg.norm(time_reversal.residual)),
                "prescribed_parity_channel_pole": _record(parity_root),
                "transmission_channel_pole": _record(pole_root),
                "transmission_pole_residue_estimate": _pair(residue),
                "reflection_to_transmission_pole_residue_ratio": _pair(residue_ratio),
                "pole_parity_inferred_from_channel_residues": inferred_parity,
                "reflection_zeros": zero_records,
            }
            records.append(record)
            print(
                f"{stage['name']} N{cells}: transmission pole "
                f"{pole_root.frequency.real:.6f}{pole_root.frequency.imag:+.6f}i, "
                f"root residual={pole_root.residual_norm:.2e}, "
                f"Chen residual={np.linalg.norm(time_reversal.residual):.2e}",
                flush=True,
            )
    summary = {
        "milestone": "M126",
        "input_sha256": {str(path): _digest(path) for path in
                         (config_path, optimization_path, material_path, baseline_path,
                          *stage_paths, Path(__file__))},
        "claim_limit": config["claim_limit"],
        "records": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
