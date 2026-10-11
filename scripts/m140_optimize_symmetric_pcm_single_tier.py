"""M140: true three-coordinate one-tier Sb2Se3 HCG pole/zero adjoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from scripts.m117_symmetric_pcm_hcg_feasibility import ROOT, _build_scaled_baseline, _complex_index, _load_config
from scripts.m120_symmetric_pcm_pole_zero_correction import (
    _constraint_system, _finite_difference_audit, _held_out,
    _reference_fraction, _write_held_out,
)
from scripts.m121_verify_symmetric_pcm_adjoint_mesh import _digest
from pzid.hcg import SymmetricPairedHCGGeometryParameterization
from pzid.poles import PoleTarget


def _one_tier_reference(material_config: dict, grid: object, scale: float) -> SymmetricPairedHCGGeometryParameterization:
    m58d, _ = _load_config(ROOT / material_config["baseline_config"])
    bounds = m58d["two_tier_geometry"]["bounds_um"]
    thickness = (
        float(bounds["inner_height"][0] + bounds["outer_height"][0]) * scale,
        float(bounds["inner_height"][1] + bounds["outer_height"][1]) * scale,
    )
    return SymmetricPairedHCGGeometryParameterization(
        grid=grid,
        port_index=float(m58d["geometry"]["port_index"]),
        grating_index=float(m58d["geometry"]["grating_index"]),
        thickness_bounds_um=thickness,
        cavity_bounds_um=tuple(float(v) * scale for v in bounds["cavity"]),
        width_bounds_um=tuple(float(v) * scale for v in bounds["inner_width"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    material_path = ROOT / config["m117_config"]
    material_config = json.loads(material_path.read_text(encoding="utf-8"))
    baseline_path = ROOT / material_config["baseline_summary"]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    seed_path = ROOT / config["seed_stage_summary"]
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    setup, _, grid, scale, _ = _build_scaled_baseline(
        material_config, baseline, cells_per_period=int(config["cells_per_period"])
    )
    reference = _one_tier_reference(material_config, grid, scale)
    physical_seed = np.asarray(seed["geometry_um"], dtype=float) + np.asarray(config["seed_perturbation_um"], dtype=float) * scale
    raw = reference.encode(*physical_seed)
    m58d, _ = _load_config(ROOT / material_config["baseline_config"])
    pole_spec = m58d["target_root_guesses"]["poles"][
        int(config["pole_anchor"]["target_pole_index_in_m58d_config"])
    ]
    target = PoleTarget(
        complex(*pole_spec["frequency"]) / scale,
        -1.0 if pole_spec["parity"] == "even" else 1.0,
    )
    material = material_config["source_material"]
    amorphous = _complex_index(material["amorphous_index"])
    crystalline = _complex_index(material["crystalline_index"])
    reference_fraction = _reference_fraction(
        amorphous, crystalline,
        float(config["pole_anchor"]["reference_grating_index"]),
    )
    omega = 2.0 * math.pi / float(config["wavelength_um"])
    pole_weight = float(config["pole_anchor"]["weight"])
    first_anchors = [float(v) for v in config["stages"][0]["zero_state_fractions"]]

    def initial_system(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _constraint_system(values, reference, setup, target, pole_weight,
                                  first_anchors, amorphous, crystalline, omega)

    derivative_error = _finite_difference_audit(raw, initial_system)
    if derivative_error > 5e-4:
        raise RuntimeError("single-tier PCM complex-frequency adjoint gradient audit failed")
    print(f"single-tier Jacobian relative error={derivative_error:.3e}", flush=True)
    initial = _held_out(raw, reference, setup, amorphous, crystalline, omega,
                        int(config["held_out_state_points"]))
    _write_held_out(output, "initial", initial)
    initial_record = {
        "geometry_um": list(reference.decode(raw)),
        "minimum_transmittance": initial["minimum_transmittance"],
        "phase_span_pi": initial["phase_span_pi"],
    }
    span = float(config["maximum_normalized_coordinate_change_from_seed"])
    bounds = (np.maximum(0.0, raw - span), np.minimum(1.0, raw + span))
    records = []
    for stage in config["stages"]:
        anchors = [float(v) for v in stage["zero_state_fractions"]]
        cache_values = None
        cache_system = None
        evaluations = []

        def evaluate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            nonlocal cache_values, cache_system
            if cache_values is None or not np.array_equal(cache_values, values):
                cache_system = _constraint_system(
                    values, reference, setup, target, pole_weight,
                    anchors, amorphous, crystalline, omega,
                )
                cache_values = values.copy()
                evaluations.append({
                    "evaluation": len(evaluations),
                    "residual_norm": float(np.linalg.norm(cache_system[0])),
                    "raw": values.tolist(),
                })
            return cache_system

        solution = least_squares(
            lambda values: evaluate(values)[0], raw,
            jac=lambda values: evaluate(values)[1],
            bounds=bounds, method="trf", tr_solver="exact",
            max_nfev=int(stage["maximum_evaluations"]),
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        raw = np.asarray(solution.x)
        held = _held_out(raw, reference, setup, amorphous, crystalline, omega,
                         int(config["held_out_state_points"]))
        name = str(stage["name"])
        _write_held_out(output, name, held)
        record = {
            "stage": name,
            "zero_state_fractions": anchors,
            "geometry_um": list(reference.decode(raw)),
            "raw": raw.tolist(),
            "constraint_residual_norm": float(np.linalg.norm(evaluate(raw)[0])),
            "solver_success": bool(solution.success),
            "solver_message": str(solution.message),
            "evaluations": evaluations,
            "held_out": {key: value for key, value in held.items()
                         if key not in {"alpha", "t", "r", "T", "R", "phase_change_pi"}},
            "gate_pass": bool(
                held["minimum_transmittance"] >= float(config["minimum_transmittance_gate"])
                and held["phase_span_pi"] >= float(config["minimum_phase_span_pi_gate"])
            ),
        }
        records.append(record)
        (output / f"{name}_summary.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{name}: Tmin={held['minimum_transmittance']:.6f}, "
              f"phase={held['phase_span_pi']:.6f}pi, "
              f"residual={record['constraint_residual_norm']:.4e}", flush=True)
    summary = {
        "milestone": "M140",
        "input_sha256": {str(path): _digest(path) for path in
                         (config_path, material_path, baseline_path, seed_path, Path(__file__))},
        "grid_shape": list(grid.shape),
        "period_um": float(grid.period),
        "alpha_at_reference_index": reference_fraction,
        "pole_target_rad_per_um": [target.frequency.real, target.frequency.imag],
        "jacobian_relative_error": derivative_error,
        "initial": initial_record,
        "stages": records,
        "claim_limit": config["claim_limit"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
