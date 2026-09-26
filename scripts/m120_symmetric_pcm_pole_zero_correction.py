"""M120: symmetric, shared-state pole/zero adjoint correction for Sb2Se3 HCGs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from scripts.m117_symmetric_pcm_hcg_feasibility import ROOT, _build_scaled_baseline, _complex_index, _load_config
from pzid.constraints import channel_zero_density_constraint, pole_density_constraint
from pzid.hcg import SymmetricPairedTwoTierHCGGeometryParameterization
from pzid.pcm import SymmetricPCMStateHCGParameterization, lorentz_lorenz_permittivity
from pzid.poles import PoleTarget
from pzid.twoport import solve_two_port_scattering


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_fraction(amorphous: complex, crystalline: complex, target_index: float) -> float:
    def polarizability(epsilon: complex) -> complex:
        return (epsilon - 1.0) / (epsilon + 2.0)

    fraction = (
        polarizability(target_index**2) - polarizability(amorphous**2)
    ) / (
        polarizability(crystalline**2) - polarizability(amorphous**2)
    )
    if abs(fraction.imag) > 1e-10 or not 0.0 <= fraction.real <= 1.0:
        raise ValueError("the reference index is not inside the lossless PCM interval")
    return float(fraction.real)


def _build_reference_parameterization(
    m117: dict[str, Any], grid: Any, scale: float
) -> tuple[SymmetricPairedTwoTierHCGGeometryParameterization, dict[str, Any]]:
    m58d, _ = _load_config(ROOT / m117["baseline_config"])
    bounds = m58d["two_tier_geometry"]["bounds_um"]
    scaled = lambda name: tuple(float(value) * scale for value in bounds[name])
    reference = SymmetricPairedTwoTierHCGGeometryParameterization(
        grid=grid,
        port_index=float(m58d["geometry"]["port_index"]),
        grating_index=float(m58d["geometry"]["grating_index"]),
        inner_height_bounds_um=scaled("inner_height"),
        outer_height_bounds_um=scaled("outer_height"),
        cavity_bounds_um=scaled("cavity"),
        inner_width_bounds_um=scaled("inner_width"),
        outer_width_bounds_um=scaled("outer_width"),
    )
    return reference, m58d


def _constraint_system(
    raw: np.ndarray,
    reference: SymmetricPairedTwoTierHCGGeometryParameterization,
    setup: Any,
    target: PoleTarget,
    pole_weight: float,
    zero_fractions: list[float],
    amorphous: complex,
    crystalline: complex,
    omega: float,
) -> tuple[np.ndarray, np.ndarray]:
    pole = pole_density_constraint(
        raw,
        SymmetricPCMStateHCGParameterization(reference, reference.grating_index**2),
        target,
        setup,
    )
    residual = [pole_weight * pole.residual]
    jacobian = [pole_weight * pole.jacobian]
    for fraction in zero_fractions:
        epsilon_material = complex(lorentz_lorenz_permittivity(amorphous, crystalline, fraction))
        state = SymmetricPCMStateHCGParameterization(reference, epsilon_material)
        reflection = channel_zero_density_constraint(raw, state, omega, "reflection", setup)
        residual.append(reflection.residual)
        jacobian.append(reflection.jacobian)
    return np.concatenate(residual), np.vstack(jacobian)


def _finite_difference_audit(
    raw: np.ndarray,
    evaluate: Any,
    step: float = 1e-6,
) -> float:
    _, analytic = evaluate(raw)
    columns = []
    for index in range(raw.size):
        plus = raw.copy()
        minus = raw.copy()
        plus[index] += step
        minus[index] -= step
        columns.append((evaluate(plus)[0] - evaluate(minus)[0]) / (2.0 * step))
    finite = np.column_stack(columns)
    return float(np.linalg.norm(analytic - finite) / max(np.linalg.norm(finite), 1e-12))


def _held_out(
    raw: np.ndarray,
    reference: SymmetricPairedTwoTierHCGGeometryParameterization,
    setup: Any,
    amorphous: complex,
    crystalline: complex,
    omega: float,
    state_points: int,
) -> dict[str, Any]:
    alpha = np.linspace(0.0, 1.0, state_points)
    epsilon_states = lorentz_lorenz_permittivity(amorphous, crystalline, alpha)
    transmission = np.empty(state_points, dtype=np.complex128)
    reflection = np.empty_like(transmission)
    for index, epsilon_material in enumerate(epsilon_states):
        state = SymmetricPCMStateHCGParameterization(reference, complex(epsilon_material))
        matrix = solve_two_port_scattering(omega, state.map(raw).epsilon_r, setup).matrix
        transmission[index] = matrix[1, 0]
        reflection[index] = matrix[0, 0]
    T = np.abs(transmission) ** 2
    R = np.abs(reflection) ** 2
    phase = np.unwrap(np.angle(transmission))
    phase -= phase[0]
    candidate_mask = alpha <= 0.9 + 1e-12
    worst_index = int(np.argmax(R))
    return {
        "alpha": alpha,
        "t": transmission,
        "r": reflection,
        "T": T,
        "R": R,
        "phase_change_pi": phase / math.pi,
        "minimum_transmittance": float(T.min()),
        "maximum_reflectance": float(R.max()),
        "phase_span_pi": float(abs(phase[-1]) / math.pi),
        "worst_reflection_fraction": float(alpha[worst_index]),
        "below_90pct_fraction_minimum_transmittance": float(np.min(T[candidate_mask])),
        "below_90pct_fraction_phase_span_pi": float(abs(phase[candidate_mask][-1]) / math.pi),
        "maximum_energy_error": float(np.max(np.abs(T + R - 1.0))),
    }


def _write_held_out(output: Path, name: str, held: dict[str, Any]) -> None:
    np.savez_compressed(
        output / f"{name}_held_out.npz",
        alpha=held["alpha"], transmission=held["t"], reflection=held["r"],
        transmittance=held["T"], reflectance=held["R"],
        phase_change_pi=held["phase_change_pi"],
    )
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True, constrained_layout=True)
    axes[0].plot(held["alpha"], held["T"], label="T")
    axes[0].plot(held["alpha"], held["R"], label="R")
    axes[1].plot(held["alpha"], held["phase_change_pi"], color="tab:blue")
    axes[0].set(ylabel="Power", ylim=(-0.03, 1.03))
    axes[1].set(xlabel="Crystalline fraction", ylabel="Phase change / π")
    for axis in axes:
        axis.axvline(0.9, color="black", linestyle="--", linewidth=0.8)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.savefig(output / f"{name}_response.png", dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    m117_path = ROOT / config["m117_config"]
    m117 = json.loads(m117_path.read_text(encoding="utf-8"))
    baseline_path = ROOT / m117["baseline_summary"]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    setup, _, grid, scale, geometry = _build_scaled_baseline(
        m117, baseline, cells_per_period=int(config["cells_per_period"])
    )
    reference, m58d = _build_reference_parameterization(m117, grid, scale)
    raw_seed = reference.encode(*(
        value + float(delta) * scale
        for value, delta in zip(geometry, config["geometry_seed_perturbation_um"], strict=True)
    ))
    material = m117["source_material"]
    amorphous = _complex_index(material["amorphous_index"])
    crystalline = _complex_index(material["crystalline_index"])
    alpha_reference = _reference_fraction(
        amorphous, crystalline, float(config["pole_anchor"]["reference_grating_index"])
    )
    pole_specification = m58d["target_root_guesses"]["poles"][
        int(config["pole_anchor"]["target_pole_index_in_m58d_config"])
    ]
    pole_frequency = complex(*pole_specification["frequency"]) / scale
    target = PoleTarget(
        frequency=pole_frequency,
        coupling_ratio=1.0 if pole_specification["parity"] == "odd" else -1.0,
    )
    pole_weight = float(config["pole_anchor"]["weight"])
    omega = 2.0 * math.pi / float(config["target_wavelength_um"])
    first_stage = config["stages"][0]

    def initial_system(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _constraint_system(
            raw, reference, setup, target, pole_weight,
            first_stage["zero_state_fractions"], amorphous, crystalline, omega,
        )

    derivative_error = _finite_difference_audit(raw_seed, initial_system)
    audit = {
        "alpha_at_m58d_index": alpha_reference,
        "pole_target_frequency_rad_per_um": [pole_frequency.real, pole_frequency.imag],
        "pole_target_parity_at_reference_state": pole_specification["parity"],
        "initial_constraint_jacobian_relative_error": derivative_error,
        "initial_raw": raw_seed.tolist(),
        "initial_geometry_um": list(reference.decode(raw_seed)),
        "input_sha256": {
            "config": _hash(config_path),
            "m117_config": _hash(m117_path),
            "m58d_summary": _hash(baseline_path),
        },
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"initial complete Jacobian relative error={derivative_error:.3e}", flush=True)
    if derivative_error > 5e-4:
        raise RuntimeError("complete complex-frequency adjoint Jacobian failed audit")
    if args.audit_only:
        return

    raw = raw_seed.copy()
    span = float(config["maximum_normalized_coordinate_change_from_seed"])
    lower_bounds = np.maximum(0.0, raw_seed - span)
    upper_bounds = np.minimum(1.0, raw_seed + span)
    records = []
    for stage_number, stage in enumerate(config["stages"]):
        name = str(stage["name"])
        anchors = [float(value) for value in stage["zero_state_fractions"]]
        cache_raw: np.ndarray | None = None
        cache_system: tuple[np.ndarray, np.ndarray] | None = None
        history = []

        def evaluate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            nonlocal cache_raw, cache_system
            if cache_raw is None or not np.array_equal(values, cache_raw):
                cache_system = _constraint_system(
                    values, reference, setup, target, pole_weight, anchors,
                    amorphous, crystalline, omega,
                )
                cache_raw = values.copy()
                history.append({
                    "evaluation": len(history),
                    "residual_norm": float(np.linalg.norm(cache_system[0])),
                    "raw": values.tolist(),
                })
            assert cache_system is not None
            return cache_system

        started = time.perf_counter()
        initial_residual = float(np.linalg.norm(evaluate(raw)[0]))
        solution = least_squares(
            lambda values: evaluate(values)[0], raw,
            jac=lambda values: evaluate(values)[1],
            bounds=(lower_bounds, upper_bounds),
            method="trf", tr_solver="exact", max_nfev=int(stage["maximum_evaluations"]),
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        raw = np.asarray(solution.x)
        final_residual = float(np.linalg.norm(evaluate(raw)[0]))
        held = _held_out(
            raw, reference, setup, amorphous, crystalline, omega,
            int(config["held_out_state_points"]),
        )
        _write_held_out(output, name, held)
        np.savetxt(output / f"{name}_raw.csv", raw, delimiter=",")
        (output / f"{name}_evaluations.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        record = {
            "stage": stage_number,
            "name": name,
            "zero_state_fractions": anchors,
            "initial_residual_norm": initial_residual,
            "final_residual_norm": final_residual,
            "evaluations": len(history),
            "solver_success": bool(solution.success),
            "solver_message": str(solution.message),
            "elapsed_seconds": float(time.perf_counter() - started),
            "raw": raw.tolist(),
            "geometry_um": list(reference.decode(raw)),
            "held_out": {key: value for key, value in held.items()
                         if key not in {"alpha", "t", "r", "T", "R", "phase_change_pi"}},
        }
        records.append(record)
        (output / f"{name}_summary.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{name}: residual={final_residual:.4e}, "
            f"Tmin={held['minimum_transmittance']:.6f}, "
            f"phase={held['phase_span_pi']:.6f}pi, "
            f"worst_alpha={held['worst_reflection_fraction']:.3f}",
            flush=True,
        )
    limits = config["response_gate"]
    gate_records = [
        record for record in records
        if record["held_out"]["minimum_transmittance"] >= float(limits["minimum_transmittance"])
        and record["held_out"]["phase_span_pi"] >= float(limits["minimum_phase_span_pi"])
    ]
    summary = {
        "stage": "M120 symmetric PCM multistate pole/zero adjoint correction",
        "audit": audit,
        "records": records,
        "full_state_gate_pass": bool(gate_records),
        "accepted_stage": gate_records[0]["name"] if gate_records else None,
        "claim_limit": config["claim_limit"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"full_state_gate_pass": summary["full_state_gate_pass"],
                      "accepted_stage": summary["accepted_stage"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
