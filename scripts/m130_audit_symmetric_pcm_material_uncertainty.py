"""M130: fixed-wavelength PCM mixing-law and absorption sensitivity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.m117_symmetric_pcm_hcg_feasibility import ROOT, _build_scaled_baseline, _complex_index
from scripts.m120_symmetric_pcm_pole_zero_correction import _build_reference_parameterization
from scripts.m121_verify_symmetric_pcm_adjoint_mesh import _digest
from pzid.pcm import SymmetricPCMStateHCGParameterization, lorentz_lorenz_permittivity
from pzid.twoport import solve_two_port_scattering


def _permittivity(alpha: np.ndarray, amorphous: complex, crystalline: complex,
                  model: str) -> np.ndarray:
    if model == "Lorentz-Lorenz":
        return np.asarray(lorentz_lorenz_permittivity(amorphous, crystalline, alpha))
    if model == "linear permittivity":
        return (1.0 - alpha) * amorphous**2 + alpha * crystalline**2
    if model == "linear index":
        return ((1.0 - alpha) * amorphous + alpha * crystalline) ** 2
    raise ValueError(f"unknown PCM mixing model: {model}")


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
    stage_path = ROOT / config["stage_summary"]
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    cells = int(config["cells_per_period"])
    setup, _, grid, scale, _ = _build_scaled_baseline(
        material_config, baseline, cells_per_period=cells
    )
    reference, _ = _build_reference_parameterization(material_config, grid, scale)
    raw = reference.encode(*[float(value) for value in stage["geometry_um"]])
    material = material_config["source_material"]
    n_amorphous = _complex_index(material["amorphous_index"])
    n_crystalline = _complex_index(material["crystalline_index"])
    alpha = np.linspace(0.0, 1.0, int(config["crystalline_fraction_points"]))
    omega = 2.0 * math.pi / float(config["wavelength_um"])
    cases = [(str(rule), n_amorphous, n_crystalline, str(rule))
             for rule in config["mixing_rules"]]
    for k in config["loss_stress_k_both_endpoints"]:
        k = float(k)
        cases.append((f"Lorentz-Lorenz k={k:g}",
                      complex(n_amorphous.real, k),
                      complex(n_crystalline.real, k), "Lorentz-Lorenz"))
    records = []
    traces = {}
    for label, amorphous, crystalline, rule in cases:
        epsilon_states = _permittivity(alpha, amorphous, crystalline, rule)
        transmission = np.empty(alpha.size, dtype=np.complex128)
        reflection = np.empty_like(transmission)
        for index, epsilon in enumerate(epsilon_states):
            mapped = SymmetricPCMStateHCGParameterization(reference, complex(epsilon)).map(raw)
            matrix = solve_two_port_scattering(omega, mapped.epsilon_r, setup).matrix
            transmission[index] = matrix[1, 0]
            reflection[index] = matrix[0, 0]
        T = np.abs(transmission) ** 2
        R = np.abs(reflection) ** 2
        A = 1.0 - T - R
        phase = np.unwrap(np.angle(transmission))
        phase -= phase[0]
        records.append({
            "case": label,
            "mixing_rule": rule,
            "amorphous_index": [amorphous.real, amorphous.imag],
            "crystalline_index": [crystalline.real, crystalline.imag],
            "minimum_transmittance": float(T.min()),
            "maximum_reflectance": float(R.max()),
            "maximum_absorptance": float(A.max()),
            "minimum_absorptance": float(A.min()),
            "phase_span_pi": float(abs(phase[-1]) / math.pi),
            "gate_pass": bool(
                T.min() >= float(config["minimum_transmittance_gate"])
                and abs(phase[-1]) / math.pi >= float(config["minimum_phase_span_pi_gate"])
            ),
        })
        traces[label] = (T, R, A, phase / math.pi)
        print(f"{label}: Tmin={T.min():.6f}, phase={abs(phase[-1])/math.pi:.6f}pi, "
              f"Amax={A.max():.6f}", flush=True)
    np.savez_compressed(
        output / "material_uncertainty.npz", alpha=alpha,
        **{f"case_{index}_{quantity}": trace[position]
           for index, (_, trace) in enumerate(traces.items())
           for position, quantity in enumerate(("T", "R", "A", "phase_pi"))},
    )
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.0), sharex=True,
                             constrained_layout=True)
    for label, (T, R, A, phase) in traces.items():
        axes[0].plot(alpha, T, label=label)
        axes[1].plot(alpha, phase, label=label)
    axes[0].set(ylabel="Transmittance", ylim=(-0.03, 1.03))
    axes[1].set(xlabel="Crystalline fraction", ylabel="Phase change / π")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(output / "m130_material_uncertainty.png", dpi=220)
    plt.close(fig)
    summary = {
        "milestone": "M130",
        "input_sha256": {str(path): _digest(path) for path in
                         (config_path, material_path, baseline_path, stage_path, Path(__file__))},
        "cells_per_period": cells,
        "grid_shape": list(grid.shape),
        "wavelength_um": float(config["wavelength_um"]),
        "geometry_um": stage["geometry_um"],
        "cases": records,
        "claim_limit": config["claim_limit"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
