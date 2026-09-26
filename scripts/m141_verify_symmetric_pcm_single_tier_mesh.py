"""M141: independent mesh replay of a one-tier PCM pole/zero adjoint design."""

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
from scripts.m121_verify_symmetric_pcm_adjoint_mesh import _digest, _metric
from scripts.m140_optimize_symmetric_pcm_single_tier import _one_tier_reference
from pzid.pcm import SymmetricPCMStateHCGParameterization, lorentz_lorenz_permittivity
from pzid.twoport import solve_two_port_scattering


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
    stage_path = ROOT / config["m140_stage_summary"]
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    geometry = [float(value) for value in stage["geometry_um"]]
    alpha = np.linspace(0.0, 1.0, int(config["crystalline_fraction_points"]))
    material = material_config["source_material"]
    epsilon_states = lorentz_lorenz_permittivity(
        _complex_index(material["amorphous_index"]),
        _complex_index(material["crystalline_index"]), alpha,
    )
    omega = 2.0 * math.pi / float(config["wavelength_um"])
    records = []
    traces = {}
    for cells in config["mesh_cells_per_period"]:
        cells = int(cells)
        setup, _, grid, scale, _ = _build_scaled_baseline(
            material_config, baseline, cells_per_period=cells
        )
        reference = _one_tier_reference(material_config, grid, scale)
        raw = reference.encode(*geometry)
        t = np.empty(alpha.size, dtype=np.complex128)
        r = np.empty_like(t)
        for index, epsilon in enumerate(epsilon_states):
            mapped = SymmetricPCMStateHCGParameterization(reference, complex(epsilon)).map(raw)
            matrix = solve_two_port_scattering(omega, mapped.epsilon_r, setup).matrix
            t[index] = matrix[1, 0]
            r[index] = matrix[0, 0]
        metric = _metric(alpha, t, r, 1.0)
        records.append({
            "cells_per_period": cells,
            "grid_shape": list(grid.shape),
            "period_um": float(grid.period),
            "geometry_um": geometry,
            "full_state": metric,
            "gate_pass": bool(
                metric["minimum_transmittance"] >= float(config["minimum_transmittance_gate"])
                and metric["phase_span_pi"] >= float(config["minimum_phase_span_pi_gate"])
            ),
        })
        traces[f"N{cells}"] = (t, r)
        print(f"N{cells}: Tmin={metric['minimum_transmittance']:.6f}, "
              f"phase={metric['phase_span_pi']:.6f}pi", flush=True)
    np.savez_compressed(
        output / "single_tier_mesh.npz", alpha=alpha,
        **{f"transmission_{name}": pair[0] for name, pair in traces.items()},
        **{f"reflection_{name}": pair[1] for name, pair in traces.items()},
    )
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                             constrained_layout=True)
    for name, (t, r) in traces.items():
        phase = np.unwrap(np.angle(t))
        axes[0].plot(alpha, abs(t) ** 2, label=name)
        axes[1].plot(alpha, (phase - phase[0]) / math.pi)
    axes[0].set(ylabel="Transmittance", ylim=(0.9, 1.005))
    axes[1].set(xlabel="Crystalline fraction", ylabel="Phase change / π")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(output / "m141_one_tier_mesh.png", dpi=220)
    plt.close(fig)
    summary = {
        "milestone": "M141",
        "input_sha256": {str(path): _digest(path) for path in
                         (config_path, material_path, baseline_path, stage_path, Path(__file__))},
        "records": records,
        "all_mesh_gates_pass": all(x["gate_pass"] for x in records),
        "claim_limit": config["claim_limit"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
