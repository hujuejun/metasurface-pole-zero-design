"""M121: independent mesh validation of an adjoint-designed symmetric PCM HCG.

The five physical interfaces are decoded anew on each grid.  No permittivity
array is interpolated from the N24 optimization grid.  This intentionally
separates optimization from held-out state and mesh validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.m117_symmetric_pcm_hcg_feasibility import ROOT, _build_scaled_baseline, _complex_index
from scripts.m120_symmetric_pcm_pole_zero_correction import _build_reference_parameterization
from pzid.pcm import SymmetricPCMStateHCGParameterization, lorentz_lorenz_permittivity
from pzid.twoport import solve_two_port_scattering


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(alpha: np.ndarray, transmission: np.ndarray, reflection: np.ndarray,
            maximum_fraction: float) -> dict[str, float]:
    chosen = alpha <= maximum_fraction + 1e-12
    t = transmission[chosen]
    r = reflection[chosen]
    phase = np.unwrap(np.angle(t))
    return {
        "maximum_fraction": maximum_fraction,
        "minimum_transmittance": float(np.min(np.abs(t) ** 2)),
        "maximum_reflectance": float(np.max(np.abs(r) ** 2)),
        "phase_span_pi": float(abs(phase[-1] - phase[0]) / math.pi),
        "maximum_energy_balance_error": float(np.max(np.abs(np.abs(t) ** 2 + np.abs(r) ** 2 - 1.0))),
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
    m117_path = ROOT / config["m117_config"]
    m117 = json.loads(m117_path.read_text(encoding="utf-8"))
    baseline_path = ROOT / m117["baseline_summary"]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    stage_path = ROOT / config["m120_stage_summary"]
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    adjoint_geometry = [float(value) for value in stage["geometry_um"]]
    baseline_label = str(config.get("baseline_design_label", "M58D frozen"))
    adjoint_label = str(config.get("adjoint_design_label", "M120 pole/zero adjoint"))
    wavelength = float(config["wavelength_um"])
    omega = 2.0 * math.pi / wavelength
    alpha = np.linspace(0.0, 1.0, int(config["crystalline_fraction_points"]))
    material = m117["source_material"]
    epsilon_states = lorentz_lorenz_permittivity(
        _complex_index(material["amorphous_index"]),
        _complex_index(material["crystalline_index"]), alpha,
    )
    candidate_limit = float(config["candidate_maximum_crystalline_fraction"])
    traces: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    records = []
    for cells in config["mesh_cells_per_period"]:
        cells = int(cells)
        setup, _, grid, scale, baseline_geometry = _build_scaled_baseline(
            m117, baseline, cells_per_period=cells
        )
        reference, _ = _build_reference_parameterization(m117, grid, scale)
        for label, geometry in ((baseline_label, baseline_geometry),
                                (adjoint_label, adjoint_geometry)):
            raw = reference.encode(*geometry)
            t = np.empty(alpha.size, dtype=np.complex128)
            r = np.empty_like(t)
            for index, epsilon in enumerate(epsilon_states):
                state = SymmetricPCMStateHCGParameterization(reference, complex(epsilon))
                matrix = solve_two_port_scattering(omega, state.map(raw).epsilon_r, setup).matrix
                t[index] = matrix[1, 0]
                r[index] = matrix[0, 0]
            traces[(label, cells)] = {"t": t, "r": r}
            candidate = _metric(alpha, t, r, candidate_limit)
            full = _metric(alpha, t, r, 1.0)
            record = {
                "design": label,
                "cells_per_period": cells,
                "grid_shape": list(grid.shape),
                "period_um": float(grid.period),
                "geometry_um": geometry,
                "candidate_window": candidate,
                "full_material_range": full,
                "candidate_gate_pass": bool(
                    candidate["minimum_transmittance"] >= float(config["minimum_transmittance_gate"])
                    and candidate["phase_span_pi"] >= float(config["minimum_phase_span_pi_gate"])
                ),
            }
            records.append(record)
            print(
                f"{label}, N{cells}: alpha<= {candidate_limit:.2f} "
                f"Tmin={candidate['minimum_transmittance']:.6f} "
                f"phase={candidate['phase_span_pi']:.6f}pi; "
                f"full Tmin={full['minimum_transmittance']:.6f}", flush=True,
            )

    np.savez_compressed(
        output / "fixed_wavelength_mesh.npz", alpha=alpha,
        **{
            f"{label.replace(' ', '_').replace('/', '_')}_N{cells}_{channel}": data[channel]
            for (label, cells), data in traces.items()
            for channel in ("t", "r")
        },
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.0), sharex=True, constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(config["mesh_cells_per_period"])))
    for index, cells in enumerate(config["mesh_cells_per_period"]):
        cells = int(cells)
        for column, label in enumerate((baseline_label, adjoint_label)):
            trace = traces[(label, cells)]
            T = np.abs(trace["t"]) ** 2
            phase = np.unwrap(np.angle(trace["t"]))
            axes[0, column].plot(alpha, T, color=colors[index], label=f"N{cells}")
            axes[1, column].plot(alpha, (phase - phase[0]) / math.pi, color=colors[index])
    for column, label in enumerate((baseline_label, adjoint_label)):
        axes[0, column].set_title(label)
        axes[0, column].set_ylim(-0.03, 1.03)
        axes[1, column].set_xlabel("Crystalline fraction, shared by both HCGs")
        axes[0, column].axvline(candidate_limit, color="black", linestyle="--", linewidth=0.9)
        axes[1, column].axvline(candidate_limit, color="black", linestyle="--", linewidth=0.9)
        for row in range(2):
            axes[row, column].grid(alpha=0.25)
    axes[0, 0].set_ylabel("Transmittance")
    axes[1, 0].set_ylabel("Transmission phase change / π")
    axes[0, 1].legend(frameon=False)
    figure.savefig(output / "m121_state_response_mesh.png", dpi=220)
    plt.close(figure)
    summary = {
        "milestone": "M121",
        "input_sha256": {str(path): _digest(path) for path in
                         (config_path, m117_path, baseline_path, stage_path, Path(__file__))},
        "wavelength_um": wavelength,
        "claim_limit": config["claim_limit"],
        "records": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
