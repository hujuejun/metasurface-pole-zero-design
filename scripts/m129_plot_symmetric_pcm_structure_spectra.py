"""M129: geometry, R/T, and phase figures for a symmetric PCM adjoint design."""

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
    geometry = [float(value) for value in stage["geometry_um"]]
    raw = reference.encode(*geometry)
    material = material_config["source_material"]
    amorphous = _complex_index(material["amorphous_index"])
    crystalline = _complex_index(material["crystalline_index"])
    alpha = np.linspace(0.0, 1.0, int(config["state_points"]))
    wavelength = np.linspace(
        float(config["spectrum_wavelength_min_um"]),
        float(config["spectrum_wavelength_max_um"]),
        int(config["spectrum_points"]),
    )
    lambda0 = float(config["fixed_wavelength_um"])
    omega0 = 2.0 * math.pi / lambda0
    state_t = np.empty(alpha.size, dtype=np.complex128)
    state_r = np.empty_like(state_t)
    for index, fraction in enumerate(alpha):
        epsilon = complex(lorentz_lorenz_permittivity(amorphous, crystalline, fraction))
        mapped = SymmetricPCMStateHCGParameterization(reference, epsilon).map(raw)
        matrix = solve_two_port_scattering(omega0, mapped.epsilon_r, setup).matrix
        state_t[index] = matrix[1, 0]
        state_r[index] = matrix[0, 0]
    state_phase = np.unwrap(np.angle(state_t))
    state_phase -= state_phase[0]
    selected = [float(value) for value in config["selected_crystalline_fractions"]]
    spectrum_t = np.empty((len(selected), wavelength.size), dtype=np.complex128)
    spectrum_r = np.empty_like(spectrum_t)
    for state_index, fraction in enumerate(selected):
        epsilon = complex(lorentz_lorenz_permittivity(amorphous, crystalline, fraction))
        mapped = SymmetricPCMStateHCGParameterization(reference, epsilon).map(raw)
        for frequency_index, value in enumerate(wavelength):
            matrix = solve_two_port_scattering(
                2.0 * math.pi / float(value), mapped.epsilon_r, setup
            ).matrix
            spectrum_t[state_index, frequency_index] = matrix[1, 0]
            spectrum_r[state_index, frequency_index] = matrix[0, 0]
        print(f"spectrum {state_index + 1}/{len(selected)} at alpha={fraction:.2f}", flush=True)
    np.savez_compressed(
        output / "response.npz", alpha=alpha, state_transmission=state_t,
        state_reflection=state_r, state_phase_change_rad=state_phase,
        wavelength_um=wavelength, spectrum_alpha=np.asarray(selected),
        spectrum_transmission=spectrum_t, spectrum_reflection=spectrum_r,
    )

    profiles = [float(value) for value in config["index_profile_fractions"]]
    profile_indices = []
    for fraction in profiles:
        epsilon = complex(lorentz_lorenz_permittivity(amorphous, crystalline, fraction))
        mapped = SymmetricPCMStateHCGParameterization(reference, epsilon).map(raw)
        profile_indices.append(np.real(np.sqrt(mapped.epsilon_r)))
    np.savez_compressed(
        output / "index_profiles.npz", x_um=grid.x, z_um=grid.z,
        fractions=np.asarray(profiles), refractive_index=np.asarray(profile_indices),
    )
    occupied = np.flatnonzero(np.any(profile_indices[0] > 1.01, axis=1))
    lower = max(0, int(occupied.min()) - 4)
    upper = min(grid.nz, int(occupied.max()) + 5)
    figure, axes = plt.subplots(1, len(profiles), figsize=(11.4, 5.2),
                                sharey=True, constrained_layout=True)
    for axis, fraction, profile in zip(axes, profiles, profile_indices, strict=True):
        image = axis.imshow(
            profile[lower:upper], origin="lower", aspect="equal", interpolation="nearest",
            extent=(0.0, grid.period, lower * grid.dz, upper * grid.dz),
            vmin=1.0, vmax=4.05, cmap="viridis",
        )
        axis.set(title=f"Crystalline fraction {fraction:.1f}", xlabel="x (µm)")
    axes[0].set_ylabel("z (µm)")
    figure.colorbar(image, ax=axes, label="Subcell-average refractive index", shrink=0.8)
    figure.savefig(output / "m129_index_profiles.png", dpi=240)
    plt.close(figure)

    colors = plt.cm.plasma(np.linspace(0.05, 0.92, len(selected)))
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True,
                                constrained_layout=True)
    for color, fraction, t, r in zip(colors, selected, spectrum_t, spectrum_r, strict=True):
        axes[0].plot(wavelength, np.abs(t) ** 2, color=color,
                     label=f"T, α={fraction:.1f}")
        axes[0].plot(wavelength, np.abs(r) ** 2, color=color, linestyle="--", alpha=0.65)
        phase = np.unwrap(np.angle(t) - np.angle(spectrum_t[0]))
        state_index = int(np.argmin(abs(alpha - fraction)))
        center_index = int(np.argmin(abs(wavelength - lambda0)))
        phase += 2.0 * math.pi * round(
            (state_phase[state_index] - phase[center_index]) / (2.0 * math.pi)
        )
        axes[1].plot(wavelength, phase / math.pi, color=color)
    axes[0].set(ylabel="T (solid), R (dashed)", ylim=(-0.03, 1.03))
    axes[1].set(xlabel="Wavelength (µm)", ylabel="Phase shift vs amorphous / π")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    for axis in axes:
        axis.axvline(lambda0, color="black", linestyle=":", linewidth=1)
        axis.grid(alpha=0.25)
    figure.savefig(output / "m129_rt_phase_spectra.png", dpi=240)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    axes[0].plot(state_t.real, state_t.imag, color="tab:purple")
    for fraction in selected:
        index = int(np.argmin(abs(alpha - fraction)))
        axes[0].plot(state_t[index].real, state_t[index].imag, "o", color="tab:purple")
        axes[0].annotate(f"{fraction:.1f}",
                         (state_t[index].real, state_t[index].imag), fontsize=8)
    axes[0].set(xlabel="Re(t)", ylabel="Im(t)", title="Complex transmission")
    axes[0].axis("equal")
    axes[1].plot(alpha, np.abs(state_t) ** 2, label="T")
    axes[1].plot(alpha, np.abs(state_r) ** 2, label="R")
    axes[1].plot(alpha, state_phase / math.pi, label="Phase / π")
    axes[1].set(xlabel="Crystalline fraction", title="At 1.55 µm")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(output / "m129_complex_t_and_state_response.png", dpi=240)
    plt.close(figure)
    summary = {
        "milestone": "M129",
        "input_sha256": {str(path): _digest(path) for path in
                         (config_path, material_path, baseline_path, stage_path, Path(__file__))},
        "cells_per_period": cells,
        "grid_shape": list(grid.shape),
        "period_um": float(grid.period),
        "geometry_um": geometry,
        "fixed_wavelength_um": lambda0,
        "minimum_transmittance": float(np.min(np.abs(state_t) ** 2)),
        "phase_span_pi": float(abs(state_phase[-1]) / math.pi),
        "claim_limit": config["claim_limit"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
