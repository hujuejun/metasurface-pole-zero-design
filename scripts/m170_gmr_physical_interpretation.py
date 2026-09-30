"""Plot transmission-channel singularities and a GMR standing-wave field."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from pzid.twoport import solve_two_port_scattering

from scripts.m164_gmr_pole_adjoint import build_setup, physical_epsilon_and_derivatives


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m170_gmr_physical_interpretation.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/m170_gmr_physical_interpretation"))
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    high_path = Path(protocol["source_high_fidelity_summary"])
    fdfd_path = Path(protocol["source_fdfd_corrected_summary"])
    base_path = Path(protocol["base_adjoint_config"])
    roots = json.loads(high_path.read_text(encoding="utf-8"))["cases"]
    fdfd = json.loads(fdfd_path.read_text(encoding="utf-8"))["cases"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(("git", "rev-parse", "HEAD"), check=False, capture_output=True, text=True)
    (output / "metadata.json").write_text(json.dumps({
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__, "platform": platform.platform(),
        "command": sys.argv,
        "input_sha256": {str(p): _sha256(p) for p in (args.config, high_path, fdfd_path, base_path)},
        "script_sha256": _sha256(Path(__file__)),
    }, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(roots)))
    for row, color in zip(roots, colors, strict=True):
        q = row["achieved_quality_factor"]
        pole = complex(**row["channel_pole"])
        zero = complex(**row["channel_zero"])
        axes[0].plot(pole.real, pole.imag, "x", color=color, ms=8)
        axes[0].plot(zero.real, zero.imag, "o", color=color, mfc="none", ms=7)
        axes[0].annotate(f"Q≈{q:.0f}", (pole.real, pole.imag), (4, -4),
                         textcoords="offset points", fontsize=7, color=color)
        axes[1].plot((zero.real - pole.real) / abs(pole.imag),
                     zero.imag / abs(pole.imag), "o", color=color,
                     label=f"Q≈{q:.0f}")
    axes[0].axhline(0, color="0.4", lw=0.8)
    axes[0].set_xlabel("Re ω (rad/µm)")
    axes[0].set_ylabel("Im ω (rad/µm)")
    axes[0].set_title("Transmission-channel poles × and zeros ○")
    axes[1].plot(0, -1, "kx", ms=8, label="Normalized pole")
    axes[1].axhline(0, color="0.4", lw=0.8)
    axes[1].set_xlabel("(Re zero − Re pole) / pole depth")
    axes[1].set_ylabel("Im frequency / pole depth")
    axes[1].set_title("How the real zero approaches the pole")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(output / "gmr_complex_transmission_channel_roots.png", dpi=220)
    fig.savefig(output / "gmr_complex_transmission_channel_roots.pdf")
    plt.close(fig)

    quality = int(protocol["nearfield_quality_factor"])
    source = next(x for x in fdfd if x["quality_factor"] == quality)
    config = dict(base)
    config["period_um"] = float(source["period_um"])
    setup = build_setup(config)
    coordinates = np.asarray(source["final_coordinates_um"])
    epsilon, _ = physical_epsilon_and_derivatives(coordinates, config, setup)
    pole = complex(**source["final_fdfd"]["verified_pole"])
    solution = solve_two_port_scattering(pole.real, epsilon, setup)
    field = solution.fields[:, 0].reshape(setup.grid.shape)
    reference = solution.system.reference_fields[:, 0].reshape(setup.grid.shape)
    scattered = field - reference
    harmonic = np.fft.fft(scattered, axis=1) / setup.grid.nx
    m1 = harmonic[:, 1]
    slab_lower = float(config["base_z_um"])
    slab_upper = slab_lower + coordinates.sum()
    inside = (setup.grid.z >= slab_lower) & (setup.grid.z <= slab_upper)
    outside = ((setup.grid.z >= slab_lower - 0.2) & (setup.grid.z < slab_lower - 0.05))
    intensity_ratio = float(np.mean(abs(m1[inside])**2) / max(np.mean(abs(m1[outside])**2), 1e-16))
    np.savez_compressed(output / "gmr_q200_nearfield.npz", epsilon=epsilon,
                        total_field=field, reference_field=reference,
                        scattered_field=scattered, first_lateral_harmonic=m1,
                        x_um=setup.grid.x, z_um=setup.grid.z)
    (output / "summary.json").write_text(json.dumps({
        "quality_target": quality,
        "fdfd_transmission_channel_pole": source["final_fdfd"]["verified_pole"],
        "first_lateral_harmonic_slab_to_outside_intensity_ratio": intensity_ratio,
        "interpretation": "Standing periodic guided field in the continuous slab; the ridge supplies radiative coupling. This supports GMR lineage but is not a full eigenmode-overlap decomposition.",
    }, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    shown = np.log10(abs(scattered)**2 / max(np.max(abs(scattered)**2), 1e-16) + 1e-6)
    axes[0].imshow(shown, origin="lower", aspect="auto", extent=(0, setup.grid.period, 0, setup.grid.height),
                   vmin=-6, vmax=0, cmap="magma")
    axes[0].contour(setup.grid.x, setup.grid.z, epsilon, levels=[1.5], colors="cyan", linewidths=0.7)
    axes[0].set_xlabel("x (µm)")
    axes[0].set_ylabel("z (µm)")
    axes[0].set_title("Log resonant scattered |Hy|²")
    axes[1].plot(setup.grid.z, abs(m1)**2 / max(np.max(abs(m1)**2), 1e-16))
    axes[1].axvspan(slab_lower, slab_upper, alpha=0.15, color="orange", label="Slab + ridge")
    axes[1].set_xlabel("z (µm)")
    axes[1].set_ylabel("Normalized |lateral harmonic m=1|²")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.savefig(output / "gmr_q200_resonant_nearfield.png", dpi=220)
    fig.savefig(output / "gmr_q200_resonant_nearfield.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
