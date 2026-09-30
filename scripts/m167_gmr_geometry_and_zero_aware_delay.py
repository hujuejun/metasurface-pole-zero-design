"""Plot exact binary GMR structures and audit phase-step proximity to zeros."""

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

from scripts.m164_gmr_pole_adjoint import FS_PER_UM_OVER_C
from scripts.m166_gmr_high_fidelity_analysis import _solver


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m167_gmr_geometry_and_zero_aware_delay.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/m167_gmr_geometry_and_zero_aware_delay"))
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    summary_path = Path(protocol["source_high_fidelity_summary"])
    config_path = Path(protocol["source_high_fidelity_config"])
    rows = json.loads(summary_path.read_text(encoding="utf-8"))["cases"]
    physics = json.loads(config_path.read_text(encoding="utf-8"))
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(("git", "rev-parse", "HEAD"), check=False, capture_output=True, text=True)
    (output / "metadata.json").write_text(json.dumps({
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__, "platform": platform.platform(),
        "command": sys.argv,
        "input_sha256": {str(p): _sha256(p) for p in (args.config, summary_path, config_path)},
        "script_sha256": _sha256(Path(__file__)),
    }, indent=2) + "\n", encoding="utf-8")
    updated = []
    for row in rows:
        pole = complex(**row["channel_pole"])
        zero = complex(**row["channel_zero"])
        gamma = abs(pole.imag)
        separation = abs(pole.real - zero.real)
        step = min(
            float(protocol["maximum_fraction_of_pole_depth"]) * gamma,
            float(protocol["zero_aware_fraction_of_pole_zero_separation"]) * separation,
        )
        solver = _solver(physics, float(row["slab_thickness_um"]),
                         float(row["ridge_height_um"]), float(row["final_period_um"]),
                         int(row["spectrum_harmonics"]))
        plus = solver(pole.real + step).transmission
        minus = solver(pole.real - step).transmission
        delay = float(np.angle(plus / minus) / (2 * step) * FS_PER_UM_OVER_C)
        updated.append({
            "quality_target": row["quality_target"],
            "achieved_quality_factor": row["achieved_quality_factor"],
            "normalized_zero_pole_real_separation": separation / gamma,
            "zero_aware_step_in_pole_depths": step / gamma,
            "zero_aware_delay_fs": delay,
            "reference_delay_fs": row["delay_reference_fs"],
            "zero_aware_relative_delay_error": abs(delay - row["delay_reference_fs"]) / max(
                abs(row["delay_reference_fs"]), 1e-12
            ),
            "gamma_only_adaptive_relative_delay_error": row["adaptive_step_relative_delay_error"],
            "fixed_step_relative_delay_error": row["fixed_step_relative_delay_error"],
            "transmittance_at_pole_real": row["center_T"],
        })
    (output / "summary.json").write_text(json.dumps({"cases": updated}, indent=2) + "\n", encoding="utf-8")

    selected = [next(x for x in rows if x["quality_target"] == q)
                for q in protocol["structure_quality_factors"]]
    fig, axes = plt.subplots(1, len(selected), figsize=(11.6, 3.1), constrained_layout=True)
    for axis, row in zip(axes, selected, strict=True):
        slab = float(row["slab_thickness_um"])
        ridge = float(row["ridge_height_um"])
        fill = float(physics["ridge_fill_factor"])
        nx, nz = 320, 200
        x = (np.arange(nx) + 0.5) / nx
        z = (np.arange(nz) + 0.5) * (slab + ridge + 0.08) / nz
        high = (z[:, None] < slab) | (
            (z[:, None] >= slab) & (z[:, None] < slab + ridge)
            & (abs(x[None, :] - 0.5) < fill / 2)
        )
        n = np.where(high, float(physics["high_index"]), float(physics["port_index"]))
        axis.imshow(n, origin="lower", aspect="auto", extent=(0, 1, 0, z[-1]),
                    vmin=float(physics["port_index"]), vmax=float(physics["high_index"]),
                    cmap="viridis")
        axis.set_title(f"Q≈{row['achieved_quality_factor']:.0f}\np={row['final_period_um']*1000:.1f} nm")
        axis.set_xlabel("x / period")
        axis.set_ylabel("z (µm)")
    fig.savefig(output / "gmr_binary_structure_q_series.png", dpi=220)
    fig.savefig(output / "gmr_binary_structure_q_series.pdf")
    plt.close(fig)

    quality = np.asarray([x["achieved_quality_factor"] for x in rows])
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), constrained_layout=True)
    axes[0].semilogx(quality, 1000 * np.asarray([x["ridge_height_um"] for x in rows]), "o-", label="ridge height")
    axes[0].semilogx(quality, 1000 * np.asarray([x["final_period_um"] for x in rows]), "s-", label="period")
    axes[0].set_ylabel("Physical size (nm)")
    axes[0].legend(frameon=False)
    axes[1].loglog(quality, [x["delay_reference_fs"] for x in rows], "o-", label="Transmission delay")
    axes[1].loglog(quality, [x["pole_depth_delay_scale_fs"] for x in rows], "s--", label="1 / pole depth")
    axes[1].set_ylabel("Delay scale (fs)")
    axes[1].legend(frameon=False)
    axes[2].loglog(quality, [x["fixed_step_relative_delay_error"] for x in updated], "o-", label="Fixed Δω")
    axes[2].loglog(quality, [x["gamma_only_adaptive_relative_delay_error"] for x in updated], "s-", label="0.2γ")
    axes[2].loglog(quality, [x["zero_aware_relative_delay_error"] for x in updated], "^-", label="Zero-aware")
    axes[2].set_ylabel("Relative delay error")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xlabel("Verified channel-pole Q")
        axis.grid(alpha=0.25)
    fig.savefig(output / "gmr_geometry_delay_sampling.png", dpi=220)
    fig.savefig(output / "gmr_geometry_delay_sampling.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
