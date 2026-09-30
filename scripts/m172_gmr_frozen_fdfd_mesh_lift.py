"""Frozen physical GMR geometry: FDFD mesh trend versus Fourier roots.

No geometry optimization occurs here. Each mesh decodes the same physical
binary interfaces directly, avoiding interpolation of a coarser epsilon grid.
The separate Fourier H101 transmission-channel pole is the reference.
"""

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

from pzid.gmr2d import SingleToothGMR
from pzid.grid import UniformGrid2D
from pzid.roots import refine_scalar_root
from pzid.twoport import TwoPortScatteringSetup, solve_two_port_scattering


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complex(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _setup(period: float, nx: int, config: dict, base: dict) -> TwoPortScatteringSetup:
    dz = float(config["reference_axial_cell_size_um"]) * float(config["reference_lateral_cells"]) / nx
    nz = int(round(float(config["reference_total_domain_um"]) / dz))
    grid = UniformGrid2D(nx, nz, period / nx, dz)
    old_dz = float(base["axial_cell_size_um"])
    old_nz = int(base["axial_cells"])

    def rescale(plane: int) -> int:
        old_z = (plane + 0.5) * old_dz
        return int(round(old_z / dz - 0.5))

    return TwoPortScatteringSetup(
        grid=grid,
        lower_source_plane=rescale(int(base["lower_source_plane"])),
        upper_source_plane=rescale(int(base["upper_source_plane"])),
        lower_monitor_planes=tuple(rescale(int(p)) for p in base["lower_monitor_planes"]),
        upper_monitor_planes=tuple(rescale(int(p)) for p in base["upper_monitor_planes"]),
        port_index=float(base["port_index"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m172_gmr_frozen_fdfd_mesh_lift.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/m172_gmr_frozen_fdfd_mesh_lift"))
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    high_path = Path(protocol["source_high_fidelity_summary"])
    adjoint_path = Path(protocol["source_adjoint_summary"])
    base_path = Path(protocol["base_adjoint_config"])
    high = json.loads(high_path.read_text(encoding="utf-8"))["cases"]
    adjoint = json.loads(adjoint_path.read_text(encoding="utf-8"))["cases"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    revision = subprocess.run(("git", "rev-parse", "HEAD"), check=False, capture_output=True, text=True)
    (output / "metadata.json").write_text(json.dumps({
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__, "platform": platform.platform(),
        "command": sys.argv,
        "input_sha256": {str(p): _sha256(p) for p in (args.config, high_path, adjoint_path, base_path)},
        "script_sha256": _sha256(Path(__file__)),
        "physical_geometry_sha256": _sha256(Path("src/pzid/gmr2d.py")),
    }, indent=2) + "\n", encoding="utf-8")
    rows = []
    fig, axes = plt.subplots(1, len(protocol["quality_factors"]), figsize=(11, 3.4), constrained_layout=True)
    for axis, quality in zip(axes, protocol["quality_factors"], strict=True):
        geometry = next(x for x in high if x["quality_target"] == quality)
        source = next(x for x in adjoint if x["quality_factor"] == quality)
        reference = complex(**geometry["held_out_pole"])
        original_seed = complex(**source["final_fdfd"]["verified_pole"])
        period = float(geometry["final_period_um"])
        grating = SingleToothGMR(
            period, float(geometry["slab_thickness_um"]), float(geometry["ridge_height_um"]),
            float(base["ridge_fill_factor"]), float(base["high_index"]), float(base["port_index"]),
        )
        case = {"quality_target": quality, "fourier_h101_reference_pole": _complex(reference),
                "period_um": period, "slab_um": grating.slab_thickness_um,
                "ridge_um": grating.ridge_height_um, "meshes": []}
        seed = original_seed
        for nx in protocol["lateral_cells"]:
            setup = _setup(period, int(nx), protocol, base)
            epsilon, _ = grating.rasterize_with_derivatives(setup.grid,
                                                             base_z_um=float(base["base_z_um"]))
            response = lambda omega: complex(solve_two_port_scattering(omega, epsilon, setup).matrix[1, 0])
            root = refine_scalar_root(
                response, seed, kind="pole", tolerance=1e-9,
                max_evaluations=int(protocol["maximum_root_evaluations"]),
                search_radius=max(0.1, 15 * abs(reference.imag)),
            )
            if root.residual_norm > 1e-6 or root.frequency.imag >= 0:
                raise RuntimeError(f"fixed-geometry mesh root failed Q={quality}, Nx={nx}: {root}")
            seed = root.frequency
            t_center = response(reference.real)
            row = {
                "nx": nx, "nz": setup.grid.nz, "dx_um": setup.grid.dx, "dz_um": setup.grid.dz,
                "transmission_channel_pole": _complex(root.frequency),
                "root_residual": root.residual_norm,
                "root_motion_from_fourier_in_reference_depths": float(abs(root.frequency - reference) / abs(reference.imag)),
                "transmittance_at_fourier_center": float(abs(t_center)**2),
                "epsilon_min": float(epsilon.min()), "epsilon_max": float(epsilon.max()),
            }
            case["meshes"].append(row)
            (output / "summary.json").write_text(json.dumps({"cases": [*rows, case]}, indent=2) + "\n", encoding="utf-8")
            print(f"Q={quality} Nx={nx}: FDFD-Fourier root separation "
                  f"{row['root_motion_from_fourier_in_reference_depths']:.3g} depths", flush=True)
        rows.append(case)
        axis.plot([x["nx"] for x in case["meshes"]],
                  [x["root_motion_from_fourier_in_reference_depths"] for x in case["meshes"]], "o-")
        axis.set_title(f"Q≈{quality}")
        axis.set_xlabel("FDFD cells per period")
        axis.set_ylabel("Channel-pole error / Fourier pole depth")
        axis.grid(alpha=0.25)
    fig.savefig(output / "gmr_fdfd_mesh_to_fourier_root.png", dpi=220)
    fig.savefig(output / "gmr_fdfd_mesh_to_fourier_root.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
