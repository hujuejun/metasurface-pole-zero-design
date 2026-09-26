"""Audit same-grid pole and adaptive direct-response runs without overclaiming."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/m161_fine_adjoint_cost_audit"))
    args = parser.parse_args()
    pole_path = Path("runs/m142i_fine_mesh_adjoint/summary.json")
    direct_path = Path("runs/m159b_fine_mesh_matched_start_direct/summary.json")
    pole = json.loads(pole_path.read_text(encoding="utf-8"))
    direct = json.loads(direct_path.read_text(encoding="utf-8"))
    if len(direct["cases"]) != 3:
        raise RuntimeError("fine-mesh direct run is incomplete")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for quality in (50.0, 100.0, 200.0):
        p = next(item for item in pole["cases"] if item["target_quality_factor"] == quality)
        d = next(item for item in direct["cases"] if item["target_quality_factor"] == quality)
        if not d["start_coordinates_exactly_match"] or not np.array_equal(d["start_used"], p["start_used"]):
            raise RuntimeError(f"Q{quality:g} direct/pole optimization starts differ")
        rows.append({
            "target_quality_factor": quality,
            "pole_target_complex_frequency_count": 1,
            "direct_complex_t_target_sample_count": d["direct_sample_count"],
            "pole_adjoint_frequency_factorizations": p["equivalent_frequency_factorizations"],
            "direct_adjoint_frequency_factorizations": d["frequency_factorizations"],
            "pole_optimization_wall_seconds": p["optimization_wall_seconds"],
            "direct_optimization_wall_seconds": d["optimization_wall_seconds"],
            "pole_verified_target_error_in_depths": p["verification"]["pole_error_in_target_depths"],
            "direct_verified_target_error_in_depths": d["verification"]["pole_error_in_target_depths"],
            "pole_design_coordinates": p["optimized_coordinates"],
            "direct_design_coordinates": d["optimized_coordinates"],
            "direct_final_complex_t_training_residual_norm": d["final_direct_residual_norm"],
        })
    result = {
        "comparison_scope": "one common physical start per Q, same 5-nm Nx24 FDFD mesh; different design targets, not equal-functions or globally representative",
        "input_sha256": {str(path): sha256(path) for path in (pole_path, direct_path)},
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    qualities = np.asarray([row["target_quality_factor"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), constrained_layout=True)
    axes[0].plot(qualities, [row["pole_adjoint_frequency_factorizations"] for row in rows], "o-", label="Pole-adjoint target")
    axes[0].plot(qualities, [row["direct_adjoint_frequency_factorizations"] for row in rows], "s--", label="Adaptive 3-sample direct target")
    axes[0].set(xlabel="Target Q", ylabel="Optimization frequency factorizations")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(qualities, [row["pole_optimization_wall_seconds"] for row in rows], "o-", label="Pole-adjoint target")
    axes[1].plot(qualities, [row["direct_optimization_wall_seconds"] for row in rows], "s--", label="Adaptive 3-sample direct target")
    axes[1].set(xlabel="Target Q", ylabel="Optimization wall time (s)")
    for axis in axes:
        axis.grid(True, alpha=0.25, linewidth=0.5)
    fig.savefig(output / "m161_fine_adjoint_cost_audit.png", dpi=240)
    fig.savefig(output / "m161_fine_adjoint_cost_audit.pdf")
    plt.close(fig)
    for row in rows:
        print(f"Q={row['target_quality_factor']:.0f}: pole/direct LU={row['pole_adjoint_frequency_factorizations']}/{row['direct_adjoint_frequency_factorizations']}, wall={row['pole_optimization_wall_seconds']:.1f}/{row['direct_optimization_wall_seconds']:.1f}s, direct pole error={row['direct_verified_target_error_in_depths']:.3g} depths", flush=True)


if __name__ == "__main__":
    main()
