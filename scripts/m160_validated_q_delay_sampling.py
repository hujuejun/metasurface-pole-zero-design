"""M160: delay and sampling control for numerically validated Q50–400 designs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.m142_single_transmission_q_screen import FS_PER_UM_OVER_C


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_t(curve: np.ndarray, detuning: float) -> complex:
    x = np.asarray(curve["detuning_in_target_depths"], dtype=float)
    index = int(np.argmin(abs(x - detuning)))
    if abs(x[index] - detuning) > 1e-8:
        raise RuntimeError(f"saved spectrum lacks detuning {detuning}")
    return complex(curve["t_real"][index], curve["t_imag"][index])


def measured_delay(curve: np.ndarray, half_step_in_depths: float, depth: float) -> float:
    plus = sample_t(curve, half_step_in_depths)
    minus = sample_t(curve, -half_step_in_depths)
    return float(np.angle(plus / minus) / (2 * half_step_in_depths * depth) * FS_PER_UM_OVER_C)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/m160_validated_q_delay_sampling"))
    args = parser.parse_args()
    inputs = {
        50: Path("runs/m142i_fine_mesh_adjoint"),
        100: Path("runs/m142i_fine_mesh_adjoint"),
        200: Path("runs/m142i_fine_mesh_adjoint"),
        400: Path("runs/m158_q400_fine_pole_adjoint"),
    }
    gate_paths = (
        Path("runs/m154b_frozen_five_pair_joint_32_2p5nm/summary.json"),
        Path("runs/m153b_fourier_modal_high_order_check/summary.json"),
        Path("runs/m158_q400_independent_gates/summary.json"),
    )
    q400_gate = json.loads(gate_paths[-1].read_text(encoding="utf-8"))
    if not q400_gate["publication_gate_passed"]:
        raise RuntimeError("Q400 independent numerical gate did not pass")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    records = []
    source_hashes = {str(path): sha256(path) for path in gate_paths}
    for quality, run in inputs.items():
        summary_path = run / "summary.json"
        curve_path = run / f"spectrum_q{quality:03d}.csv"
        source_hashes[str(summary_path)] = sha256(summary_path)
        source_hashes[str(curve_path)] = sha256(curve_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        case = next(item for item in summary["cases"] if item["target_quality_factor"] == quality)
        curve = np.genfromtxt(curve_path, delimiter=",", names=True)
        depth = abs(complex(**case["verification"]["verified_pole"]).imag)
        exact_delay = float(case["verification"]["group_delay_fs_at_pole_real"])
        fixed_detuning = quality / 200.0  # h = gamma_Q50/4, common absolute step.
        adaptive_detuning = 0.25
        fixed_delay = measured_delay(curve, fixed_detuning, depth)
        adaptive_delay = measured_delay(curve, adaptive_detuning, depth)
        record = {
            "target_quality_factor": quality,
            "verified_quality_factor_on_design_mesh": float(case["verification"]["verified_quality_factor"]),
            "verified_pole": case["verification"]["verified_pole"],
            "target_pole_depth_rad_per_um": abs(complex(**case["target_pole"]).imag),
            "group_delay_fs_at_pole_real": exact_delay,
            "isolated_pole_delay_contribution_fs": float(FS_PER_UM_OVER_C / depth),
            "nonpole_delay_remainder_fs": float(exact_delay - FS_PER_UM_OVER_C / depth),
            "fixed_physical_half_step_rad_per_um": float(depth * fixed_detuning),
            "fixed_half_step_in_pole_depths": fixed_detuning,
            "fixed_step_delay_fs": fixed_delay,
            "fixed_step_relative_delay_error": float(abs(fixed_delay - exact_delay) / abs(exact_delay)),
            "adaptive_half_step_in_pole_depths": adaptive_detuning,
            "adaptive_step_delay_fs": adaptive_delay,
            "adaptive_step_relative_delay_error": float(abs(adaptive_delay - exact_delay) / abs(exact_delay)),
        }
        records.append(record)
    physical_steps = [record["fixed_physical_half_step_rad_per_um"] for record in records]
    if max(physical_steps) - min(physical_steps) > 1e-12:
        raise RuntimeError("fixed sample steps were not physically identical")
    result = {
        "method": "saved-spectrum derivative-sampling control on numerically validated designs",
        "input_sha256": source_hashes,
        "phase_winding_comparison_excluded": True,
        "direct_adaptive_three_sample_method_not_excluded": True,
        "cases": records,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), constrained_layout=True)
    qs = np.asarray([case["target_quality_factor"] for case in records])
    axes[0].plot(qs, [case["group_delay_fs_at_pole_real"] for case in records], "o-", label="Measured group delay")
    axes[0].plot(qs, [case["isolated_pole_delay_contribution_fs"] for case in records], "s--", label="Pole factor 1/gamma")
    axes[0].set(xlabel="Prescribed Q", ylabel="Delay at pole real frequency (fs)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(qs, 100 * np.asarray([case["fixed_step_relative_delay_error"] for case in records]), "o-", label="Fixed physical step")
    axes[1].plot(qs, 100 * np.asarray([case["adaptive_step_relative_delay_error"] for case in records]), "s--", label="Step = 0.25 pole depth")
    axes[1].set(xlabel="Prescribed Q", ylabel="Delay-estimation error (%)")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.25, linewidth=0.5)
    fig.savefig(output / "m160_validated_q_delay_sampling.png", dpi=240)
    fig.savefig(output / "m160_validated_q_delay_sampling.pdf")
    plt.close(fig)
    for case in records:
        print(f"Q={case['target_quality_factor']}: delay={case['group_delay_fs_at_pole_real']:.2f} fs, fixed/adaptive errors={100*case['fixed_step_relative_delay_error']:.1f}%/{100*case['adaptive_step_relative_delay_error']:.1f}%", flush=True)


if __name__ == "__main__":
    main()
