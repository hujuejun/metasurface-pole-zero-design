"""Census hard-binary parity poles across the M27E validation meshes.

This is a deliberately independent post-optimization audit.  It scans a broad
real-frequency interval before refining roots, so the reported mesh comparison
does not depend on continuation guesses propagated from one mesh to the next.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pzid.plotting import save_figure
from pzid.roots import ScalarRootResult, refine_scalar_root

from scripts.m4b_analyze_eigenchannels import resonance_estimates
from scripts.m4b_optimize_kerker_bar import _parity_response
from scripts.m11_screen_fp_tm1_hybrid import _root_json, _sweep
from scripts.m27e_multimesh_binary_reopt import _physical_system


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _frequency(record: dict[str, Any]) -> complex:
    return complex(*record["frequency"])


def _sweep_with_progress(
    omega: np.ndarray,
    epsilon: np.ndarray,
    setup: Any,
    cells: int,
    chunk_size: int = 40,
) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    for start in range(0, omega.size, chunk_size):
        stop = min(start + chunk_size, omega.size)
        chunks.append(_sweep(2.0 * np.pi / omega[start:stop], epsilon, setup))
        print(f"N={cells} broad sweep {stop}/{omega.size}", flush=True)
    return {
        "even": np.concatenate([np.asarray(item["even"]) for item in chunks]),
        "odd": np.concatenate([np.asarray(item["odd"]) for item in chunks]),
        "matrices": np.concatenate(
            [np.asarray(item["matrices"]) for item in chunks], axis=0
        ),
        "empty": np.concatenate([np.asarray(item["empty"]) for item in chunks]),
        "energy_balance_max_abs_error": max(
            float(item["energy_balance_max_abs_error"]) for item in chunks
        ),
        "reciprocity_max_abs_error": max(
            float(item["reciprocity_max_abs_error"]) for item in chunks
        ),
        "mirror_scattering_max_abs_error": max(
            float(item["mirror_scattering_max_abs_error"]) for item in chunks
        ),
    }


def _delay_trace(omega: np.ndarray, response: np.ndarray) -> np.ndarray:
    phase = np.unwrap(np.angle(response))
    delay = np.abs(np.gradient(phase, omega))
    return np.maximum(delay - np.percentile(delay, 15.0), 0.0)


def _refine_estimate(
    estimate: dict[str, float],
    parity: str,
    epsilon: np.ndarray,
    setup: Any,
    tolerance: float,
    max_evaluations: int,
) -> ScalarRootResult | None:
    center = float(estimate["omega_rad_per_um"])
    linewidth = max(float(estimate["linewidth_rad_per_um"]), 1e-4)
    radius = min(max(3.0 * linewidth, 0.0025), 0.008)
    depths = sorted({0.5 * linewidth, 2.5e-4, 6.0e-4, 1.5e-3})
    candidates = []
    for depth in depths:
        result = refine_scalar_root(
            lambda value: _parity_response(value, epsilon, setup, parity),
            complex(center, -depth),
            kind="pole",
            tolerance=tolerance,
            max_evaluations=max_evaluations,
            search_radius=radius,
        )
        if result.success:
            candidates.append(result)
    if not candidates:
        return None
    return min(candidates, key=lambda item: item.residual_norm)


def _deduplicate_roots(
    roots: list[ScalarRootResult],
    start: float,
    stop: float,
    residual_gate: float,
) -> list[ScalarRootResult]:
    accepted = [
        item
        for item in roots
        if item.residual_norm <= residual_gate
        and item.frequency.imag < 0.0
        and start - 0.002 <= item.frequency.real <= stop + 0.002
    ]
    accepted.sort(key=lambda item: (item.frequency.real, item.frequency.imag))
    unique: list[ScalarRootResult] = []
    for item in accepted:
        duplicate = next(
            (
                previous
                for previous in unique
                if abs(item.frequency - previous.frequency) <= 1e-5
            ),
            None,
        )
        if duplicate is None:
            unique.append(item)
        elif item.residual_norm < duplicate.residual_norm:
            unique[unique.index(duplicate)] = item
    return unique


def _pair_records(
    even_roots: list[ScalarRootResult], odd_roots: list[ScalarRootResult]
) -> list[dict[str, Any]]:
    pairs = []
    for even in even_roots:
        for odd in odd_roots:
            mean_linewidth = -even.frequency.imag - odd.frequency.imag
            if mean_linewidth <= 0.0:
                continue
            center = 0.5 * (even.frequency + odd.frequency)
            pairs.append(
                {
                    "even_frequency": [even.frequency.real, even.frequency.imag],
                    "odd_frequency": [odd.frequency.real, odd.frequency.imag],
                    "center_frequency": [center.real, center.imag],
                    "real_separation_rad_per_um": float(
                        even.frequency.real - odd.frequency.real
                    ),
                    "center_mismatch_over_mean_linewidth": float(
                        abs(even.frequency.real - odd.frequency.real) / mean_linewidth
                    ),
                    "linewidth_ratio_even_over_odd": float(
                        even.frequency.imag / odd.frequency.imag
                    ),
                }
            )
    return sorted(
        pairs, key=lambda item: item["center_mismatch_over_mean_linewidth"]
    )


def _write_spectrum(
    path: Path, omega: np.ndarray, spectrum: dict[str, Any]
) -> None:
    matrices = np.asarray(spectrum["matrices"])
    even_delay = _delay_trace(omega, np.asarray(spectrum["even"]))
    odd_delay = _delay_trace(omega, np.asarray(spectrum["odd"]))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "omega_rad_per_um",
                "even_real",
                "even_imag",
                "odd_real",
                "odd_imag",
                "even_resonant_delay_um",
                "odd_resonant_delay_um",
                "transmittance",
                "reflectance",
                "transmission_phase_rad",
            )
        )
        phase = np.unwrap(np.angle(matrices[:, 1, 0]))
        for index, frequency in enumerate(omega):
            writer.writerow(
                (
                    frequency,
                    spectrum["even"][index].real,
                    spectrum["even"][index].imag,
                    spectrum["odd"][index].real,
                    spectrum["odd"][index].imag,
                    even_delay[index],
                    odd_delay[index],
                    abs(matrices[index, 1, 0]) ** 2,
                    abs(matrices[index, 0, 0]) ** 2,
                    phase[index],
                )
            )


def _plot_census(records: list[dict[str, Any]], target: complex) -> plt.Figure:
    figure, axes = plt.subplots(
        2, 1, figsize=(9.0, 7.3), constrained_layout=True
    )
    colors = ("tab:blue", "tab:orange", "tab:green")
    for record, color in zip(records, colors, strict=True):
        cells = int(record["cells_per_period"])
        suffix = " held out" if record["held_out"] else ""
        omega = np.asarray(record["omega"])
        axes[0].semilogy(
            omega,
            np.maximum(np.asarray(record["even_delay"]), 1e-2),
            color=color,
            label=f"N={cells}{suffix}, even",
        )
        axes[0].semilogy(
            omega,
            np.maximum(np.asarray(record["odd_delay"]), 1e-2),
            color=color,
            linestyle="--",
            label=f"N={cells}{suffix}, odd",
        )
        for parity, marker, filled in (("even", "o", True), ("odd", "s", False)):
            roots = [_frequency(item) for item in record[f"{parity}_roots"]]
            for index, root in enumerate(roots):
                axes[1].scatter(
                    root.real,
                    root.imag,
                    marker=marker,
                    s=40,
                    color=color if filled else None,
                    facecolors=color if filled else "none",
                    edgecolors=color,
                    label=(
                        f"N={cells}{suffix}, {parity} pole" if index == 0 else None
                    ),
                )
    axes[0].axvline(target.real, color="black", linestyle=":", linewidth=0.9)
    axes[0].set_ylabel("Resonant eigenphase delay (µm)")
    axes[0].set_title("Independent broad-band pole census of one hard-binary contour")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].scatter(
        target.real,
        target.imag,
        marker="x",
        s=85,
        linewidths=2,
        color="black",
        label="optimization target",
    )
    axes[1].axhline(0.0, color="0.45", linewidth=0.7)
    axes[1].set_xlabel("Re frequency (rad/µm)")
    axes[1].set_ylabel("Im frequency (rad/µm)")
    axes[1].set_title("All independently refined parity poles in scan interval")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    for axis in axes:
        axis.grid(True, linewidth=0.4, alpha=0.35)
    return figure


def _plot_response(records: list[dict[str, Any]], target: complex) -> plt.Figure:
    figure, axes = plt.subplots(
        2, 1, figsize=(9.0, 6.6), sharex=True, constrained_layout=True
    )
    colors = ("tab:blue", "tab:orange", "tab:green")
    for record, color in zip(records, colors, strict=True):
        cells = int(record["cells_per_period"])
        suffix = " held out" if record["held_out"] else ""
        label = f"N={cells}{suffix}"
        omega = np.asarray(record["omega"])
        axes[0].plot(omega, record["transmittance"], color=color, label=f"T, {label}")
        axes[0].plot(
            omega,
            record["reflectance"],
            color=color,
            linestyle="--",
            label=f"R, {label}",
        )
        phase = np.asarray(record["transmission_phase"])
        axes[1].plot(omega, (phase - phase[0]) / np.pi, color=color, label=label)
    for axis in axes:
        axis.axvline(target.real, color="black", linestyle=":", linewidth=0.9)
        axis.grid(True, linewidth=0.4, alpha=0.35)
        axis.legend(frameon=False, fontsize=8, ncol=2)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Power")
    axes[0].set_title("Broad-band hard-binary scattering audit")
    axes[1].set_xlabel("Real frequency (rad/µm)")
    axes[1].set_ylabel("Transmission phase change / π")
    return figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/m27e_fan_1d_multimesh_binary_tm.json"),
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("runs/m27e_fan_1d_multimesh_binary"),
    )
    parser.add_argument("--start", type=float, default=4.28)
    parser.add_argument("--stop", type=float, default=4.37)
    parser.add_argument("--step", type=float, default=1.25e-4)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    parent_config = json.loads(
        Path(config["parent_config"]).read_text(encoding="utf-8")
    )
    summary = json.loads((arguments.run / "summary.json").read_text(encoding="utf-8"))
    target = complex(*summary["common_pole_target"])
    omega = np.arange(
        arguments.start, arguments.stop + 0.5 * arguments.step, arguments.step
    )
    controls = config["verification"]
    records = []
    for cells_value in controls["validation_cells_per_period"]:
        cells = int(cells_value)
        epsilon_path = arguments.run / f"hard_binary_permittivity_n{cells}.csv"
        epsilon = np.loadtxt(epsilon_path, delimiter=",")
        setup, _, _ = _physical_system(config, parent_config, cells, 64.0)
        if epsilon.shape != setup.grid.shape:
            raise ValueError(
                f"N={cells} epsilon shape {epsilon.shape} != grid {setup.grid.shape}"
            )
        spectrum = _sweep_with_progress(omega, epsilon, setup, cells)
        parity_records: dict[str, Any] = {}
        for parity in ("even", "odd"):
            response = np.asarray(spectrum[parity])
            estimates = resonance_estimates(
                2.0 * np.pi / omega, response, maximum_count=24
            )
            refined = []
            for estimate in estimates:
                result = _refine_estimate(
                    estimate,
                    parity,
                    epsilon,
                    setup,
                    float(controls["root_tolerance"]),
                    int(controls["root_max_evaluations"]),
                )
                if result is not None:
                    refined.append(result)
            roots = _deduplicate_roots(
                refined,
                arguments.start,
                arguments.stop,
                float(controls["root_residual_gate"]),
            )
            parity_records[f"{parity}_estimates"] = estimates
            parity_records[f"{parity}_roots"] = [_root_json(root) for root in roots]
            print(
                f"N={cells} {parity}: {len(estimates)} peaks, "
                f"{len(roots)} unique validated poles",
                flush=True,
            )
            for root in roots:
                print(
                    f"  {root.frequency.real:.10f} {root.frequency.imag:+.10f}i "
                    f"residual={root.residual_norm:.3e}",
                    flush=True,
                )
        matrices = np.asarray(spectrum["matrices"])
        even_delay = _delay_trace(omega, np.asarray(spectrum["even"]))
        odd_delay = _delay_trace(omega, np.asarray(spectrum["odd"]))
        record = {
            "cells_per_period": cells,
            "held_out": cells == int(controls["held_out_cells_per_period"]),
            "epsilon_sha256": _sha256(epsilon_path),
            **parity_records,
            "opposite_parity_pairs": _pair_records(
                [
                    ScalarRootResult(
                        frequency=_frequency(item),
                        kind="pole",
                        success=bool(item["success"]),
                        message=str(item["message"]),
                        evaluations=int(item["evaluations"]),
                        residual_norm=float(item["residual_norm"]),
                    )
                    for item in parity_records["even_roots"]
                ],
                [
                    ScalarRootResult(
                        frequency=_frequency(item),
                        kind="pole",
                        success=bool(item["success"]),
                        message=str(item["message"]),
                        evaluations=int(item["evaluations"]),
                        residual_norm=float(item["residual_norm"]),
                    )
                    for item in parity_records["odd_roots"]
                ],
            ),
            "energy_balance_max_abs_error": float(
                spectrum["energy_balance_max_abs_error"]
            ),
            "reciprocity_max_abs_error": float(
                spectrum["reciprocity_max_abs_error"]
            ),
            "mirror_scattering_max_abs_error": float(
                spectrum["mirror_scattering_max_abs_error"]
            ),
            "omega": omega,
            "even_delay": even_delay,
            "odd_delay": odd_delay,
            "transmittance": np.abs(matrices[:, 1, 0]) ** 2,
            "reflectance": np.abs(matrices[:, 0, 0]) ** 2,
            "transmission_phase": np.unwrap(np.angle(matrices[:, 1, 0])),
        }
        records.append(record)
        _write_spectrum(
            arguments.run / f"hard_binary_broad_census_n{cells}.csv", omega, spectrum
        )

    json_records = []
    for record in records:
        json_records.append(
            {
                key: (
                    [float(value) for value in item]
                    if isinstance(item, np.ndarray)
                    else item
                )
                for key, item in record.items()
            }
        )
    audit = {
        "scope": "Independent broad-band pole census of the final M27E hard-binary contour.",
        "scan": {
            "start_rad_per_um": arguments.start,
            "stop_rad_per_um": arguments.stop,
            "step_rad_per_um": arguments.step,
            "point_count": int(omega.size),
        },
        "common_target": [target.real, target.imag],
        "n64_excluded_from_optimization": True,
        "records": json_records,
        "provenance": {
            "script_sha256": _sha256(Path(__file__).resolve()),
            "config_sha256": _sha256(arguments.config),
            "input_summary_sha256": _sha256(arguments.run / "summary.json"),
        },
    }
    output_path = arguments.run / "hard_binary_pole_census.json"
    output_path.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    figures = arguments.run / "figures"
    figures.mkdir(exist_ok=True)
    figure = _plot_census(records, target)
    save_figure(figure, figures / "m27e_hard_binary_pole_census.png", dpi=230)
    plt.close(figure)
    figure = _plot_response(records, target)
    save_figure(figure, figures / "m27e_hard_binary_broad_response.png", dpi=230)
    plt.close(figure)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
