"""Independent TM Fourier-modal check of frozen M142I patterned cavities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from pzid.rcwa2d import (
    FourierLayer,
    centered_binary_grating_layer,
    homogeneous_layer,
    solve_tm_stack,
)
from pzid.roots import refine_scalar_root_complex_secant


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> str | None:
    result = subprocess.run(("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def physical_layers(
    coordinates: list[float],
    geometry: dict,
    material: dict,
    harmonic_half_width: int,
) -> tuple[tuple[FourierLayer, ...], float]:
    """Encode physical interface lengths without reading an FDFD epsilon grid."""

    if len(coordinates) != 2:
        raise ValueError("expected mirror and cavity physical thickness scales")
    mirror_scale, cavity_scale = map(float, coordinates)
    n_hi = float(material["high_index"])
    n_lo = float(material["low_index"])
    wavelength = float(material["wavelength_um"])
    pairs = int(geometry["fixed_mirror_pairs_per_side"])
    ridge_height = float(geometry["ridge_height_um"])
    cavity_length = wavelength / (2 * n_lo) * cavity_scale
    if cavity_length <= ridge_height:
        raise ValueError("cavity is thinner than the ridge")
    upper_mirror = []
    for _ in range(pairs):
        upper_mirror.extend((
            homogeneous_layer(n_hi, wavelength / (4 * n_hi) * mirror_scale, harmonic_half_width),
            homogeneous_layer(n_lo, wavelength / (4 * n_lo) * mirror_scale, harmonic_half_width),
        ))
    half_gap = homogeneous_layer(n_lo, (cavity_length - ridge_height) / 2, harmonic_half_width)
    ridge = centered_binary_grating_layer(
        n_lo, n_hi, float(geometry["ridge_fill_factor"]), ridge_height, harmonic_half_width
    )
    layers = tuple(upper_mirror + [half_gap, ridge, half_gap] + list(reversed(upper_mirror)))
    length = float(sum(layer.thickness_um for layer in layers))
    return layers, length


def source_curve(path: Path, detunings: np.ndarray) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    x = np.asarray(data["detuning_in_target_depths"], dtype=float)
    real = np.interp(detunings, x, data["t_real"])
    imag = np.interp(detunings, x, data["t_imag"])
    return real + 1j * imag


def write_curve(path: Path, columns: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns.keys())
        writer.writerows(zip(*columns.values(), strict=True))


def solve_case(
    quality: float,
    harmonics: int,
    source: dict,
    geometry: dict,
    material: dict,
    protocol: dict,
    source_run: Path,
    output: Path,
) -> dict:
    started = time.perf_counter()
    coordinates = list(map(float, source["optimized_coordinates"]))
    layers, length = physical_layers(coordinates, geometry, material, harmonics)
    source_pole = complex(**source["verification"]["verified_pole"])
    target_pole = complex(**source["target_pole"])
    target_depth = abs(target_pole.imag)
    period = float(geometry["period_um"])
    n_port = float(material["port_index"])
    detuning = np.asarray(protocol["spectrum_detunings_in_target_depths"], dtype=float)
    omega = target_pole.real + target_depth * detuning
    responses = [
        solve_tm_stack(value, layers, period_um=period, port_index=n_port, harmonic_half_width=harmonics)
        for value in omega
    ]
    t_raw = np.asarray([item.transmission for item in responses])
    # The FDFD transmission is calibrated to an empty homogeneous cell.
    t = t_raw * np.exp(-1j * omega * n_port * length)
    r = np.asarray([item.reflection for item in responses])
    fdfd_path = source_run / f"spectrum_q{int(quality):03d}.csv"
    t_fdfd = source_curve(fdfd_path, detuning)
    real_energy_error = float(np.max(abs(abs(t) ** 2 + abs(r) ** 2 - 1)))

    def channel_transmission(value: complex) -> complex:
        return solve_tm_stack(value, layers, period_um=period, port_index=n_port, harmonic_half_width=harmonics).transmission

    root = refine_scalar_root_complex_secant(
        channel_transmission,
        source_pole,
        source_pole + (0.08 + 0.05j) * target_depth,
        kind="pole",
        tolerance=float(protocol["root_tolerance"]),
        max_evaluations=int(protocol["max_root_evaluations"]),
        search_radius=max(0.1, 10 * target_depth),
    )
    valid_root = bool(
        np.isfinite(root.frequency)
        and root.frequency.imag < 0
        and root.residual_norm <= float(protocol["accepted_root_residual"])
    )
    record = {
        "target_quality_factor": quality,
        "harmonic_half_width": harmonics,
        "total_harmonics": 2 * harmonics + 1,
        "physical_coordinates_unchanged": coordinates,
        "physical_stack_length_um": length,
        "source_fdfd_pole": {"real": source_pole.real, "imag": source_pole.imag},
        "source_fdfd_spectrum_sha256": sha256(fdfd_path),
        "verified_channel_pole": valid_root,
        "root_success_flag": bool(root.success),
        "root_residual_norm": float(root.residual_norm),
        "root_evaluations": int(root.evaluations),
        "real_energy_balance_max_abs_error": real_energy_error,
        "spectrum_max_abs_transmittance_difference_from_fdfd": float(np.max(abs(abs(t) ** 2 - abs(t_fdfd) ** 2))),
        "spectrum_complex_t_rmse_from_fdfd": float(np.sqrt(np.mean(abs(t - t_fdfd) ** 2))),
        "spectrum_relative_phase_mae_rad_from_fdfd": float(np.mean(abs(np.angle((t / t[len(t)//2]) / (t_fdfd / t_fdfd[len(t_fdfd)//2]))))),
        "wall_seconds": time.perf_counter() - started,
    }
    if valid_root:
        record.update({
            "rcwa_pole": {"real": float(root.frequency.real), "imag": float(root.frequency.imag)},
            "rcwa_quality_factor": float(root.frequency.real / (-2 * root.frequency.imag)),
            "pole_shift_from_fdfd_in_target_depths": float(abs(root.frequency - source_pole) / target_depth),
        })
    write_curve(output / f"spectrum_q{int(quality):03d}_m{harmonics:02d}.csv", {
        "omega_rad_per_um": omega,
        "detuning_in_target_depths": detuning,
        "rcwa_t_real_deembedded": t.real,
        "rcwa_t_imag_deembedded": t.imag,
        "rcwa_transmittance": abs(t) ** 2,
        "rcwa_reflectance": abs(r) ** 2,
        "rcwa_relative_phase_rad": np.unwrap(np.angle(t / t[len(t)//2])),
        "fdfd_t_real": t_fdfd.real,
        "fdfd_t_imag": t_fdfd.imag,
        "fdfd_transmittance": abs(t_fdfd) ** 2,
        "fdfd_relative_phase_rad": np.unwrap(np.angle(t_fdfd / t_fdfd[len(t_fdfd)//2])),
    })
    return record


def plot_summary(summary: dict, protocol: dict, output: Path) -> None:
    cases = summary["cases"]
    qualities = [float(value) for value in protocol["target_quality_factors"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.8), constrained_layout=True)
    for quality in qualities:
        subset = sorted((item for item in cases if item["target_quality_factor"] == quality), key=lambda item: item["harmonic_half_width"])
        axes[0].plot([item["harmonic_half_width"] for item in subset], [item.get("pole_shift_from_fdfd_in_target_depths", np.nan) for item in subset], "o-", label=f"Q={quality:.0f}")
        axes[1].plot([item["harmonic_half_width"] for item in subset], [item["spectrum_max_abs_transmittance_difference_from_fdfd"] for item in subset], "o-", label=f"Q={quality:.0f}")
    axes[0].set_ylabel("RCWA–FDFD pole separation / target depth")
    axes[1].set_ylabel("Max |T(RCWA) - T(FDFD)|")
    for axis in axes:
        axis.set_xlabel("Highest Fourier order M")
        axis.grid(True, alpha=0.3, linewidth=0.5)
    axes[0].legend(frameon=False)
    fig.savefig(output / "m153_fourier_order_trends.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.2), constrained_layout=True)
    for col, quality in enumerate(qualities):
        subset = [item for item in cases if item["target_quality_factor"] == quality]
        if not subset:
            continue
        finest = max(subset, key=lambda item: item["harmonic_half_width"])
        path = output / f"spectrum_q{int(quality):03d}_m{finest['harmonic_half_width']:02d}.csv"
        data = np.genfromtxt(path, delimiter=",", names=True)
        x = data["detuning_in_target_depths"]
        axes[0, col].plot(x, data["fdfd_transmittance"], "k-", label="FDFD T")
        axes[0, col].plot(x, data["rcwa_transmittance"], "C0--", label="Fourier-modal T")
        axes[0, col].plot(x, data["rcwa_reflectance"], "C1:", label="Fourier-modal R")
        axes[0, col].set_title(f"Q target={quality:.0f}")
        axes[1, col].plot(x, data["fdfd_relative_phase_rad"] / np.pi, "k-", label="FDFD")
        axes[1, col].plot(x, data["rcwa_relative_phase_rad"] / np.pi, "C0--", label="Fourier-modal")
        axes[1, col].set_xlabel("Detuning / target pole depth")
        if col == 0:
            axes[0, col].set_ylabel("Power")
            axes[1, col].set_ylabel("Relative t phase / pi")
        for axis in (axes[0, col], axes[1, col]):
            axis.grid(True, alpha=0.3, linewidth=0.5)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)
    fig.savefig(output / "m153_rt_phase_cross_solver.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m153_fourier_modal_reference_check.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/m153_fourier_modal_reference_check"))
    parser.add_argument("--only-quality", type=float)
    parser.add_argument("--max-harmonic", type=int)
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    geometry_path = Path(protocol["source_geometry_config"])
    material_path = Path(protocol["source_material_config"])
    source_path = Path(protocol["source_run"]) / "summary.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    material = json.loads(material_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "script_sha256": sha256(Path(__file__)),
        "solver_source_sha256": sha256(Path("src/pzid/rcwa2d.py")),
        "input_sha256": {str(path): sha256(path) for path in (args.config, geometry_path, material_path, source_path)},
        "command": sys.argv,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    results = []
    for quality in protocol["target_quality_factors"]:
        if args.only_quality is not None and float(quality) != args.only_quality:
            continue
        source_case = next(item for item in source["cases"] if item["target_quality_factor"] == quality)
        for harmonics in protocol["harmonic_half_widths"]:
            if args.max_harmonic is not None and harmonics > args.max_harmonic:
                continue
            result = solve_case(float(quality), int(harmonics), source_case, geometry, material, protocol, Path(protocol["source_run"]), output)
            results.append(result)
            (output / "summary.json").write_text(json.dumps({"method": "independent Fourier-modal TM frozen physical geometry check", "cases": results}, indent=2) + "\n", encoding="utf-8")
            print(
                f"Q={quality:.0f} M={harmonics}: root={result['verified_channel_pole']} "
                f"shift={result.get('pole_shift_from_fdfd_in_target_depths', float('nan')):.4f} depths "
                f"maxTdiff={result['spectrum_max_abs_transmittance_difference_from_fdfd']:.4g} "
                f"energy={result['real_energy_balance_max_abs_error']:.1e} "
                f"wall={result['wall_seconds']:.1f}s",
                flush=True,
            )
    if args.only_quality is None and args.max_harmonic is None:
        plot_summary({"cases": results}, protocol, output)


if __name__ == "__main__":
    main()
