"""Resolve parasitic transmission-channel poles of the frozen low-index contour.

This fixed-geometry diagnostic is independent of the design optimizer. It
computes the published coarse grid plus pole-depth-scaled local windows,
including the empty-cell transmission needed for a common phase reference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from pzid.roots import refine_scalar_root
from pzid.twoport import solve_two_port_scattering
from scripts.replay_final_2d import build


ROOT = Path(__file__).resolve().parents[1]


def half_max_width(k: np.ndarray, reflectance: np.ndarray) -> float:
    """Interpolate the absolute R=R_peak/2 crossings around one isolated peak."""
    peak = int(np.argmax(reflectance))
    threshold = 0.5 * float(reflectance[peak])
    left = np.flatnonzero(reflectance[:peak] < threshold)
    right = np.flatnonzero(reflectance[peak + 1:] < threshold)
    if left.size == 0 or right.size == 0:
        raise ValueError("Peak half-maximum crossings lie outside the local window")
    i = int(left[-1])
    j = peak + 1 + int(right[0])
    left_crossing = float(np.interp(threshold, reflectance[i:i + 2], k[i:i + 2]))
    right_crossing = float(np.interp(
        threshold, reflectance[j - 1:j + 1][::-1], k[j - 1:j + 1][::-1]
    ))
    return right_crossing - left_crossing


def write_spectrum(path: Path, rows: list[tuple[float, complex, complex, complex]]) -> dict:
    ordered = sorted(rows, key=lambda row: row[0])
    k = np.asarray([row[0] for row in ordered])
    t = np.asarray([row[1] for row in ordered])
    r = np.asarray([row[2] for row in ordered])
    empty = np.asarray([row[3] for row in ordered])
    phase = np.unwrap(np.angle(t / empty)) / np.pi
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("k_rad_per_um", "T", "R", "t_real", "t_imag",
                         "r_real", "r_imag", "empty_t_real", "empty_t_imag",
                         "deembedded_phase_pi"))
        writer.writerows(zip(k, np.abs(t)**2, np.abs(r)**2, t.real, t.imag,
                             r.real, r.imag, empty.real, empty.imag, phase,
                             strict=True))
    return {
        "points": len(rows),
        "minimum_transmittance": float(np.min(np.abs(t)**2)),
        "maximum_reflectance": float(np.max(np.abs(r)**2)),
        "unwrapped_phase_span_pi": float(np.ptp(phase)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/low_index_resonance_audit.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/low_index_resonance_audit"))
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["design"] != "low_index_binary":
        raise ValueError("This diagnostic is specific to the frozen low-index contour")
    eps, setup = build(config["design"], int(config["mesh_cells_per_period"]))
    empty_eps = np.full_like(eps, setup.port_index**2)
    response_cache: dict[float, tuple[float, complex, complex, complex]] = {}

    def response(k: float) -> tuple[float, complex, complex, complex]:
        key = float(k)
        if key not in response_cache:
            s = solve_two_port_scattering(key, eps, setup).matrix
            empty = solve_two_port_scattering(key, empty_eps, setup).matrix[1, 0]
            response_cache[key] = (key, complex(s[1, 0]), complex(s[0, 0]), complex(empty))
        return response_cache[key]

    def transmission(k: complex) -> complex:
        return complex(solve_two_port_scattering(k, eps, setup).matrix[1, 0])

    band = [float(value) for value in config["band_rad_per_um"]]
    features = []
    local_grids = []
    for item in config["features"]:
        guess = complex(*[float(value) for value in item["pole_guess_rad_per_um"]])
        root = refine_scalar_root(
            transmission, guess, kind="pole", tolerance=1e-10,
            max_evaluations=200,
            search_radius=float(item["search_radius_rad_per_um"]),
        )
        pole = complex(root.frequency)
        if (not root.success or root.residual_norm > config["root_residual_limit"]
                or pole.imag >= 0 or not band[0] < pole.real < band[1]):
            raise RuntimeError(f"Unverified in-band pole for {item['name']}: {root}")
        depth = -pole.imag
        halfwidth = float(item["window_halfwidth_in_pole_depths"]) * depth
        local_grids.append((item["name"], np.linspace(
            pole.real - halfwidth, pole.real + halfwidth,
            int(item["window_points"]),
        )))
        features.append({
            "name": item["name"],
            "pole_rad_per_um": [float(pole.real), float(pole.imag)],
            "pole_residual": float(root.residual_norm),
            "quality_factor": float(pole.real / (2 * depth)),
        })

    coarse_grid = np.linspace(*band, int(config["coarse_points"]))
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    coarse_metrics = write_spectrum(output / "coarse_spectrum.csv",
                                    [response(k) for k in coarse_grid])
    combined = list(response_cache.values())
    for feature, (name, grid) in zip(features, local_grids, strict=True):
        local_rows = [response(k) for k in grid]
        write_spectrum(output / f"{name}_spectrum.csv", local_rows)
        combined.extend(local_rows)
        k = np.asarray([row[0] for row in local_rows])
        r = np.asarray([abs(row[2])**2 for row in local_rows])
        feature["sampled_half_maximum_width_rad_per_um"] = half_max_width(k, r)
        feature["sampled_peak_reflectance"] = float(np.max(r))
        feature["sampled_minimum_transmittance"] = float(min(abs(row[1])**2 for row in local_rows))
        fit = minimize_scalar(
            lambda value: -abs(response(float(value))[2])**2,
            bounds=(k[int(np.argmax(r))] - 2 * (-feature["pole_rad_per_um"][1]),
                    k[int(np.argmax(r))] + 2 * (-feature["pole_rad_per_um"][1])),
            method="bounded", options={"xatol": 1e-10},
        )
        if not fit.success:
            raise RuntimeError(f"Real-axis peak refinement failed for {name}")
        peak = response(float(fit.x))
        feature["peak_k_rad_per_um"] = float(peak[0])
        feature["peak_reflectance"] = float(abs(peak[2])**2)
        feature["transmittance_at_peak"] = float(abs(peak[1])**2)
        combined.append(peak)
    # Exact duplicate coarse/window frequencies share the same cached response.
    unique = {row[0]: row for row in combined}
    adaptive_metrics = write_spectrum(output / "adaptive_spectrum.csv",
                                      list(unique.values()))
    summary = {
        "design": config["design"],
        "mesh_cells_per_period": config["mesh_cells_per_period"],
        "band_rad_per_um": band,
        "coarse_grid": coarse_metrics,
        "adaptive_grid": adaptive_metrics,
        "features": features,
        "scope": config["scope"],
        "inputs_sha256": {
            "config": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "permittivity": hashlib.sha256((ROOT / "inputs/final_permittivity/low_index_binary_n56.csv").read_bytes()).hexdigest(),
        },
        "caution": "Window locations move with FDFD mesh; this N56 audit does not establish mesh convergence of individual parasitic poles.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
