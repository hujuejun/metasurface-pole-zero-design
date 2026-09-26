"""Independent post-optimization verification of the M37 Huygens disk."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import numpy as np

from pzid.residues import fit_fixed_poles
from scripts.m37_huygens_adjoint import _as_complex
from scripts.m37_huygens_forward import run_forward
from scripts.m37_huygens_roots import _root_worker


def _root_comparison(
    target_records: dict[str, dict[str, Any]],
    final_records: dict[str, dict[str, Any]],
    names: list[str],
) -> tuple[dict[str, float], float]:
    linewidths = [
        abs(_as_complex(target_records[name]["frequency"]).imag)
        for name in names
        if target_records[name]["kind"] == "pole"
    ]
    scale = min(linewidths)
    errors = {
        name: float(
            abs(
                _as_complex(final_records[name]["frequency"])
                - _as_complex(target_records[name]["frequency"])
            )
            / scale
        )
        for name in names
    }
    return errors, max(errors.values())


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    arguments.output.mkdir(parents=True, exist_ok=False)
    _json(arguments.output / "config.json", config)
    with np.load(arguments.design) as design:
        final_raw = np.asarray(design["final_raw"], dtype=float)

    adjoint_config = json.loads(
        Path(config["adjoint_config"]).read_text(encoding="utf-8")
    )
    parameterization = adjoint_config["parameterization"]
    physical = {}
    for name, raw, bounds in zip(
        ("radius", "height", "refractive_index"),
        final_raw,
        (
            parameterization["radius_range"],
            parameterization["height_range"],
            parameterization["disk_index_range"],
        ),
        strict=True,
    ):
        low, high = map(float, bounds)
        physical[name] = low + float(raw) * (high - low)

    forward = json.loads(
        Path(config["verification_forward_config"]).read_text(encoding="utf-8")
    )
    forward["nanodisk"].update(physical)
    forward_directory = arguments.output / "forward"
    forward_directory.mkdir()
    _json(forward_directory / "config.json", forward)
    forward_summary = run_forward(forward, forward_directory)

    root_config = json.loads(
        Path(config["verification_root_config"]).read_text(encoding="utf-8")
    )
    root_directory = arguments.output / "roots"
    root_directory.mkdir()
    _json(root_directory / "config.json", root_config)
    controls = {
        "tolerance": root_config["tolerance"],
        "max_evaluations": root_config["max_evaluations"],
    }
    final_roots: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=int(root_config.get("workers", 1))) as pool:
        futures = {
            pool.submit(
                _root_worker,
                forward,
                root,
                controls,
                str(root_directory / root["name"]),
            ): root["name"]
            for root in root_config["roots"]
        }
        for future in as_completed(futures):
            record = future.result()
            final_roots[record["name"]] = record
            print(
                f"verified {record['name']}: {record['frequency']} "
                f"residual={record['residual_norm']:.3e}",
                flush=True,
            )
    ordered = [final_roots[item["name"]] for item in root_config["roots"]]
    _json(root_directory / "summary.json", {"roots": ordered})

    target_data = json.loads(Path(config["root_summary"]).read_text(encoding="utf-8"))
    target_roots = {item["name"]: item for item in target_data["roots"]}
    required = [*config["pole_names"], config["reflection_zero_name"]]
    errors, maximum_error = _root_comparison(target_roots, final_roots, required)

    with np.load(forward_directory / "forward_scan.npz") as scan:
        omega = np.asarray(scan["omega"])
        reflection = np.asarray(scan["reflection"])
        transmission = np.asarray(scan["transmission"])
    residue_interval = config["verification"]["residue_omega_interval"]
    selected = (omega >= residue_interval[0]) & (omega <= residue_interval[1])
    order = np.argsort(omega[selected])
    selected_omega = omega[selected][order]
    pole_values = [
        _as_complex(final_roots[name]["frequency"])
        for name in config["pole_names"]
    ]
    residue_records = {}
    for name, values in (
        ("reflection", reflection[selected][order]),
        ("transmission", transmission[selected][order]),
    ):
        fit = fit_fixed_poles(
            selected_omega,
            values,
            pole_values,
            polynomial_degree=int(config["verification"]["polynomial_degree"]),
        )
        residue_records[name] = {
            "residues": [[value.real, value.imag] for value in fit.residues],
            "residue_magnitudes": np.abs(fit.residues).tolist(),
            "normalized_rms_error": fit.normalized_rms_error,
        }
    minimum_residue = min(
        min(record["residue_magnitudes"])
        for record in residue_records.values()
    )
    phase_span = float(
        np.ptp(np.unwrap(np.angle(transmission[np.argsort(omega)]))) / np.pi
    )
    limits = config["verification"]
    passed = bool(
        all(final_roots[name]["success"] for name in required)
        and maximum_error <= float(limits["maximum_root_shift_linewidths"])
        and forward_summary["maximum_power_error"]
        <= float(limits["maximum_power_error"])
        and forward_summary["minimum_reflectance"]
        <= float(limits["maximum_minimum_reflectance"])
        and minimum_residue >= float(limits["minimum_scalar_residue_magnitude"])
    )
    summary = {
        "passed": passed,
        "final_raw": final_raw.tolist(),
        "final_physical": physical,
        "forward": forward_summary,
        "transmission_phase_span_pi": phase_span,
        "root_errors_in_narrower_pole_linewidths": errors,
        "maximum_root_error_in_narrower_pole_linewidths": maximum_error,
        "scalar_channel_residue_fits": residue_records,
        "minimum_scalar_residue_magnitude": minimum_residue,
    }
    _json(arguments.output / "summary.json", summary)
    if not passed:
        raise SystemExit("independent M37 verification gate failed")


if __name__ == "__main__":
    main()

