"""Refine M37 channel poles and zeros with resumable complex secants."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import numpy as np

from pzid.huygens3d import axial_eigenchannels, copolarized_coefficients
from pzid.roots import refine_scalar_root_complex_secant
from pzid.scattering3d import deembed_scattering_matrix_3d, solve_scattering_3d
from scripts.m37_huygens_forward import build_benchmark


def _complex(value: list[float]) -> complex:
    if len(value) != 2:
        raise ValueError("complex values must be [real, imag]")
    return complex(float(value[0]), float(value[1]))


def _selected_response(matrix: np.ndarray, name: str) -> complex:
    reflection, transmission = copolarized_coefficients(matrix)
    symmetric, antisymmetric = axial_eigenchannels(matrix)
    values = {
        "reflection": reflection,
        "transmission": transmission,
        "symmetric": symmetric,
        "antisymmetric": antisymmetric,
    }
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"unknown response {name!r}") from error


def _reported_root(
    root_config: dict[str, Any], result: Any
) -> tuple[complex, str]:
    """Convert a time-reversed channel zero into its physical pole label.

    For a reciprocal lossless cell, the zero evaluated in the upper half-plane
    is the complex conjugate of the outgoing pole.  Retaining both frequencies
    in the run record makes the numerical operation explicit and auditable.
    """

    evaluation_frequency = complex(result.frequency)
    if bool(root_config.get("report_conjugate_as_pole", False)):
        if str(root_config["kind"]) != "zero":
            raise ValueError("conjugate pole reporting requires a zero solve")
        return np.conj(evaluation_frequency), "pole"
    return evaluation_frequency, str(root_config["kind"])


def _root_worker(
    forward_config: dict[str, Any],
    root_config: dict[str, Any],
    controls: dict[str, Any],
    output: str,
) -> dict[str, Any]:
    setup, _, epsilon = build_benchmark(forward_config)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []

    def response(frequency: complex) -> complex:
        raw = solve_scattering_3d(frequency, epsilon, setup).matrix
        local = deembed_scattering_matrix_3d(raw, frequency, setup)
        value = _selected_response(local, str(root_config["response"]))
        index = len(samples)
        sample_path = directory / f"evaluation_{index:03d}.npz"
        np.savez_compressed(
            sample_path,
            omega=np.asarray(frequency),
            raw_scattering_matrix=raw,
            scattering_matrix=local,
            response=np.asarray(value),
        )
        sample = {
            "index": index,
            "omega": [frequency.real, frequency.imag],
            "response": [value.real, value.imag],
            "file": sample_path.name,
        }
        samples.append(sample)
        (directory / "evaluations.json").write_text(
            json.dumps(samples, indent=2) + "\n", encoding="utf-8"
        )
        return value

    result = refine_scalar_root_complex_secant(
        response,
        _complex(root_config["first"]),
        _complex(root_config["second"]),
        kind=str(root_config["kind"]),
        tolerance=float(controls["tolerance"]),
        max_evaluations=int(controls["max_evaluations"]),
        search_radius=float(root_config["search_radius"]),
    )
    frequency, reported_kind = _reported_root(root_config, result)
    record = {
        "name": root_config["name"],
        "response": root_config["response"],
        "kind": reported_kind,
        "solver_kind": root_config["kind"],
        "frequency": [frequency.real, frequency.imag],
        "evaluation_frequency": [result.frequency.real, result.frequency.imag],
        "reported_conjugate": bool(
            root_config.get("report_conjugate_as_pole", False)
        ),
        "success": result.success,
        "message": result.message,
        "evaluations": result.evaluations,
        "residual_norm": result.residual_norm,
    }
    (directory / "result.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    forward_path = Path(config["forward_config"])
    forward = json.loads(forward_path.read_text(encoding="utf-8"))
    if arguments.resume:
        saved = json.loads(
            (arguments.output / "config.json").read_text(encoding="utf-8")
        )
        if saved != config:
            raise ValueError("resume config differs from the saved run config")
    else:
        arguments.output.mkdir(parents=True, exist_ok=False)
        (arguments.output / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        (arguments.output / "forward_config.json").write_text(
            json.dumps(forward, indent=2) + "\n", encoding="utf-8"
        )

    completed: dict[str, dict[str, Any]] = {}
    pending = []
    for root_config in config["roots"]:
        result_path = arguments.output / root_config["name"] / "result.json"
        if result_path.is_file():
            completed[root_config["name"]] = json.loads(
                result_path.read_text(encoding="utf-8")
            )
        else:
            pending.append(root_config)
    controls = {
        "tolerance": config["tolerance"],
        "max_evaluations": config["max_evaluations"],
    }
    with ProcessPoolExecutor(max_workers=int(config.get("workers", 1))) as executor:
        futures = {
            executor.submit(
                _root_worker,
                forward,
                root_config,
                controls,
                str(arguments.output / root_config["name"]),
            ): root_config["name"]
            for root_config in pending
        }
        for future in as_completed(futures):
            record = future.result()
            completed[record["name"]] = record
            print(
                f"{record['name']}: omega={record['frequency']} "
                f"residual={record['residual_norm']:.3e}",
                flush=True,
            )
    ordered = [completed[item["name"]] for item in config["roots"]]
    (arguments.output / "summary.json").write_text(
        json.dumps({"roots": ordered}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
