"""Stage-2 physical-coordinate adjoint audit for the M37 nanodisk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pzid.huygens3d import SmoothNanodiskParameterization3D
from pzid.poles import PoleTarget
from pzid.scattering3d import local_to_global_incoming_3d
from pzid.workflow3d import (
    ChannelConstraintSpec3D,
    PoleConstraintSpec3D,
    PoleZeroInverseProblem3D,
)
from scripts.m37_huygens_forward import build_benchmark


def _as_complex(value: list[float]) -> complex:
    return complex(float(value[0]), float(value[1]))


def _root_records(
    path: Path, required_names: list[str] | tuple[str, ...] | None = None
) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = {item["name"]: item for item in data["roots"]}
    required = records.keys() if required_names is None else required_names
    missing = [name for name in required if name not in records]
    if missing:
        raise RuntimeError(f"root summary is missing {missing}")
    failed = [name for name in required if not records[name]["success"]]
    if failed:
        raise RuntimeError(f"root gate failed for {failed}")
    return records


def _normalized_coordinate(value: float, bounds: list[float]) -> float:
    low, high = map(float, bounds)
    if not low <= value <= high:
        raise ValueError("physical point lies outside its declared range")
    return (value - low) / (high - low)


def _coupling_ratio_for_local_parity(
    frequency: complex,
    local_upper_amplitude: float,
    setup: Any,
) -> complex:
    """Return the raw-basis Chen ratio for an exact local parity channel."""

    local = np.asarray((1.0, 0.0, local_upper_amplitude, 0.0))
    incoming = local_to_global_incoming_3d(local, np.conj(frequency), setup)
    if abs(incoming[0]) < 1e-12:
        raise RuntimeError("time-reversed pole input has a vanishing lower component")
    # Workflow convention is global incoming [1, sigma*].
    return complex(np.conj(incoming[2] / incoming[0]))


def build_audit(
    config: dict[str, Any],
) -> tuple[PoleZeroInverseProblem3D, np.ndarray, list[dict[str, Any]]]:
    forward = json.loads(
        Path(config["forward_config"]).read_text(encoding="utf-8")
    )
    setup, _, _ = build_benchmark(forward)
    specification = config["parameterization"]
    parameterization = SmoothNanodiskParameterization3D(
        grid=setup.grid,
        host_index=setup.port_index,
        radius_range=tuple(specification["radius_range"]),
        height_range=tuple(specification["height_range"]),
        disk_index_range=tuple(specification["disk_index_range"]),
        transverse_boundary_width=float(
            specification["transverse_boundary_width"]
        ),
        axial_boundary_width=float(specification["axial_boundary_width"]),
    )
    point = config["physical_point"]
    raw = np.asarray(
        (
            _normalized_coordinate(point["radius"], specification["radius_range"]),
            _normalized_coordinate(point["height"], specification["height_range"]),
            _normalized_coordinate(
                point["disk_index"], specification["disk_index_range"]
            ),
        )
    )
    mapping = parameterization.map(raw)
    required_roots = [*config["pole_names"], config["reflection_zero_name"]]
    roots = _root_records(Path(config["root_summary"]), required_roots)
    pole_specs = []
    coupling_records = []
    parity_amplitudes = (1.0, -1.0)
    if len(config["pole_names"]) != len(parity_amplitudes):
        raise ValueError("M37 requires one symmetric and one antisymmetric pole")
    for name, local_upper_amplitude in zip(
        config["pole_names"], parity_amplitudes, strict=True
    ):
        frequency = _as_complex(roots[name]["frequency"])
        coupling = _coupling_ratio_for_local_parity(
            frequency, local_upper_amplitude, setup
        )
        pole_specs.append(
            PoleConstraintSpec3D(
                target=PoleTarget(
                    frequency=frequency,
                    coupling_ratio=coupling,
                ),
                polarization="x",
            )
        )
        coupling_records.append(
            {
                "name": name,
                "frequency": [frequency.real, frequency.imag],
                "coupling_ratio": [coupling.real, coupling.imag],
                "local_upper_amplitude": local_upper_amplitude,
                "source": "exact local parity transformed to raw port basis",
            }
        )
    zero_frequency = _as_complex(
        roots[config["reflection_zero_name"]]["frequency"]
    )
    problem = PoleZeroInverseProblem3D(
        parameterization=parameterization,
        setup=setup,
        poles=tuple(pole_specs),
        channels=(
            ChannelConstraintSpec3D(
                frequency=zero_frequency,
                channel="reflection",
                target_value=0.0,
                polarization="x",
            ),
        ),
    )
    return problem, raw, coupling_records


def _component_errors(
    analytic: np.ndarray,
    numerical: np.ndarray,
    *,
    absolute_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.abs(analytic - numerical)
    scale = np.maximum(np.maximum(np.abs(analytic), np.abs(numerical)), absolute_floor)
    return absolute, absolute / scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    arguments.output.mkdir(parents=True, exist_ok=False)
    (arguments.output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    problem, raw, couplings = build_audit(config)
    direction = np.asarray(config["direction"], dtype=float)
    direction /= np.linalg.norm(direction)
    step = float(config["difference_step"])
    if np.any(raw - step * direction < 0.0) or np.any(
        raw + step * direction > 1.0
    ):
        raise ValueError("finite-difference direction crosses a coordinate bound")

    evaluation = problem.evaluate(raw)
    analytic = evaluation.complex_jacobian.reshape(
        evaluation.complex_residual.size, -1
    ) @ direction
    plus = problem.evaluate(raw + step * direction)
    minus = problem.evaluate(raw - step * direction)
    numerical = (plus.complex_residual - minus.complex_residual) / (2.0 * step)
    absolute, relative = _component_errors(
        analytic,
        numerical,
        absolute_floor=float(config["absolute_tolerance"]),
    )
    active = np.maximum(np.abs(analytic), np.abs(numerical)) >= float(
        config["absolute_tolerance"]
    )
    passed = bool(
        np.all(absolute[~active] <= float(config["absolute_tolerance"]))
        and np.all(relative[active] <= float(config["relative_tolerance"]))
    )
    mapping = problem.parameterization.map(raw)
    np.savez_compressed(
        arguments.output / "audit.npz",
        raw=raw,
        direction=direction,
        epsilon_r=mapping.epsilon_r,
        complex_residual=evaluation.complex_residual,
        complex_jacobian=evaluation.complex_jacobian,
        analytic_directional_derivative=analytic,
        numerical_directional_derivative=numerical,
        absolute_error=absolute,
        relative_error=relative,
    )
    summary = {
        "passed": passed,
        "raw_coordinates": raw.tolist(),
        "physical_coordinates": {
            "radius": mapping.radius,
            "height": mapping.height,
            "disk_index": mapping.disk_index,
        },
        "couplings": couplings,
        "maximum_absolute_error": float(np.max(absolute)),
        "maximum_active_relative_error": float(
            np.max(relative[active]) if np.any(active) else 0.0
        ),
        "active_complex_rows": int(np.count_nonzero(active)),
        "complex_rows": int(active.size),
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if not passed:
        raise SystemExit("physical-coordinate adjoint gate failed")


if __name__ == "__main__":
    main()
