"""Stage-3 physical-coordinate pole-zero inverse design for M37."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pzid.huygens3d import SmoothNanodiskParameterization3D
from pzid.optimize import LMOptions, solve_underdetermined_lm
from pzid.poles import PoleTarget
from pzid.workflow3d import (
    ChannelConstraintSpec3D,
    PoleConstraintSpec3D,
    PoleZeroInverseProblem3D,
)
from scripts.m37_huygens_adjoint import _as_complex, _normalized_coordinate, _root_records
from scripts.m37_huygens_forward import build_benchmark


def _coupling_records(path: Path) -> dict[str, complex]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data["passed"]:
        raise RuntimeError("the prerequisite physical-coordinate adjoint gate failed")
    return {
        item["name"]: _as_complex(item["coupling_ratio"])
        for item in data["couplings"]
    }


def build_inverse_problem(
    config: dict[str, Any],
) -> tuple[PoleZeroInverseProblem3D, np.ndarray]:
    adjoint_config = json.loads(
        Path(config["adjoint_config"]).read_text(encoding="utf-8")
    )
    forward = json.loads(
        Path(adjoint_config["forward_config"]).read_text(encoding="utf-8")
    )
    setup, _, _ = build_benchmark(forward)
    specification = adjoint_config["parameterization"]
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
    required_roots = [*config["pole_names"], config["reflection_zero_name"]]
    roots = _root_records(Path(config["root_summary"]), required_roots)
    couplings = _coupling_records(Path(config["adjoint_summary"]))
    weights = config["weights"]
    poles = tuple(
        PoleConstraintSpec3D(
            target=PoleTarget(
                frequency=_as_complex(roots[name]["frequency"]),
                coupling_ratio=couplings[name],
            ),
            polarization="x",
            weight=float(weights["pole"]),
        )
        for name in config["pole_names"]
    )
    zero_frequency = _as_complex(
        roots[config["reflection_zero_name"]]["frequency"]
    )
    problem = PoleZeroInverseProblem3D(
        parameterization=parameterization,
        setup=setup,
        poles=poles,
        channels=(
            ChannelConstraintSpec3D(
                frequency=zero_frequency,
                channel="reflection",
                target_value=0.0,
                polarization="x",
                weight=float(weights["reflection_zero"]),
            ),
        ),
    )
    initial = config["initial_physical_point"]
    raw = np.asarray(
        (
            _normalized_coordinate(initial["radius"], specification["radius_range"]),
            _normalized_coordinate(initial["height"], specification["height_range"]),
            _normalized_coordinate(
                initial["disk_index"], specification["disk_index_range"]
            ),
        )
    )
    return problem, raw


def _physical_record(mapping: Any) -> dict[str, float]:
    return {
        "radius": mapping.radius,
        "height": mapping.height,
        "disk_index": mapping.disk_index,
    }


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
    evaluations = arguments.output / "evaluations"
    evaluations.mkdir()
    problem, initial = build_inverse_problem(config)
    records: list[dict[str, Any]] = []

    def logged_system(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        evaluation = problem.evaluate(raw)
        index = len(records)
        mapping = evaluation.mapping
        np.savez_compressed(
            evaluations / f"evaluation_{index:03d}.npz",
            raw=np.asarray(raw),
            epsilon_r=mapping.epsilon_r,
            complex_residual=evaluation.complex_residual,
            complex_jacobian=evaluation.complex_jacobian,
            residual=evaluation.residual,
            jacobian=evaluation.jacobian,
        )
        record = {
            "index": index,
            "raw": np.asarray(raw).tolist(),
            "physical": _physical_record(mapping),
            "residual_norm": float(np.linalg.norm(evaluation.residual)),
        }
        records.append(record)
        (arguments.output / "evaluation_history.json").write_text(
            json.dumps(records, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"evaluation {index}: residual={record['residual_norm']:.6e} "
            f"physical={record['physical']}",
            flush=True,
        )
        return evaluation.residual, evaluation.jacobian.reshape(
            evaluation.residual.size, -1
        )

    options = LMOptions(**config["optimizer"])
    result = solve_underdetermined_lm(
        logged_system,
        initial,
        lower_bound=0.0,
        upper_bound=1.0,
        options=options,
        allow_overdetermined=True,
    )
    final_mapping = problem.parameterization.map(result.x)
    np.savez_compressed(
        arguments.output / "design.npz",
        initial_raw=initial,
        final_raw=result.x,
        initial_epsilon=problem.parameterization.map(initial).epsilon_r,
        final_epsilon=final_mapping.epsilon_r,
        final_residual=result.residual,
        final_jacobian=result.jacobian,
    )
    history = [item.__dict__ for item in result.history]
    (arguments.output / "accepted_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "converged": result.converged,
        "reason": result.reason,
        "initial_physical": _physical_record(
            problem.parameterization.map(initial)
        ),
        "final_physical": _physical_record(final_mapping),
        "initial_residual_norm": records[0]["residual_norm"],
        "final_residual_norm": float(np.linalg.norm(result.residual)),
        "evaluations": len(records),
        "accepted_iterations": len(history),
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
