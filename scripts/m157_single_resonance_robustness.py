"""M157: held-out pole/zero motion for the converged single-resonance family."""

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
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from pzid.adjoint import scattering_matrix_with_jacobian
from pzid.rcwa2d import centered_binary_grating_layer, homogeneous_layer, solve_tm_stack
from pzid.roots import refine_scalar_root_complex_secant
from pzid.twoport import solve_two_port_scattering

from scripts.m142e_optimize_patterned_q_poles import build_fixed_grid
from scripts.m144a_systematic_pole_zero_robustness import (
    PARAMETERS,
    _adjoint_pole_gradient,
    _epsilon,
    _root,
)
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def decode(value: dict[str, float]) -> complex:
    return complex(value["real"], value["imag"])


def fdfd_zero_gradient(
    values: np.ndarray,
    zero: complex,
    geometry: dict[str, Any],
    material: dict[str, Any],
    scalar: Any,
    twoport: Any,
    protocol: dict[str, Any],
    depth: float,
) -> tuple[np.ndarray, float, float]:
    """Differentiate the *reflection-channel* zero, not an S-matrix zero."""

    epsilon, thickness_derivatives = _epsilon(values, geometry, material, scalar)
    derivatives = np.empty((len(PARAMETERS), *epsilon.shape), dtype=float)
    derivatives[:2] = thickness_derivatives
    steps = (
        (2, float(protocol["fill_mapping_derivative_step"])),
        (3, float(protocol["height_mapping_derivative_step_um"])),
    )
    mapping_errors = []
    for index, step in steps:
        displacement = np.zeros(len(PARAMETERS))
        displacement[index] = step
        plus, _ = _epsilon(values + displacement, geometry, material, scalar)
        minus, _ = _epsilon(values - displacement, geometry, material, scalar)
        derivatives[index] = (plus - minus) / (2 * step)
        half_step, _ = _epsilon(values + displacement / 2, geometry, material, scalar)
        half_minus, _ = _epsilon(values - displacement / 2, geometry, material, scalar)
        half_slope = (half_step - half_minus) / step
        mapping_errors.append(float(np.linalg.norm(derivatives[index] - half_slope) / max(np.linalg.norm(half_slope), 1)))
    low = float(material["low_index"])
    high = float(values[4])
    derivatives[4] = 2 * high * (epsilon - low**2) / (high**2 - low**2)

    evaluation = scattering_matrix_with_jacobian(zero, epsilon, twoport)
    residual = complex(evaluation.matrix[0, 0])
    parameter_gradient = np.einsum("zx,pzx->p", evaluation.jacobian[0, 0], derivatives)
    frequency_step = float(protocol["frequency_derivative_step_in_pole_depths"]) * depth
    plus = solve_two_port_scattering(zero + frequency_step, epsilon, twoport).matrix[0, 0]
    minus = solve_two_port_scattering(zero - frequency_step, epsilon, twoport).matrix[0, 0]
    frequency_gradient = (plus - minus) / (2 * frequency_step)
    if abs(frequency_gradient) < 1e-8:
        raise RuntimeError("reflection-zero frequency derivative is nearly singular")
    return -parameter_gradient / frequency_gradient, float(abs(residual)), float(max(mapping_errors))


def fourier_response(
    frequency: complex,
    values: np.ndarray,
    geometry: dict[str, Any],
    material: dict[str, Any],
    harmonics: int,
):
    # Index disorder changes epsilon, not the manufactured layer thickness.
    # The M153 helper computes a quarter-wave thickness using its *current*
    # high index and is therefore unsuitable for an index perturbation.
    reference_high = float(material["high_index"])
    changed_high = float(values[4])
    low = float(material["low_index"])
    wavelength = float(material["wavelength_um"])
    mirror_scale = float(values[0])
    cavity_scale = float(values[1])
    ridge_height = float(values[3])
    cavity_length = wavelength / (2 * low) * cavity_scale
    if cavity_length <= ridge_height:
        raise ValueError("cavity is thinner than its ridge")
    upper_mirror = []
    for _ in range(int(geometry["fixed_mirror_pairs_per_side"])):
        upper_mirror.extend((
            homogeneous_layer(changed_high, wavelength / (4 * reference_high) * mirror_scale, harmonics),
            homogeneous_layer(low, wavelength / (4 * low) * mirror_scale, harmonics),
        ))
    half_gap = homogeneous_layer(low, (cavity_length - ridge_height) / 2, harmonics)
    ridge = centered_binary_grating_layer(low, changed_high, float(values[2]), ridge_height, harmonics)
    layers = tuple(upper_mirror + [half_gap, ridge, half_gap] + list(reversed(upper_mirror)))
    return solve_tm_stack(
        frequency,
        layers,
        period_um=float(geometry["period_um"]),
        port_index=float(material["port_index"]),
        harmonic_half_width=harmonics,
    )


def fourier_root(
    kind: str,
    values: np.ndarray,
    initial: complex,
    depth: float,
    geometry: dict[str, Any],
    material: dict[str, Any],
    harmonics: int,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    def response(omega: complex) -> complex:
        scattering = fourier_response(omega, values, geometry, material, harmonics)
        return scattering.transmission if kind == "pole" else scattering.reflection

    root = refine_scalar_root_complex_secant(
        response,
        initial,
        initial + (0.08 + 0.05j) * depth,
        kind=kind,
        tolerance=float(protocol["root_tolerance"]),
        max_evaluations=int(protocol["root_max_evaluations"]),
        search_radius=float(protocol["root_search_radius_in_base_depths"]) * depth,
    )
    valid = bool(
        np.isfinite(root.frequency)
        and root.residual_norm <= float(protocol["root_accept_residual"])
        and (kind != "pole" or root.frequency.imag < 0)
    )
    return {
        "verified": valid,
        "frequency": encode(root.frequency),
        "residual_norm": root.residual_norm,
        "success_flag": bool(root.success),
        "evaluations": root.evaluations,
        "message": root.message,
    }


def wilson(successes: int, total: int, z: float = 1.95996398454) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return [float(max(0, center - radius)), float(min(1, center + radius))]


def summarize(errors: list[float], protocol: dict[str, Any]) -> dict[str, Any]:
    array = np.asarray(errors, dtype=float)
    if len(array) == 0:
        return {"count": 0, "median": None, "p95": None, "maximum": None, "passed_error_gates": False}
    median = float(np.median(array))
    p95 = float(np.percentile(array, 95))
    return {
        "count": len(array),
        "median": median,
        "p95": p95,
        "maximum": float(np.max(array)),
        "passed_error_gates": bool(
            median <= float(protocol["prediction_median_error_gate_in_base_depths"])
            and p95 <= float(protocol["prediction_p95_error_gate_in_base_depths"])
        ),
    }


def run_case(
    source_case: dict[str, Any],
    geometry: dict[str, Any],
    material: dict[str, Any],
    protocol: dict[str, Any],
    scalar: Any,
    twoport: Any,
    normalized_draws: np.ndarray,
) -> dict[str, Any]:
    started = time.perf_counter()
    quality = float(source_case["target_quality_factor"])
    values = np.asarray([
        *source_case["optimized_coordinates"],
        geometry["ridge_fill_factor"],
        geometry["ridge_height_um"],
        material["high_index"],
    ], dtype=float)
    fdfd_pole = decode(source_case["verification"]["verified_pole"])
    depth = abs(fdfd_pole.imag)
    pole_gradient, pole_null, base_epsilon = _adjoint_pole_gradient(
        values, fdfd_pole, int(source_case["pole_parity_sign"]),
        geometry, material, scalar, twoport,
        {"fill_derivative_step": protocol["fill_mapping_derivative_step"],
         "height_derivative_step_um": protocol["height_mapping_derivative_step_um"],
         "frequency_derivative_step_in_pole_depths": protocol["frequency_derivative_step_in_pole_depths"]},
    )
    fdfd_zero_result = _root(
        "reflection", base_epsilon, scalar, complex(fdfd_pole.real, 0), depth,
        {"root_tolerance": protocol["root_tolerance"], "root_max_evaluations": protocol["root_max_evaluations"]},
    )
    if not fdfd_zero_result["verified"]:
        raise RuntimeError(f"Q={quality} FDFD reflection zero not verified")
    fdfd_zero = decode(fdfd_zero_result["frequency"])
    zero_gradient, zero_null, mapping_error = fdfd_zero_gradient(
        values, fdfd_zero, geometry, material, scalar, twoport, protocol, depth
    )
    if max(abs(pole_null), zero_null) > 1e-5 or mapping_error > 0.02:
        raise RuntimeError(f"Q={quality} baseline adjoint or mapping audit failed")

    orders = [int(value) for value in protocol["fourier_harmonic_half_widths"]]
    baselines = []
    for harmonics in orders:
        pole_root = fourier_root("pole", values, fdfd_pole, depth, geometry, material, harmonics, protocol)
        zero_root = fourier_root("zero", values, fdfd_zero, depth, geometry, material, harmonics, protocol)
        if not pole_root["verified"] or not zero_root["verified"]:
            raise RuntimeError(f"Q={quality} M={harmonics} baseline channel root failed")
        baselines.append({"harmonic_half_width": harmonics, "pole": pole_root, "reflection_zero": zero_root})
    baseline_pole = decode(baselines[-1]["pole"]["frequency"])
    baseline_zero = decode(baselines[-1]["reflection_zero"]["frequency"])
    harmonic_motion = {
        name: float(abs(decode(baselines[-1][name]["frequency"]) - decode(baselines[-2][name]["frequency"])) / depth)
        for name in ("pole", "reflection_zero")
    }
    if max(harmonic_motion.values()) > float(protocol["harmonic_root_motion_gate_in_base_depths"]):
        raise RuntimeError(f"Q={quality} Fourier-order root motion failed: {harmonic_motion}")

    sigma = protocol["independent_one_sigma_perturbations"]
    scales = np.asarray([
        values[0] * sigma["mirror_scale_relative"],
        values[1] * sigma["cavity_scale_relative"],
        sigma["ridge_fill_absolute"],
        sigma["ridge_height_um_absolute"],
        values[4] * sigma["high_index_relative"],
    ], dtype=float)
    samples = []
    for index, draw in enumerate(normalized_draws):
        delta = draw * scales
        shifted = values + delta
        pole_prediction = baseline_pole + np.dot(pole_gradient, delta)
        zero_prediction = baseline_zero + np.dot(zero_gradient, delta)
        pole = fourier_root("pole", shifted, pole_prediction, depth, geometry, material, orders[-1], protocol)
        zero = fourier_root("zero", shifted, zero_prediction, depth, geometry, material, orders[-1], protocol)
        response = fourier_response(baseline_pole.real, shifted, geometry, material, orders[-1])
        sample = {
            "index": index,
            "normalized_draw": draw.tolist(),
            "physical_delta": delta.tolist(),
            "pole_prediction": encode(pole_prediction),
            "zero_prediction": encode(zero_prediction),
            "actual_pole": pole,
            "actual_reflection_zero": zero,
            "transmittance_at_unperturbed_pole_real": float(abs(response.transmission) ** 2),
            "reflectance_at_unperturbed_pole_real": float(abs(response.reflection) ** 2),
        }
        if pole["verified"]:
            sample["pole_prediction_error_in_base_depths"] = float(abs(decode(pole["frequency"]) - pole_prediction) / depth)
        if zero["verified"]:
            sample["zero_prediction_error_in_base_depths"] = float(abs(decode(zero["frequency"]) - zero_prediction) / depth)
        samples.append(sample)
        if (index + 1) % 12 == 0:
            print(f"Q={quality:.0f}: {index + 1}/{len(normalized_draws)} held-out perturbations", flush=True)

    pole_errors = [item["pole_prediction_error_in_base_depths"] for item in samples if "pole_prediction_error_in_base_depths" in item]
    zero_errors = [item["zero_prediction_error_in_base_depths"] for item in samples if "zero_prediction_error_in_base_depths" in item]
    yield_count = sum(item["transmittance_at_unperturbed_pole_real"] >= float(protocol["illustrative_yield_center_transmittance_floor"]) for item in samples)
    return {
        "target_quality_factor": quality,
        "base_parameters": dict(zip(PARAMETERS, values.tolist(), strict=True)),
        "fdfd_pole": encode(fdfd_pole),
        "fdfd_reflection_zero": fdfd_zero_result,
        "fdfd_parity_null_abs_residual": float(abs(pole_null)),
        "fdfd_reflection_null_abs_residual": zero_null,
        "mapping_step_halving_relative_error": mapping_error,
        "fdfd_adjoint_pole_slopes": dict(zip(PARAMETERS, map(encode, pole_gradient), strict=True)),
        "fdfd_adjoint_reflection_zero_slopes": dict(zip(PARAMETERS, map(encode, zero_gradient), strict=True)),
        "fourier_baselines": baselines,
        "harmonic_root_motion_in_base_depths": harmonic_motion,
        "pole_prediction_error": summarize(pole_errors, protocol),
        "zero_prediction_error": summarize(zero_errors, protocol),
        "root_verification_complete": bool(len(pole_errors) == len(samples) and len(zero_errors) == len(samples)),
        "illustrative_center_transmittance_yield": {
            "passes": yield_count,
            "total": len(samples),
            "fraction": yield_count / len(samples),
            "wilson_95_interval": wilson(yield_count, len(samples)),
        },
        "samples": samples,
        "wall_seconds": time.perf_counter() - started,
    }


def plot(cases: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.3, 6.0), constrained_layout=True)
    for column, case in enumerate(cases):
        samples = case["samples"]
        depth = abs(decode(case["fdfd_pole"]).imag)
        baseline_p = decode(case["fourier_baselines"][-1]["pole"]["frequency"])
        baseline_z = decode(case["fourier_baselines"][-1]["reflection_zero"]["frequency"])
        for row, (predicted_key, actual_key, baseline) in enumerate((
            ("pole_prediction", "actual_pole", baseline_p),
            ("zero_prediction", "actual_reflection_zero", baseline_z),
        )):
            valid = [sample for sample in samples if sample[actual_key]["verified"]]
            predicted = np.asarray([(decode(item[predicted_key]) - baseline) / depth for item in valid])
            actual = np.asarray([(decode(item[actual_key]["frequency"]) - baseline) / depth for item in valid])
            axis = axes[row, column]
            axis.scatter(predicted.real, actual.real, s=14, alpha=0.65, label="Real-frequency motion")
            axis.scatter(predicted.imag, actual.imag, s=14, marker="x", alpha=0.65, label="Imaginary-frequency motion")
            limit = max(0.2, np.max(abs(np.r_[predicted.real, predicted.imag, actual.real, actual.imag])) * 1.08)
            axis.plot([-limit, limit], [-limit, limit], "k--", linewidth=0.8, label="Exact prediction")
            axis.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
            axis.grid(True, alpha=0.25, linewidth=0.5)
            if row == 0:
                axis.set_title(f"Q={case['target_quality_factor']:.0f}")
            if column == 0:
                axis.set_ylabel(("Pole" if row == 0 else "Reflection zero") + " actual shift / pole depth")
            if row == 1:
                axis.set_xlabel("Adjoint-predicted shift / pole depth")
    axes[0, 2].legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(output / "m157_heldout_pole_zero_motion.png", dpi=240)
    fig.savefig(output / "m157_heldout_pole_zero_motion.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m157_single_resonance_robustness.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/m157_single_resonance_robustness"))
    parser.add_argument("--only-quality", type=float)
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
    revision = subprocess.run(("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=False)
    metadata = {
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "command": sys.argv,
        "source_sha256": {str(path): sha256(path) for path in (args.config, geometry_path, material_path, source_path)},
        "code_sha256": {str(path): sha256(path) for path in (Path(__file__), Path("src/pzid/rcwa2d.py"), Path("src/pzid/adjoint.py"))},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    rng = np.random.default_rng(int(protocol["random_seed"]))
    draws = np.clip(rng.standard_normal((int(protocol["samples_per_quality"]), len(PARAMETERS))), -float(protocol["normal_draw_clip_sigma"]), float(protocol["normal_draw_clip_sigma"]))
    np.savetxt(output / "normalized_draws.csv", draws, delimiter=",", header=",".join(PARAMETERS), comments="")
    _, scalar, twoport = build_fixed_grid(geometry, material)
    cases = []
    for quality in protocol["target_quality_factors"]:
        if args.only_quality is not None and float(quality) != args.only_quality:
            continue
        source_case = next(item for item in source["cases"] if item["target_quality_factor"] == quality)
        result = run_case(source_case, geometry, material, protocol, scalar, twoport, draws)
        cases.append(result)
        (output / "summary.json").write_text(json.dumps({"method": "FDFD-adjoint root sensitivities; held-out Fourier-modal perturbed roots", "cases": cases}, indent=2) + "\n", encoding="utf-8")
        print(f"Q={quality:.0f}: pole median={result['pole_prediction_error']['median']:.4f}, zero median={result['zero_prediction_error']['median']:.4f} depths; root_complete={result['root_verification_complete']}", flush=True)
    if len(cases) == len(protocol["target_quality_factors"]):
        plot(cases, output)


if __name__ == "__main__":
    main()
