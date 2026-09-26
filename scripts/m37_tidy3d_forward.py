"""Independent Tidy3D FDTD replay of the M37 Huygens forward benchmark.

Tidy3D is intentionally an optional external validator rather than a project
dependency.  Run this entry point from an isolated environment containing the
authenticated Tidy3D client.  ``--estimate-only`` creates no solver result and
must be used before the two paid reference/structure tasks.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np


SPEED_OF_LIGHT_UM_PER_SECOND = 299_792_458.0 * 1e6


def frequency_grid(config: dict[str, Any]) -> np.ndarray:
    wavelengths = np.linspace(
        float(config["wavelength_max"]),
        float(config["wavelength_min"]),
        int(config["frequency_points"]),
    )
    return SPEED_OF_LIGHT_UM_PER_SECOND / wavelengths


def build_simulation(config: dict[str, Any], *, with_disk: bool) -> Any:
    import tidy3d as td

    frequencies = frequency_grid(config)
    center_frequency = 0.5 * (frequencies[0] + frequencies[-1])
    bandwidth = 1.2 * (frequencies[-1] - frequencies[0])
    period = float(config["period"])
    height = float(config["domain_height"])
    host = td.Medium(permittivity=float(config["host_index"]) ** 2)
    disk_geometry = td.Cylinder(
        center=(0.0, 0.0, 0.0),
        radius=float(config["disk_radius"]),
        length=float(config["disk_height"]),
        axis=2,
    )
    disk_medium = td.Medium(
        permittivity=float(config["disk_index"]) ** 2
    )
    disk_grid_template = td.Structure(
        geometry=disk_geometry,
        medium=disk_medium,
    )
    structures = []
    if with_disk:
        structures.append(
            td.Structure(
                geometry=disk_geometry,
                medium=disk_medium,
                name="silicon_nanodisk",
            )
        )
    source_z = -0.30 * height
    reflection_z = -0.40 * height
    transmission_z = 0.32 * height
    monitors = [
        td.FluxMonitor(
            center=(0.0, 0.0, reflection_z),
            size=(td.inf, td.inf, 0.0),
            freqs=frequencies,
            name="reflection_flux",
        ),
        td.FluxMonitor(
            center=(0.0, 0.0, transmission_z),
            size=(td.inf, td.inf, 0.0),
            freqs=frequencies,
            name="transmission_flux",
        ),
        td.FieldMonitor(
            center=(0.0, 0.0, transmission_z),
            size=(td.inf, td.inf, 0.0),
            freqs=frequencies,
            fields=("Ex",),
            colocate=True,
            name="transmission_field",
        ),
    ]
    if bool(config.get("record_complex_reflection", False)):
        monitors.append(
            td.FieldMonitor(
                center=(0.0, 0.0, reflection_z),
                size=(td.inf, td.inf, 0.0),
                freqs=frequencies,
                fields=("Ex",),
                colocate=True,
                name="reflection_field",
            )
        )
    grid_mode = str(config.get("grid_mode", "uniform"))
    if grid_mode == "uniform":
        grid_spec = td.GridSpec.uniform(
            dl=float(config["uniform_grid_dl"])
        )
    elif grid_mode == "auto_interface_aligned":
        half_disk_height = 0.5 * float(config["disk_height"])
        grid_spec = td.GridSpec.auto(
            wavelength=float(config["mesh_wavelength_um"]),
            min_steps_per_wvl=float(config["min_steps_per_wvl"]),
            max_scale=float(config.get("grid_max_scale", 1.4)),
            snapping_points=(
                (None, None, -half_disk_height),
                (None, None, half_disk_height),
            ),
            override_structures=(disk_grid_template,),
        )
    else:
        raise ValueError(f"unsupported grid_mode: {grid_mode}")
    return td.Simulation(
        center=(0.0, 0.0, 0.0),
        size=(period, period, height),
        medium=host,
        grid_spec=grid_spec,
        structures=structures,
        sources=[
            td.PlaneWave(
                center=(0.0, 0.0, source_z),
                size=(td.inf, td.inf, 0.0),
                source_time=td.GaussianPulse(
                    freq0=center_frequency,
                    fwidth=bandwidth,
                ),
                direction="+",
                pol_angle=0.0,
                name="x_polarized_plane_wave",
            )
        ],
        monitors=monitors,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary.periodic(),
            z=td.Boundary.pml(num_layers=int(config["pml_layers"])),
        ),
        run_time=float(config["run_time_seconds"]),
        shutoff=float(config.get("shutoff", 1e-8)),
    )


def _mean_transmission_field(data: Any) -> np.ndarray:
    field = data["transmission_field"].Ex
    for dimension in tuple(field.dims):
        if dimension != "f":
            field = field.mean(dim=dimension)
    return np.asarray(field.values)


def _mean_reflection_field(data: Any) -> np.ndarray:
    field = data["reflection_field"].Ex
    for dimension in tuple(field.dims):
        if dimension != "f":
            field = field.mean(dim=dimension)
    return np.asarray(field.values)


def _extrapolated_incident_field_at_reflection_plane(
    config: dict[str, Any], reference_data: Any
) -> np.ndarray:
    """Extrapolate the one-way incident field to the reflection plane.

    The reflection monitor is below the unidirectional source, where the
    reference simulation contains essentially no field.  Consequently that
    local reference field cannot normalize the reflected amplitude.  The
    homogeneous reference field above the source is instead propagated back
    to the reflection plane using the host-medium wavenumber.
    """

    frequencies = frequency_grid(config)
    height = float(config["domain_height"])
    reflection_z = -0.40 * height
    transmission_z = 0.32 * height
    wavenumber = (
        2.0
        * np.pi
        * float(config["host_index"])
        * frequencies
        / SPEED_OF_LIGHT_UM_PER_SECOND
    )
    propagation = np.exp(1j * wavenumber * (reflection_z - transmission_z))
    return _mean_transmission_field(reference_data) * propagation


def _mesh_variant_config(
    config: dict[str, Any], variant: dict[str, Any]
) -> dict[str, Any]:
    result = deepcopy(config)
    for key, value in variant.items():
        if key != "name":
            result[key] = value
    result.pop("mesh_variants", None)
    result.pop("convergence_window_um", None)
    result.pop("gate", None)
    return result


def _mesh_record(variant: dict[str, Any]) -> dict[str, float]:
    if "uniform_grid_dl" in variant:
        return {"uniform_grid_dl_um": float(variant["uniform_grid_dl"])}
    return {"minimum_steps_per_wavelength": float(variant["min_steps_per_wvl"])}


def _mesh_label(variant: dict[str, Any]) -> str:
    if "uniform_grid_dl" in variant:
        return f"{1e3 * float(variant['uniform_grid_dl']):g} nm"
    return f"{float(variant['min_steps_per_wvl']):g} steps/λ"


def build_convergence_simulations(config: dict[str, Any]) -> dict[str, Any]:
    simulations = {}
    prefix = str(config["task_prefix"])
    for variant in config["mesh_variants"]:
        variant_config = _mesh_variant_config(config, variant)
        name = str(variant["name"])
        simulations[f"{prefix}_{name}_reference"] = build_simulation(
            variant_config, with_disk=False
        )
        simulations[f"{prefix}_{name}_structure"] = build_simulation(
            variant_config, with_disk=True
        )
    return simulations


def audit_convergence_batch(config: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    simulations = build_convergence_simulations(config)
    records = {}
    for name, simulation in simulations.items():
        simulation.validate_pre_upload()
        simulation.to_file(output / f"{name}_simulation.hdf5")
        sizes = simulation.grid.sizes
        records[name] = {
            "cells": [len(sizes.x), len(sizes.y), len(sizes.z)],
            "total_cells": int(len(sizes.x) * len(sizes.y) * len(sizes.z)),
            "minimum_dl_um": [
                float(np.min(sizes.x)),
                float(np.min(sizes.y)),
                float(np.min(sizes.z)),
            ],
            "maximum_dl_um": [
                float(np.max(sizes.x)),
                float(np.max(sizes.y)),
                float(np.max(sizes.z)),
            ],
        }
    prefix = str(config["task_prefix"])
    for variant in config["mesh_variants"]:
        name = str(variant["name"])
        reference_grid = simulations[f"{prefix}_{name}_reference"].grid
        structure_grid = simulations[f"{prefix}_{name}_structure"].grid
        for axis in ("x", "y", "z"):
            np.testing.assert_allclose(
                getattr(reference_grid.boundaries, axis),
                getattr(structure_grid.boundaries, axis),
                rtol=0.0,
                atol=1e-15,
            )
        records[f"{prefix}_{name}_grid_pair"] = {"identical": True}
    _write_json(output / "mesh_audit.json", records)

    finest_name = str(config["mesh_variants"][-1]["name"])
    simulation = simulations[f"{prefix}_{finest_name}_structure"]
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    simulation.plot(x=0.0, ax=axes[0])
    simulation.plot_grid(x=0.0, ax=axes[0])
    simulation.plot(y=0.0, ax=axes[1])
    simulation.plot_grid(y=0.0, ax=axes[1])
    simulation.plot(z=0.0, ax=axes[2])
    simulation.plot_grid(z=0.0, ax=axes[2])
    axes[0].set_title("x = 0")
    axes[1].set_title("y = 0")
    axes[2].set_title("z = 0")
    figure.tight_layout()
    figure.savefig(output / "geometry_mesh_audit.png", dpi=200)
    plt.close(figure)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare_convergence_batch(
    config: dict[str, Any], output: Path
) -> Any:
    import tidy3d.web as web

    batch_path = output / "batch.hdf5"
    if batch_path.is_file():
        return web.Batch.from_file(batch_path)
    simulations = build_convergence_simulations(config)
    for name, simulation in simulations.items():
        simulation.validate_pre_upload()
        simulation.to_file(output / f"{name}_simulation.hdf5")
    batch = web.Batch(simulations=simulations, verbose=False)
    batch.upload()
    batch.to_file(batch_path)
    _write_json(
        output / "task_ids.json",
        {name: job.task_id for name, job in batch.jobs.items()},
    )
    return batch


def estimate_convergence_batch(config: dict[str, Any], output: Path) -> None:
    import tidy3d as td
    import tidy3d.web as web

    batch = prepare_convergence_batch(config, output)
    per_task = {
        name: float(web.estimate_cost(job.task_id, verbose=False))
        for name, job in batch.jobs.items()
    }
    estimate = {
        "tidy3d_version": td.__version__,
        "per_task_flexcredits": per_task,
        "total_flexcredits": float(batch.estimate_cost(verbose=False)),
    }
    _write_json(output / "estimate.json", estimate)
    print(json.dumps(estimate, indent=2))


def _normalized_response(
    *,
    config: dict[str, Any],
    reference_data: Any,
    structure_data: Any,
) -> dict[str, np.ndarray]:
    reference_transmission = np.asarray(
        reference_data["transmission_flux"].flux.values
    )
    reference_reflection = np.asarray(
        reference_data["reflection_flux"].flux.values
    )
    structure_transmission = np.asarray(
        structure_data["transmission_flux"].flux.values
    )
    structure_reflection = np.asarray(
        structure_data["reflection_flux"].flux.values
    )
    transmission = structure_transmission / reference_transmission
    reflection = -(
        structure_reflection - reference_reflection
    ) / reference_transmission
    incident_reflection_plane = _extrapolated_incident_field_at_reflection_plane(
        config, reference_data
    )
    reference_reflection_plane = _mean_reflection_field(reference_data)
    reflection_complex = (
        _mean_reflection_field(structure_data) - reference_reflection_plane
    ) / incident_reflection_plane
    transmission_complex = _mean_transmission_field(
        structure_data
    ) / _mean_transmission_field(reference_data)
    frequencies = frequency_grid(config)
    return {
        "frequency_hz": frequencies,
        "wavelength_um": SPEED_OF_LIGHT_UM_PER_SECOND / frequencies,
        "transmittance": transmission,
        "reflectance": reflection,
        "transmission_complex": transmission_complex,
        "reflection_complex": reflection_complex,
        "reference_transmission_flux": reference_transmission,
        "reference_reflection_flux": reference_reflection,
    }


def run_convergence_batch(config: dict[str, Any], output: Path) -> None:
    import tidy3d as td
    import tidy3d.web as web

    batch = prepare_convergence_batch(config, output)
    batch_data = batch.run(path_dir=output / "data", replace_existing=False)
    prefix = str(config["task_prefix"])
    summaries = {}
    for variant in config["mesh_variants"]:
        name = str(variant["name"])
        variant_config = _mesh_variant_config(config, variant)
        response = _normalized_response(
            config=variant_config,
            reference_data=batch_data[f"{prefix}_{name}_reference"],
            structure_data=batch_data[f"{prefix}_{name}_structure"],
        )
        np.savez_compressed(output / f"response_{name}.npz", **response)
        phase_span = float(
            np.ptp(np.unwrap(np.angle(response["transmission_complex"])))
            / np.pi
        )
        summaries[name] = {
            **_mesh_record(variant),
            "minimum_reflectance": float(np.min(response["reflectance"])),
            "maximum_reflectance": float(np.max(response["reflectance"])),
            "minimum_transmittance": float(np.min(response["transmittance"])),
            "maximum_transmittance": float(np.max(response["transmittance"])),
            "maximum_power_defect": float(
                np.max(
                    np.abs(
                        response["transmittance"]
                        + response["reflectance"]
                        - 1.0
                    )
                )
            ),
            "transmission_phase_span_pi": phase_span,
            "maximum_complex_transmission_power_mismatch": float(
                np.max(
                    np.abs(
                        np.abs(response["transmission_complex"]) ** 2
                        - response["transmittance"]
                    )
                )
            ),
            "maximum_complex_reflection_power_mismatch": float(
                np.max(
                    np.abs(
                        np.abs(response["reflection_complex"]) ** 2
                        - response["reflectance"]
                    )
                )
            ),
        }
    _write_json(output / "run_summary.json", summaries)
    cloud_metadata = {"tidy3d_version": td.__version__, "tasks": {}}
    for name, job in batch.jobs.items():
        info = web.get_info(job.task_id, verbose=False)
        progress, field_decay = web.get_run_info(job.task_id)
        cloud_metadata["tasks"][name] = {
            "task_id": job.task_id,
            "task_name": info.taskName,
            "status": info.status,
            "estimated_maximum_flexcredits": info.estFlexUnit,
            "billed_flexcredits": web.real_cost(
                job.task_id, verbose=False
            ),
            "time_steps": info.timeSteps,
            "solver_version": info.solverVersion,
            "solver_time_metadata": info.totalSolverTime,
            "time_stepping_percentage": progress,
            "final_field_decay": field_decay,
        }
    _write_json(output / "cloud_task_metadata.json", cloud_metadata)


def _quadratic_minimum(
    wavelength: np.ndarray,
    values: np.ndarray,
    window: tuple[float, float],
) -> tuple[float, float]:
    selected = np.flatnonzero(
        (wavelength >= window[0]) & (wavelength <= window[1])
    )
    if selected.size < 3:
        raise ValueError("convergence window requires at least three samples")
    index = int(selected[np.argmin(values[selected])])
    if index == 0 or index == wavelength.size - 1:
        return float(wavelength[index]), float(values[index])
    neighbors = slice(index - 1, index + 2)
    coefficients = np.polyfit(wavelength[neighbors], values[neighbors], 2)
    if coefficients[0] <= 0.0:
        return float(wavelength[index]), float(values[index])
    location = float(-coefficients[1] / (2.0 * coefficients[0]))
    if not wavelength[index - 1] <= location <= wavelength[index + 1]:
        return float(wavelength[index]), float(values[index])
    return location, float(np.polyval(coefficients, location))


def analyze_convergence_batch(config: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    responses: dict[str, dict[str, np.ndarray]] = {}
    records: dict[str, dict[str, float]] = {}
    window = tuple(map(float, config["convergence_window_um"]))
    for variant in config["mesh_variants"]:
        name = str(variant["name"])
        with np.load(output / f"response_{name}.npz") as loaded:
            order = np.argsort(loaded["wavelength_um"])
            response = {
                key: np.asarray(loaded[key])[order]
                for key in (
                    "wavelength_um",
                    "transmittance",
                    "reflectance",
                    "transmission_complex",
                    "reflection_complex",
                )
            }
        responses[name] = response
        zero_wavelength, zero_reflectance = _quadratic_minimum(
            response["wavelength_um"], response["reflectance"], window
        )
        records[name] = {
            **_mesh_record(variant),
            "reflection_zero_wavelength_um": zero_wavelength,
            "reflection_zero_reflectance": zero_reflectance,
            "reflection_zero_transmittance": float(
                np.interp(
                    zero_wavelength,
                    response["wavelength_um"],
                    response["transmittance"],
                )
            ),
            "transmission_phase_span_pi": float(
                np.ptp(np.unwrap(np.angle(response["transmission_complex"])))
                / np.pi
            ),
            "maximum_power_defect": float(
                np.max(
                    np.abs(
                        response["transmittance"]
                        + response["reflectance"]
                        - 1.0
                    )
                )
            ),
            "maximum_complex_transmission_power_mismatch": float(
                np.max(
                    np.abs(
                        np.abs(response["transmission_complex"]) ** 2
                        - response["transmittance"]
                    )
                )
            ),
            "maximum_complex_reflection_power_mismatch": float(
                np.max(
                    np.abs(
                        np.abs(response["reflection_complex"]) ** 2
                        - response["reflectance"]
                    )
                )
            ),
        }

    coarse_name, fine_name = map(str, config["gate"]["finest_pair"])
    coarse = responses[coarse_name]
    fine = responses[fine_name]
    np.testing.assert_allclose(
        coarse["wavelength_um"], fine["wavelength_um"], atol=1e-12, rtol=0.0
    )
    power_difference = float(
        max(
            np.max(np.abs(fine["transmittance"] - coarse["transmittance"])),
            np.max(np.abs(fine["reflectance"] - coarse["reflectance"])),
        )
    )
    zero_shift = abs(
        records[fine_name]["reflection_zero_wavelength_um"]
        - records[coarse_name]["reflection_zero_wavelength_um"]
    )
    phase_span_difference = abs(
        records[fine_name]["transmission_phase_span_pi"]
        - records[coarse_name]["transmission_phase_span_pi"]
    )
    gate = config["gate"]
    passed = bool(
        zero_shift <= float(gate["maximum_reflection_zero_shift_um"])
        and power_difference <= float(gate["maximum_power_difference"])
        and phase_span_difference
        <= float(gate["maximum_phase_span_difference_pi"])
    )
    summary = {
        "passed": passed,
        "variants": records,
        "finest_pair": [coarse_name, fine_name],
        "finest_reflection_zero_shift_um": zero_shift,
        "finest_maximum_power_difference": power_difference,
        "finest_phase_span_difference_pi": phase_span_difference,
        "gate": gate,
    }
    _write_json(output / "convergence_summary.json", summary)

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.5))
    for variant in config["mesh_variants"]:
        name = str(variant["name"])
        response = responses[name]
        label = _mesh_label(variant)
        wavelength = response["wavelength_um"]
        axes[0, 0].plot(wavelength, response["transmittance"], label=f"T {label}")
        axes[0, 0].plot(
            wavelength, response["reflectance"], "--", label=f"R {label}"
        )
        phase = np.unwrap(np.angle(response["transmission_complex"])) / np.pi
        axes[0, 1].plot(wavelength, phase, label=label)
        axes[1, 0].plot(
            response["transmission_complex"].real,
            response["transmission_complex"].imag,
            label=label,
        )
        if name != fine_name:
            axes[1, 1].plot(
                wavelength,
                np.abs(
                    response["transmittance"]
                    - responses[fine_name]["transmittance"]
                ),
                label=rf"$|\Delta T|$ {label}",
            )
            axes[1, 1].plot(
                wavelength,
                np.abs(
                    response["reflectance"]
                    - responses[fine_name]["reflectance"]
                ),
                "--",
                label=rf"$|\Delta R|$ {label}",
            )
    axes[0, 0].set_ylabel("Power")
    axes[0, 0].set_xlabel("Wavelength (µm)")
    axes[0, 0].set_title("Spectral convergence")
    axes[0, 1].set_ylabel(r"Unwrapped $\arg(t)/\pi$")
    axes[0, 1].set_xlabel("Wavelength (µm)")
    axes[0, 1].set_title("Transmission phase")
    axes[1, 0].set_xlabel(r"Re $t$")
    axes[1, 0].set_ylabel(r"Im $t$")
    axes[1, 0].set_title("Transmission complex plane")
    axes[1, 1].set_xlabel("Wavelength (µm)")
    axes[1, 1].set_ylabel("Absolute power difference")
    axes[1, 1].set_title(f"Mesh error relative to {fine_name}")
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[1, 0].set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    figure.savefig(output / "convergence.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for variant in config["mesh_variants"]:
        name = str(variant["name"])
        response = responses[name]
        label = _mesh_label(variant)
        axes[0].plot(
            response["transmission_complex"].real,
            response["transmission_complex"].imag,
            label=label,
        )
        axes[1].plot(
            response["reflection_complex"].real,
            response["reflection_complex"].imag,
            label=label,
        )
    axes[0].set_xlabel(r"Re $t$")
    axes[0].set_ylabel(r"Im $t$")
    axes[0].set_title("Transmission complex plane")
    axes[1].set_xlabel(r"Re $r$")
    axes[1].set_ylabel(r"Im $r$")
    axes[1].set_title("Reflection complex plane")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    figure.savefig(output / "complex_planes.png", dpi=200)
    plt.close(figure)


def _local_extrema(
    wavelength: np.ndarray,
    values: np.ndarray,
    *,
    minimum: bool,
) -> list[dict[str, float]]:
    center = values[1:-1]
    if minimum:
        selected = (center < values[:-2]) & (center < values[2:])
    else:
        selected = (center > values[:-2]) & (center > values[2:])
    indices = np.flatnonzero(selected) + 1
    return [
        {"wavelength_um": float(wavelength[index]), "value": float(values[index])}
        for index in indices
    ]


def analyze_independent_validation(
    *,
    tidy3d_response_path: Path,
    fdfd_scan_path: Path,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tidy3d_response = np.load(tidy3d_response_path)
    fdfd_response = np.load(fdfd_scan_path)
    tidy_order = np.argsort(tidy3d_response["wavelength_um"])
    fdfd_order = np.argsort(fdfd_response["wavelength"])
    wavelength = np.asarray(fdfd_response["wavelength"])[fdfd_order]
    tidy_wavelength = np.asarray(tidy3d_response["wavelength_um"])[tidy_order]
    if not np.allclose(wavelength, tidy_wavelength, atol=1e-12, rtol=0.0):
        raise ValueError("Tidy3D and FDFD wavelength grids differ")

    tidy_transmittance = np.asarray(tidy3d_response["transmittance"])[tidy_order]
    tidy_reflectance = np.asarray(tidy3d_response["reflectance"])[tidy_order]
    tidy_transmission = np.asarray(
        tidy3d_response["transmission_complex"]
    )[tidy_order]
    fdfd_transmission = np.asarray(fdfd_response["transmission"])[fdfd_order]
    fdfd_reflection = np.asarray(fdfd_response["reflection"])[fdfd_order]
    fdfd_transmittance = np.abs(fdfd_transmission) ** 2
    fdfd_reflectance = np.abs(fdfd_reflection) ** 2

    fdfd_phase = np.unwrap(np.angle(fdfd_transmission))
    tidy_phase = np.unwrap(np.angle(tidy_transmission))
    phase_candidates: list[tuple[float, float, float, np.ndarray]] = []
    for sign in (1.0, -1.0):
        signed_phase = sign * tidy_phase
        offset = float(np.mean(fdfd_phase - signed_phase))
        aligned = signed_phase + offset
        rms = float(np.sqrt(np.mean((aligned - fdfd_phase) ** 2)))
        phase_candidates.append((rms, sign, offset, aligned))
    _, phase_sign, phase_offset, tidy_phase_aligned = min(
        phase_candidates, key=lambda item: item[0]
    )
    phase_error_pi = (tidy_phase_aligned - fdfd_phase) / np.pi

    delta_transmittance = tidy_transmittance - fdfd_transmittance
    delta_reflectance = tidy_reflectance - fdfd_reflectance
    fdfd_peak_index = int(np.argmax(fdfd_reflectance))
    tidy_peak_index = int(np.argmax(tidy_reflectance))
    metrics = {
        "tidy3d_response": str(tidy3d_response_path),
        "fdfd_response": str(fdfd_scan_path),
        "wavelength_points": int(wavelength.size),
        "wavelength_range_um": [float(wavelength[0]), float(wavelength[-1])],
        "transmittance_rms_difference": float(
            np.sqrt(np.mean(delta_transmittance**2))
        ),
        "transmittance_maximum_absolute_difference": float(
            np.max(np.abs(delta_transmittance))
        ),
        "reflectance_rms_difference": float(
            np.sqrt(np.mean(delta_reflectance**2))
        ),
        "reflectance_maximum_absolute_difference": float(
            np.max(np.abs(delta_reflectance))
        ),
        "tidy3d_maximum_power_defect": float(
            np.max(np.abs(tidy_transmittance + tidy_reflectance - 1.0))
        ),
        "fdfd_maximum_power_defect": float(
            np.max(np.abs(fdfd_transmittance + fdfd_reflectance - 1.0))
        ),
        "phase_convention_sign": phase_sign,
        "phase_constant_offset_pi": phase_offset / np.pi,
        "phase_rms_difference_pi": float(
            np.sqrt(np.mean(phase_error_pi**2))
        ),
        "phase_maximum_absolute_difference_pi": float(
            np.max(np.abs(phase_error_pi))
        ),
        "fdfd_phase_span_pi": float(np.ptp(fdfd_phase) / np.pi),
        "tidy3d_phase_span_pi": float(np.ptp(tidy_phase) / np.pi),
        "fdfd_strongest_reflection_peak": {
            "wavelength_um": float(wavelength[fdfd_peak_index]),
            "reflectance": float(fdfd_reflectance[fdfd_peak_index]),
        },
        "tidy3d_strongest_reflection_peak": {
            "wavelength_um": float(wavelength[tidy_peak_index]),
            "reflectance": float(tidy_reflectance[tidy_peak_index]),
        },
        "fdfd_local_reflection_minima": _local_extrema(
            wavelength, fdfd_reflectance, minimum=True
        ),
        "tidy3d_local_reflection_minima": _local_extrema(
            wavelength, tidy_reflectance, minimum=True
        ),
    }
    (output / "independent_comparison.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 10.0), sharex=True)
    axes[0].plot(wavelength, fdfd_transmittance, label="FDFD T", color="C0")
    axes[0].plot(
        wavelength,
        tidy_transmittance,
        "--",
        label="Tidy3D T",
        color="C0",
    )
    axes[0].plot(wavelength, fdfd_reflectance, label="FDFD R", color="C1")
    axes[0].plot(
        wavelength,
        tidy_reflectance,
        "--",
        label="Tidy3D R",
        color="C1",
    )
    axes[0].set_ylabel("Power")
    axes[0].set_title("Independent FDFD–Tidy3D validation")
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=0.25)

    axes[1].plot(wavelength, fdfd_phase / np.pi, label="FDFD")
    axes[1].plot(
        wavelength,
        tidy_phase_aligned / np.pi,
        "--",
        label="Tidy3D (constant phase aligned)",
    )
    axes[1].set_ylabel(r"Unwrapped $\arg(t)/\pi$")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(wavelength, delta_transmittance, label=r"$\Delta T$")
    axes[2].plot(wavelength, delta_reflectance, label=r"$\Delta R$")
    axes[2].plot(wavelength, phase_error_pi, label=r"$\Delta\arg(t)/\pi$")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Wavelength (µm)")
    axes[2].set_ylabel("Tidy3D − FDFD")
    axes[2].legend(ncol=3)
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "independent_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--batch-estimate", action="store_true")
    parser.add_argument("--batch-audit", action="store_true")
    parser.add_argument("--batch-run", action="store_true")
    parser.add_argument("--batch-analyze", action="store_true")
    parser.add_argument("--fdfd-scan", type=Path)
    arguments = parser.parse_args()
    if sum(
        (
            arguments.estimate_only,
            arguments.run,
            arguments.analyze_only,
            arguments.batch_estimate,
            arguments.batch_audit,
            arguments.batch_run,
            arguments.batch_analyze,
        )
    ) != 1:
        raise ValueError(
            "select exactly one execution mode"
        )
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    if arguments.analyze_only and not arguments.output.exists():
        raise FileNotFoundError(arguments.output)
    if arguments.output.exists():
        saved = json.loads(
            (arguments.output / "config.json").read_text(encoding="utf-8")
        )
        if saved != config:
            raise ValueError("run config differs from the uploaded estimate config")
    else:
        arguments.output.mkdir(parents=True)
        (arguments.output / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
    if arguments.batch_estimate:
        estimate_convergence_batch(config, arguments.output)
        return
    if arguments.batch_audit:
        audit_convergence_batch(config, arguments.output)
        return
    if arguments.batch_run:
        run_convergence_batch(config, arguments.output)
        return
    if arguments.batch_analyze:
        analyze_convergence_batch(config, arguments.output)
        return
    if arguments.analyze_only:
        if arguments.fdfd_scan is None:
            raise ValueError("--fdfd-scan is required with --analyze-only")
        analyze_independent_validation(
            tidy3d_response_path=arguments.output / "tidy3d_response.npz",
            fdfd_scan_path=arguments.fdfd_scan,
            output=arguments.output,
        )
        return
    reference = build_simulation(config, with_disk=False)
    structure = build_simulation(config, with_disk=True)
    reference.to_file(arguments.output / "reference_simulation.hdf5")
    structure.to_file(arguments.output / "structure_simulation.hdf5")

    import tidy3d as td
    import tidy3d.web as web

    prefix = str(config["task_prefix"])
    task_path = arguments.output / "task_ids.json"
    if task_path.is_file():
        task_ids = json.loads(task_path.read_text(encoding="utf-8"))
    else:
        task_ids = {}
    for name, simulation in (
        ("reference", reference),
        ("structure", structure),
    ):
        if name in task_ids:
            continue
        task_ids[name] = web.upload(
            simulation,
            task_name=f"{prefix}-{name}",
            verbose=False,
        )
        task_path.write_text(
            json.dumps(task_ids, indent=2) + "\n", encoding="utf-8"
        )
    if arguments.estimate_only:
        estimates = {
            "tidy3d_version": td.__version__,
            "reference_flexcredits": float(
                web.estimate_cost(task_ids["reference"], verbose=False)
            ),
            "structure_flexcredits": float(
                web.estimate_cost(task_ids["structure"], verbose=False)
            ),
        }
        estimates["total_flexcredits"] = (
            estimates["reference_flexcredits"]
            + estimates["structure_flexcredits"]
        )
        (arguments.output / "estimate.json").write_text(
            json.dumps(estimates, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(estimates, indent=2))
        return

    web.start(task_ids["reference"])
    web.start(task_ids["structure"])
    web.monitor(task_ids["reference"])
    web.monitor(task_ids["structure"])
    reference_data = web.load(
        task_ids["reference"],
        path=str(arguments.output / "reference_data.hdf5"),
    )
    structure_data = web.load(
        task_ids["structure"],
        path=str(arguments.output / "structure_data.hdf5"),
    )
    reference_transmission = np.asarray(
        reference_data["transmission_flux"].flux.values
    )
    reference_reflection = np.asarray(
        reference_data["reflection_flux"].flux.values
    )
    structure_transmission = np.asarray(
        structure_data["transmission_flux"].flux.values
    )
    structure_reflection = np.asarray(
        structure_data["reflection_flux"].flux.values
    )
    transmission = structure_transmission / reference_transmission
    reflection = -(
        structure_reflection - reference_reflection
    ) / reference_transmission
    t_complex = _mean_transmission_field(structure_data) / _mean_transmission_field(
        reference_data
    )
    frequencies = frequency_grid(config)
    wavelengths = SPEED_OF_LIGHT_UM_PER_SECOND / frequencies
    np.savez_compressed(
        arguments.output / "tidy3d_response.npz",
        frequency_hz=frequencies,
        wavelength_um=wavelengths,
        transmittance=transmission,
        reflectance=reflection,
        transmission_complex=t_complex,
        reference_transmission_flux=reference_transmission,
        reference_reflection_flux=reference_reflection,
    )
    summary = {
        "tidy3d_version": td.__version__,
        "minimum_reflectance": float(np.min(reflection)),
        "maximum_reflectance": float(np.max(reflection)),
        "minimum_transmittance": float(np.min(transmission)),
        "maximum_transmittance": float(np.max(transmission)),
        "transmission_phase_span_pi": float(
            np.ptp(np.unwrap(np.angle(t_complex))) / np.pi
        ),
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
