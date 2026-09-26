"""Stage-1 forward benchmark for a symmetric embedded Huygens nanodisk.

The output directory is immutable: the script refuses to overwrite an
existing run.  It records the input, Git revision, full four-port scattering
matrices, mesh diagnostics, structure slices, and real-axis response figures.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from pzid.grid3d import UniformGrid3D
from pzid.huygens3d import (
    EmbeddedNanodisk3D,
    SmoothNanodiskParameterization3D,
    axial_eigenchannels,
    axial_symmetry_defect,
    copolarized_coefficients,
)
from pzid.plotting3d import plot_refractive_index_slices_3d
from pzid.pml3d import AxialPML3D
from pzid.scattering3d import (
    ScatteringSetup3D,
    deembed_scattering_matrix_3d,
    solve_scattering_3d,
)


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def build_benchmark(
    config: dict[str, Any],
) -> tuple[ScatteringSetup3D, EmbeddedNanodisk3D, np.ndarray]:
    mesh = config["mesh"]
    grid = UniformGrid3D(
        nx=int(mesh["nx"]),
        ny=int(mesh["ny"]),
        nz=int(mesh["nz"]),
        dx=float(mesh["dx"]),
        dy=float(mesh["dy"]),
        dz=float(mesh["dz"]),
    )
    ports = config["ports"]
    pml = config["pml"]
    setup = ScatteringSetup3D(
        grid=grid,
        lower_source_plane=int(ports["lower_source_plane"]),
        upper_source_plane=int(ports["upper_source_plane"]),
        lower_monitor_planes=tuple(ports["lower_monitor_planes"]),
        upper_monitor_planes=tuple(ports["upper_monitor_planes"]),
        port_index=float(ports["refractive_index"]),
        pml=AxialPML3D(
            cells=int(pml["cells"]),
            target_round_trip_amplitude=float(
                pml["target_round_trip_amplitude"]
            ),
            polynomial_order=int(pml.get("polynomial_order", 3)),
        ),
        speed_of_light=float(config.get("speed_of_light", 1.0)),
    )
    specification = config["nanodisk"]
    disk = EmbeddedNanodisk3D(
        radius=float(specification["radius"]),
        height=float(specification["height"]),
        disk_index=float(specification["refractive_index"]),
        host_index=setup.port_index,
        transverse_supersampling=int(
            specification.get("transverse_supersampling", 8)
        ),
    )
    representation = str(specification.get("representation", "subcell"))
    if representation == "subcell":
        epsilon = disk.epsilon_r(grid)
    elif representation == "smooth":
        smooth = specification["smooth_parameterization"]
        parameterization = SmoothNanodiskParameterization3D(
            grid=grid,
            host_index=setup.port_index,
            radius_range=tuple(smooth["radius_range"]),
            height_range=tuple(smooth["height_range"]),
            disk_index_range=tuple(smooth["disk_index_range"]),
            transverse_boundary_width=float(smooth["transverse_boundary_width"]),
            axial_boundary_width=float(smooth["axial_boundary_width"]),
        )

        def normalized(value: float, bounds: list[float]) -> float:
            low, high = map(float, bounds)
            return (value - low) / (high - low)

        raw = np.asarray(
            (
                normalized(disk.radius, smooth["radius_range"]),
                normalized(disk.height, smooth["height_range"]),
                normalized(disk.disk_index, smooth["disk_index_range"]),
            )
        )
        epsilon = parameterization.map(raw).epsilon_r
    else:
        raise ValueError("nanodisk.representation must be 'subcell' or 'smooth'")
    return setup, disk, epsilon


def _candidate_index(phase: np.ndarray, omega: np.ndarray) -> int:
    delay = np.gradient(np.unwrap(np.angle(phase)), omega)
    baseline = np.median(delay)
    return int(np.argmax(np.abs(delay - baseline)))


def _solve_frequency(
    config: dict[str, Any], index: int, frequency: float
) -> tuple[int, np.ndarray]:
    """Independent, picklable frequency worker for local or EC2 batching."""

    setup, _, epsilon = build_benchmark(config)
    matrix = solve_scattering_3d(frequency, epsilon, setup).matrix
    return index, np.asarray(matrix)


def _atomic_checkpoint(
    path: Path,
    *,
    omega: np.ndarray,
    matrices: np.ndarray,
    completed: np.ndarray,
) -> None:
    temporary = path.with_suffix(".tmp.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            omega=omega,
            scattering_matrix=matrices,
            completed=completed,
        )
    os.replace(temporary, path)


def run_forward(config: dict[str, Any], output: Path) -> dict[str, Any]:
    setup, disk, epsilon = build_benchmark(config)
    scan = config["scan"]
    wavelengths = np.linspace(
        float(scan["wavelength_max"]),
        float(scan["wavelength_min"]),
        int(scan["points"]),
    )
    omega = 2.0 * np.pi * setup.speed_of_light / wavelengths
    checkpoint_path = output / "forward_checkpoint.npz"
    matrices = np.full(
        (omega.size, 4, 4), np.nan + 1j * np.nan, dtype=np.complex128
    )
    completed = np.zeros(omega.size, dtype=bool)
    if checkpoint_path.is_file():
        checkpoint = np.load(checkpoint_path)
        np.testing.assert_allclose(checkpoint["omega"], omega, rtol=0.0, atol=0.0)
        matrices[...] = checkpoint["scattering_matrix"]
        completed[...] = checkpoint["completed"]
    reflection = np.empty(omega.size, dtype=np.complex128)
    transmission = np.empty(omega.size, dtype=np.complex128)
    symmetric = np.empty(omega.size, dtype=np.complex128)
    antisymmetric = np.empty(omega.size, dtype=np.complex128)
    symmetry_defect = np.empty(omega.size, dtype=float)
    cross_power = np.empty(omega.size, dtype=float)

    pending = np.flatnonzero(~completed)
    workers = int(scan.get("workers", 1))
    if workers < 1:
        raise ValueError("scan.workers must be positive")
    if pending.size:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _solve_frequency, config, int(index), float(omega[index])
                ): int(index)
                for index in pending
            }
            for count, future in enumerate(as_completed(futures), start=1):
                index, matrix = future.result()
                matrices[index] = matrix
                completed[index] = True
                _atomic_checkpoint(
                    checkpoint_path,
                    omega=omega,
                    matrices=matrices,
                    completed=completed,
                )
                print(
                    f"completed {np.count_nonzero(completed)}/{omega.size}: "
                    f"omega={omega[index]:.9g}",
                    flush=True,
                )

    local_matrices = np.empty_like(matrices)
    for index, raw_matrix in enumerate(matrices):
        matrix = deembed_scattering_matrix_3d(raw_matrix, omega[index], setup)
        local_matrices[index] = matrix
        reflection[index], transmission[index] = copolarized_coefficients(matrix)
        symmetric[index], antisymmetric[index] = axial_eigenchannels(matrix)
        symmetry_defect[index] = axial_symmetry_defect(matrix)
        lower_cross = matrix[np.ix_((1, 3), (0, 2))]
        cross_power[index] = float(np.sum(np.abs(lower_cross[:, 0]) ** 2))

    reflectance = np.abs(reflection) ** 2
    transmittance = np.abs(transmission) ** 2
    power_error = np.abs(reflectance + transmittance - 1.0)
    symmetric_index = _candidate_index(symmetric, omega)
    antisymmetric_index = _candidate_index(antisymmetric, omega)

    np.savez_compressed(
        output / "forward_scan.npz",
        wavelength=wavelengths,
        omega=omega,
        raw_scattering_matrix=matrices,
        scattering_matrix=local_matrices,
        reflection=reflection,
        transmission=transmission,
        symmetric_eigenchannel=symmetric,
        antisymmetric_eigenchannel=antisymmetric,
        epsilon_r=epsilon,
        symmetry_defect=symmetry_defect,
        cross_power=cross_power,
        power_error=power_error,
    )

    figure, _ = plot_refractive_index_slices_3d(
        epsilon,
        setup.grid,
        length_unit=config.get("length_unit", "normalized length"),
    )
    figure.savefig(output / "structure.png", dpi=220)
    plt.close(figure)

    order = np.argsort(wavelengths)
    plotted_wavelength = wavelengths[order]
    figure, axes = plt.subplots(
        2, 2, figsize=(9.4, 6.7), constrained_layout=True
    )
    axes[0, 0].plot(plotted_wavelength, transmittance[order], label="T")
    axes[0, 0].plot(plotted_wavelength, reflectance[order], label="R")
    axes[0, 0].set_ylabel("Power")
    axes[0, 0].legend()
    axes[0, 1].plot(
        plotted_wavelength,
        np.unwrap(np.angle(transmission))[order] / np.pi,
    )
    axes[0, 1].set_ylabel("arg(t) / pi")
    axes[1, 0].plot(
        plotted_wavelength,
        np.unwrap(np.angle(symmetric))[order] / np.pi,
        label="symmetric",
    )
    axes[1, 0].plot(
        plotted_wavelength,
        np.unwrap(np.angle(antisymmetric))[order] / np.pi,
        label="antisymmetric",
    )
    axes[1, 0].set_ylabel("Eigenphase / pi")
    axes[1, 0].legend()
    axes[1, 1].semilogy(plotted_wavelength, power_error[order], label="|R+T-1|")
    axes[1, 1].semilogy(plotted_wavelength, cross_power[order], label="cross power")
    axes[1, 1].set_ylabel("Numerical defect")
    axes[1, 1].legend()
    for axis in axes[-1]:
        axis.set_xlabel(f"Wavelength ({config.get('length_unit', 'length')})")
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    figure.savefig(output / "real_axis_response.png", dpi=220)
    plt.close(figure)

    summary = {
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "grid_shape": list(setup.grid.shape),
        "field_unknowns": setup.grid.field_size,
        "first_diffraction_omega": setup.first_diffraction_frequency,
        "minimum_transmittance": float(np.min(transmittance)),
        "maximum_transmittance": float(np.max(transmittance)),
        "minimum_reflectance": float(np.min(reflectance)),
        "maximum_reflectance": float(np.max(reflectance)),
        "maximum_power_error": float(np.max(power_error)),
        "maximum_cross_power": float(np.max(cross_power)),
        "maximum_symmetry_defect": float(np.max(symmetry_defect)),
        "symmetric_candidate": {
            "index": symmetric_index,
            "wavelength": float(wavelengths[symmetric_index]),
            "omega": float(omega[symmetric_index]),
        },
        "antisymmetric_candidate": {
            "index": antisymmetric_index,
            "wavelength": float(wavelengths[antisymmetric_index]),
            "omega": float(omega[antisymmetric_index]),
        },
    }
    _write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an output containing the identical saved config",
    )
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    if arguments.resume:
        saved = json.loads(
            (arguments.output / "config.json").read_text(encoding="utf-8")
        )
        if saved != config:
            raise ValueError("resume config differs from the saved run config")
    else:
        arguments.output.mkdir(parents=True, exist_ok=False)
        _write_json(arguments.output / "config.json", config)
    run_forward(config, arguments.output)


if __name__ == "__main__":
    main()
