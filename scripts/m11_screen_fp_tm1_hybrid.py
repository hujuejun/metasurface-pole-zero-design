from __future__ import annotations

import csv

from dataclasses import asdict

from pathlib import Path

from typing import Any

import numpy as np

from pzid.roots import ScalarRootResult, refine_scalar_root

from pzid.twoport import TwoPortScatteringSetup, solve_two_port_scattering

def _complex_json(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]

def _root_json(result: ScalarRootResult) -> dict[str, Any]:
    record = asdict(result)
    record["frequency"] = _complex_json(result.frequency)
    if result.frequency != 0.0:
        record["wavelength_um"] = _complex_json(2.0 * np.pi / result.frequency)
    if result.kind == "pole" and result.frequency.imag < 0.0:
        record["quality_factor"] = float(
            result.frequency.real / (-2.0 * result.frequency.imag)
        )
    return record

def _sweep(
    wavelengths: np.ndarray,
    epsilon: np.ndarray,
    setup: TwoPortScatteringSetup,
) -> dict[str, np.ndarray | float]:
    even = np.empty(wavelengths.size, dtype=np.complex128)
    odd = np.empty_like(even)
    matrices = np.empty((wavelengths.size, 2, 2), dtype=np.complex128)
    empty = np.empty(wavelengths.size, dtype=np.complex128)
    energy_error = 0.0
    reciprocity_error = 0.0
    mirror_error = 0.0
    for index, wavelength in enumerate(wavelengths):
        omega = 2.0 * np.pi / wavelength
        solution = solve_two_port_scattering(omega, epsilon, setup)
        matrix = solution.matrix
        empty_t = solution.system.reference_scattering[1, 0]
        matrices[index] = matrix
        empty[index] = empty_t
        even[index] = (matrix[0, 0] - matrix[1, 0]) / (-empty_t)
        odd[index] = (matrix[0, 0] + matrix[1, 0]) / empty_t
        energy_error = max(
            energy_error,
            abs(float(np.sum(np.abs(matrix[:, 0]) ** 2)) - 1.0),
            abs(float(np.sum(np.abs(matrix[:, 1]) ** 2)) - 1.0),
        )
        reciprocity_error = max(reciprocity_error, abs(matrix[0, 1] - matrix[1, 0]))
        mirror_error = max(mirror_error, abs(matrix[0, 0] - matrix[1, 1]))
    return {
        "even": even,
        "odd": odd,
        "matrices": matrices,
        "empty": empty,
        "energy_balance_max_abs_error": float(energy_error),
        "reciprocity_max_abs_error": float(reciprocity_error),
        "mirror_scattering_max_abs_error": float(mirror_error),
    }

def _write_spectrum(
    path: Path, wavelengths: np.ndarray, spectrum: dict[str, Any]
) -> None:
    fields = (
        "wavelength_um",
        "omega_rad_per_um",
        "even_real",
        "even_imag",
        "odd_real",
        "odd_imag",
        "s11_real",
        "s11_imag",
        "s21_real",
        "s21_imag",
        "reflectance",
        "transmittance",
    )
    matrices = np.asarray(spectrum["matrices"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, wavelength in enumerate(wavelengths):
            s11 = matrices[index, 0, 0]
            s21 = matrices[index, 1, 0]
            writer.writerow(
                {
                    "wavelength_um": wavelength,
                    "omega_rad_per_um": 2.0 * np.pi / wavelength,
                    "even_real": spectrum["even"][index].real,
                    "even_imag": spectrum["even"][index].imag,
                    "odd_real": spectrum["odd"][index].real,
                    "odd_imag": spectrum["odd"][index].imag,
                    "s11_real": s11.real,
                    "s11_imag": s11.imag,
                    "s21_real": s21.real,
                    "s21_imag": s21.imag,
                    "reflectance": abs(s11) ** 2,
                    "transmittance": abs(s21) ** 2,
                }
            )

def _json_radiation(diagnostics: dict[str, float | complex]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in diagnostics.items():
        result[key] = _complex_json(value) if isinstance(value, complex) else float(value)
    return result
