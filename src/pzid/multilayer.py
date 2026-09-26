"""Analytic normal-incidence multilayer scattering and channel residues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pzid.roots import ScalarRootResult, refine_scalar_root


@dataclass(frozen=True)
class MultilayerResponse:
    """Reflection/transmission amplitudes and their common denominator."""

    reflection: complex
    transmission: complex
    numerator: complex
    denominator: complex
    denominator_derivative: complex


def normal_incidence_multilayer(
    omega: complex,
    layers: Sequence[tuple[float, float]],
    *,
    incident_index: float = 1.0,
    substrate_index: float = 1.0,
) -> MultilayerResponse:
    """Evaluate a nondispersive dielectric stack with exp(-i omega t).

    Each layer is ``(refractive_index, thickness_um)`` and ``omega`` is in
    rad/um with vacuum light speed normalized to one. The characteristic
    matrix and its frequency derivative are accumulated analytically.
    """

    if incident_index <= 0.0 or substrate_index <= 0.0:
        raise ValueError("port indices must be positive")
    matrix = np.eye(2, dtype=np.complex128)
    derivative = np.zeros((2, 2), dtype=np.complex128)
    for index, thickness in layers:
        n = float(index)
        d = float(thickness)
        if n <= 0.0 or d < 0.0:
            raise ValueError("layer indices must be positive and thicknesses nonnegative")
        delta = complex(omega) * n * d
        cosine = np.cos(delta)
        sine = np.sin(delta)
        layer = np.asarray(
            ((cosine, -1j * sine / n), (-1j * n * sine, cosine)),
            dtype=np.complex128,
        )
        layer_derivative = np.asarray(
            (
                (-n * d * sine, -1j * d * cosine),
                (-1j * n * n * d * cosine, -n * d * sine),
            ),
            dtype=np.complex128,
        )
        derivative = derivative @ layer + matrix @ layer_derivative
        matrix = matrix @ layer
    n0 = float(incident_index)
    ns = float(substrate_index)
    denominator = (
        n0 * matrix[0, 0]
        + n0 * ns * matrix[0, 1]
        + matrix[1, 0]
        + ns * matrix[1, 1]
    )
    numerator = (
        n0 * matrix[0, 0]
        + n0 * ns * matrix[0, 1]
        - matrix[1, 0]
        - ns * matrix[1, 1]
    )
    denominator_derivative = (
        n0 * derivative[0, 0]
        + n0 * ns * derivative[0, 1]
        + derivative[1, 0]
        + ns * derivative[1, 1]
    )
    return MultilayerResponse(
        reflection=complex(numerator / denominator),
        transmission=complex(2.0 * n0 / denominator),
        numerator=complex(numerator),
        denominator=complex(denominator),
        denominator_derivative=complex(denominator_derivative),
    )


def refine_multilayer_pole(
    layers: Sequence[tuple[float, float]],
    guess: complex,
    *,
    incident_index: float = 1.0,
    substrate_index: float = 1.0,
    tolerance: float = 1e-11,
    max_evaluations: int = 200,
    search_radius: float = 0.5,
) -> ScalarRootResult:
    """Refine an outgoing multilayer pole as a denominator zero."""

    result = refine_scalar_root(
        lambda value: normal_incidence_multilayer(
            value,
            layers,
            incident_index=incident_index,
            substrate_index=substrate_index,
        ).denominator,
        guess,
        kind="zero",
        tolerance=tolerance,
        max_evaluations=max_evaluations,
        search_radius=search_radius,
    )
    return ScalarRootResult(
        frequency=result.frequency,
        kind="pole",
        success=result.success,
        message=result.message,
        evaluations=result.evaluations,
        residual_norm=result.residual_norm,
    )


def multilayer_reflection_residue(
    pole: complex,
    layers: Sequence[tuple[float, float]],
    *,
    incident_index: float = 1.0,
    substrate_index: float = 1.0,
    reference_offset_um: float = 0.0,
) -> complex:
    """Return the reflection-channel residue at a fixed external reference."""

    if reference_offset_um < 0.0:
        raise ValueError("reference offset must be nonnegative")
    response = normal_incidence_multilayer(
        pole,
        layers,
        incident_index=incident_index,
        substrate_index=substrate_index,
    )
    residue = response.numerator / response.denominator_derivative
    return complex(
        residue
        * np.exp(2j * complex(pole) * float(incident_index) * reference_offset_um)
    )
