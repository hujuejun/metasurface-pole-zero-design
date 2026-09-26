"""Analytic scattering models used as numerical ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexArray: TypeAlias = NDArray[np.complex128]


@dataclass(frozen=True)
class ScalarScattering:
    """Reflection and transmission amplitudes at one or more frequencies."""

    reflection: complex | ComplexArray
    transmission: complex | ComplexArray


def slab_scattering(
    omega: ArrayLike,
    *,
    thickness: float,
    layer_index: float,
    left_index: float = 1.0,
    right_index: float = 1.0,
    speed_of_light: float = 1.0,
    power_normalized: bool = True,
) -> ScalarScattering:
    """Normal-incidence scattering of one homogeneous dielectric layer.

    The amplitudes use the repository's ``exp(-i omega t)`` convention and
    reference planes at the two layer interfaces. The indices are restricted
    to positive real values in the initial lossless validation problem, while
    ``omega`` may be complex.
    """

    if thickness < 0.0:
        raise ValueError("thickness must be nonnegative")
    if min(layer_index, left_index, right_index, speed_of_light) <= 0.0:
        raise ValueError("indices and speed_of_light must be positive")

    frequencies = np.asarray(omega, dtype=np.complex128)
    r_left_layer = (left_index - layer_index) / (left_index + layer_index)
    r_layer_right = (layer_index - right_index) / (layer_index + right_index)
    t_left_layer = 2.0 * left_index / (left_index + layer_index)
    t_layer_right = 2.0 * layer_index / (layer_index + right_index)

    propagation = np.exp(
        1j * layer_index * frequencies * thickness / speed_of_light
    )
    denominator = 1.0 + r_left_layer * r_layer_right * propagation**2
    reflection = (
        r_left_layer + r_layer_right * propagation**2
    ) / denominator
    transmission = (
        t_left_layer * t_layer_right * propagation / denominator
    )

    if power_normalized:
        transmission = transmission * np.sqrt(right_index / left_index)

    if frequencies.ndim == 0:
        return ScalarScattering(
            reflection=complex(reflection),
            transmission=complex(transmission),
        )
    return ScalarScattering(reflection=reflection, transmission=transmission)

