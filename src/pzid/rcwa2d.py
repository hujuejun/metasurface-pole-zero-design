"""Independent Fourier-modal solver for 2D TM, 1D-periodic dielectric stacks.

The harmonic convention is exp(i kx x + i beta z - i omega t).  Fields are
the Fourier coefficients of Hy and (1/epsilon) dHy/dz.  Layers are combined
with scattering matrices, avoiding exponentially growing transfer products.
Only nondispersive isotropic dielectric media are supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import eig, solve


@dataclass(frozen=True)
class FourierLayer:
    """A uniform-z layer with Fourier matrices of epsilon and 1/epsilon."""

    thickness_um: float
    epsilon: np.ndarray
    inverse_epsilon: np.ndarray


@dataclass(frozen=True)
class _Modes:
    w: np.ndarray
    v: np.ndarray
    beta: np.ndarray


@dataclass(frozen=True)
class _Scattering:
    s11: np.ndarray
    s12: np.ndarray
    s21: np.ndarray
    s22: np.ndarray


@dataclass(frozen=True)
class FourierResponse:
    transmission: complex
    reflection: complex
    transmitted_orders: np.ndarray
    reflected_orders: np.ndarray


def homogeneous_layer(index: float, thickness_um: float, harmonic_half_width: int) -> FourierLayer:
    if index <= 0 or thickness_um < 0 or harmonic_half_width < 0:
        raise ValueError("index and thickness must be positive, harmonics nonnegative")
    identity = np.eye(2 * harmonic_half_width + 1, dtype=complex)
    return FourierLayer(thickness_um, index**2 * identity, identity / index**2)


def centered_binary_grating_layer(
    low_index: float,
    high_index: float,
    fill_factor: float,
    thickness_um: float,
    harmonic_half_width: int,
) -> FourierLayer:
    """Exact Fourier coefficients for a centered high-index ridge."""

    if not 0 <= fill_factor <= 1 or low_index <= 0 or high_index <= 0 or thickness_um < 0:
        raise ValueError("invalid grating material or geometry")
    orders = np.arange(-harmonic_half_width, harmonic_half_width + 1)
    difference = orders[:, None] - orders[None, :]
    centered_shape = fill_factor * np.sinc(difference * fill_factor) * (-1.0) ** difference
    identity = np.eye(len(orders), dtype=complex)
    epsilon = low_index**2 * identity + (high_index**2 - low_index**2) * centered_shape
    inverse_epsilon = identity / low_index**2 + (1 / high_index**2 - 1 / low_index**2) * centered_shape
    return FourierLayer(thickness_um, epsilon.astype(complex), inverse_epsilon.astype(complex))


def _outgoing_beta(beta_squared: np.ndarray) -> np.ndarray:
    beta = np.sqrt(np.asarray(beta_squared, dtype=complex))
    # At a nearby lower-half-plane complex frequency, a propagating mode has
    # positive Re(beta) and negative Im(beta), whereas an evanescent mode must
    # retain positive Im(beta) by analytic continuation from the real axis.
    evanescent = beta_squared.real < 0
    flip = (evanescent & (beta.imag < 0)) | (~evanescent & (beta.real < 0))
    beta[flip] *= -1
    return beta


def _modes(
    omega: complex,
    layer: FourierLayer,
    transverse_k: np.ndarray,
    *,
    factorization: Literal["li", "legacy"] = "li",
) -> _Modes:
    epsilon = np.asarray(layer.epsilon, dtype=complex)
    inverse_epsilon = np.asarray(layer.inverse_epsilon, dtype=complex)
    n = len(transverse_k)
    if epsilon.shape != (n, n) or inverse_epsilon.shape != (n, n):
        raise ValueError("Fourier layer matrix size does not match harmonic basis")
    k_matrix = np.diag(transverse_k)
    if factorization == "li":
        # In the lamellar TM problem, H_y and (1/epsilon) dH_y/dx are
        # continuous at vertical material walls. Li's inverse rule therefore
        # uses inv([[epsilon]]) for the normal-x flux. The tangential E_x
        # coupling instead uses [[1/epsilon]] and its matrix inverse.
        normal_flux = solve(epsilon, k_matrix, assume_a="gen")
        beta_operator = solve(
            inverse_epsilon,
            omega**2 * np.eye(n) - k_matrix @ normal_flux,
            assume_a="gen",
        )
    elif factorization == "legacy":
        # Preserved solely for exact reproduction of pre-audit M153-M187 runs.
        # It applies direct products at both TM discontinuities and converges
        # slowly; do not use it as a high-accuracy reference.
        beta_operator = epsilon @ (omega**2 * np.eye(n) - k_matrix @ inverse_epsilon @ k_matrix)
    else:
        raise ValueError("factorization must be 'li' or 'legacy'")
    beta_squared, w = eig(beta_operator)
    beta = _outgoing_beta(beta_squared)
    v = (
        inverse_epsilon @ (w * beta[None, :])
        if factorization == "li"
        else solve(epsilon, w * beta[None, :], assume_a="gen")
    )
    return _Modes(w, v, beta)


def _interface(left: _Modes, right: _Modes) -> _Scattering:
    n = len(left.beta)
    matrix = np.block([[left.w, -right.w], [-left.v, -right.v]])
    forcing = np.block([[-left.w, right.w], [-left.v, -right.v]])
    result = solve(matrix, forcing, assume_a="gen")
    return _Scattering(result[:n, :n], result[:n, n:], result[n:, :n], result[n:, n:])


def _propagation(modes: _Modes, thickness_um: float) -> _Scattering:
    n = len(modes.beta)
    p = np.diag(np.exp(1j * modes.beta * thickness_um))
    zero = np.zeros((n, n), dtype=complex)
    return _Scattering(zero, p, p, zero)


def _cascade(left: _Scattering, right: _Scattering) -> _Scattering:
    n = left.s11.shape[0]
    matrix = np.eye(n, dtype=complex) - left.s22 @ right.s11
    intermediate = solve(matrix, np.column_stack((left.s21, left.s22 @ right.s12)), assume_a="gen")
    from_left = intermediate[:, :n]
    from_right = intermediate[:, n:]
    return _Scattering(
        left.s11 + left.s12 @ right.s11 @ from_left,
        left.s12 @ (right.s11 @ from_right + right.s12),
        right.s21 @ from_left,
        right.s21 @ from_right + right.s22,
    )


def solve_tm_stack(
    omega: complex,
    layers: tuple[FourierLayer, ...],
    *,
    period_um: float,
    port_index: float,
    harmonic_half_width: int,
    factorization: Literal["li", "legacy"] = "li",
) -> FourierResponse:
    """Return specular Hy amplitudes for one incident zeroth-order plane wave.

    ``li`` applies the inverse Fourier factorization required for patterned
    TM layers. ``legacy`` reproduces the pre-audit direct-product solver and
    is retained only to make historical simulation results reproducible.
    """

    if period_um <= 0 or port_index <= 0 or harmonic_half_width < 0:
        raise ValueError("invalid period, port index, or harmonic count")
    n = 2 * harmonic_half_width + 1
    kx = 2 * np.pi * np.arange(-harmonic_half_width, harmonic_half_width + 1) / period_um
    port = homogeneous_layer(port_index, 0.0, harmonic_half_width)
    port_modes = _modes(omega, port, kx, factorization=factorization)
    previous = port_modes
    total: _Scattering | None = None
    for layer in layers:
        modes = _modes(omega, layer, kx, factorization=factorization)
        pieces = (_interface(previous, modes), _propagation(modes, layer.thickness_um))
        for piece in pieces:
            total = piece if total is None else _cascade(total, piece)
        previous = modes
    final = _interface(previous, port_modes)
    total = final if total is None else _cascade(total, final)
    incident = np.zeros(n, dtype=complex)
    incident[harmonic_half_width] = 1.0
    reflected = total.s11 @ incident
    transmitted = total.s21 @ incident
    return FourierResponse(
        transmission=complex(transmitted[harmonic_half_width]),
        reflection=complex(reflected[harmonic_half_width]),
        transmitted_orders=transmitted,
        reflected_orders=reflected,
    )
