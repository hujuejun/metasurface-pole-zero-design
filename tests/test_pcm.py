"""Checks for the phase-change pilot material model."""

import numpy as np
import pytest

from pzid.grid import UniformGrid2D
from pzid.hcg import (
    SymmetricPairedHCGGeometryParameterization,
    SymmetricPairedTwoTierHCGGeometryParameterization,
)
from pzid.pcm import SymmetricPCMStateHCGParameterization, lorentz_lorenz_permittivity


def test_lorentz_lorenz_recovers_both_endpoints() -> None:
    amorphous = 3.285 + 0.001j
    crystalline = 4.05 + 0.01j
    assert lorentz_lorenz_permittivity(amorphous, crystalline, 0.0) == pytest.approx(amorphous**2)
    assert lorentz_lorenz_permittivity(amorphous, crystalline, 1.0) == pytest.approx(crystalline**2)


def test_lorentz_lorenz_intermediate_is_passive_and_monotone() -> None:
    fractions = np.linspace(0.0, 1.0, 21)
    epsilon = lorentz_lorenz_permittivity(3.285 + 0.001j, 4.05 + 0.01j, fractions)
    assert isinstance(epsilon, np.ndarray)
    assert np.all(np.diff(epsilon.real) > 0.0)
    assert np.all(epsilon.imag >= 0.0)


@pytest.mark.parametrize("fraction", [-0.01, 1.01, float("nan")])
def test_lorentz_lorenz_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="crystalline_fraction"):
        lorentz_lorenz_permittivity(3.285, 4.05, fraction)


def test_symmetric_pcm_state_geometry_retains_exact_interface_derivatives() -> None:
    grid = UniformGrid2D(nx=24, nz=112, dx=0.77 / 24, dz=0.77 / 24)
    reference = SymmetricPairedTwoTierHCGGeometryParameterization(
        grid=grid,
        port_index=1.0,
        grating_index=3.48,
        inner_height_bounds_um=(0.15, 0.55),
        outer_height_bounds_um=(0.15, 0.55),
        cavity_bounds_um=(0.15, 0.65),
        inner_width_bounds_um=(0.10, 0.45),
        outer_width_bounds_um=(0.10, 0.45),
    )
    # Avoid the intentionally grid-snapped M58D coordinates when checking a
    # central difference: the cell-coverage derivative has a kink there.
    raw = reference.encode(0.3621, 0.3397, 0.4473, 0.1971, 0.1883)
    state = SymmetricPCMStateHCGParameterization(reference, (4.05 + 0.001j) ** 2)
    mapping = state.map(raw)
    assert np.allclose(mapping.epsilon_r, np.flipud(mapping.epsilon_r))
    assert np.min(mapping.epsilon_r.imag) >= 0.0
    step = 1e-6
    plus = raw.copy()
    minus = raw.copy()
    plus[2] += step
    minus[2] -= step
    finite = (state.map(plus).epsilon_r - state.map(minus).epsilon_r) / (2.0 * step)
    assert np.linalg.norm(finite - mapping.epsilon_derivatives[2]) / np.linalg.norm(finite) < 1e-5


def test_one_tier_pcm_wrapper_matches_equal_width_two_tier_geometry() -> None:
    grid = UniformGrid2D(nx=24, nz=112, dx=0.77 / 24, dz=0.77 / 24)
    one_tier = SymmetricPairedHCGGeometryParameterization(
        grid=grid, port_index=1.0, grating_index=3.48,
        thickness_bounds_um=(0.3, 1.1), cavity_bounds_um=(0.15, 0.65),
        width_bounds_um=(0.1, 0.45),
    )
    two_tier = SymmetricPairedTwoTierHCGGeometryParameterization(
        grid=grid, port_index=1.0, grating_index=3.48,
        inner_height_bounds_um=(0.15, 0.55),
        outer_height_bounds_um=(0.15, 0.55),
        cavity_bounds_um=(0.15, 0.65),
        inner_width_bounds_um=(0.1, 0.45),
        outer_width_bounds_um=(0.1, 0.45),
    )
    epsilon = (3.7 + 0.0001j) ** 2
    one = SymmetricPCMStateHCGParameterization(one_tier, epsilon).map(
        one_tier.encode(0.7, 0.45, 0.19)
    )
    two = SymmetricPCMStateHCGParameterization(two_tier, epsilon).map(
        two_tier.encode(0.36, 0.34, 0.45, 0.19, 0.19)
    )
    assert np.allclose(one.epsilon_r, two.epsilon_r, atol=1e-12, rtol=0.0)
    assert np.allclose(one.epsilon_r, np.flipud(one.epsilon_r), atol=1e-12, rtol=0.0)
