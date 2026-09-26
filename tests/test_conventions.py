from math import pi

import pytest

from pzid.conventions import DEFAULT_CONVENTION


def test_passive_half_plane_matches_time_convention() -> None:
    assert DEFAULT_CONVENTION.passive_pole_is_valid(1.0 - 0.1j)
    assert not DEFAULT_CONVENTION.passive_pole_is_valid(1.0 + 0.1j)
    assert not DEFAULT_CONVENTION.passive_pole_is_valid(1.0 + 0.0j)


def test_first_diffraction_threshold() -> None:
    threshold = DEFAULT_CONVENTION.first_diffraction_angular_frequency(
        period=0.5,
        port_index=1.0,
        speed_of_light=1.0,
    )
    assert threshold == pytest.approx(4.0 * pi)


def test_subwavelength_check_uses_most_restrictive_port() -> None:
    period = 0.5
    assert DEFAULT_CONVENTION.is_subwavelength(
        period=period,
        omega_max=1.9 * pi,
        port_indices=(1.0, 2.0),
    )
    assert not DEFAULT_CONVENTION.is_subwavelength(
        period=period,
        omega_max=2.1 * pi,
        port_indices=(1.0, 2.0),
    )


@pytest.mark.parametrize(
    ("period", "index"),
    [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)],
)
def test_invalid_diffraction_parameters_raise(period: float, index: float) -> None:
    with pytest.raises(ValueError):
        DEFAULT_CONVENTION.first_diffraction_angular_frequency(period, index)
