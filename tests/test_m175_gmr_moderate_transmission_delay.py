"""Minimal analytic checks for the M175 local phase-derivative diagnostic."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.m164_gmr_pole_adjoint import FS_PER_UM_OVER_C
from scripts.m175_gmr_moderate_transmission_delay import phase_delay


def test_phase_delay_recovers_single_pole_scale_off_resonance() -> None:
    gamma = 0.01
    center = 6.77

    def response(frequency: float) -> SimpleNamespace:
        return SimpleNamespace(transmission=1 / (frequency - (center - 1j * gamma)))

    actual = phase_delay(response, center - 2 * gamma, 1e-5 * gamma)
    expected = FS_PER_UM_OVER_C / (5 * gamma)
    assert np.isclose(actual, expected, rtol=1e-8)


def test_phase_delay_rejects_a_near_zero_sample() -> None:
    def response(frequency: float) -> SimpleNamespace:
        return SimpleNamespace(transmission=frequency)

    with pytest.raises(RuntimeError, match="near-zero"):
        phase_delay(response, 0.0, 1e-6)
