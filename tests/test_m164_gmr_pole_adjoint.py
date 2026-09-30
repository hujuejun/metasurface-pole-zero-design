"""Regression tests for the physical GMR Chen pole-adjoint chain rule."""

import json
from pathlib import Path

import numpy as np

from scripts.m164_gmr_pole_adjoint import build_setup, check_adjoint_gradient


def test_physical_gmr_maxwell_adjoint_matches_finite_difference() -> None:
    config = json.loads(Path("configs/m164_gmr_pole_adjoint.json").read_text(encoding="utf-8"))
    setup = build_setup(config)
    coordinates = np.asarray((0.0988, 0.0937))
    target = 6.77 - 0.016925j
    audit = check_adjoint_gradient(coordinates, target, config, setup)
    assert max(audit["relative_errors"]) < 0.003
