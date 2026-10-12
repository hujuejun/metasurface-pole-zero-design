"""Cheap non-solver checks for the narrow-feature replay diagnostic."""
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.low_index_resonance_audit import half_max_width


def test_half_max_width_interpolates_crossings():
    k = np.linspace(-2.0, 2.0, 9)
    r = np.maximum(0.0, 1.0 - np.abs(k))
    assert np.isclose(half_max_width(k, r), 1.0)


def test_frozen_low_index_audit_resolves_missed_reflection_peaks():
    root = Path(__file__).resolve().parents[1]
    data = root / "reference_results/low_index_resonance_audit"
    result = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    assert result["mesh_cells_per_period"] == 56
    assert result["coarse_grid"]["points"] == 241
    assert result["coarse_grid"]["maximum_reflectance"] < 0.003
    assert result["adaptive_grid"]["maximum_reflectance"] > 0.9999
    assert len(result["features"]) == 2
    assert all(item["pole_residual"] < 1e-6 for item in result["features"])
    assert all(item["peak_reflectance"] > 0.9999 for item in result["features"])
    permittivity = root / "inputs/final_permittivity/low_index_binary_n56.csv"
    assert hashlib.sha256(permittivity.read_bytes()).hexdigest() == (
        result["inputs_sha256"]["permittivity"]
    )
    for name, count in (("coarse", 241), ("adaptive", 725),
                        ("parasitic_1", 241), ("parasitic_2", 241)):
        rows = (data / f"{name}_spectrum.csv").read_text(encoding="utf-8").splitlines()
        assert len(rows) == count + 1
