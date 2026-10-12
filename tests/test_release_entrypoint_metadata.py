"""Validate packaged inputs before expensive optimization entry points run."""

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reproduction_configs_include_required_provenance():
    cases = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))
    for name in ("low-index-binary", "planar-hcg", "qbic", "qbic-binary-negative-control"):
        config = json.loads((ROOT / cases[name]["config"]).read_text(encoding="utf-8"))
        assert isinstance(config["provenance"], dict) and config["provenance"]
        if name == "planar-hcg":
            assert config["registration"]["milestone"] == "M58D"
            source = config["initial_geometry"]
            path = ROOT / source["source_raw"]
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source["source_raw_sha256"]


def test_ipdip_provenance_inputs_are_packaged():
    config = json.loads((ROOT / "configs/ipdip_optimize.json").read_text(encoding="utf-8"))
    assert config["note"]
    for key in ("source_config", "parent_geometry", "parent_summary",
                "material_model", "optical_constants"):
        assert (ROOT / config[key]).is_file(), key


def test_qbic_fresh_seed_limit_is_archived_separately_from_frozen_geometry():
    audit = json.loads((ROOT / "reference_results/qbic_fresh_seed_audit.json")
                       .read_text(encoding="utf-8"))
    assert audit["full_common_pole_reached"]
    assert audit["reflection_zero_reached"]
    assert not audit["passes_performance_gate"]
    assert audit["metrics"]["minimum_transmittance_in_performance_window"] < 0.85
