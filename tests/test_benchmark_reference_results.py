"""The published GMR cost figure can be traced to every saved attempt."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reference_results/gmr_benchmark"


def test_complete_cost_benchmark_archive_matches_original_checksums():
    audit = json.loads((DATA / "m198_maintext_audit.json").read_text(encoding="utf-8"))
    assert audit["records"] == 72
    trials = sorted((DATA / "trials").glob("*.json"))
    assert len(trials) == 72
    for path in (*trials, DATA / "preparation_q0200.json",
                 DATA / "preparation_q0800.json", DATA / "cost_ledger.csv"):
        original = next((digest for name, digest in audit["sha256"].items()
                         if Path(name.replace("\\", "/")).name == path.name), None)
        assert original is not None, path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == original, path


def test_pole_error_ledger_has_one_row_per_attempt():
    lines = (DATA / "gmr_cost_pole_errors.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 73
