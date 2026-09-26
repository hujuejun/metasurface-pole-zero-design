from __future__ import annotations

import hashlib

import json

import platform

import subprocess

import sys

from datetime import UTC, datetime

from pathlib import Path

import matplotlib

import numpy as np

import scipy

def _git(*command: str) -> str:
    completed = subprocess.run(
        ("git", *command), check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()

def write_metadata(
    destination: Path, config_path: Path, literature_path: Path
) -> None:
    metadata = {
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "git_revision": _git("rev-parse", "HEAD"),
        "worktree_clean": _git("status", "--porcelain") == "",
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "config_sha256": _sha256(config_path),
        "literature_metrics_sha256": _sha256(literature_path) if literature_path.exists() else None,
    }
    destination.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
