"""Shared numerical definitions required by the published studies."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import numpy as np

def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _metric(alpha: np.ndarray, transmission: np.ndarray, reflection: np.ndarray, maximum_fraction: float) -> dict[str, float]:
    chosen = alpha <= maximum_fraction + 1e-12
    t = transmission[chosen]
    r = reflection[chosen]
    phase = np.unwrap(np.angle(t))
    return {'maximum_fraction': maximum_fraction, 'minimum_transmittance': float(np.min(np.abs(t) ** 2)), 'maximum_reflectance': float(np.max(np.abs(r) ** 2)), 'phase_span_pi': float(abs(phase[-1] - phase[0]) / math.pi), 'maximum_energy_balance_error': float(np.max(np.abs(np.abs(t) ** 2 + np.abs(r) ** 2 - 1.0)))}
