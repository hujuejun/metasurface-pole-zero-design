"""Shared numerical definitions required by the published studies."""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks, peak_widths

def resonance_estimates(wavelengths: np.ndarray, eigenvalue: np.ndarray, *, maximum_count: int=4) -> list[dict[str, float]]:
    order = np.argsort(2.0 * np.pi / wavelengths)
    omega = 2.0 * np.pi / wavelengths[order]
    phase = np.unwrap(np.angle(eigenvalue[order]))
    delay = np.abs(np.gradient(phase, omega))
    baseline = float(np.percentile(delay, 15.0))
    resonant_delay = np.maximum(delay - baseline, 0.0)
    prominence = max(0.5, 0.08 * float(np.max(resonant_delay)))
    indices, properties = find_peaks(resonant_delay, prominence=prominence, distance=max(3, wavelengths.size // 50))
    if not indices.size:
        return []
    widths = peak_widths(resonant_delay, indices, rel_height=0.5)[0]
    omega_step = float(np.median(np.diff(omega)))
    records = []
    for local, index in enumerate(indices):
        records.append({'omega_rad_per_um': float(omega[index]), 'wavelength_um': float(2.0 * np.pi / omega[index]), 'linewidth_rad_per_um': float(widths[local] * omega_step), 'peak_delay_um': float(resonant_delay[index]), 'prominence': float(properties['prominences'][local])})
    return sorted(records, key=lambda record: record['prominence'], reverse=True)[:maximum_count]
