from __future__ import annotations

import numpy as np

def _best_high_transmission_segment(
    omega: np.ndarray,
    transmission: np.ndarray,
    transmittance: np.ndarray,
    threshold: float,
    minimum_points: int,
) -> dict[str, float]:
    order = np.argsort(omega)
    ordered_omega = omega[order]
    ordered_t = transmission[order]
    ordered_power = transmittance[order]
    phase = np.unwrap(np.angle(ordered_t))
    valid = ordered_power >= threshold
    boundaries = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    records: list[dict[str, float]] = []
    for start, stop in boundaries.reshape(-1, 2):
        if stop - start < minimum_points:
            continue
        local_phase = phase[start:stop]
        local_power = ordered_power[start:stop]
        records.append(
            {
                "phase_span_pi": float(np.ptp(local_phase) / np.pi),
                "minimum_transmittance": float(np.min(local_power)),
                "maximum_reflectance": float(1.0 - np.min(local_power)),
                "omega_min_rad_per_um": float(ordered_omega[start]),
                "omega_max_rad_per_um": float(ordered_omega[stop - 1]),
                "point_count": int(stop - start),
            }
        )
    if not records:
        return {
            "phase_span_pi": 0.0,
            "minimum_transmittance": 0.0,
            "maximum_reflectance": 1.0,
            "omega_min_rad_per_um": float("nan"),
            "omega_max_rad_per_um": float("nan"),
            "point_count": 0,
        }
    return max(records, key=lambda item: item["phase_span_pi"])
