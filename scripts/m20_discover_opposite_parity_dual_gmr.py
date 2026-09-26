from __future__ import annotations

from typing import Any

import numpy as np

def _pair_record(even: dict[str, Any], odd: dict[str, Any]) -> dict[str, Any]:
    even_pole = complex(*even["root"]["frequency"])
    odd_pole = complex(*odd["root"]["frequency"])
    even_residue = complex(*even["boundary_reflection_residue"])
    odd_residue = complex(*odd["boundary_reflection_residue"])
    linewidth_even = float(even["full_linewidth_rad_per_um"])
    linewidth_odd = float(odd["full_linewidth_rad_per_um"])
    mean_linewidth = 0.5 * (linewidth_even + linewidth_odd)
    center = 0.5 * (even_pole.real + odd_pole.real)
    even_contribution = even_residue / (center - even_pole)
    odd_contribution = odd_residue / (center - odd_pole)
    denominator = max(
        abs(even_contribution) + abs(odd_contribution), np.finfo(float).tiny
    )
    relative_phase = float(np.angle(odd_residue / even_residue) / np.pi)
    phase_error = float(1.0 - abs(relative_phase))
    mismatch = float(abs(odd_pole.real - even_pole.real) / mean_linewidth)
    linewidth_ratio = float(linewidth_even / linewidth_odd)
    residue_ratio = float(abs(even_residue) / abs(odd_residue))
    coherent_sum = float(
        abs(even_contribution + odd_contribution) / denominator
    )
    ranking_score = float(
        np.log1p(mismatch)
        + abs(np.log(linewidth_ratio))
        + phase_error
        + abs(np.log(residue_ratio))
        + coherent_sum
    )
    return {
        "pole_center_rad_per_um": float(center),
        "real_frequency_separation_rad_per_um": float(
            odd_pole.real - even_pole.real
        ),
        "center_mismatch_over_mean_linewidth": mismatch,
        "linewidth_ratio_even_over_odd": linewidth_ratio,
        "boundary_residue_magnitude_ratio_even_over_odd": residue_ratio,
        "relative_boundary_residue_phase_pi": relative_phase,
        "opposite_residue_phase_error_pi": phase_error,
        "normalized_resonant_reflection_sum_at_pair_center": coherent_sum,
        "exploratory_ranking_score": ranking_score,
    }
