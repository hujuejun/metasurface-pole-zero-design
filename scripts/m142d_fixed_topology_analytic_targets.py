from __future__ import annotations

from scripts.m142_single_transmission_q_screen import (
    FS_PER_UM_OVER_C,
    deembedded_response,
    symmetric_cavity_layers,
)

def scaled_cavity_layers(
    pair_count: int,
    wavelength_um: float,
    high_index: float,
    low_index: float,
    mirror_scale: float,
    cavity_scale: float,
) -> tuple[tuple[float, float], ...]:
    """Scale every mirror layer and the central cavity independently."""

    reference = symmetric_cavity_layers(pair_count, wavelength_um, high_index, low_index)
    middle = 2 * pair_count
    return tuple(
        (index, thickness * (cavity_scale if position == middle else mirror_scale))
        for position, (index, thickness) in enumerate(reference)
    )
