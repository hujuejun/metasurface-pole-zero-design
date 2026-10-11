import numpy as np

from pzid.feature_projection2d import (
    area_averaged_binary_permittivity,
    conservative_binary_permittivity,
    feature_opening_audit,
    periodic_disk_opening,
    project_dual_phase_minimum_feature,
)


def test_undersized_solid_and_air_tips_are_removed() -> None:
    solid = np.zeros((40, 32), dtype=bool)
    solid[10:30, :] = True
    solid[9, 12] = True
    solid[10, 13] = False
    before = feature_opening_audit(solid, cell_um=0.02, diameter_um=0.1,
                                   design_rows=(5, 35))
    assert before["solid_removed_pixels"] > 0
    assert before["air_removed_pixels"] > 0
    projected, audit = project_dual_phase_minimum_feature(
        solid, cell_um=0.02, diameter_um=0.1, design_rows=(5, 35),
    )
    assert audit["passes_dual_phase_opening"]
    assert np.array_equal(projected, project_dual_phase_minimum_feature(
        projected, cell_um=0.02, diameter_um=0.1, design_rows=(5, 35),
    )[0])


def test_periodic_seam_is_not_an_artificial_edge() -> None:
    mask = np.zeros((30, 24), dtype=bool)
    mask[10:20, :3] = True
    mask[10:20, -3:] = True
    opened = periodic_disk_opening(mask, cell_um=0.02, diameter_um=0.08)
    assert np.any(opened[:, 0])
    assert np.any(opened[:, -1])


def test_area_average_uses_binary_permittivity_not_index_average() -> None:
    fine = np.zeros((4, 4), dtype=bool)
    fine[:2, :2] = True
    epsilon = area_averaged_binary_permittivity(
        fine, coarse_shape=(2, 2), low_index=1.0, high_index=1.5,
    )
    assert epsilon[0, 0] == 2.25
    assert epsilon[0, 1] == 1.0


def test_noninteger_binary_contour_transfer_preserves_area_and_seam() -> None:
    fine = np.zeros((8, 8), dtype=bool)
    fine[2:6, :2] = True
    fine[2:6, -2:] = True
    epsilon = conservative_binary_permittivity(
        fine, target_shape=(6, 3), low_index=1.0, high_index=1.5,
    )
    fraction = (epsilon - 1.0) / 1.25
    assert abs(float(np.mean(fraction)) - float(np.mean(fine))) < 1e-12
    assert np.any(fraction[:, 0] > 0.0)
    assert np.any(fraction[:, -1] > 0.0)
    assert np.all((fraction >= 0.0) & (fraction <= 1.0))


def test_integer_binary_contour_transfer_reproduces_existing_average() -> None:
    fine = np.arange(12 * 8).reshape(12, 8) % 7 < 3
    old = area_averaged_binary_permittivity(
        fine, coarse_shape=(3, 2), low_index=1.0, high_index=1.5,
    )
    new = conservative_binary_permittivity(
        fine, target_shape=(3, 2), low_index=1.0, high_index=1.5,
    )
    assert np.array_equal(old, new)


def test_100nm_rule_rejects_undersized_long_solid_and_air_necks() -> None:
    cell = 0.008
    diameter = 0.1
    narrow_solid = np.zeros((60, 40), dtype=bool)
    narrow_solid[10:50, 14:26] = True  # 12 cells, 96 nm wide.
    assert not np.any(periodic_disk_opening(
        narrow_solid, cell_um=cell, diameter_um=diameter,
    )[20:40, 14:26])
    wide_solid = np.zeros((60, 40), dtype=bool)
    wide_solid[10:50, 14:27] = True  # 13 cells, 104 nm wide.
    assert np.any(periodic_disk_opening(
        wide_solid, cell_um=cell, diameter_um=diameter,
    )[20:40, 14:27])
    narrow_air = np.ones((60, 40), dtype=bool)
    narrow_air[10:50, 14:26] = False
    audit = feature_opening_audit(narrow_air, cell_um=cell, diameter_um=diameter,
                                  design_rows=(10, 50))
    assert audit["air_removed_pixels"] > 0
