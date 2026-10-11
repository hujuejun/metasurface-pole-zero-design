"""Regression checks for physical-array rather than quotient connectivity."""

import numpy as np

from pzid.printability2d import (
    backbone_width_audit,
    fill_axially_enclosed_voids,
    tiled_topology_audit,
)


def test_boundary_straddling_rod_is_not_array_connected() -> None:
    mask = np.zeros((20, 12), dtype=bool)
    mask[3:17, :3] = True
    mask[3:17, -3:] = True
    audit = tiled_topology_audit(mask)
    assert audit["central_solid_component_count"] == 2
    assert audit["array_spanning_solid_component_count"] == 0
    assert audit["axially_through_void_present"]
    assert not audit["passes_periodic_cell_topology"]


def test_web_connects_rods_and_vents_both_exteriors() -> None:
    mask = np.zeros((20, 12), dtype=bool)
    mask[3:17, :3] = True
    mask[3:17, -3:] = True
    mask[8:12, :] = True
    audit = tiled_topology_audit(mask)
    assert audit["central_solid_component_count"] == 1
    assert audit["array_spanning_solid_component_count"] == 1
    assert not audit["axially_through_void_present"]
    assert audit["central_unvented_void_count"] == 0
    assert audit["passes_periodic_cell_topology"]


def test_fill_enclosed_void_but_not_exterior() -> None:
    mask = np.zeros((20, 12), dtype=bool)
    mask[6:14, :] = True
    mask[9:11, 5:7] = False
    assert tiled_topology_audit(mask)["central_unvented_void_count"] == 1
    repaired = fill_axially_enclosed_voids(mask)
    assert np.all(repaired[9:11, 5:7])
    assert not np.any(repaired[0])
    assert not np.any(repaired[-1])
    assert tiled_topology_audit(repaired)["passes_periodic_cell_topology"]


def test_backbone_width_gate_distinguishes_thin_web() -> None:
    thin = np.zeros((20, 12), dtype=bool)
    thin[9:11, :] = True
    thick = np.zeros((20, 12), dtype=bool)
    thick[7:13, :] = True
    thin_audit = backbone_width_audit(
        thin, dx_um=0.1, dz_um=0.1, minimum_neck_um=0.5
    )
    thick_audit = backbone_width_audit(
        thick, dx_um=0.1, dz_um=0.1, minimum_neck_um=0.5
    )
    assert not thin_audit["eroded_scaffold_spans_array"]
    assert thick_audit["eroded_scaffold_spans_array"]
    assert thick_audit["all_load_bearing_cores_connected"]


def test_spanning_web_does_not_certify_detached_load_bearing_lobe() -> None:
    mask = np.zeros((30, 20), dtype=bool)
    mask[12:18, :] = True
    mask[2:9, 7:14] = True
    mask[9:12, 10] = True  # one-cell attachment to a substantial lobe
    audit = backbone_width_audit(
        mask, dx_um=0.1, dz_um=0.1, minimum_neck_um=0.5
    )
    assert audit["eroded_scaffold_spans_array"]
    assert audit["eroded_core_central_component_count"] == 2
    assert not audit["all_load_bearing_cores_connected"]
