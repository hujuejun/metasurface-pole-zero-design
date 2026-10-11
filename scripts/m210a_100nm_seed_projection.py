"""Shared numerical definitions required by the published studies."""
from __future__ import annotations
import numpy as np
from pzid.feature_projection2d import feature_opening_audit
from pzid.printability2d import backbone_width_audit, tiled_topology_audit
from scripts.m205i_contour_feature_audit import _widest_spanning_width

def geometry_audit(mask_fine: np.ndarray, *, fine_cell_um: float, config: dict) -> dict:
    factor = int(config['contour_refinement'])
    start, stop = (24 * factor, 168 * factor)
    feature = feature_opening_audit(mask_fine, cell_um=fine_cell_um, diameter_um=float(config['minimum_solid_and_air_feature_um']), design_rows=(start, stop))
    topology = tiled_topology_audit(mask_fine)
    width = backbone_width_audit(mask_fine, dx_um=fine_cell_um, dz_um=fine_cell_um, minimum_neck_um=float(config['minimum_load_bearing_neck_um']))
    spanning = _widest_spanning_width(mask_fine, fine_cell_um)
    opposite_start, opposite_stop = map(int, config['opposite_gap_rows'])
    opposite_web = bool(np.any(np.all(mask_fine[opposite_start * factor:opposite_stop * factor], axis=1)))
    passes = bool(feature['passes_dual_phase_opening'] and topology['passes_periodic_cell_topology'] and width['all_load_bearing_cores_connected'] and (spanning['estimated_maximum_spanning_neck_um'] >= float(config['minimum_load_bearing_neck_um'])) and (not opposite_web))
    return {'passes': passes, 'feature': feature, 'topology': topology, 'backbone': width, 'spanning_neck': spanning, 'opposite_gap_full_web': opposite_web}
