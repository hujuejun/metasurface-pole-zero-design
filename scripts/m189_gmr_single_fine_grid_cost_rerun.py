"""Shared numerical definitions required by the published studies."""
from __future__ import annotations
from scripts.m172_gmr_frozen_fdfd_mesh_lift import _setup

def setup_for_quality(quality: int, plan: dict, base: dict, reference: dict):
    config = {**base, 'period_um': float(reference['final_period_um'])}
    mesh = {'reference_axial_cell_size_um': plan['reference_axial_cell_size_um'], 'reference_lateral_cells': plan['reference_lateral_cells'], 'reference_total_domain_um': plan['reference_total_domain_um']}
    setup = _setup(float(config['period_um']), int(plan['lateral_cells']), mesh, base)
    if setup.grid.nz != int(plan['expected_axial_cells']):
        raise RuntimeError(f'Q{quality} axial cell count differs from frozen protocol')
    return (config, setup)
