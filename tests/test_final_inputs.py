"""Validate final material grids and geometric seeds without production sweeps."""
import json
from pathlib import Path
import pytest
import numpy as np
from scripts.replay_final_2d import build
from scripts.m58d_optimize_two_tier_planar import _build
from scripts.m117_symmetric_pcm_hcg_feasibility import _build_scaled_baseline
from scripts.m120_symmetric_pcm_pole_zero_correction import _build_reference_parameterization

@pytest.fixture(autouse=True)
def repo_cwd(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])

@pytest.mark.parametrize('case,mesh',[('high_index_continuous',32),('qbic',32),*[('high_index_binary',n) for n in [64,72,80,88]],*[('low_index_binary',n) for n in [32,40,48,56]]])
def test_final_grid_shape(case,mesh):
    epsilon,setup=build(case,mesh)
    assert np.isfinite(epsilon).all()
    assert epsilon.shape==setup.grid.shape
    assert np.min(epsilon)>0

def test_parametric_final_geometries():
    config=json.loads(Path('configs/m58d_two_tier_planar_singularity_homotopy_tm.json').read_text())
    baseline=json.loads(Path('inputs/planar_hcg_geometry.json').read_text())
    grid,setup,param,_=_build(config)
    mapped=param.map(param.encode(*baseline['final_geometry']))
    assert mapped.epsilon_r.shape==grid.shape
    material=json.loads(Path('configs/m123_symmetric_sb2se3_hcg_scaled_1550_material.json').read_text())
    _,_,grid,scale,_=_build_scaled_baseline(material,baseline,cells_per_period=24)
    param,_=_build_reference_parameterization(material,grid,scale)
    geometry=json.loads(Path('inputs/pcm_geometry.json').read_text())['geometry_um']
    assert np.isfinite(param.map(param.encode(*geometry)).epsilon_r).all()
