"""Publication scope and protocol checks without costly production reruns."""
import json
from pathlib import Path
import numpy as np
from scripts.m164_gmr_pole_adjoint import build_setup
from scripts.m218b_ipdip_200nm_adjoint import physical_contour
from scripts.m7_binary_mesh_reopt import _master_parameterization

ROOT=Path(__file__).resolve().parents[1]

def read(path):return json.loads((ROOT/path).read_text())

def test_benchmark_is_final_three_method_fine_grid():
    plan=read('inputs/gmr/benchmark_inputs.json')
    assert plan['lateral_cells']==384 and plan['expected_axial_cells']==1280
    assert plan['methods']==['pole','direct3','direct5']
    assert len(plan['cases'])==2
    assert all(len(case['starts'])==12 for case in plan['cases'])
    offsets=[]
    for case in plan['cases']:
        ref=case['reference']
        offsets.append(np.array(case['starts'])-[ref['slab_thickness_um'],ref['ridge_height_um']])
    np.testing.assert_allclose(offsets[0],offsets[1],atol=2e-17,rtol=0)

def test_release_has_no_superseded_study_cases():
    cases=read('cases.json')
    assert not any(k.startswith('fp-') for k in cases)
    assert not any(k.startswith('pcm-') and not k.startswith('pcm-rectangular') for k in cases)
    assert {'ipdip-optimize','ipdip-replay','gmr-benchmark'} <= cases.keys()

def test_ipdip_seed_and_final_contours_are_exactly_reconstructible():
    source=read('configs/ipdip_geometry.json');config=read('configs/ipdip_optimize.json')
    setup,_=_master_parameterization(source)
    cell=read('inputs/ipdip/dimensions.json')['fine_contour_cell_um']
    for filename in ['seed_geometry.npz','final_geometry.npz']:
        with np.load(ROOT/'inputs/ipdip'/filename) as saved:
            contour,_,_,gate=physical_contour(saved['density'],setup,config,cell)
            assert gate['passes']
            np.testing.assert_array_equal(contour,saved['mask_fine'])
