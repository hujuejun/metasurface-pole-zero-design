"""Recompute the final planar HCG spectrum from physical interface coordinates."""
import argparse
import json
from pathlib import Path
from scripts.m58d_optimize_two_tier_planar import _build, _spectrum, _write_spectrum

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(Path('configs/m58d_two_tier_planar_singularity_homotopy_tm.json').read_text())
    geometry=json.loads(Path('inputs/planar_hcg_geometry.json').read_text())['final_geometry']
    _,setup,parameterization,_=_build(config)
    epsilon=parameterization.map(parameterization.encode(*geometry)).epsilon_r
    spectrum=_spectrum(epsilon,setup,config)
    args.output.mkdir(parents=True,exist_ok=False)
    _write_spectrum(args.output/'spectrum.csv',spectrum)

if __name__=='__main__':main()
