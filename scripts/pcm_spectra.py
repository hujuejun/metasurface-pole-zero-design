"""Selected-state spectra for the final rectangular PCM pair (Fig. 4)."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.m117_symmetric_pcm_hcg_feasibility import _build_scaled_baseline, _complex_index
from scripts.m140_optimize_symmetric_pcm_single_tier import _one_tier_reference
from pzid.pcm import SymmetricPCMStateHCGParameterization, lorentz_lorenz_permittivity
from pzid.twoport import solve_two_port_scattering


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mesh',type=int,default=48)
    p.add_argument('--points',type=int,default=181)
    args=p.parse_args()
    material=json.loads(Path('configs/m123_symmetric_sb2se3_hcg_scaled_1550_material.json').read_text())
    baseline=json.loads(Path(material['baseline_summary']).read_text())
    setup,_,grid,scale,_=_build_scaled_baseline(material,baseline,cells_per_period=args.mesh)
    param=_one_tier_reference(material,grid,scale)
    geometry=json.loads(Path('inputs/pcm_rectangular_geometry.json').read_text())['geometry_um']
    raw=param.encode(*geometry)
    alpha=np.array([0,.3,.6,.9,1])
    optical=material['source_material']
    eps=lorentz_lorenz_permittivity(_complex_index(optical['amorphous_index']),
                                   _complex_index(optical['crystalline_index']),alpha)
    wavelength=np.linspace(1.46,1.64,args.points)
    t=np.empty((len(alpha),len(wavelength)),complex); r=np.empty_like(t)
    args.output.mkdir(parents=True,exist_ok=False)
    for i,e in enumerate(eps):
        mapped=SymmetricPCMStateHCGParameterization(param,complex(e)).map(raw)
        for j,lam in enumerate(wavelength):
            s=solve_two_port_scattering(2*np.pi/lam,mapped.epsilon_r,setup).matrix
            t[i,j],r[i,j]=s[1,0],s[0,0]
        print('Completed alpha',alpha[i],flush=True)
    np.savez_compressed(args.output/'spectra.npz',alpha=alpha,wavelength_um=wavelength,
                        transmission=t,reflection=r)


if __name__=='__main__':main()
