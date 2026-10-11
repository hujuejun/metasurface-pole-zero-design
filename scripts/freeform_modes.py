"""Refine parity poles and extract the normalized Hy mode maps in Fig. 2."""
import argparse
import json
from pathlib import Path
import numpy as np
from pzid.grid import UniformGrid2D
from pzid.twoport import TwoPortScatteringSetup, solve_two_port_scattering
from pzid.roots import refine_scalar_root
from pzid.modes import resonant_residue_field, parity_errors
from scripts.replay_final_2d import build, CONFIGS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case',choices=['high_index_continuous','high_index_binary','low_index_binary'])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    mesh={'high_index_continuous':32,'high_index_binary':88,'low_index_binary':56}[args.case]
    eps,old_setup=build(args.case,mesh)
    config=json.loads(Path('configs/'+CONFIGS[args.case]+'.json').read_text())
    target=config['target']
    guess=complex(target['common_pole_real_rad_per_um'],target['common_pole_imag_rad_per_um'])
    nz,nx=eps.shape; grid=UniformGrid2D(nx,nz,old_setup.grid.dx,old_setup.grid.dz)
    setup=TwoPortScatteringSetup(grid,1,nz-2,(2,3),(nz-4,nz-3),old_setup.port_index)
    fields={}; diagnostics={}
    for parity,coupling in [('even',-1),('odd',1)]:
        def response(k):
            s=solve_two_port_scattering(k,eps,setup).matrix
            return s[0,0]+coupling*s[0,1]
        root=refine_scalar_root(response,guess,kind='pole',tolerance=1e-10,
                               max_evaluations=100,search_radius=max(.2,2*abs(guess.imag)))
        if root.residual_norm>1e-6:raise RuntimeError('Parity pole refinement failed')
        samples=[resonant_residue_field(root.frequency,coupling,eps,setup,relative_offset=d) for d in [1e-5,5e-6]]
        change=float(np.linalg.norm(samples[0]-samples[1])/np.linalg.norm(samples[1]))
        field=samples[-1]; anchor=np.unravel_index(np.argmax(abs(field[nz//2:])),(nz-nz//2,nx))
        anchor=(anchor[0]+nz//2,anchor[1]);field=field*np.exp(-1j*np.angle(field[anchor]))/np.max(abs(field))
        fields[parity]=field
        diagnostics[parity]={'pole_real':root.frequency.real,'pole_imag':root.frequency.imag,
            'offset_halving_relative_change':change,'parity_errors':parity_errors(field)}
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/'modes.npz',epsilon=eps,**fields)
    (args.output/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')

if __name__=='__main__':main()
