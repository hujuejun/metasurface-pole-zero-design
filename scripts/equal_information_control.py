"""SI equal-information compact-model control, restricted to Q=100 and 200."""
import json
from pathlib import Path
import numpy as np
from scripts.m152_analyze_saved_single_resonance_controls import (
    load_curve, sample_indices, pole_model, direct_pade_2_1, errors,
)

def main():
    source=Path('runs/m142i_fine_mesh_adjoint')
    results=[]
    for quality in [100,200]:
        _,detuning,t=load_curve(source/f'spectrum_q{quality:03d}.csv','t_real','t_imag')
        pole_train=sample_indices(detuning,[-3.,0.,3.])
        direct_train=sample_indices(detuning,[-3.,-1.,1.,3.])
        heldout=np.ones(len(detuning),dtype=bool)
        heldout[np.union1d(pole_train,direct_train)]=False
        results.append({'target_quality_factor':quality,
            'pole_informed':errors(t,pole_model(detuning,t,pole_train),heldout,.1),
            'direct_rational_2_1':errors(t,direct_pade_2_1(detuning,t,direct_train),heldout,.1)})
    output=Path('runs/equal_information_control')
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps({'cases':results},indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
