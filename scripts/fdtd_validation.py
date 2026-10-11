"""Reproduce final 2D FDTD validations from portable, account-free setup inputs.

inspect and analyze are local. estimate uploads non-running drafts. run requires
an explicit total cost ceiling and never starts a task above that ceiling.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def amplitude(data):
    if 'transmitted_specular_field' in data.monitor_data:
        field = data['transmitted_specular_field'].Ex
        x = np.asarray(field.coords['x'].values)
        return np.trapezoid(np.asarray(field.values)[:,0,0,:], x=x, axis=0)/(x[-1]-x[0])
    return np.asarray(data['transmitted_complex_field'].Ex.values).ravel()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case')
    parser.add_argument('--config',type=Path,default=Path('configs/fdtd_validation.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--action',choices=['inspect','estimate','run','analyze'],default='inspect')
    parser.add_argument('--max-cost',type=float,help='Maximum TOTAL FlexCredits explicitly authorized by the reader')
    args=parser.parse_args()
    import tidy3d as td
    cases=json.loads(args.config.read_text())
    if args.case not in cases:parser.error('Choose one of: '+', '.join(cases))
    paths={k:Path(v) for k,v in cases[args.case].items()}
    simulations={k:td.Simulation.from_file(v) for k,v in paths.items()}
    hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}
    args.output.mkdir(parents=True,exist_ok=True)
    if args.action=='inspect':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for role,sim in simulations.items():
            sim.validate_pre_upload()
            sim.plot_eps(y=0)
            plt.savefig(args.output/f'{role}-geometry.png',dpi=150,bbox_inches='tight');plt.close()
            print(role,'size',sim.size,'run_time',sim.run_time,'grid cells',sim.num_cells)
        return
    if args.action=='analyze':
        device=td.SimulationData.from_file(args.output/'device.hdf5')
        reference=td.SimulationData.from_file(args.output/'reference.hdf5')
        f=np.asarray(device['transmitted_flux'].flux.f.values)
        if not np.array_equal(f,np.asarray(reference['transmitted_flux'].flux.f.values)):
            raise ValueError('Device/reference frequency grids differ')
        incident=np.asarray(reference['transmitted_flux'].flux.values)
        if np.any(incident<=0):raise ValueError('Nonpositive incident normalization')
        t=amplitude(device)/amplitude(reference)
        T=np.asarray(device['transmitted_flux'].flux.values)/incident
        R=-np.asarray(device['reflected_flux'].flux.values)/incident
        wavelength=td.C_0/f; order=np.argsort(wavelength)
        np.savetxt(args.output/'spectrum.csv', np.c_[wavelength[order],T[order],R[order],
            1-T[order]-R[order],np.unwrap(np.angle(t[order]))], delimiter=',',
            header='wavelength_um,T,R,A,phase_rad',comments='')
        return
    import tidy3d.web as web
    quote_path=args.output/'quote.json'
    if args.action=='estimate':
        if quote_path.exists():raise FileExistsError('Use a new directory for a new quote')
        tasks={}
        for role,sim in simulations.items():
            sim.validate_pre_upload()
            task_id=web.upload(sim,task_name=f'{args.case}-{role}',verbose=True)
            cost=web.estimate_cost(task_id,verbose=True)
            if cost is None:raise RuntimeError('No valid cost estimate returned; no tasks started')
            tasks[role]={'task_id':task_id,'maximum_cost':float(cost)}
        quote={'setup_sha256':hashes,'tasks':tasks,'total_maximum_cost':sum(t['maximum_cost'] for t in tasks.values())}
        quote_path.write_text(json.dumps(quote,indent=2)+'\n')
        print('Total maximum FlexCredits:',quote['total_maximum_cost'],'; no tasks started.')
        return
    quote=json.loads(quote_path.read_text())
    if hashes!=quote['setup_sha256']:raise ValueError('Setups changed after estimation')
    if args.max_cost is None or not np.isfinite(args.max_cost) or args.max_cost<quote['total_maximum_cost']:
        parser.error('Explicit --max-cost must cover the complete saved quote')
    for role,task in quote['tasks'].items():
        task_id=task['task_id']
        state=str(web.get_info(task_id).status).lower()
        if state=='draft':web.start(task_id)
        elif state not in ('success','running','queued','preprocess'):
            raise RuntimeError(f'Unexpected task state: {state}')
        web.monitor(task_id)
        web.load(task_id,path=args.output/f'{role}.hdf5',replace_existing=False)


if __name__=='__main__':main()
