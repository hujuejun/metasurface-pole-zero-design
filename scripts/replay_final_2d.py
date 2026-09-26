"""Solve final freeform permittivity designs without rerunning optimization."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from pzid.twoport import solve_two_port_scattering
from scripts.m5_optimize_coupled_bars import build_optimization_problem
from scripts.m7_binary_mesh_reopt import _master_parameterization, _target_parameterization, _homogeneous_monitor_setup
from scripts.m27d_optimize_fan_1d_pole_zero import _build_problem

CONFIGS={'high_index_continuous':'m5_coupled_bar_optimize_tm','high_index_binary':'m7_binary_mesh_reopt_n64_tm','low_index_binary':'low_index_binary','qbic':'m27d_fan_1d_pole_zero_adjoint_tm'}

def build(case,mesh):
    config=json.loads(Path(f'configs/{CONFIGS[case]}.json').read_text())
    suffix=''
    if case.endswith('_binary'):
        if mesh not in config['continuation']['validation_cells_per_period']:
            raise ValueError('Choose a bundled final validation mesh')
        master,reference=_master_parameterization(config)
        setup,_=_target_parameterization(config,master,reference.design_rows,mesh,config['continuation']['stages'][-1]['projection_beta'])
        suffix=f'_n{mesh}'
    elif case=='qbic':
        _,_,_,setup,_,_,_=_build_problem(config)
    else:
        _,_,setup,_,_=build_optimization_problem(config,cells_per_period=32)
    epsilon=np.loadtxt(f'inputs/final_permittivity/{case}{suffix}.csv',delimiter=',')
    if case.endswith('_binary'):setup,_=_homogeneous_monitor_setup(epsilon,setup)
    assert epsilon.shape==setup.grid.shape,(epsilon.shape,setup.grid.shape)
    return epsilon,setup

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=CONFIGS)
    parser.add_argument('--mesh',type=int,default=64)
    parser.add_argument('--k-min',type=float,required=True,help='vacuum angular frequency/c in radians per micrometer')
    parser.add_argument('--k-max',type=float,required=True)
    parser.add_argument('--points',type=int,default=241)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.points<2 or args.k_max<=args.k_min:parser.error('Need at least two points and k-max > k-min')
    epsilon,setup=build(args.case,args.mesh)
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for k in np.linspace(args.k_min,args.k_max,args.points):
        response=solve_two_port_scattering(float(k),epsilon,setup).matrix
        empty=solve_two_port_scattering(float(k),np.full_like(epsilon,setup.port_index**2),setup).matrix[1,0]
        t=response[1,0];r=response[0,0]
        rows.append([k,abs(t)**2,abs(r)**2,t.real,t.imag,r.real,r.imag,empty.real,empty.imag])
    a=np.asarray(rows)
    phase=np.unwrap(np.angle(a[:,3]+1j*a[:,4]))/np.pi
    deembedded=np.unwrap(np.angle((a[:,3]+1j*a[:,4])/(a[:,7]+1j*a[:,8])))/np.pi
    with (args.output/'spectrum.csv').open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['k_rad_per_um','T','R','t_real','t_imag','r_real','r_imag','empty_t_real','empty_t_imag','phase_pi','deembedded_phase_pi'])
        writer.writerows(np.column_stack([a,phase,deembedded]))
    print(args.output/'spectrum.csv')

if __name__=='__main__':main()
