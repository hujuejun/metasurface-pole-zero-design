"""TM scattering of the final connected IP-Dip contour using measured n,k."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from pzid.feature_projection2d import conservative_binary_permittivity
from pzid.measured_dispersion import load_nk_csv
from pzid.twoport import solve_two_port_scattering
from scripts.m5_optimize_coupled_bars import build_optimization_problem
from scripts.m6_fabrication_continue import _scaled_port_setup
from scripts.m7_binary_mesh_reopt import _homogeneous_monitor_setup, _master_parameterization
from scripts.m214_m210b_index_sensitivity import reference_factor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/ipdip_geometry.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mesh', type=int, default=96, choices=[32,48,64,96,128])
    parser.add_argument('--wavelengths', type=float, nargs='+', help='Optional spot checks, in micrometers')
    args = parser.parse_args()
    source = json.loads(args.config.read_text())
    dimensions = json.loads(Path('inputs/ipdip/dimensions.json').read_text())
    with np.load('inputs/ipdip/final_geometry.npz') as saved:
        contour = saved['mask_fine'].astype(bool)
    material = load_nk_csv(Path('inputs/ipdip/optical_constants.csv'))
    reference, _ = _master_parameterization(source)
    _, _, candidate, _, _ = build_optimization_problem(source, cells_per_period=args.mesh)
    setup = _scaled_port_setup(reference, candidate.grid)
    binary = conservative_binary_permittivity(contour, target_shape=setup.grid.shape,
                                               low_index=1.0, high_index=1.5)
    fraction = (binary-1)/(1.5**2-1)
    setup, shifts = _homogeneous_monitor_setup(binary, setup)
    scale = dimensions['physical_period_um']/source['benchmark']['period_um']
    wavelength = np.array(args.wavelengths) if args.wavelengths else (
        np.array([4.70,4.85,5,5.10,5.15,5.20,5.25,5.30,5.40]) if args.mesh==128
        else np.unique(np.r_[np.linspace(4.65,5.44,161),4.7,5,5.4]))
    if np.any(wavelength < material.wavelength_um.min()) or np.any(wavelength > material.wavelength_um.max()):
        raise ValueError('Requested wavelength outside measured material range')
    empty = reference_factor(wavelength, setup, scale)
    args.output.mkdir(parents=True, exist_ok=False)
    t, r = [], []
    for j, lam in enumerate(wavelength):
        index = np.interp(lam, material.wavelength_um, material.n) + 1j*np.interp(lam, material.wavelength_um, material.k)
        matrix = solve_two_port_scattering(2*np.pi*scale/lam, 1+fraction*(index**2-1), setup).matrix
        t.append(matrix[1,0]/empty[j]); r.append(matrix[0,0])
        print(f'{j+1}/{len(wavelength)} wavelength={lam:.6f} um T={abs(t[-1])**2:.8f} R={abs(r[-1])**2:.8f}', flush=True)
    t, r = np.array(t), np.array(r)
    np.savez_compressed(args.output/'spectrum.npz', wavelength_um=wavelength,
                        transmission=t, reflection=r, grid_shape=setup.grid.shape)
    np.savetxt(args.output/'spectrum.csv', np.c_[wavelength, abs(t)**2, abs(r)**2,
        1-abs(t)**2-abs(r)**2, np.unwrap(np.angle(t))], delimiter=',',
        header='wavelength_um,T,R,A,phase_rad', comments='')
    (args.output/'metadata.json').write_text(json.dumps({'mesh':args.mesh,
        'period_um':dimensions['physical_period_um'], 'monitor_shifts':shifts,
        'polarization':'TM (Hy)', 'material':'measured n,k; direct interpolation'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
