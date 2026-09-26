"""Fresh Maxwell replay of final FP geometries; never reports optimizer costs."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.m142e_optimize_patterned_q_poles import build_fixed_grid, _verify_final, _write_curve

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality',type=int,choices=[50,100,200,400],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    high=args.quality==400
    config=json.loads(Path('configs/'+('m158_q400_fine_pole_adjoint' if high else 'm142i_patterned_q_fine_mesh_adjoint')+'.json').read_text())
    material=json.loads(Path(config['source_analytic_config']).read_text())
    seed=json.loads(Path('inputs/fp/'+('q400_start_final' if high else 'fine_starts_final')+'.json').read_text())
    case=next(x for x in seed['cases'] if x['target_quality_factor']==args.quality)
    _,scalar,_=build_fixed_grid(config,material)
    verification,curves,epsilon=_verify_final(np.asarray(case['optimized_coordinates']),complex(**case['target_pole']),config,material,scalar)
    args.output.mkdir(parents=True,exist_ok=False)
    _write_curve(args.output/f'spectrum_q{args.quality:03d}.csv',curves)
    np.save(args.output/f'permittivity_q{args.quality:03d}.npy',epsilon)
    (args.output/'summary.json').write_text(json.dumps({'method':'final geometry replay; no optimization cost measurement','cases':[{**case,'verification':verification}]},indent=2)+'\n')

if __name__=='__main__':main()
