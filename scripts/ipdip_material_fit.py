"""Refit the passive Lorentz model used for complex-frequency IP-Dip design."""
import argparse
import json
from pathlib import Path
from pzid.measured_dispersion import fit_ipdip_lorentz, load_nk_csv

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    model,diagnostics=fit_ipdip_lorentz(load_nk_csv(Path('inputs/ipdip/optical_constants.csv')))
    if not diagnostics['fit_success']:raise RuntimeError('Material fit did not converge')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'model.json').write_text(json.dumps({'model':model.as_dict(),
        'fit_diagnostics':diagnostics},indent=2)+'\n')

if __name__=='__main__':main()
