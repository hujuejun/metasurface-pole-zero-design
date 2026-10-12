"""Named entry points. No solve runs unless --run is explicitly supplied."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def main():
    cases=json.loads((ROOT/'cases.json').read_text())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=sorted(cases))
    parser.add_argument('--run',action='store_true',help='Execute, otherwise print the command only')
    parser.add_argument('--output',type=Path,help='Override output directory (dependent cases still use documented defaults)')
    parser.add_argument('--nk-csv',type=Path,help='User-supplied IP-Dip n,k table for ipdip-replay')
    args=parser.parse_args()
    case=cases[args.case]
    if args.nk_csv is not None and args.case!='ipdip-replay':
        parser.error('--nk-csv applies only to ipdip-replay')
    if args.run and args.case=='ipdip-replay' and args.nk_csv is None:
        parser.error('ipdip-replay requires --nk-csv PATH; measured data are not bundled')
    if args.case=='disk-fdtd' and args.run:
        parser.error('Cloud execution is not supported by this wrapper. See README for explicit cost-estimation and run commands.')
    command=[sys.executable,'-m',case['module'],'--config',case['config'],'--output',str(args.output or case['output'])]
    command += case.get('extra_args', [])
    if args.nk_csv is not None:
        command += ['--nk-csv',str(args.nk_csv.resolve())]
    elif args.case=='ipdip-replay':
        command += ['--nk-csv','PATH_TO_IPDIP_NK.csv']
    print(subprocess.list2cmdline(command),flush=True)
    if args.run:
        env=os.environ.copy()
        env['PYTHONPATH']=str(ROOT/'src')
        env['MPLBACKEND']='Agg'
        subprocess.run(command,cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
