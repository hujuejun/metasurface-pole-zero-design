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
    parser.add_argument('--preflight-only',action='store_true',help='FP adjoint derivative audit only')
    args=parser.parse_args()
    case=cases[args.case]
    if args.case=='disk-fdtd' and args.run:
        parser.error('Cloud execution is not supported by this wrapper. See README for explicit cost-estimation and run commands.')
    command=[sys.executable,'-m',case['module'],'--config',case['config'],'--output',str(args.output or case['output'])]
    if args.case in ['fp-optimize','fp-q400-optimize']:
        seed='fine_starts' if args.case=='fp-optimize' else 'q400_start'
        command+=['--start-from-summary',f'inputs/fp/{seed}.json']
    if args.preflight_only:
        if args.case not in ['fp-optimize','fp-q400-optimize']:
            parser.error('--preflight-only is available only for the two FP optimization cases')
        command+=['--preflight-only']
    print(subprocess.list2cmdline(command),flush=True)
    if args.run:
        env=os.environ.copy()
        env['PYTHONPATH']=str(ROOT/'src')
        env['MPLBACKEND']='Agg'
        subprocess.run(command,cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
