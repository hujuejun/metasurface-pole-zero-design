"""Check source checksums, publication scope, and accidental private information."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
FOLDERS=('src','scripts','configs','inputs','reference_results','tests','.github')
EXTENSIONS={'.py','.json','.csv','.npz','.hdf5','.yml'}

def inventory():
    files=[p for folder in FOLDERS for p in (ROOT/folder).rglob('*')
           if p.is_file() and p.suffix in EXTENSIONS and '__pycache__' not in p.parts]
    files.extend(p for p in ROOT.iterdir() if p.is_file() and (p.suffix in {'.py','.md','.toml','.txt','.cff'} or p.name in {'LICENSE','cases.json'}))
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())

def digest(path):
    data=path.read_bytes()
    if path.suffix not in {'.npz','.hdf5'}:data=data.replace(b'\r\n',b'\n')
    return hashlib.sha256(data).hexdigest()

def check_manifest(records, expected):
    """Compare exact filename/hash mappings, independent of platform ordering."""
    actual={row['file']:row['sha256'] for row in records}
    saved={row['file']:row['sha256'] for row in expected}
    if len(actual)!=len(records) or len(saved)!=len(expected):
        raise ValueError('Duplicate filename in source manifest')
    missing=sorted(saved.keys()-actual.keys())
    added=sorted(actual.keys()-saved.keys())
    changed=sorted(name for name in actual.keys() & saved.keys() if actual[name]!=saved[name])
    if missing or added or changed:
        raise ValueError('Source manifest mismatch; review then refresh it. '
                         f'Missing: {missing}; added: {added}; changed: {changed}')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest',action='store_true',help='Refresh after an intentional source change')
    args=parser.parse_args()
    files=inventory()
    records=[{'file':p.relative_to(ROOT).as_posix(),'sha256':digest(p)} for p in files]
    manifest=ROOT/'source_manifest.json'
    if args.write_manifest:
        manifest.write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    else:
        check_manifest(records, json.loads(manifest.read_text(encoding='utf-8')))
    patterns=[r'gh[pousr]_[A-Za-z0-9]{30,}',r'AKIA[0-9A-Z]{16}',
              r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'[A-Z]:[\\/]Users[\\/]',
              r'arn:aws:',r'fdve-[0-9a-f-]{20,}']
    for path in files:
        if path.suffix in {'.npz','.hdf5'} or path.resolve()==Path(__file__).resolve():continue
        content=path.read_text(encoding='utf-8-sig')
        for pattern in patterns:
            assert not re.search(pattern,content),f'Possible private information in {path.relative_to(ROOT)}'
    cases=json.loads((ROOT/'cases.json').read_text())
    assert not any(name.startswith('fp-') for name in cases)
    assert (ROOT/'LICENSE').read_text().startswith('MIT License')
    assert not (ROOT/'inputs/pcm_geometry.json').exists()
    assert not (ROOT/'inputs/ipdip/optical_constants.csv').exists()
    print(f'PASS: {len(files)} files match the release manifest; scope and text hygiene checks passed.')

if __name__=='__main__':main()
