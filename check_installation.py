"""Read-only source/import/input check; does not launch any simulation."""
from __future__ import annotations
import ast
import importlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))

def main():
    count=0
    for path in sorted((ROOT/'scripts').glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'))
        importlib.import_module('scripts.'+path.stem)
        count+=1
    def check(value):
        if isinstance(value,dict):
            for v in value.values():check(v)
        elif isinstance(value,list):
            for v in value:check(v)
        elif isinstance(value,str) and value.startswith(('configs/','inputs/')):
            assert (ROOT/value).exists(),f'Missing input: {value}'
    for path in (ROOT/'configs').glob('*.json'):check(json.loads(path.read_text()))
    import pzid.fdfd,pzid.fdfd3d
    for module in [pzid.fdfd,pzid.fdfd3d]:
        assert Path(module.__file__).resolve().is_relative_to(ROOT),module.__file__
    for name,module in list(sys.modules.items()):
        if name.startswith(('pzid.','scripts.')) and getattr(module,'__file__',None):
            assert Path(module.__file__).resolve().is_relative_to(ROOT),f'External dependency leaked in: {name}: {module.__file__}'
    print(f'PASS: {count} modules import; explicit bundled input paths resolve; both FDFD engines are local.')

if __name__=='__main__':main()
