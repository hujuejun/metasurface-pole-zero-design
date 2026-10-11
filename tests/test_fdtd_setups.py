"""Optional, local-only validation of portable FDTD definitions."""
import json
from pathlib import Path
import pytest

def test_all_saved_fdtd_definitions_are_simulations():
    td=pytest.importorskip('tidy3d')
    root=Path(__file__).resolve().parents[1]
    pairs=json.loads((root/'configs/fdtd_validation.json').read_text())
    for name,pair in pairs.items():
        sims={role:td.Simulation.from_file(root/path) for role,path in pair.items()}
        for sim in sims.values():
            sim.validate_pre_upload()
            assert sim.type=='Simulation'
            assert not sim.attrs
            assert {'reflected_flux','transmitted_flux'} <= {m.name for m in sim.monitors}
        assert sims['device'].size==sims['reference'].size
        assert len(sims['reference'].structures)==0
