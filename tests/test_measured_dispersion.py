"""Material-table and analytic-continuation regression checks for M218."""

from pathlib import Path

import numpy as np

from pzid.measured_dispersion import (
    PassiveLorentzModel, fit_ipdip_lorentz, load_nk_csv,
)


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "inputs/ipdip/optical_constants.csv"


def test_supplied_table_is_numeric_passive_and_ordered() -> None:
    data = load_nk_csv(CSV)
    assert len(data.wavelength_um) == 301
    assert np.allclose(data.wavelength_um[[0, -1]], [3.0, 6.0])
    at_five = np.flatnonzero(np.isclose(data.wavelength_um, 5.0))
    assert len(at_five) == 1
    assert np.isclose(data.n[at_five[0]], 1.50555)
    assert np.isclose(data.k[at_five[0]], 0.004815)


def test_passive_fit_matches_device_band_and_round_trips() -> None:
    data = load_nk_csv(CSV)
    model, report = fit_ipdip_lorentz(data)
    assert report["fit_success"]
    assert report["max_abs_n_error_assessment"] < 2e-4
    assert report["max_abs_k_error_assessment"] < 1e-4
    assert all(value > 0 for value in model.oscillator_strength)
    assert all(value > 0 for value in model.damping_omega)
    omega = 2 * np.pi / np.linspace(4.65, 5.44, 151)
    index = np.asarray(model.index(omega))
    assert np.all(index.imag >= 0)
    restored = PassiveLorentzModel.from_dict(model.as_dict())
    assert np.allclose(restored.epsilon(omega), model.epsilon(omega))


def test_time_reversed_medium_is_conjugate_of_physical_lower_partner() -> None:
    model, _ = fit_ipdip_lorentz(load_nk_csv(CSV))
    upper = complex(2 * np.pi / 5.0, 0.025)
    reversed_medium = model.reversed_epsilon(upper)
    physical_lower = model.epsilon(upper.conjugate())
    assert np.isclose(reversed_medium, np.conj(physical_lower))
    assert not np.isclose(reversed_medium, model.epsilon(upper))
