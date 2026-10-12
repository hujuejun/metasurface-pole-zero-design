"""Self-contained material parser and analytic-continuation regression checks."""

import json
from pathlib import Path

import numpy as np

from pzid.measured_dispersion import (
    OpticalConstants, PassiveLorentzModel, fit_ipdip_lorentz, load_nk_csv,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = PassiveLorentzModel.from_dict(json.loads(
    (ROOT / "inputs/ipdip/material_model.json").read_text(encoding="utf-8"))['model'])


def synthetic_data() -> OpticalConstants:
    wavelengths = np.linspace(3.0, 6.0, 301)
    index = np.asarray(MODEL.index(2 * np.pi / wavelengths))
    return OpticalConstants(wavelengths, index.real, index.imag)


def test_user_supplied_table_parser_is_numeric_passive_and_ordered(tmp_path: Path) -> None:
    source = synthetic_data()
    csv_path = tmp_path / "synthetic_nk.csv"
    csv_path.write_text("lambda_um,n,k\n" + "".join(
        f"{lam:.8f},{real:.12f},{imag:.12f}\n"
        for lam, real, imag in zip(source.wavelength_um, source.n, source.k, strict=True)
    ), encoding="utf-8")
    data = load_nk_csv(csv_path)
    assert len(data.wavelength_um) == 301
    assert np.allclose(data.wavelength_um, source.wavelength_um)
    assert np.allclose(data.n, source.n)
    assert np.allclose(data.k, source.k)


def test_passive_fit_recovers_synthetic_device_band_and_round_trips() -> None:
    data = synthetic_data()
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
    model = MODEL
    upper = complex(2 * np.pi / 5.0, 0.025)
    reversed_medium = model.reversed_epsilon(upper)
    physical_lower = model.epsilon(upper.conjugate())
    assert np.isclose(reversed_medium, np.conj(physical_lower))
    assert not np.isclose(reversed_medium, model.epsilon(upper))
