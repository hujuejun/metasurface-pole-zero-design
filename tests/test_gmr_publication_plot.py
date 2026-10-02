import numpy as np

from scripts.plot_gmr_publication import plot_family


def test_figure_targets_use_prescribed_poles_and_held_out_verification(tmp_path):
    family, delays = [], []
    for quality in (50, 200, 800):
        gamma = 6.77 / (2 * quality)
        family.append({'quality_target': quality,
                       'target_pole': {'real': 6.77, 'imag': -gamma},
                       'channel_pole': {'real': 6.77, 'imag': -3 * gamma},
                       'held_out_pole': {'real': 6.77, 'imag': -1.05 * gamma}})
        delays.append({'target_Q': quality, 'held_out': {'delay_fs': 3.33564095198152 / (5 * gamma)}})
        np.savetxt(tmp_path / f'spectrum_q{quality:04d}.csv', [[6.7, .9], [6.77, .01], [6.8, .8]],
                   delimiter=',', header='omega_rad_per_um,T', comments='')
    plot_family(family, delays, tmp_path, tmp_path)
    result = np.genfromtxt(tmp_path / 'gmr_family_comparison.csv', delimiter=',', names=True)
    assert np.allclose(result['verified_linewidth_THz'] / result['target_linewidth_THz'], 1.05)
    assert np.allclose(result['verified_delay_fs'], result['pole_implied_target_delay_fs'])
    assert (tmp_path / 'gmr_family.pdf').is_file()
