"""Plot final GMR family and six-method cost results from freshly generated files."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def plot_family(family, delays, spectra_dir, output, recompute_spectra=False):
    """Compare independent responses with specifications from target poles."""
    delay_by_q = {r['target_Q']: r for r in delays}
    # c in micrometres / picosecond: full pole linewidth in THz is c*gamma/pi.
    target_gamma = np.array([abs(r['target_pole']['imag']) for r in family])
    verified_gamma = np.array([abs(r['held_out_pole']['imag']) for r in family])
    target_width = 299.792458 * target_gamma / np.pi
    verified_width = 299.792458 * verified_gamma / np.pi
    target_delay = 3.33564095198152 / (5 * target_gamma)
    verified_delay = np.array([delay_by_q[r['quality_target']]['held_out']['delay_fs'] for r in family])
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.8), layout='constrained')
    for quality, color in zip((50, 200, 800), ('#247d85', '#d08432', '#75549b')):
        if recompute_spectra:
            from scripts.m166_gmr_high_fidelity_analysis import _solver
            row = next(r for r in family if r['quality_target'] == quality)
            physics = {'high_index': 2., 'port_index': 1., 'ridge_fill_factor': .66}
            response = _solver(physics, row['slab_thickness_um'], row['ridge_height_um'], row['final_period_um'], 81)
            pole = complex(row['channel_pole']['real'], row['channel_pole']['imag'])
            grid = np.unique(np.r_[2000*np.pi/np.linspace(905,955,161),
                                    pole.real + abs(pole.imag)*np.linspace(-8,8,161)])
            grid = grid[(grid >= 2000*np.pi/955) & (grid <= 2000*np.pi/905)]
            transmission = np.array([abs(response(float(k)).transmission)**2 for k in grid])
            np.savetxt(output/f'spectrum_q{quality:04d}.csv', np.c_[grid,transmission],
                       delimiter=',', header='omega_rad_per_um,T', comments='')
            print(f'RCWA spectrum regenerated for target Q={quality}', flush=True)
            spectra_dir = output
        source = spectra_dir / f'spectrum_q{quality:04d}.csv'
        if not source.exists():
            source = spectra_dir / f'q{quality}_spectrum.csv'
        spectrum = np.genfromtxt(source, delimiter=',', names=True)
        frequency = spectrum['omega_rad_per_um'] if 'omega_rad_per_um' in spectrum.dtype.names else spectrum['k_per_um']
        wavelength = 2000 * np.pi / frequency
        order = np.argsort(wavelength)
        ax[0].plot(wavelength[order], spectrum['T'][order], lw=1.8,
                   color=color, label=rf'$Q_{{\rm target}}={quality}$')
    ax[0].set(xlim=(905, 955), ylim=(-.03, 1.48), yticks=np.linspace(0,1,6), xlabel='Wavelength (nm)',
              ylabel='Transmittance', title='(a) Transmission spectra')
    ax[0].legend(frameon=False, fontsize=8, loc='upper right')
    inset = ax[0].inset_axes([.03, .74, .43, .25])
    inset.set(xlim=(-.65, .65), ylim=(-.18, .73))
    inset.add_patch(Rectangle((-.65, 0), 1.3, .17, facecolor='#8bb4c5', edgecolor='#285b70'))
    for x in (-1, 0, 1):
        inset.add_patch(Rectangle((x-.33, .17), .66, .20, facecolor='#8bb4c5', edgecolor='#285b70'))
    inset.text(0, .06, '$n=2$', ha='center', va='center', fontsize=8)
    inset.text(-.58, .51, 'air', fontsize=8)
    inset.annotate('', xy=(.10,.40), xytext=(.10,.72), arrowprops={'arrowstyle':'->','lw':1})
    inset.annotate('', xy=(.50,-.08), xytext=(-.50,-.08), arrowprops={'arrowstyle':'<->','lw':.8})
    inset.text(0, -.11, '$a$', va='top', ha='center', fontsize=8)
    inset.axis('off')
    for axis, target, verified, title, xlabel, ylabel in (
        (ax[1], target_width, verified_width, '(b) Linewidth specification', 'Target linewidth (THz)', 'Verified linewidth (THz)'),
        (ax[2], target_delay, verified_delay, '(c) Group-delay specification', 'Pole-implied target delay (fs)', 'Verified group delay (fs)'),
    ):
        bounds = (min(target.min(), verified.min())*.78, max(target.max(), verified.max())*1.25)
        axis.loglog(bounds, bounds, '--', color='.5', lw=1.2, label='Target = verified')
        axis.loglog(target, verified, 'o', color='#246c94', ms=6, label='RCWA verification')
        axis.set(xlim=bounds, ylim=bounds, title=title, xlabel=xlabel, ylabel=ylabel)
        axis.legend(frameon=False, fontsize=8, loc='upper left')
        axis.grid(which='major', alpha=.16)
    fig.savefig(output/'gmr_family.pdf')
    fig.savefig(output/'gmr_family.png', dpi=220)
    plt.close(fig)
    np.savetxt(output/'gmr_family_comparison.csv',
               np.c_[[r['quality_target'] for r in family], target_width, verified_width, target_delay, verified_delay],
               delimiter=',', header='target_Q,target_linewidth_THz,verified_linewidth_THz,pole_implied_target_delay_fs,verified_delay_fs', comments='')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--delay", type=Path, help="Separate operating-point results from the full design workflow")
    parser.add_argument("--cost", type=Path, help="Omit to regenerate only the family figure")
    parser.add_argument("--spectra-dir", type=Path, help="Defaults to the family summary directory")
    parser.add_argument("--recompute-spectra", action="store_true", help="Fresh RCWA exemplar spectra on a common wavelength window")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    family = json.loads(args.family.read_text())["cases"]
    if args.delay:
        delays = json.loads(args.delay.read_text())["cases"]
    else:
        delays = [dict(target_Q=r["quality_target"], **r["moderate_transmission"]) for r in family]
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                        "pdf.fonttype": 42, "ps.fonttype": 42})
    plot_family(family, delays, args.spectra_dir or args.family.parent, args.output, args.recompute_spectra)
    if args.cost is None:
        return
    costs = list(csv.DictReader(args.cost.open()))
    methods = ["pole", "pole+background", "direct3", "direct5", "direct3_physical", "direct5_physical_delay"]
    labels = ["Pole", "Pole +\nbackground", "Direct 3", "Direct 5", "Pinned\ndirect 3", "Pinned\ndirect 5"]
    colors = ["#246c94", "#7a5698", "#ce7937", "#4b8b57", "#b74e48", "#8b7b45"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6), layout="constrained", sharex="col")
    for col, quality in enumerate((200, 800)):
        rows = {r["method"]: r for r in costs if int(r.get("quality_target", 0)) == quality}
        qualified = [int(rows[m]["qualified"]) for m in methods]
        lu = [int(rows[m].get("fdfd_factorizations", rows[m].get("total_method_specific_fdfd_frequency_factorizations"))) for m in methods]
        axes[0,col].bar(range(6), qualified, color=colors)
        axes[1,col].bar(range(6), lu, color=colors)
        for i, (a,b) in enumerate(zip(qualified,lu)):
            axes[0,col].text(i,a+.2,str(a),ha="center",fontsize=9)
            axes[1,col].text(i,b+90,str(b),ha="center",fontsize=9)
        axes[0,col].set(ylim=(0,13), title=f"Target Q = {quality}", ylabel="Qualified designs / 12")
        axes[1,col].set(ylim=(0,4700), ylabel="FDFD factorizations", xticks=range(6), xticklabels=labels)
        axes[1,col].tick_params(axis="x", labelsize=8)
    fig.savefig(args.output/"gmr_cost.pdf")
    fig.savefig(args.output/"gmr_cost.png", dpi=180)


if __name__ == "__main__":
    main()
