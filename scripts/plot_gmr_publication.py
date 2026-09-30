"""Plot final GMR family and six-method cost results from freshly generated files."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--delay", type=Path, help="Separate operating-point results from the full design workflow")
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    family = json.loads(args.family.read_text())["cases"]
    if args.delay:
        delays = json.loads(args.delay.read_text())["cases"]
    else:
        delays = [dict(target_Q=r["quality_target"], **r["moderate_transmission"]) for r in family]
    costs = list(csv.DictReader(args.cost.open()))
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                        "pdf.fonttype": 42, "ps.fonttype": 42})
    q = np.asarray([r["achieved_quality_factor"] for r in family])
    fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.4), layout="constrained")
    for key, label in (("final_period_um", "Period"), ("slab_thickness_um", "Slab"), ("ridge_height_um", "Ridge")):
        ax[0].plot(q, [1000*r[key] for r in family], "o-", label=label)
    ax[0].set(xscale="log", yscale="log", xlabel="Verified Q", ylabel="Dimension (nm)", title="(a) Binary geometry")
    ax[0].legend(frameon=False)
    ax[1].plot(q, [r["held_out"]["T"] for r in delays], "o-", color="#247d85")
    ax[1].axhline(.5, color="0.55", linestyle=":")
    ax[1].set(xscale="log", ylim=(0, 1.05), xlabel="Verified Q", ylabel="Transmittance", title="(b) Operating-point transmission")
    ax[2].loglog(q, [r["held_out"]["delay_fs"] for r in delays], "o-", label="Transmitted delay")
    ax[2].loglog(q, [3.33564095198152/(5*abs(r["channel_pole"]["imag"])) for r in family], "--", label="Isolated pole")
    ax[2].set(xlabel="Verified Q", ylabel="Group delay (fs)", title="(c) Delay at the same point")
    ax[2].legend(frameon=False)
    fig.savefig(args.output/"gmr_family.pdf")
    fig.savefig(args.output/"gmr_family.png", dpi=180)
    plt.close(fig)
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
