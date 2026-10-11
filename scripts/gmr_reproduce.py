"""Fresh final-geometry replay and matched-start GMR comparison, without run archives."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from pzid.roots import refine_scalar_root
from scripts.m164_gmr_pole_adjoint import channel_transmission, physical_epsilon_and_derivatives
from scripts.m166_gmr_high_fidelity_analysis import _solver
from scripts.m175_gmr_moderate_transmission_delay import phase_delay
from scripts.m177_gmr_preregistered_cost_benchmark import (
    as_complex, run_method,
)
from scripts.m189_gmr_single_fine_grid_cost_rerun import setup_for_quality


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def complex_record(value):
    return {"real": float(value.real), "imag": float(value.imag)}


def physical_reference(row, physics, operating=None):
    """Recompute both Fourier poles and the held-out delay, not load results."""
    result = dict(row)
    for width, key in ((81, "channel_pole"), (101, "held_out_pole")):
        response = _solver(physics, row["slab_thickness_um"], row["ridge_height_um"], row["final_period_um"], width)
        guess = as_complex(row[key])
        pole = refine_scalar_root(lambda k: response(k).transmission, guess, kind="pole",
                                  tolerance=1e-10, max_evaluations=80,
                                  search_radius=max(0.05, 12 * abs(guess.imag)))
        if pole.residual_norm > 1e-6 or pole.frequency.imag >= 0:
            raise RuntimeError("Reference pole refinement failed")
        result[key] = complex_record(pole.frequency)
    pole = as_complex(result["channel_pole"])
    gamma = abs(pole.imag)
    frequency = float(operating) if operating is not None else pole.real - 2 * gamma
    response = _solver(physics, row["slab_thickness_um"], row["ridge_height_um"], row["final_period_um"], 101)
    center = response(frequency)
    moderate = {"operating_frequency_rad_per_um": frequency,
                "held_out": {"T": float(abs(center.transmission)**2),
                             "delay_fs": phase_delay(response, frequency, 0.01 * gamma)}}
    held_out = as_complex(result["held_out_pole"])
    result["achieved_quality_factor"] = float(held_out.real / (-2 * held_out.imag))
    zero_response = _solver(physics, row["slab_thickness_um"], row["ridge_height_um"], row["final_period_um"], 81)
    zero = refine_scalar_root(lambda k: zero_response(k).transmission, complex(pole.real),
                              kind="zero", tolerance=1e-10, max_evaluations=100,
                              search_radius=max(0.08,10*gamma))
    if zero.residual_norm > 1e-6:
        raise RuntimeError("Reference transmission zero refinement failed")
    result["channel_zero"] = complex_record(zero.frequency)
    result["moderate_transmission"] = moderate
    return result, moderate


def summarize(case):
    rows = []
    for method, prep_count in case["preparation_factorizations"].items():
        records = [t["methods"][method] for t in case["trials"] if method in t["methods"]]
        keys = ("objective_fdfd_frequency_factorizations", "target_generation_fdfd_frequency_factorizations",
                "real_recentering_fdfd_frequency_factorizations", "fdfd_root_verification_frequency_factorizations")
        count = prep_count + sum(sum(int(r.get(k, 0)) for k in keys) for r in records)
        rows.append({"quality_target": case["quality_target"], "method": method,
                     "completed": len(records), "qualified": sum(bool(r["qualified"]) for r in records),
                     "fdfd_factorizations": count,
                     "fourier_response_evaluations": sum(int(r.get("fourier_response_evaluations", 0)) for r in records)})
    return rows


def reference_preparation(item, mesh_plan, reference, protocol, base):
    """Characterize the fixed reference on the published 384 x 1280 grid."""
    config, setup = setup_for_quality(reference["quality_target"], mesh_plan, base, reference)
    coordinates = np.array([reference["slab_thickness_um"], reference["ridge_height_um"]])
    epsilon, _ = physical_epsilon_and_derivatives(coordinates, config, setup)
    count = 0
    def transmission(k):
        nonlocal count
        count += 1
        return channel_transmission(k, epsilon, setup)
    began = time.perf_counter()
    seed = as_complex(item["fdfd_pole_seed"])
    root = refine_scalar_root(transmission, seed, kind="pole", tolerance=1e-10,
                              max_evaluations=80, search_radius=max(0.1, 15*abs(seed.imag)))
    if root.residual_norm > 1e-6 or root.frequency.imag >= 0:
        raise RuntimeError("Fine-grid reference pole refinement failed")
    prep = {"reference_fdfd_pole": complex_record(root.frequency),
            "reference_pole_root_evaluations": count,
            "reference_pole_preparation_wall_seconds": time.perf_counter()-began,
            "reference_geometry_slab_ridge_um": coordinates.tolist(),
            "reference_geometry_period_um": float(config["period_um"]),
            "reference_h81_pole": reference["channel_pole"],
            "reference_h101_pole": reference["held_out_pole"],
            "mesh": {"nx": setup.grid.nx, "nz": setup.grid.nz},
            "direct_target_preparation": {}}
    for method in ("direct3", "direct5"):
        offsets = np.array(protocol["methods"][method]["sample_offsets_in_reference_pole_depths"])
        frequencies = root.frequency.real + offsets*abs(root.frequency.imag)
        began = time.perf_counter()
        values = [channel_transmission(float(k), epsilon, setup) for k in frequencies]
        prep["direct_target_preparation"][method] = {
            "sample_frequencies": frequencies.tolist(),
            "sample_targets": [complex_record(v) for v in values],
            "offsets_in_reference_pole_depths": offsets.tolist(),
            "preparation_fdfd_frequency_factorizations": len(values),
            "preparation_wall_seconds": time.perf_counter()-began}
    return prep, config, setup, coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/gmr_replay.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality", type=int)
    parser.add_argument("--starts-limit", type=int, help="Short implementation check only; omit for final comparison")
    parser.add_argument("--methods", nargs="+", choices=["pole", "direct3", "direct5"], help="Subset for checks only")
    args = parser.parse_args()
    plan = read(args.config)
    physics = read("configs/m166_gmr_high_fidelity_analysis.json")
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "protocol.json", plan)
    inputs_path = Path(plan["inputs"])
    save(args.output / "metadata.json", {"inputs_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         "subset_check": args.starts_limit is not None or args.methods is not None})
    result = {"cases": []}
    mesh_plan = read(inputs_path)
    for item in mesh_plan["cases"]:
        quality = item.get("quality_target", item.get("reference", {}).get("quality_target"))
        if args.quality is not None and quality != args.quality:
            continue
        if plan["mode"] == "replay":
            reference, moderate = physical_reference(item, physics)
            response = _solver(physics, item["slab_thickness_um"], item["ridge_height_um"], item["final_period_um"], 81)
            p = as_complex(reference["channel_pole"])
            grid = np.linspace(p.real - 8*abs(p.imag), p.real + 8*abs(p.imag), 81)
            t = np.asarray([response(float(k)).transmission for k in grid])
            np.savetxt(args.output / f"q{quality}_spectrum.csv", np.c_[grid, t.real, t.imag, abs(t)**2],
                       delimiter=",", header="k_per_um,t_real,t_imag,T", comments="")
            result["cases"].append(reference)
            save(args.output / "summary.json", result)
            print(f"Q{quality}: Q={reference['achieved_quality_factor']:.3f}, T={moderate['held_out']['T']:.5f}, delay={moderate['held_out']['delay_fs']:.3f} fs", flush=True)
            continue

        base = read("configs/m164_gmr_pole_adjoint.json")
        protocol = read("configs/gmr_benchmark_protocol.json")
        reference, moderate = physical_reference(item["reference"], physics, item["operating_frequency_rad_per_um"])
        prep, config, setup, coordinates = reference_preparation(item, mesh_plan, reference, protocol, base)
        target = as_complex(prep["reference_fdfd_pole"])
        methods = args.methods or mesh_plan["methods"]
        shared = prep["reference_pole_root_evaluations"]
        preparation_counts = {"pole": shared, "direct3": shared+3, "direct5": shared+5}
        bounds = (np.asarray((config["slab_thickness_bounds_um"][0], config["ridge_height_bounds_um"][0])),
                  np.asarray((config["slab_thickness_bounds_um"][1], config["ridge_height_bounds_um"][1])))
        case = {"quality_target": quality, "reference_preparation": prep,
                "shared_reference_pole_preparation_factorizations": prep["reference_pole_root_evaluations"],
                "preparation_factorizations": {m: preparation_counts[m] for m in methods}, "trials": []}
        result["cases"].append(case)
        for index, start in enumerate(item["starts"][:args.starts_limit]):
            start = np.asarray(start)
            if np.any(start <= bounds[0]) or np.any(start >= bounds[1]):
                raise ValueError("Starting coordinates outside physical bounds")
            order = methods[index % len(methods):] + methods[:index % len(methods)]
            trial = {"start_index": index, "start_coordinates_um": start.tolist(), "methods": {}}
            case["trials"].append(trial)
            for method in order:
                record = run_method(method, start, target, reference, moderate, coordinates, prep,
                                    config, setup, physics, bounds, protocol)
                trial["methods"][method] = record
                save(args.output / "summary.json", result)
                rows = [row for c in result["cases"] for row in summarize(c)]
                with (args.output / "cost_table.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"Q{quality} start {index} {method}: qualified={record['qualified']}", flush=True)


if __name__ == "__main__":
    main()
