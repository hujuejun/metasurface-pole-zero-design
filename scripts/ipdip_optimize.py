"""Measured-IP-Dip lower-Q pole/zero throughput search by grouped contour moves."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from pzid.fabrication import subpixel_binary_permittivity
from pzid.measured_dispersion import PassiveLorentzModel
from scripts.m206a_connected_parasitic_pole_polish import _boundary
from scripts.m218b_ipdip_200nm_adjoint import objective, physical_contour, soft_pullback
from scripts.m7_binary_mesh_reopt import _master_parameterization


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contour_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()


def separated_sites(ranked: list[tuple], count: int, nx: int, minimum_distance: int
                    ) -> list[tuple]:
    chosen = []
    for entry in ranked:
        _, row, col, _ = entry
        if all(abs(row - old_row) + min(abs(col - old_col), nx - abs(col - old_col))
               >= minimum_distance for _, old_row, old_col, _ in chosen):
            chosen.append(entry)
            if len(chosen) == count:
                break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/ipdip_optimize.json")
    parser.add_argument("--output", type=Path, default=ROOT/"runs/ipdip_optimize")
    args = parser.parse_args()
    config_path = args.config
    out = args.output
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = json.loads((ROOT / config["source_config"]).read_text(encoding="utf-8"))
    parent = json.loads((ROOT / config["parent_summary"]).read_text(encoding="utf-8"))
    model = PassiveLorentzModel.from_dict(
        json.loads((ROOT / config["material_model"]).read_text(encoding="utf-8"))["model"]
    )
    setup, parameterization = _master_parameterization(source)
    with np.load(ROOT / config["parent_geometry"]) as saved:
        density = np.asarray(saved["density"], float)
        expected_mask = np.asarray(saved["mask_fine"], bool)
    fine_cell = float(parent["fine_contour_cell_um"])
    contour, fraction, projection, gate = physical_contour(density, setup, config, fine_cell)
    if not gate["passes"] or not np.array_equal(contour, expected_mask):
        raise RuntimeError("M218F best contour fails exact 200-nm reproducibility")
    scale = parent["physical_period_um"] / source["benchmark"]["period_um"]
    target = complex(source["target"]["common_pole_real_rad_per_um"],
                     config["target_pole_imag_native"])
    objective_args = (setup, model, scale, target,
                      np.asarray(config["reflection_offsets_in_huygens_depths"]),
                      config["reflection_weight"], config["absorptance_weight"])
    _, _, baseline = objective(fraction, *objective_args, with_jacobian=False)
    if baseline["time_reversed_pole_residual"] > config["pole_residual_guard"]:
        raise RuntimeError("M218F seed exceeds pole residual guard")
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "seed_geometry.npz", density=density,
                        mask_fine=contour, fill_fraction=fraction)
    history = []
    attempts = []
    subpixels = int(source["continuation"]["binary_rasterization_subpixels"])
    protected_start, protected_stop = map(int, config["protected_neck_rows"])
    for step in range(int(config["maximum_accepted_steps"])):
        residual, jacobian, current = objective(fraction, *objective_args,
                                                with_jacobian=True)
        assert jacobian is not None
        center_absorptance_jac = np.real(jacobian[5] / config["absorptance_weight"])
        center_reflection_jac = 2 * np.real(np.conj(residual[4]) * jacobian[4])
        center_transmission_jac = -center_absorptance_jac - center_reflection_jac
        gradient = soft_pullback(np.array([0.5 + 0j]),
                                 (-center_transmission_jac)[np.newaxis],
                                 density, subpixels, 16.0)
        ranked = []
        start, stop = parameterization.design_rows
        for row, col in _boundary(density >= 0.5, start, stop, protected_start):
            if protected_start <= row < protected_stop:
                continue
            direction = -np.sign(float(gradient[row, col]))
            if direction == 0:
                continue
            room = density[row, col] if direction < 0 else 1.0 - density[row, col]
            if room >= 0.03:
                ranked.append((abs(float(gradient[row, col])) * min(room, 0.32),
                               int(row), int(col), float(direction)))
        ranked.sort(reverse=True)
        coarse_fraction = subpixel_binary_permittivity(
            density, setup.grid.shape, 0.5, 1.0, 1.5, subpixels=subpixels
        )[1]
        proposals = []
        for group_size in config["group_sizes"]:
            for selection in ("ranked", "spaced"):
                sites = (ranked[:group_size] if selection == "ranked" else
                         separated_sites(ranked, group_size, setup.grid.nx,
                                         int(config["nonmaximum_site_distance_cells"])))
                if len(sites) < group_size:
                    continue
                for amplitude in config["group_density_steps"]:
                    changed = density.copy()
                    for _, row, col, direction in sites:
                        changed[row, col] = np.clip(
                            changed[row, col] + direction * float(amplitude), 0, 1
                        )
                    coarse_trial = subpixel_binary_permittivity(
                        changed, setup.grid.shape, 0.5, 1.0, 1.5,
                        subpixels=subpixels
                    )[1]
                    delta = coarse_trial - coarse_fraction
                    if not np.any(delta):
                        continue
                    predicted_residual = residual + np.einsum("kij,ij->k", jacobian, delta)
                    predicted_T = float(current["center"]["T"]
                                        + np.sum(center_transmission_jac * delta))
                    predicted_pole = float(np.linalg.norm(predicted_residual[:4]))
                    proposals.append({"group_size": group_size, "selection": selection,
                                      "amplitude": float(amplitude),
                                      "predicted_T": predicted_T,
                                      "predicted_pole_residual": predicted_pole,
                                      "changed": changed})
        # Keep one best amplitude per group/selection; this prevents a single
        # large, inadmissible group from filling the exact-evaluation shortlist.
        shortlist = []
        for size in config["group_sizes"]:
            for selection in ("ranked", "spaced"):
                group = [p for p in proposals
                         if p["group_size"] == size and p["selection"] == selection]
                if group:
                    shortlist.append(max(group, key=lambda p: (
                        p["predicted_pole_residual"] <= config["pole_residual_guard"] * 1.3,
                        p["predicted_T"])))
        shortlist.sort(key=lambda p: p["predicted_T"], reverse=True)
        records = []
        seen: set[str] = set()
        best = None
        for proposal in shortlist:
            trial_contour, trial_fraction, _, trial_gate = physical_contour(
                proposal["changed"], setup, config, fine_cell
            )
            key = contour_hash(trial_contour)
            record = {key_: proposal[key_] for key_ in (
                "group_size", "selection", "amplitude", "predicted_T",
                "predicted_pole_residual")}
            record["mask_sha256"] = key
            record["geometry_passes"] = bool(trial_gate["passes"])
            if key in seen or np.array_equal(trial_contour, contour):
                record["decision"] = "duplicate contour"
            elif not trial_gate["passes"]:
                record["decision"] = "geometry gate"
            else:
                _, _, exact = objective(trial_fraction, *objective_args,
                                        with_jacobian=False)
                record["exact"] = exact
                if exact["time_reversed_pole_residual"] > config["pole_residual_guard"]:
                    record["decision"] = "pole guard"
                elif exact["maximum_background_reflectance"] > config["maximum_background_reflectance"]:
                    record["decision"] = "reflection guard"
                elif exact["center"]["A"] > config["maximum_center_absorptance"]:
                    record["decision"] = "center absorption guard"
                elif (exact["center"]["T"] - current["center"]["T"]
                      < config["minimum_center_T_gain_per_step"]):
                    record["decision"] = "insufficient throughput gain"
                else:
                    record["decision"] = "feasible throughput gain"
                    if best is None or exact["center"]["T"] > best[0]["center"]["T"]:
                        best = (exact, proposal["changed"], trial_contour,
                                trial_fraction, trial_gate, record)
            seen.add(key)
            records.append(record)
        attempts.append({"step": step, "current": current,
                         "ranked_sites": len(ranked),
                         "linearized_proposals": len(proposals),
                         "candidate_tests": records})
        (out / "candidate_audit.json").write_text(
            json.dumps(attempts, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        if best is None:
            print("STOP", json.dumps({"step": step, "reason": "no feasible lower-Q throughput move",
                                      "center": current["center"]}), flush=True)
            break
        exact, density, contour, fraction, gate, selected = best
        accepted = {"step": step, "selected": selected, "center": exact["center"],
                    "pole_residual": exact["time_reversed_pole_residual"],
                    "maximum_background_reflectance": exact["maximum_background_reflectance"],
                    "spanning_neck_um": gate["spanning_neck"]["estimated_maximum_spanning_neck_um"]}
        history.append(accepted)
        np.savez_compressed(out / f"accepted_{step:02d}_geometry.npz",
                            density=density, mask_fine=contour,
                            fill_fraction=fraction)
        (out / "history.json").write_text(
            json.dumps(history, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print("ACCEPTED", json.dumps({"step": step, "group": selected["group_size"],
                                      "center": exact["center"],
                                      "pole_residual": accepted["pole_residual"]}), flush=True)
    _, _, final = objective(fraction, *objective_args, with_jacobian=False)
    final_gate = physical_contour(density, setup, config, fine_cell)[3]
    if not final_gate["passes"]:
        raise RuntimeError("final contour fails hard physical geometry gates")
    final_file = out / "final_geometry.npz"
    np.savez_compressed(final_file, density=density,
                        mask_fine=contour, fill_fraction=fraction)
    summary = {
        "stage": "M218I lower-Q measured-IP-Dip throughput-aware pole/zero adjoint redesign",
        "target_pole_native": [target.real, target.imag],
        "accepted_steps": len(history), "baseline": baseline, "final": final,
        "final_geometry_audit": final_gate,
        "physical_period_um": parent["physical_period_um"],
        "fine_contour_cell_um": fine_cell,
        "provenance_sha256": {str(path.relative_to(ROOT)): digest(path)
                              for path in (config_path, ROOT / config["source_config"],
                                           ROOT / config["parent_geometry"],
                                           ROOT / config["material_model"],
                                           ROOT / "data/materials/IP-Dip_nk_3-6um.csv",
                                           Path(__file__), final_file)},
        "limitations": config["note"],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("FINAL", json.dumps({"accepted_steps": len(history),
                                "center": final["center"],
                                "pole_residual": final["time_reversed_pole_residual"]}),
          flush=True)


if __name__ == "__main__":
    main()
