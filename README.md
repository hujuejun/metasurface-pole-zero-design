# Metasurface pole-zero design

Simulation source accompanying **Inverse design of poles and zeros: tailoring phase control in metasurfaces**, including the main text and Supplementary Information (SI).

This is a private collaboration repository, separate from the research workspace. It contains the selected simulation drivers, their required helper functions, the local 2D/3D FDFD engines, an independent 2D Fourier-modal solver, configurations, and essential design/initialization inputs. It does **not** contain research-history archives, intermediate run outputs, saved spectra, manuscript PDFs, cloud credentials, or unrelated device studies.

## Installation and checks

Use Python 3.12 or newer, from the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python check_installation.py
python -m pytest -q
```

The local preparation was checked with Python 3.12, NumPy 2.4.6, SciPy 1.18.1, and Matplotlib 3.11.1. `requirements-tested.txt` records the tested numerical stack. A compatible installation does not require reproducing the original Windows environment. Floating-point solver/BLAS differences can change an optimization trajectory; independently replay the frozen final geometry to distinguish this from electromagnetic differences.

The preparation checks cover imports, bundled input paths, 2D/3D engine tests, final-grid compatibility, parametric geometry decoding, and an FP adjoint finite-difference audit. They are **not** a fresh rerun of every production optimization, mesh sweep, or cloud calculation.

## What to run

`python reproduce.py CASE` prints the exact command without launching a solve. Add `--run` to execute. Always run commands from this repository's root. Most drivers require a new output directory; do not overwrite another run. `--output` changes only that case's destination, not the paths used by downstream cases.

| Manuscript result | Named cases | Inputs and validation |
| --- | --- | --- |
| Single FP resonance, Q=50/100/200 | `fp-optimize` | Exact fine-grid optimization starts, fixed five-pair topology |
| Single FP resonance, Q=400 | `fp-q400-optimize`, `fp-q400-verify` | The independently checked fine-grid Q400 design, not the superseded endpoint |
| FP independent numerical checks | `fp-axial-mesh`, `fp-joint-mesh`, `fp-fourier`, `fp-solver-comparison` | Frozen physical geometries; joint FDFD and 27/35-harmonic comparisons |
| Matched-start direct-transmission comparison | `fp-direct-control` | Requires newly generated `fp-optimize` outputs; no common-start pilot |
| SI synthetic perturbations | `fp-robustness` | Corrected fixed-thickness index perturbation, 48 draws per Q |
| High-index continuous surface | `high-index-continuous` | Continuous density optimization and final-grid replay |
| High-index binary surface | `high-index-binary` | Necessary continuation seed only; final N64/72/80/88 grids |
| Low-index connected binary surface | `low-index-binary` | Necessary density seed only; final N32/40/48/56 grids |
| SI continuous qBIC and failed binarization | `qbic`, `qbic-binary-negative-control` | Run the continuous case first; the binary result is a reported negative control, not a validated binary design |
| Planar HCG | `planar-hcg` | Five physical interface coordinates, prescribed singularities |
| PCM HCG | `pcm-optimize`, `pcm-mesh`, `pcm-roots` | Final geometry and lossless effective-medium material model |
| SI PCM sensitivity and rectangular alternative | `pcm-material-sensitivity`, `pcm-rectangular-optimize`, `pcm-rectangular-mesh` | Alternative mixing/loss assumptions and equal-width geometry |
| SI 3D nanodisk | `disk-smooth-roots`, `disk-adjoint`, `disk-optimize`, `disk-forward` | Smooth-interface inverse design and distinct physical-cylinder replay |
| SI independent nanodisk FDTD | `disk-fdtd` (instructions below) | Optional Tidy3D source; requires a separate account and paid compute |

Source module names retain their original identifiers to avoid breaking imports and make scientific provenance auditable. Earlier modules under `scripts/` contain **only the helper definitions needed by these final drivers**, not executable exploratory studies. `source_manifest.json` records source hashes and roles. Configurations have been flattened so no inherited campaign configuration is required. Necessary starting densities are numerical initial conditions, not an archive of intermediate runs.

### Single-resonance reproduction order

```bash
python reproduce.py fp-optimize --run
python reproduce.py fp-q400-optimize --run
python reproduce.py fp-axial-mesh --run
python reproduce.py fp-joint-mesh --run
python reproduce.py fp-fourier --run
python reproduce.py fp-solver-comparison --run
python reproduce.py fp-q400-verify --run
python reproduce.py fp-direct-control --run
python reproduce.py fp-robustness --run
python -m scripts.equal_information_control
python -m scripts.m160_validated_q_delay_sampling
python -m scripts.m161_compare_fine_adjoint_costs
```

For a much shorter derivative check:

```bash
python reproduce.py fp-optimize --preflight-only --run --output runs/fp_gradient_check
```

For fresh electromagnetic verification of a final FP geometry without optimization:

```bash
python -m scripts.replay_fp --quality 100 --output runs/fp_q100_replay
```

Replay outputs are intentionally separate from optimization outputs: a replay cannot establish optimization cost. The SI equal-information model comparison uses only Q100 and Q200; it does not reproduce discarded cross-geometry pilot studies.

### Final freeform and HCG geometry replay

The numerical permittivity arrays in `inputs/final_permittivity/` are final device definitions, not saved electromagnetic solutions. Every replay performs new Maxwell solves. For example:

```bash
python -m scripts.replay_final_2d high_index_continuous --k-min 6.9330725 --k-max 8.8829665 --points 241 --output runs/high_index_replay
python -m scripts.replay_final_2d high_index_binary --mesh 80 --k-min 6.9330725 --k-max 8.8829665 --points 241 --output runs/high_binary_replay
python -m scripts.replay_planar_hcg --output runs/planar_hcg_replay
python reproduce.py pcm-mesh --run
python reproduce.py pcm-mesh-coarse --run
python reproduce.py pcm-mesh-intermediate --run
python reproduce.py pcm-spectra --run
python reproduce.py pcm-material-sensitivity --run
python reproduce.py pcm-rectangular-mesh --run
```

For low-index/qBIC replay, choose the frequency window specified in the manuscript and corresponding configuration. `k` is angular frequency divided by vacuum light speed, in rad/µm; lengths are µm. Replay exports both raw and empty-cell-de-embedded phase. Use **raw S21 phase for qBIC**, and the manuscript's de-embedding convention for the high-/low-index surfaces. Do not compare full-window and restricted-window phase spans interchangeably.

The qBIC full replay window is 4.323489–4.346749 rad/µm (1201 points). The low-index ±3 target-depth window is approximately 13.207054–15.608987 rad/µm (241 points). After rerunning the qBIC binary negative control, its independent pole census is available through `python -m scripts.m27e_audit_hard_binary_pole_census --help`.

PCM verification uses the bundled final physical coordinates, so it does not require first repeating optimization. To verify a newly optimized geometry instead, change the relevant config's `stage_summary` (or `m140_stage_summary`) to that run's newly generated final-stage JSON. The same distinction applies to the bundled planar-HCG geometry used to initialize the PCM design.

The original coarse PCM sweep uses 501 states at N24/32/40/48; the intermediate N40/48/56/64 sweep and fine N72/80 sweep use 301 states. This preserves the actual simulation configurations. The current SI caption saying 501 states for all N≤64 needs correction before publication.

### Three-dimensional reproduction

Run `disk-smooth-roots`, then `disk-adjoint`, then `disk-optimize`. To independently verify that newly optimized smooth design:

```bash
python -m scripts.m37_huygens_verify --config configs/m37d_huygens_inverse_coarse.json --design runs/disk_optimize/design.npz --output runs/disk_inverse_verification
```

`disk-forward` is the physical cylinder on the N30 FDFD grid, not the smooth-interface inverse-design surrogate. These vector sparse solves can require substantial RAM and runtime. Small-grid engine tests are not substitutes for the production mesh checks.

Optional independent FDTD requires `python -m pip install -e ".[fdtd]"` and the reader's own Tidy3D authentication. No account, key, task download, or precomputed field file is included. The wrapper refuses cloud execution. Explicitly inspect the source and obtain a cost estimate before opting into a run:

```bash
python -m scripts.m37_tidy3d_forward --config configs/m38b_tidy3d_mesh_convergence.json --output runs/disk_fdtd --batch-estimate
# The following launches paid cloud simulations; run only after reviewing cost:
python -m scripts.m37_tidy3d_forward --config configs/m38b_tidy3d_mesh_convergence.json --output runs/disk_fdtd --batch-run
python -m scripts.m37_tidy3d_forward --config configs/m38b_tidy3d_mesh_convergence.json --output runs/disk_fdtd --batch-analyze
```

The estimate step uploads simulation definitions to the provider. No cloud calculation was launched while preparing this repository.

## Scientific scope

Reproducibility does not upgrade the manuscript's validation status. In particular:

- High-index binary optical stability is not exact recovery of all prescribed singularities.
- The qBIC is a continuous-index feasibility example, not a successfully binarized device.
- The planar HCG remains a feasibility result at its stated mesh.
- PCM assumes idealized homogeneous, lossless intermediate material states; artificial-loss tests are not measured material data.
- The 3D disk comparison uses FDFD and FDTD at the mesh resolutions reported in the SI. Smooth-interface inverse-design recovery and physical-cylinder replay are distinct calculations.
- Device-specific FP solve counts do not demonstrate universal algorithmic superiority.

## Sharing and licensing

Keep this repository private until the authors approve release. No collaborators are invited automatically. A public-release license has not yet been selected; no open-source license grant is implied by this preparation. NumPy, SciPy, Matplotlib, pytest, and optional Tidy3D are installed dependencies, not vendored third-party source.
