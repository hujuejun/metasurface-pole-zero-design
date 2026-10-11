# Metasurface pole-zero design

Simulation source for *Inverse design of poles and zeros: tailoring phase control in metasurfaces*, including the main text and SI.

The collection includes final-study drivers, required numerical helpers, home-built 2D/3D FDFD engines, corrected RCWA verification, and portable FDTD setups. Frozen geometries, starting coordinates, and material tables are necessary inputs. Trial outputs, superseded studies, credentials, and cloud-result archives are excluded from the current tree. The repository remains private until the authors authorize publication.

## Installation

Use Python 3.12 from the repository root:

```bash
python -m venv .venv
# Activate .venv using the command appropriate to your shell.
python -m pip install -e ".[test]"
python check_installation.py
python -m pytest -q
python check_release.py
```

For FDTD only, install `python -m pip install -e ".[fdtd]"`. Tidy3D is an optional external solver. No AWS account is required for FDFD: AWS was an execution environment, not an algorithm dependency. The fine-grid benchmark and 3D calculations require substantially more memory than the tests. Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` when scheduling jobs to avoid oversubscription.

`requirements-tested.txt` records the numerical environment. Sparse-solver and floating-point differences can change optimization trajectories; replay the frozen geometry to distinguish optimizer variation from electromagnetic differences. See [VALIDATION.md](VALIDATION.md) for actual release checks.

## Manuscript-to-code map

`python reproduce.py CASE` prints a command; append `--run` to execute. Run from this repository root. Outputs belong in ignored `runs/` directories. An output override does not update downstream input paths.

| Paper result | Entry points | Scope |
|---|---|---|
| Sec. 3 / Fig. 1: seven GMRs | `gmr-replay` | Corrected Li-factorized RCWA poles, spectra, Q, and delay from frozen geometries |
| Sec. 3 / Fig. 1 and SI benchmark | `gmr-benchmark` | 384 × 1280 grid; Q targets 200/800; 12 paired starts; pole, 3-transmission-sample, and 5-transmission-sample objectives |
| Sec. 4.2–4.3 / Fig. 2 | `high-index-continuous`, `high-index-binary`, `low-index-binary` | Optimization and final grids, including SI convergence meshes |
| Fig. 2 Hy maps | `scripts.freeform_modes` | Fresh parity-pole refinement and mode extraction |
| Sec. 4.4 / Fig. 3 | `planar-hcg`; `scripts.replay_planar_hcg` | Passive HCG optimization and frozen-interface replay |
| Sec. 4.5 / Fig. 4 and SI | `pcm-rectangular-optimize`, `pcm-rectangular-mesh`; `scripts.pcm_spectra` | Rectangular PCM pair only; 501 states; selected-state spectra; meshes 24/32/40/48 |
| Sec. 5 / Fig. 5 and SI | `ipdip-optimize`, `ipdip-replay` | Connected 200-nm-feature IP-Dip TM design; measured material loss |
| SI qBIC and binarization limitation | `qbic`, then `qbic-binary-negative-control` | Negative control retained because it is a reported SI result |
| SI 3D disk | `disk-smooth-roots`, `disk-adjoint`, `disk-optimize`, `disk-forward` | Smooth-interface inverse design and physical-cylinder replay |
| SI GMR checks | `gmr-sensitivity`, `gmr-fields`, `gmr-mesh`, `gmr-delay` | Dependencies specified in each configuration |
| Independent FDTD | `scripts.fdtd_validation`; `disk-fdtd` | Final 2D setups and separate 3D disk driver |

Direct module examples:

```bash
python -m scripts.ipdip_replay --mesh 96 --wavelengths 5 --output runs/ipdip-spot
python -m scripts.ipdip_material_fit --output runs/ipdip-fit
python -m scripts.freeform_modes high_index_continuous --output runs/modes
python -m scripts.replay_planar_hcg --output runs/hcg-replay
python -m scripts.pcm_spectra --output runs/pcm-spectra
python -m scripts.replay_final_2d high_index_binary --mesh 88 --k-min 7 --k-max 9 --points 241 --output runs/high-binary-replay
```

Freeform cases are `high_index_continuous`, `high_index_binary`, `low_index_binary`, and `qbic`. The binary finest meshes are 88 and 56, respectively. Choose the paper's frequency interval when computing band metrics. Frequencies in rad/µm denote k = ω/c, not angular frequency in s⁻¹.

For IP-Dip, omit `--wavelengths` for the full spectrum. Meshes 32/48/64/96 reproduce the SI comparison; 128 uses nine sentinel wavelengths. Real-frequency spectra interpolate measured n,k directly; the Lorentz fit is for complex-frequency optimization. The optimizer preserves the original contour projection, connectivity checks, protected regions, and acceptance guards. These are geometric checks, not a mechanical buckling simulation. Nominal target poles are not independently extracted poles of the final lossy structure.

## GMR design versus verification

The shortest verification path is `gmr-replay`. Bundled root values are search seeds; Maxwell responses are recomputed.

To reproduce geometry design, run `gmr-adjoint`, `gmr-correction`, `gmr-q3200`, and `gmr-family`, in that order. The historical outer correction and period alignment used an earlier TM modal operator. It is retained as `src/pzid/rcwa2d_design_legacy.py` only because it generated the physical geometries. Published verification uses the corrected `src/pzid/rcwa2d.py`. The `gmr-family` output is historical design output, not corrected verification. Afterwards run `gmr-replay`; this does not silently reoptimize the geometries.

`gmr-benchmark` regenerates common fine-grid reference targets and runs all 72 optimizations. For a subset, invoke `scripts.gmr_reproduce` with `--config configs/gmr_benchmark.json --quality 200 --starts-limit 1 --methods pole --output runs/benchmark-check`. Even one attempt is costly. A subset is not the full comparison.

Costs include unsuccessful attempts, objectives, direct-target generation/updates, real-frequency recentering, final FDFD root extraction, and shared reference characterization once per method. RCWA evaluations are separate. Fresh reference refinement may take a different number of preparation evaluations than the original pilot; trajectories and wall times may also vary. Budgets, paired starts, and qualification criteria are preserved, not bitwise-identical iteration histories.

## Independent FDTD

`inputs/fdtd/*.hdf5` are Simulation definitions only: geometry/material arrays, grid, sources, boundaries, monitors, and runtime. They are not fields, results, cloud jobs, or credentials. `configs/fdtd_validation.json` pairs each device with its empty reference.

```bash
python -m scripts.fdtd_validation ipdip --action inspect --output runs/fdtd-ipdip
python -m scripts.fdtd_validation ipdip --action estimate --output runs/fdtd-ipdip
# Review the quote, then substitute an explicitly approved total:
python -m scripts.fdtd_validation ipdip --action run --max-cost APPROVED_TOTAL --output runs/fdtd-ipdip
python -m scripts.fdtd_validation ipdip --action analyze --output runs/fdtd-ipdip
```

`inspect` is local. `estimate` uploads non-running drafts using the reader's account. `run` requires the saved quote, unchanged setup hashes, and an explicit total FlexCredit ceiling. Never commit quotes or results. Cases: `high-continuous`, `high-binary`, `low-binary`, `planar-hcg`, `ipdip`, `pcm-alpha0/03/06/09/1`.

The continuous-high-index validation uses the SI validation geometry, not the main-text optimization endpoint. The low-index setup includes periodic contour continuation and the final long ringdown. IP-Dip uses the final stable absorber boundaries, not earlier unstable PML controls.

For the 3D disk, inspect `python -m scripts.m37_tidy3d_forward --help` and its configuration. Estimate/run switches are explicit; `reproduce.py` refuses cloud execution. The paper uses the 17.5-nm uniform-grid result.

## License and provenance

Original code uses the [MIT license](LICENSE). Material data and dependencies have separate attribution in [THIRD_PARTY.md](THIRD_PARTY.md). Cite the accompanying manuscript and repository; [CITATION.cff](CITATION.cff) provides software metadata.

Stable numerical helper filenames retain research identifiers for traceability; they do not signify additional studies. The FP/grating study, stepped-PCM results, and superseded benchmark objectives are removed from the current tree. Git history is retained for recovery, not rewritten.
