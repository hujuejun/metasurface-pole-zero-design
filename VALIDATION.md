# Release validation

The source collection was audited against the current main text and Supplementary Information on 2026-10-10. This is a source/reproducibility audit, not a complete rerun of all production calculations.

## Checks performed

- 115 local tests pass, including 2D and 3D FDFD, adjoint derivatives, radiation/port handling, corrected RCWA and its historical-operator regression, final geometry compatibility, material fitting, feature projection, connectivity, and the final benchmark protocol.
- Both IP-Dip starting and final fine contours reconstruct exactly from the bundled density inputs under the retained projection and geometric gates.
- Fresh IP-Dip TM FDFD at the final N96 grid and 5 µm gives T = 0.72194763 and R = 0.00485774, reproducing the manuscript values.
- Fresh corrected-RCWA replay of the target-Q200 frozen GMR gives Q approximately 183.97, T = 0.90584, and delay = 37.459 fs at the specified physical operating point. Verification uses the corrected Li-factorized operator, not historical design-stage values.
- Fresh high-index continuous Hy mode extraction refines both parity poles. Halving the complex-frequency field-extraction offset changes the residues by approximately 0.000128 and 0.000125; longitudinal parity errors are below 7e-11.
- Every portable 2D FDTD device/reference definition loads as a Simulation and passes local pre-upload validation. Definitions contain no result data or account attributes. The IP-Dip and rectangular-PCM geometry renders were visually inspected. Long-runtime warnings are expected for the selected resonance ringdown setups.
- Module/import and explicit bundled-input checks use this repository's own engines, not the parent research workspace.
- Source/input checksums use LF-normalized text and exact binary bytes. The release checker scans the curated text for common credential, cloud-resource, and absolute user-path patterns. This is a targeted hygiene check, not a guarantee against every possible secret format.

## Scope and provenance

The current tree removes the FP/grating study, superseded stepped-PCM result drivers, old direct-control variants, and unused exploratory entry points. Shared numerical helpers remain only where required by a retained study. Earlier versions remain recoverable in Git history.

The final GMR benchmark uses three methods, twelve paired starts per reference, and the fine 384 × 1280 grid. The wrapper regenerates reference targets and includes shared reference characterization once per method in aggregate costs. It does not load archived trial results. Fresh root-refinement evaluation counts and optimization trajectories may differ with starting root estimates and the numerical platform.

The historical RCWA design operator remains isolated solely to reproduce the original geometry-design sequence. Corrected RCWA is used for the reported frozen-geometry verification and benchmark qualification. The qBIC binarization negative control remains because it is discussed in SI.

The full seven-design optimization sequence, 72-attempt fine-grid benchmark, full-band convergence sweeps, and paid FDTD simulations were not rerun for this release preparation. Local setup validation does not replace an FDTD solve. No new cloud costs were incurred.

No manuscript source was changed during this repository audit. The MIT license applies to original code; third-party attribution is separate.
