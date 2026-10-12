# Release validation

The source collection was reconciled with the current main text and Supplementary Information on 2026-10-12 following the independent audit. This is a source/reproducibility audit, not a complete rerun of all production calculations.

## Checks performed

- 128 local tests pass, including 2D and 3D FDFD, adjoint derivatives, radiation/port handling, corrected RCWA and its historical-operator regression, final geometry compatibility, material fitting, feature projection, connectivity, the final benchmark protocol, and audits of the frozen trial, qBIC limitation, and narrow-resonance records.
- Both IP-Dip starting and final fine contours reconstruct exactly from the bundled density inputs under the retained projection and geometric gates.
- Fresh IP-Dip TM FDFD at the final N96 grid and 5 µm gives T = 0.72194763 and R = 0.00485774, reproducing the manuscript values.
- Fresh corrected-RCWA replay of the target-Q200 frozen GMR gives Q approximately 183.97, T = 0.90584, and delay = 37.459 fs at the specified physical operating point. Verification uses the corrected Li-factorized operator, not historical design-stage values.
- Fresh high-index continuous Hy mode extraction refines both parity poles. Halving the complex-frequency field-extraction offset changes the residues by approximately 0.000128 and 0.000125; longitudinal parity errors are below 7e-11.
- Every portable 2D FDTD device/reference definition loads as a Simulation and passes local pre-upload validation. Definitions contain no result data or account attributes. The IP-Dip and rectangular-PCM geometry renders were visually inspected. Long-runtime warnings are expected for the selected resonance ringdown setups.
- Module/import and explicit bundled-input checks use this repository's own engines, not the parent research workspace.
- Five previously late-failing release entry points now have the required historical provenance or registration keys and early input checks. IP-Dip optimization uses the bundled analytic material model, while exact real-frequency replay explicitly requires a user-supplied n,k CSV. The full expensive optimizations were not rerun in this audit, so these are configuration/preflight repairs rather than a guarantee that every fresh optimizer reaches its historical endpoint.
- A fresh fixed-geometry N56 pole-guided low-index diagnostic resolves two in-band reflection peaks missed by the original 241-point sweep. The archived poles have Q approximately 28,500 and 98,000, and each local lossless reflection peak approaches unity. The old high-transmission extrema are explicitly identified as sampled values, not continuous-band bounds. This N56 diagnostic does not establish mesh convergence of the additional parasitic poles.
- All 72 original fine-grid GMR benchmark trial JSON records, one-time preparation records, and audited numerical ledger are bundled under `reference_results/gmr_benchmark/` and checked against their saved hashes. Compact FDFD/FDTD comparison snapshots are under `reference_results/fdtd_validation/`; their inclusion does not change the finite-time or mesh-convergence limits in the SI.
- Frozen numerical reference files are exempt from Git line-ending conversion so their original raw SHA-256 checksums remain stable across Windows and Linux clones. Source-code text remains LF-normalized.
- Source/input checksums use LF-normalized text and exact binary bytes. The release checker scans the curated text for common credential, cloud-resource, and absolute user-path patterns. This is a targeted hygiene check, not a guarantee against every possible secret format.

## Scope and provenance

The current tree removes the FP/grating study, superseded stepped-PCM result drivers, old direct-control variants, and unused exploratory entry points. Shared numerical helpers remain only where required by a retained study. Earlier versions remain recoverable in Git history.

For this reconciliation, every remaining tracked script/configuration was checked against the current case map, retained imports, tests, or the main/SI verification path. Several research-numbered helpers (for example the M11/M20/M22 qBIC helpers, M45/M50 HCG helpers, and M117/M120/M121 rectangular-PCM helpers) look historical but are still imported by current drivers. None was deleted merely for its filename. The new frozen records are limited to the 72-trial benchmark, its ledger, the low-index pole-guided audit, and compact current FDTD comparisons; the full research archive was not added. No additional tracked file was demonstrably obsolete and safe to remove from this already-curated checkout.

The final GMR benchmark uses three methods, twelve paired starts per reference, and the fine 384 × 1280 grid. The wrapper regenerates reference targets and includes shared reference characterization once per method in aggregate costs. It does not load archived trial results. Fresh root-refinement evaluation counts and optimization trajectories may differ with starting root estimates and the numerical platform.

The historical RCWA design operator remains isolated solely to reproduce the original geometry-design sequence. Corrected RCWA is used for the reported frozen-geometry verification and benchmark qualification. The qBIC binarization negative control remains because it is discussed in SI.

The full seven-design optimization sequence, 72-attempt fine-grid benchmark, full-band convergence sweeps, and paid FDTD simulations were not rerun for this release preparation. Local setup validation does not replace an FDTD solve. No new cloud costs were incurred. The bundled qBIC frozen geometry replays its SI response, but a fresh optimization from the released seed did not recover the same endpoint. The existing 3D disk FDFD/FDTD phase curves are not yet a like-for-like quantitative phase comparison because scan windows and propagation references differ.

The manuscript source was corrected as documented in its `MAIN_TEXT_AUDIT_CHANGELOG.md`. The MIT license applies to original code; third-party attribution is separate. On the author's instruction, the measured IP-Dip table was removed from the current release. Its earlier presence in Git history is not erased by this ordinary deletion. Exact real-frequency IP-Dip replay requires a user-supplied table; the retained fitted model does not replace the underlying measured data.
