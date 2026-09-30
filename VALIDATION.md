# Preparation checks

## GMR update (2026-09-30)

- All 92 tests pass, including the added GMR geometry, adjoint, and operating-point delay tests.
- All 72 selected modules import from the bundled tree; explicit input paths resolve. The new source/configuration scan found no credential patterns or absolute user-workspace paths.
- Both revised main text and SI compile on Overleaf with zero LaTeX errors or warnings. The project default is restored to `main.tex`.
- Fresh independent Q200 replay gives Q=185.670, transmittance 0.90498, and delay 38.041 fs, consistent with the selected final result.
- A fresh Q800 matched-start subset (one start; pole and physical-operating-point direct three-sample objectives) reaches qualified designs with both methods. This checks execution, not the full published success/cost statistics.
- No full seven-design optimization sequence or complete 144-trial benchmark was rerun for this packaging update. The manuscript figures use the selected final research calculations.
- Added only final-result drivers, required helper definitions, minimal initial/reference coordinates, final geometries, and configurations. Archived outputs and exploratory benchmark batches remain excluded.

The unified GMR replay/benchmark wrapper regenerates electromagnetic targets and validations from minimal physical inputs instead of requiring saved research summaries. The Q800 starts preserve the signed Q200 geometric offsets. Helper-only modules have no exploratory command-line entry point. The full design sequence retains the outer Fourier correction and final period polish; it must not be described as a single uncorrected coarse-grid adjoint run. Fourier truncation values 81 and 101 denote half-widths (163 and 203 orders).

Completed locally before the initial private upload:

- All selected driver/helper modules import using this repository's own source tree; explicit bundled configuration/input paths resolve.
- 87 tests pass, including 2D and 3D FDFD, ports, grids, PML, adjoint, roots, Fourier-modal, PCM, final freeform grid shapes, and parametric geometry decoding.
- FP Q50 fine-grid derivative preflight: maximum relative error 1.99e-9.
- Fresh FP Q100 final-geometry replay: refined pole 4.053667940115864 - 0.02026833970057883 i rad/µm; Q = 100.00000000000242; maximum sampled energy-balance error approximately 4.04e-13.
- Fresh high-index continuous final-geometry smoke replay: finite, energy-conserving responses at k=7.90 and 7.91 rad/µm.
- Basic credential-pattern scan: no embedded GitHub/API/AWS credential patterns, account resource identifiers, or absolute research-workspace paths found in included source/configuration text.

Smoke-run outputs and caches are ignored, not committed. Full production optimizations, long mesh sweeps, and the optional paid FDTD simulations have not been rerun as part of this preparation. These checks establish packaging and representative execution, not complete independent confirmation of every scientific result.

## Adaptations from research source

Only final-result drivers and the transitive helper definitions they call were selected. Bare imports were made package-qualified. Configuration inheritance was flattened; unused campaign notes/provenance were removed. Required seeds and final geometry were extracted without run histories. Obsolete literature/parent-summary hashes are optional metadata, not simulation prerequisites. The FP Q400 configuration selects the independently checked endpoint, and Q400 is excluded from the earlier three-design fine-mesh family. Reader-facing replay/control wrappers call the retained numerical implementations.

The qBIC binary continuation is retained only because its negative result is explicitly discussed in SI. Unrelated failed campaigns are excluded.

## Resolved manuscript consistency item

The SI PCM table caption was corrected on Overleaf in manuscript commit `087380f`. It now states 501 uniformly spaced states for N24/32/40/48 and 301 for N56/64/72/80, replacing the incorrect claim of 501 states for every mesh through N64. The separate intermediate N40/48/56/64 sweep uses 301 states; the caption identifies the coarse sweep for the N40/48 table entries. Numerical results and simulation protocols are unchanged.

The corrected source was pushed to Overleaf and synchronization was verified. PDF recompilation was not verified because the local LaTeX compiler failed to initialize.
