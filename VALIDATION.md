# Preparation checks

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

## Manuscript consistency item

The SI PCM table caption currently says 501 states for every mesh through N64. The actual source configurations use 501 states for the coarse N24/32/40/48 sweep, and 301 for the intermediate N40/48/56/64 and fine N72/80 sweeps. This repository preserves the actual protocols; correct the caption before public release.
