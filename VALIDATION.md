# Validation protocol

## Evidence hierarchy

1. Published mathematical equations and documented defaults.
2. Official public Mplus guide inputs and outputs.
3. Open supplementary examples and author-posted output files.
4. Independently simulated data with declared truths.
5. Internal R reference implementation for optimized-kernel verification.

No proprietary source code, reverse engineering, or undisclosed licensed output
is used.

## Required gates for each model family

- Syntax tests compile both frontends into identical canonical models.
- Malformed, ambiguous, unidentified, or unsupported models fail early.
- Pure Rust kernels agree with the transparent R implementation at stated
  numerical tolerances on fixed inputs.
- Simulation recovery covers operating conditions over N, T, missingness,
  parameter magnitude, and prior informativeness.
- Bias, RMSE, credible-interval coverage, convergence, ESS, and MCSE are reported.
- Public comparisons agree within combined Monte Carlo uncertainty—normally two
  combined MCSEs—with compatible intervals and substantive conclusions.
- Fixed master seeds produce task-identical results across one, two, and
  scheduler-selected worker counts.
- Performance evidence identifies the timed kernel, warmup policy, hardware,
  compiler flags, and uncertainty; regressions over 10% require explanation.

## Package gates

- Unit, integration, property, malformed-input, interruption/restart, sampler,
  parity, and regression tests.
- Rust test, fmt, Clippy, dependency audit, sanitizer/Valgrind where applicable.
- Offline installation from the source tarball on supported platforms.
- R release, oldrel, and devel CI on Linux, macOS, and Windows.
- `R CMD build`, temporary-library install, vignettes/examples, and
  `R CMD check --as-cran` without errors or warnings and with explained notes.

## Foundation tolerances

The current scalar and grouped Rust sufficient-statistics kernels must match R
within `1e-12`.
Because both engines share the same R sampler, fixed-seed posterior draws must
match within `1e-12`. Recovery smoke tests use broad tolerances and are not a
substitute for the planned simulation study.
