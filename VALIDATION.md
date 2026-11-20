# Validation protocol

## Evidence hierarchy

1. Published mathematical equations and documented Bayesian defaults in the DSEM literature.
2. Published methodological benchmark datasets and reference outputs (Chapters 6 & 9 DSEM series; Asparouhov, Hamaker, & Muthén, 2018; Asparouhov & Muthén, 2020, 2024, 2025, 2026).
3. Open supplementary examples and simulation specifications.
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
- Bias, RMSE, credible-interval coverage, rank-normalized split-Rhat, bulk and
  tail ESS, and Monte Carlo standard error of the posterior mean are reported.
- Public comparisons agree within combined Monte Carlo uncertainty—normally two
  combined MCSEs—with compatible intervals and substantive conclusions.
- Fixed master seeds produce task-identical results across one, two, and
  scheduler-selected worker counts.
- Performance evidence identifies the timed kernel, warmup policy, hardware,
  compiler flags, and uncertainty; regressions over 10% require explanation.

## Package gates

- Unit, integration, property, malformed-input, interruption/restart, sampler,
  benchmark, and regression tests.
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

Convergence diagnostics use the required `posterior` package on every
installation: rank-normalized split-Rhat, bulk ESS, tail ESS, and MCSE of the
posterior mean. MCSE uses raw-draw mean ESS, not rank-normalized bulk ESS.
Degenerate or nonfinite draws return unavailable diagnostics. A Monte Carlo
replication is marked converged only with at least two chains, all finite
Rhat < 1.01, and both ESS measures >= 100 for every parameter. This screening
rule is not evidence of correctness or sufficient precision for every study.

## Statistical regression tests

- A strong independent normal prior is checked against numerical integration
  of the observed-regression posterior, catching a coefficient/variance prior
  mismatch that weak-prior recovery tests can miss.
- FFBS means and covariances are compared with an independently constructed
  dense Gaussian posterior, using five sampling-standard-error bounds.
- The stationary AR conditional is checked by numerical quadrature, and the
  innovation-variance conditional by joint-density ratios.
- Latent fitted values are reconstructed from joint state/loading draws.
- Unsupported section-based SEM data/model semantics, malformed controls,
  nonnumeric data, and invalid time indices must fail explicitly.
