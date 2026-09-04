# DSEMr 0.0.1.9000

- Added the initial R package and offline-vendored extendr/Rust build.
- Added a canonical DSEM model representation and dual syntax compilers.
- Added an experimental Bayesian N=1 Gaussian AR(1) reference estimator.
- Added an experimental two-level Gaussian AR(1) Gibbs sampler with random
  intercepts and autoregressive effects.
- Added posterior draw storage and interval summaries for person-specific
  random effects.
- Added gap-safe lag construction with configurable regular time intervals.
- Added direct clustered-data simulation and a two-level recovery harness.
- Added a grouped Rust sufficient-statistics kernel with exact R parity tests.
- Added PSOCK dispatch for independent Monte Carlo replications.
- Added Rust sufficient-statistics acceleration with exact R parity tests.
- Added deterministic chain and Monte Carlo seeds and scheduler-aware compute
  manifests.
- Added simulation, Monte Carlo, inspection, diagnostics, standard extractors,
  documentation, compatibility matrix, and release/validation gates.
