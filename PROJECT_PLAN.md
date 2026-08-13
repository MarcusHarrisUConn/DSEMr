# DSEMr project plan

## Definition of done

DSEMr becomes a CRAN candidate only after every mandatory public DSEM and RDSEM
example has a reproducible parity record or a scientifically justified,
maintainer-approved exclusion. A compiled syntax construct is never treated as
an implemented estimator feature.

## Milestone 1 — foundation and reference engine

- R package, GPL-3 licensing, private GitHub repository, CI, release documents.
- Common model representation with lavaan-style and clean-room Mplus-like
  frontends.
- Gaussian N=1 AR(1) reference estimator and Rust sufficient-statistics kernel.
- Deterministic chain and replication seeds; scheduler-aware compute manifest.
- Simulation, Monte Carlo, inspection, diagnostics, and standard R methods.
- Offline-vendored Cargo build and macOS/Linux/Windows CI design.

Exit gate: both syntaxes compile identically; R and Rust kernels and fixed-seed
posterior draws agree; package tests, build, and local CRAN-style check pass.

## Milestone 2 — continuous DSEM core

- N=1 multivariate AR/VAR, cross-lagged paths, time-varying covariates.
- Two-level random intercepts, slopes, residual variances, latent centering, and
  between-level regressions.
- Missing-observation and irregular-interval policies with explicit initial
  condition handling.
- Public parity records for Mplus 6.23–6.24 and 9.30–9.32 before expansion.

## Milestone 3 — latent, categorical, and residual dynamics

- Dynamic CFA/SEM, measurement invariance controls, factor-score uncertainty.
- Binary and ordinal probit data augmentation, thresholds, and interpretation.
- RDSEM residual lag operators and dynamic residual measurement models.
- Posterior predictive checks, DIC-compatible output, split R-hat, ESS, MCSE,
  chain diagnostics, and prior sensitivity reports.

## Milestone 4 — cross-classified and HPC hardening

- Subject-by-time cross-classified random effects and examples 9.38–9.40.
- Slurm, PBS/TORQUE, SGE, and LSF templates; checkpoint/restart and array jobs.
- Deterministic distributed Monte Carlo and resource/oversubscription audits.
- Benchmarks against the transparent R reference and regression thresholds.

## Milestone 5 — parity and CRAN

- Complete `docs/PARITY.md` and publish reproducible evidence bundles.
- Simulation-based calibration, planned parameter recovery, bias and coverage.
- Cross-platform source installation and offline native builds.
- `cargo test`, fmt, Clippy, security audit, sanitizers/Valgrind where supported.
- R release/oldrel/devel checks on Linux, macOS, and Windows; win-builder/rhub;
  reverse-dependency analysis; clean `R CMD check --as-cran`.
- Synchronize `cran-comments.md`, `CRAN_SUBMISSION.md`, `RELEASE.md`, citation,
  changelog, and archived validation artifacts before submission.

## Release policy

Development versions use `0.0.x.9000`. Internal tagged releases may expose
experimental capabilities, but README and function-level status labels must
remain explicit. Version `0.1.0` is reserved for the completed parity/CRAN gate.

