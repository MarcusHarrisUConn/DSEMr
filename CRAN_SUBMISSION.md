# CRAN submission status

**Ready for CRAN submission (version 0.1.0).**

- **Package version:** `0.1.0`
- **Maintainer:** Marcus Harris (`marcus.harris.uconn@gmail.com`)
- **Offline Rust build:** Verified with `-j 2 --offline` against vendored
  `src/rust/vendor.tar.xz` (`extendr-api = 0.9.0`, MSRV `rustc >= 1.71.0`).
  Crate attributions recorded in `inst/AUTHORS`.
- **Model & validation coverage:** Validated via simulation recovery and
  published methodological benchmarks across single-level AR/VAR/RDSEM,
  dynamic CFA/WNFS/IRT, Tobit censored DSEM, two-level location-scale and
  multivariate VAR DSEM, cross-classified DSEM, three-level DSEM (`DSEM3`),
  `TINTERVAL` grid alignment, `DIC`/`pD`, `STDYX`/`STDY`/`STD`, Bayesian
  $R^2$, partial correlations, hypothesis/Wald tests, IRFs, `VIF`, and
  penalized shrinkage priors (`PSEM`).
- **Pre-submission checks:** `R CMD build`, `R CMD check --as-cran`,
  `cargo test --offline`, `cargo clippy --offline --locked`,
  `spelling::spell_check_package()`, and `urlchecker::url_check()`.
