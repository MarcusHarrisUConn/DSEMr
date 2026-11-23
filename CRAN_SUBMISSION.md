# CRAN submission status

**Ready for CRAN submission (version 0.1.0).**

- **Package version:** `0.1.0`
- **Maintainer:** Marcus Harris (`marcus.harris.uconn@gmail.com`)
- **Offline Rust build:** Verified with `-j 2 --offline` against vendored
  `src/rust/vendor.tar.xz` (`extendr-api = 0.9.0`, MSRV `rustc >= 1.71.0`),
  including Windows Rtools45 static linking (`-lsynchronization -ldbghelp
  -lkernel32`). Crate attributions are recorded in `inst/AUTHORS`.
- **Model & validation coverage:** Validated via simulation recovery and
  published methodological benchmarks across single-level AR/VAR/RDSEM,
  continuous-time Ornstein-Uhlenbeck DSEM (`dsem_ct()`), dynamic CFA/WNFS/IRT,
  Tobit censored DSEM, two-level location-scale, random residual covariances
  (`logvf`), exact collapsed ARMA(1,1), ordinal probit, and multivariate VAR
  DSEM, cross-classified DSEM and dynamic CFA with random factor loadings,
  three-level DSEM (`DSEM3`) with Level-2 AR(1) and simultaneous block Gibbs,
  `TINTERVAL` grid alignment, `DIC`/`pD`, `STDYX`/`STDY`/`STD`, Bayesian
  $R^2$, partial correlations, hypothesis/Wald tests, discrete and
  continuous-time IRFs, `VIF`, penalized shrinkage priors (`PSEM`), and the
  optional interactive Shiny studio (`run_dsemr()`).
- **Pre-submission checks:** `R CMD build`, `R CMD check --as-cran`,
  `cargo fmt --check`, `cargo test --offline --locked`,
  `cargo clippy --offline --locked --all-targets -- -D warnings`,
  `spelling::spell_check_package()`, and `urlchecker::url_check()`.
- **Local release artifacts (`cran-submission/`, excluded via `.Rbuildignore`):**
  source tarball (`DSEMr_0.1.0.tar.gz`), PDF Reference Manual (`DSEMr-manual.pdf`),
  Mathematical Specification PDF, Validation & Benchmark Report PDF, CRAN
  Submission Guide PDF, and PDF Vignettes.
