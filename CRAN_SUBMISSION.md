# CRAN submission status

**Ready for CRAN submission (version 0.1.0).**

- **Package version:** `0.1.0`
- **Maintainer:** Marcus Harris (`marcus.harris.uconn@gmail.com`)
- **Offline Rust build:** Verified with `-j 2 --offline` against vendored
  `src/rust/vendor.tar.xz` (`extendr-api = 0.9.0`, MSRV `rustc >= 1.71.0`).
  Crate attributions recorded in `inst/AUTHORS`.
- **Parity coverage:** Validated across Mplus Ch. 6 (6.23–6.28), Ch. 9
  (9.30–9.40), RDSEM, and Mplus Version 9 Three-Level DSEM (`DSEM3`),
  `TINTERVAL`, `DIC`/`pD`, `STDYX`/`STDY`/`STD`, `VIF`, and `PSEM` priors.
- **Pre-submission checks:** `R CMD build`, `R CMD check --as-cran`,
  `cargo test --offline`, `spelling::spell_check_package()`, and
  `urlchecker::url_check()`.
