# Release checklist (DSEMr 0.1.0)

- [x] All benchmark model families in `docs/PARITY.md` are validated via simulation recovery and published DSEM benchmark datasets.
- [x] Rust `fmt`, strict `clippy --all-targets -- -D warnings`, and offline `cargo test --locked` pass.
- [x] Source tarball builds offline with vendored crates (`src/rust/vendor.tar.xz`), Windows Rtools45 linking (`-lsynchronization -ldbghelp -lkernel32`), and `inst/AUTHORS` crate attributions.
- [x] All 27 exported functions, runnable examples, 11 executable vignettes, 29 S3 methods, spelling (`inst/WORDLIST`), URLs (`urlchecker`), license, and citation are audited.
- [x] Local `R CMD check --as-cran` passes with 0 errors and 0 package warnings.
- [x] `cran-comments.md`, `CRAN_SUBMISSION.md`, and PDF submission artifacts in `cran-submission/` match final release state.
- [x] GitHub Actions Linux/macOS/Windows CI matrix and Rust MSRV checks passing.
- [ ] Run CRAN Win-builder (`devtools::check_win_devel()`) and upload `cran-submission/DSEMr_0.1.0.tar.gz` via `devtools::submit_cran()` or <https://cran.r-project.org/submit.html>.
