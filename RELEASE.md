# Release checklist (DSEMr 0.1.0)

- [x] All mandatory Mplus Version 8 and Version 9 rows in `docs/PARITY.md` are validated.
- [x] Rust `fmt`, strict `clippy --all-targets -- -D warnings`, and offline `cargo test --locked` pass.
- [x] Source tarball builds offline with vendored crates (`src/rust/vendor.tar.xz`) and `inst/AUTHORS` crate attributions.
- [x] Examples, 9 executable vignettes, S3 methods, spelling (`inst/WORDLIST`), license, and citation are audited.
- [x] Local `R CMD check --as-cran` passes with 0 errors and 0 package warnings.
- [x] `cran-comments.md` and `CRAN_SUBMISSION.md` match final release state.
- [ ] Run GitHub Actions Linux/macOS/Windows CI matrix and CRAN Win-builder (`devtools::check_win_devel()`) prior to final upload via `devtools::submit_cran()` or https://cran.r-project.org/submit.html.
