# Release checklist

The current version is an experimental development release and is not eligible
for CRAN submission.

- [ ] All mandatory rows in `docs/PARITY.md` are validated or justified.
- [ ] Public evidence bundles are reproducible from clean environments.
- [ ] Simulation-based calibration and parameter-recovery reports pass review.
- [ ] Worker-count and scheduler invariance pass on real HPC CI.
- [ ] Rust fmt, Clippy, tests, audit, sanitizer/Valgrind jobs pass.
- [ ] Linux, macOS, Windows, R release/oldrel/devel checks pass.
- [ ] Source tarball installs offline with vendored crates.
- [ ] Vignettes, examples, URLs, spelling, license, and citation are audited.
- [ ] `R CMD check --as-cran`, win-builder, and rhub-style checks are clean.
- [ ] Reverse dependencies are checked.
- [ ] `cran-comments.md` and `CRAN_SUBMISSION.md` match final evidence.

