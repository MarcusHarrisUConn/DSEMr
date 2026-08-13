# Contributing to DSEMr

Use a focused branch and include tests and documentation with every change.
Before opening a pull request, run:

```sh
cargo fmt --manifest-path src/rust/Cargo.toml -- --check
cargo clippy --manifest-path src/rust/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path src/rust/Cargo.toml
Rscript -e 'testthat::test_local()'
R CMD build .
R CMD check DSEMr_*.tar.gz
```

Estimator contributions must include the equations, identification conditions,
prior parameterization, missing-data behavior, reference implementation,
recovery design, diagnostic expectations, and parity-matrix update. Never add a
silent syntax translation or broaden a capability claim without its evidence.

Do not commit copyrighted articles or Mplus documentation. Add bibliographic
metadata to `research/manifest.csv`; local files belong in the ignored
`research/library/` directory.

