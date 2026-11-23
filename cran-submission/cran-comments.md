## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release.

## Notes on Rust dependencies

* All Rust crate dependencies (`extendr-api` 0.9.0 and its minimal transitive
  dependencies) are pinned in `src/rust/Cargo.lock` and vendored offline inside
  `src/rust/vendor.tar.xz`. No internet access is performed during compilation.
* `src/Makevars.in` and `src/Makevars.win.in` pass `-j 2 --offline` to Cargo,
  print exact `rustc` and `cargo` versions during installation, link all
  required Windows runtime libraries (`-lws2_32 -ladvapi32 -luserenv -lbcrypt
  -lntdll -ldbghelp -lsynchronization -lkernel32`), and clean up unpacked
  vendor and build artifacts after linking.
* Authors, copyrights, and licenses for all vendored Rust crates are documented
  in `inst/AUTHORS`.
* The package uses pure standard-library Rust numerical kernels with no
  external C/Fortran system libraries beyond R itself.

## Test environments

* Local macOS arm64 (R 4.6.1, rustc 1.97.1, cargo 1.97.1): `R CMD check --as-cran`
* GitHub Actions CI matrix:
  - Ubuntu latest (R release, R devel, R oldrel-1)
  - macOS latest (R release)
  - Windows latest (R release, Rtools45)
  - Rust stable and MSRV 1.71.0 (`cargo fmt`, `cargo clippy --offline --locked`, `cargo test --offline --locked`)
