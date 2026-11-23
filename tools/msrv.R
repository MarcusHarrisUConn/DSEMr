desc <- read.dcf("DESCRIPTION")
requirements <- desc[, "SystemRequirements"]
if (!grepl("cargo", requirements, ignore.case = TRUE) ||
    !grepl("rustc", requirements, ignore.case = TRUE)) {
  stop("DESCRIPTION must declare Cargo and rustc in SystemRequirements.")
}
cargo_bins <- c(
  file.path(Sys.getenv("HOME"), ".cargo", "bin"),
  if (nzchar(Sys.getenv("USERPROFILE"))) file.path(Sys.getenv("USERPROFILE"), ".cargo", "bin")
)
Sys.setenv(PATH = paste(c(Sys.getenv("PATH"), unique(cargo_bins)), collapse = .Platform$path.sep))
rustc <- Sys.which("rustc")
cargo <- Sys.which("cargo")
if (!nzchar(rustc) || !nzchar(cargo)) {
  stop("Cargo and rustc are required to build DSEMr from source. See https://rustup.rs/.")
}
message("Using ", system2(cargo, "--version", stdout = TRUE))
rust_version <- system2(rustc, "--version", stdout = TRUE)
message("Using ", rust_version)
version_number <- sub("^rustc ([0-9]+\\.[0-9]+\\.[0-9]+).*", "\\1", rust_version[1L])
if (utils::compareVersion(version_number, "1.71.0") < 0) {
  stop("DSEMr's pinned Rust dependencies require rustc >= 1.71.0.")
}
