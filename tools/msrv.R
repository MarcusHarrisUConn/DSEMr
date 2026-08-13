desc <- read.dcf("DESCRIPTION")
requirements <- desc[, "SystemRequirements"]
if (!grepl("cargo", requirements, ignore.case = TRUE) ||
    !grepl("rustc", requirements, ignore.case = TRUE)) {
  stop("DESCRIPTION must declare Cargo and rustc in SystemRequirements.")
}
Sys.setenv(PATH = paste(Sys.getenv("PATH"), file.path(Sys.getenv("HOME"), ".cargo", "bin"), sep = .Platform$path.sep))
rustc <- Sys.which("rustc")
cargo <- Sys.which("cargo")
if (!nzchar(rustc) || !nzchar(cargo)) {
  stop("Cargo and rustc are required to build DSEMr from source. See https://rustup.rs/.")
}
message("Using ", system2(cargo, "--version", stdout = TRUE))
message("Using ", system2(rustc, "--version", stdout = TRUE))

