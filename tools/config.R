source("tools/msrv.R")

debug <- nzchar(Sys.getenv("DEBUG"))
not_cran <- nzchar(Sys.getenv("NOT_CRAN")) || debug
vendor_exists <- file.exists("src/rust/vendor.tar.xz")

if (!not_cran && !vendor_exists) {
  stop("CRAN-style builds require src/rust/vendor.tar.xz for offline Cargo compilation.")
}

.cran_flags <- if (!not_cran && vendor_exists) "-j 2 --offline" else ""
.profile <- if (debug) "" else "--release"
.clean_target <- if (debug) "" else "$(TARGET_DIR)"
.libdir <- if (debug) "debug" else "release"
.target <- ""
.panic_exports <- ""

windows <- .Platform[["OS.type"]] == "windows"
input <- if (windows) "src/Makevars.win.in" else "src/Makevars.in"
output <- if (windows) "src/Makevars.win" else "src/Makevars"
text <- readLines(input, warn = FALSE)
text <- gsub("@CRAN_FLAGS@", .cran_flags, text, fixed = TRUE)
text <- gsub("@PROFILE@", .profile, text, fixed = TRUE)
text <- gsub("@CLEAN_TARGET@", .clean_target, text, fixed = TRUE)
text <- gsub("@LIBDIR@", .libdir, text, fixed = TRUE)
text <- gsub("@TARGET@", .target, text, fixed = TRUE)
text <- gsub("@PANIC_EXPORTS@", .panic_exports, text, fixed = TRUE)
writeLines(text, output, useBytes = TRUE)
message("Wrote ", output)

