source("tools/msrv.R")

debug <- nzchar(Sys.getenv("DEBUG"))
online_override <- identical(tolower(Sys.getenv("DSEMR_ONLINE_CARGO")), "true")
vendor_exists <- file.exists("src/rust/vendor.tar.xz")

if (!online_override && !vendor_exists) {
  stop("Offline Cargo build requires src/rust/vendor.tar.xz.")
}

.cran_flags <- if (!online_override && vendor_exists) "-j 2 --offline" else "-j 2"
.profile <- if (debug) "" else "--release"
.clean_target <- if (debug) "" else "\"$(TARGET_DIR)\""
.libdir <- if (debug) "debug" else "release"
.target <- ""
.panic_exports <- ""
.win_target <- if (identical(R.version$arch, "aarch64")) {
  "aarch64-pc-windows-gnullvm"
} else if (identical(.Platform$r_arch, "i386")) {
  "i686-pc-windows-gnu"
} else {
  "x86_64-pc-windows-gnu"
}

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
text <- gsub("@WIN_TARGET@", .win_target, text, fixed = TRUE)
writeLines(text, output, useBytes = TRUE)
message("Wrote ", output)
