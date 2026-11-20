# Root launcher so `shiny::runApp(".")` launches the DSEMr Shiny studio
# directly from the repository directory.
required_packages <- c("shiny", "bslib", "ggplot2")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) {
  stop(
    "Missing required package(s): ",
    paste(missing_packages, collapse = ", "),
    ". Install with install.packages(c(",
    paste(sprintf("\"%s\"", missing_packages), collapse = ", "),
    ")), then relaunch the app.",
    call. = FALSE
  )
}

if (requireNamespace("DSEMr", quietly = TRUE)) {
  DSEMr:::dsemr_shiny_app()
} else {
  stop("Install the DSEMr package first (e.g. `R CMD INSTALL .`) to compile the Rust backend.", call. = FALSE)
}
