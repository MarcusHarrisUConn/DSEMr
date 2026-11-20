#' Launch the interactive DSEMr Shiny studio
#'
#' Opens the guided interactive web interface for Bayesian Dynamic Structural
#' Equation Modeling (`DSEMr`), covering data design and `TINTERVAL` alignment,
#' visual model building (`dsem_spec()` and syntax editing) with live SVG path
#' diagrams and LaTeX equations, Rust-accelerated MCMC estimation and
#' diagnostics, post-estimation analytics (`STDYX`, Bayesian \eqn{R^2}, partial
#' correlations, hypothesis and Wald tests, impulse response functions, cluster
#' rankings, and DIC model comparison), and reproducible R/Quarto report export.
#'
#' Requires the optional suggested packages `shiny`, `bslib`, and `ggplot2`.
#'
#' @param launch.browser Logical; passed to [shiny::runApp()] (default `TRUE`).
#' @param port Optional port number passed to [shiny::runApp()].
#' @param host Host address passed to [shiny::runApp()] (default `"127.0.0.1"`).
#' @return Invisibly returns the result of [shiny::runApp()].
#' @examples
#' if (interactive()) {
#'   run_dsemr()
#' }
#' @export
run_dsemr <- function(launch.browser = TRUE,
                      port = getOption("shiny.port", NULL),
                      host = getOption("shiny.host", "127.0.0.1")) {
  .dsem_require_shiny_pkgs()
  shiny::runApp(
    dsemr_shiny_app(),
    launch.browser = launch.browser,
    port = port,
    host = host
  )
}

#' Construct the DSEMr Shiny application object
#'
#' @return A `shiny.appobj` suitable for [shiny::runApp()] or testing.
#' @keywords internal
dsemr_shiny_app <- function() {
  .dsem_require_shiny_pkgs()
  shiny::shinyApp(ui = .dsem_shiny_ui(), server = .dsem_shiny_server)
}

.dsem_require_shiny_pkgs <- function() {
  req <- c("shiny", "bslib", "ggplot2")
  miss <- req[!vapply(req, requireNamespace, logical(1), quietly = TRUE)]
  if (length(miss)) {
    .dsem_abort(
      "Launching the DSEMr Shiny studio requires suggested package(s): %s. Install with install.packages(c(%s)).",
      paste(miss, collapse = ", "),
      paste(sprintf("\"%s\"", miss), collapse = ", ")
    )
  }
  invisible(TRUE)
}
