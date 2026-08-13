#' Inspect a DSEMr object
#'
#' @param object A `DSEMmodel` or `DSEMfit`.
#' @param what Component to extract.
#' @return The requested component.
#' @export
dsem_inspect <- function(object, what = c("model", "terms", "estimates", "draws",
                                          "diagnostics", "compute", "versions", "fit")) {
  what <- match.arg(what)
  if (inherits(object, "DSEMmodel")) {
    if (what == "model") return(object)
    if (what == "terms") return(object$terms)
    .dsem_abort("`%s` is only available for a fitted model.", what)
  }
  if (!inherits(object, "DSEMfit")) .dsem_abort("`object` must be a DSEMmodel or DSEMfit.")
  if (what == "model") return(object$model)
  if (what == "terms") return(object$model$terms)
  object[[what]]
}

#' @export
print.DSEMfit <- function(x, ...) {
  cat("<DSEMfit>", x$status, "\n")
  cat(" engine:", x$engine, " chains:", dim(x$draws)[3L],
      " retained draws/chain:", dim(x$draws)[1L], "\n")
  print(x$estimates[, c("parameter", "mean", "sd", "q2.5", "q97.5", "rhat", "ess")], row.names = FALSE)
  invisible(x)
}

#' @export
summary.DSEMfit <- function(object, ...) {
  out <- list(
    call = object$call,
    estimates = object$estimates,
    fit = object$fit,
    diagnostics = object$diagnostics,
    engine = object$engine,
    elapsed = object$elapsed,
    status = object$status
  )
  class(out) <- "summary.DSEMfit"
  out
}

#' @export
print.summary.DSEMfit <- function(x, ...) {
  cat("DSEMr Bayesian dynamic model\n")
  cat("Status:", x$status, " Engine:", x$engine, " Elapsed:", round(x$elapsed, 3), "seconds\n\n")
  print(x$estimates, row.names = FALSE)
  invisible(x)
}

#' @export
coef.DSEMfit <- function(object, ...) {
  stats::setNames(object$estimates$mean, object$estimates$parameter)
}

#' @export
vcov.DSEMfit <- function(object, ...) {
  p <- dim(object$draws)[2L]
  flat <- matrix(aperm(object$draws, c(1, 3, 2)), ncol = p)
  colnames(flat) <- dimnames(object$draws)$parameter
  stats::cov(flat)
}

#' @export
fitted.DSEMfit <- function(object, ...) object$fitted.values

#' @export
residuals.DSEMfit <- function(object, ...) object$residuals

#' @export
predict.DSEMfit <- function(object, newdata = NULL, ...) {
  if (is.null(newdata)) return(object$fitted.values)
  .dsem_abort("Prediction for `newdata` is not enabled in the foundation release.")
}

#' @export
plot.DSEMfit <- function(x, type = c("trace", "density"), parameter = NULL, ...) {
  type <- match.arg(type)
  params <- dimnames(x$draws)$parameter
  if (is.null(parameter)) parameter <- params[1L]
  if (!parameter %in% params) .dsem_abort("Unknown parameter `%s`.", parameter)
  j <- match(parameter, params)
  mat <- matrix(x$draws[, j, ], nrow = dim(x$draws)[1L])
  if (type == "trace") {
    graphics::matplot(mat, type = "l", lty = 1, xlab = "Retained iteration",
                      ylab = parameter, main = paste("Trace:", parameter), ...)
  } else {
    density <- stats::density(as.numeric(mat))
    graphics::plot(density, main = paste("Posterior density:", parameter), xlab = parameter, ...)
  }
  invisible(x)
}
