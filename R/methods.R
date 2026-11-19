#' Inspect components of a DSEMr model or fitted object
#'
#' @param object A `DSEMmodel` or `DSEMfit` object.
#' @param what Component to retrieve.
#' @return The requested component from `object`.
#' @examples
#' dat <- simulate_dsem(n = 50, seed = 3)
#' fit <- dsem("y ~ lag(y, 1)", dat, chains = 2, iter = 160, warmup = 80, seed = 3)
#' dsem_inspect(fit, "diagnostics")
#' @export
dsem_inspect <- function(object, what = c("model", "terms", "estimates", "draws",
                                          "diagnostics", "compute", "versions", "fit",
                                          "random_effects", "random_effects_summary",
                                          "random_effect_draws", "latent_states",
                                          "latent_draws", "prior", "sampler",
                                          "identification", "standardized", "vif")) {
  what <- match.arg(what)
  if (inherits(object, "DSEMmodel")) {
    if (what == "model") return(object)
    if (what == "terms") return(object$terms)
    .dsem_abort("`%s` is only available for a fitted model.", what)
  }
  if (!inherits(object, "DSEMfit")) .dsem_abort("`object` must be a DSEMmodel or DSEMfit.")
  if (what == "model") return(object$model)
  if (what == "terms") return(object$model$terms)
  if (what == "standardized") return(dsem_standardize(object, type = "stdyx"))
  if (what == "vif") return(dsem_vif(object))
  if (is.null(object[[what]])) .dsem_abort("`%s` is not available for this fitted model.", what)
  object[[what]]
}

#' @exportS3Method base::print
print.DSEMfit <- function(x, ...) {
  cat("<DSEMfit>", x$status, "\n")
  cat(" engine:", x$engine, " chains:", dim(x$draws)[3L],
      " retained draws/chain:", dim(x$draws)[1L], "\n")
  if (!is.null(x$fit$dic) && is.finite(x$fit$dic)) {
    cat(" DIC:", round(x$fit$dic, 3), " pD:", round(x$fit$pd, 3), "\n")
  }
  cols <- intersect(c("parameter", "mean", "sd", "q50", "q2.5", "q97.5", "rhat", "ess"),
                    names(x$estimates))
  print(x$estimates[, cols, drop = FALSE], row.names = FALSE)
  invisible(x)
}

#' @exportS3Method base::summary
summary.DSEMfit <- function(object, standardized = FALSE, ...) {
  out <- list(
    call = object$call,
    estimates = object$estimates,
    fit = object$fit,
    diagnostics = object$diagnostics,
    standardized = if (isTRUE(standardized)) dsem_standardize(object, "stdyx") else NULL,
    engine = object$engine,
    elapsed = object$elapsed,
    status = object$status,
    level_type = object$model$level_type
  )
  class(out) <- "summary.DSEMfit"
  out
}

#' @exportS3Method base::print
print.summary.DSEMfit <- function(x, ...) {
  cat("DSEMr Bayesian Dynamic Structural Equation Model\n")
  cat("Hierarchy:", x$level_type, " Status:", x$status,
      " Engine:", x$engine, " Elapsed:", round(x$elapsed, 3), "s\n")
  if (!is.null(x$fit$dic) && is.finite(x$fit$dic)) {
    cat("Model Fit: DIC =", round(x$fit$dic, 3), "| pD =", round(x$fit$pd, 3), "\n")
  }
  cat("\nUnstandardized Posterior Summaries:\n")
  print(x$estimates, row.names = FALSE)
  if (!is.null(x$standardized)) {
    cat("\nStandardized Estimates (STDYX):\n")
    print(x$standardized, row.names = FALSE)
  }
  invisible(x)
}

#' @exportS3Method stats::coef
coef.DSEMfit <- function(object, ...) {
  stats::setNames(object$estimates$mean, object$estimates$parameter)
}

#' @exportS3Method stats::vcov
vcov.DSEMfit <- function(object, ...) {
  p <- dim(object$draws)[2L]
  flat <- matrix(aperm(object$draws, c(1L, 3L, 2L)), ncol = p)
  colnames(flat) <- dimnames(object$draws)$parameter
  stats::cov(flat)
}

#' @exportS3Method stats::confint
confint.DSEMfit <- function(object, parm, level = 0.95, ...) {
  if (!is.numeric(level) || length(level) != 1L || !is.finite(level) || level <= 0 || level >= 1) {
    .dsem_abort("`level` must be strictly between 0 and 1.")
  }
  alpha <- (1 - level) / 2
  params <- dimnames(object$draws)$parameter
  if (!missing(parm) && !is.null(parm)) {
    params <- if (is.numeric(parm)) params[parm] else intersect(parm, params)
  }
  res <- t(vapply(params, function(nm) {
    vals <- as.numeric(object$draws[, nm, ])
    stats::quantile(vals, probs = c(alpha, 1 - alpha), names = FALSE, na.rm = TRUE)
  }, numeric(2)))
  colnames(res) <- paste0(format(100 * c(alpha, 1 - alpha), trim = TRUE, digits = 3), " %")
  res
}

#' @exportS3Method stats::logLik
logLik.DSEMfit <- function(object, ...) {
  val <- if (!is.null(object$fit$log_lik)) object$fit$log_lik else NA_real_
  attr(val, "df") <- if (!is.null(object$fit$pd) && is.finite(object$fit$pd)) object$fit$pd else nrow(object$estimates)
  attr(val, "nobs") <- if (is.matrix(object$observed)) nrow(object$observed) else length(object$observed)
  class(val) <- "logLik"
  val
}

#' @exportS3Method stats::fitted
fitted.DSEMfit <- function(object, ...) object$fitted.values

#' @exportS3Method stats::residuals
residuals.DSEMfit <- function(object, ...) object$residuals

#' @exportS3Method stats::predict
predict.DSEMfit <- function(object, newdata = NULL, horizon = 0L, ...) {
  horizon <- .dsem_integer(horizon, "horizon", 0)
  if (is.null(newdata) && horizon == 0L) return(object$fitted.values)
  cf <- stats::coef(object)
  ar_idx <- grep("^(lag1_|ar1_|mean_s|mean_lag1_)", names(cf))
  phi <- if (length(ar_idx)) cf[ar_idx[1L]] else 0
  int_idx <- grep("^(\\(Intercept\\)|intercept_|mean_)", names(cf))
  mu <- if (length(int_idx)) cf[int_idx[1L]] else 0

  if (horizon > 0L) {
    last_y <- if (is.matrix(object$observed)) {
      utils::tail(object$observed[, 1L][!is.na(object$observed[, 1L])], 1L)
    } else {
      utils::tail(object$observed[!is.na(object$observed)], 1L)
    }
    fc <- numeric(horizon)
    cur <- last_y
    is_lat_cent <- any(grepl("^mean_", names(cf)))
    for (h in seq_len(horizon)) {
      cur <- if (is_lat_cent) mu + phi * (cur - mu) else mu + phi * cur
      fc[h] <- cur
    }
    return(fc)
  }

  if (!is.data.frame(newdata)) .dsem_abort("`newdata` must be a data frame.")
  regs <- object$model$terms[object$model$terms$op == "~" & object$model$terms$rhs != "1" & object$model$terms$level == "within", , drop = FALSE]
  out_nm <- unique(regs$lhs)[1L]
  if (!out_nm %in% names(newdata)) .dsem_abort("`newdata` must contain outcome column `%s`.", out_nm)
  yv <- as.double(newdata[[out_nm]])
  n_new <- length(yv)
  if (n_new < 2L) return(rep(mu, n_new))
  is_lat_cent <- any(grepl("^mean_", names(cf)))
  pred <- if (is_lat_cent) mu + phi * (yv[-n_new] - mu) else mu + phi * yv[-n_new]
  c(mu, pred)
}

#' @exportS3Method graphics::plot
plot.DSEMfit <- function(x, type = c("trace", "density", "acf", "intervals", "latent"),
                         parameter = NULL, ...) {
  type <- match.arg(type)
  if (type == "latent") {
    if (is.null(x$latent_states)) .dsem_abort("Latent state summaries are not available for this model.")
    ls_df <- x$latent_states
    graphics::plot(ls_df$time, ls_df$mean, type = "l", lwd = 2,
                   ylim = range(c(ls_df$q2.5, ls_df$q97.5), na.rm = TRUE),
                   xlab = "Time", ylab = "Latent factor score", main = "Posterior Latent Trajectory", ...)
    graphics::lines(ls_df$time, ls_df$q2.5, lty = 2)
    graphics::lines(ls_df$time, ls_df$q97.5, lty = 2)
    return(invisible(x))
  }
  if (type == "intervals") {
    est <- x$estimates
    k <- nrow(est)
    graphics::plot(est$mean, seq_len(k), xlim = range(c(est$q2.5, est$q97.5), na.rm = TRUE),
                   yaxt = "n", xlab = "Estimate (95% CrI)", ylab = "", main = "Posterior Parameter Intervals", pch = 19, ...)
    graphics::axis(2, at = seq_len(k), labels = est$parameter, las = 1)
    graphics::segments(est$q2.5, seq_len(k), est$q97.5, seq_len(k), lwd = 2)
    graphics::abline(v = 0, lty = 3)
    return(invisible(x))
  }
  params <- dimnames(x$draws)$parameter
  if (is.null(parameter)) parameter <- params[1L]
  if (!parameter %in% params) .dsem_abort("Unknown parameter `%s`.", parameter)
  j <- match(parameter, params)
  mat <- matrix(x$draws[, j, ], nrow = dim(x$draws)[1L])
  if (type == "trace") {
    graphics::matplot(mat, type = "l", lty = 1, xlab = "Retained iteration",
                      ylab = parameter, main = paste("Trace:", parameter), ...)
  } else if (type == "density") {
    dens <- stats::density(as.numeric(mat))
    graphics::plot(dens, main = paste("Posterior density:", parameter), xlab = parameter, ...)
  } else {
    stats::acf(mat[, 1L], main = paste("Autocorrelation:", parameter), ...)
  }
  invisible(x)
}
