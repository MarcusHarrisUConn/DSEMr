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
                                          "random_effect_draws",
                                          "l2_random_effects", "l2_random_effects_summary",
                                          "l2_random_effect_draws",
                                          "time_effects", "time_effects_summary",
                                          "time_effect_draws",
                                          "latent_states", "latent_draws",
                                          "prior", "sampler", "identification",
                                          "constraints", "wald_test",
                                          "standardized", "r2", "vif", "partial_cor")) {
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
  if (what == "r2") return(dsem_r2(object))
  if (what == "vif") return(dsem_vif(object))
  if (what == "partial_cor") return(dsem_partial_cor(object))
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
  } else {
    cat(" DIC: NA (categorical / probit model)\n")
  }
  cols <- intersect(c("parameter", "mean", "sd", "q50", "q2.5", "q97.5", "rhat", "ess"),
                    names(x$estimates))
  tab <- x$estimates[, cols, drop = FALSE]
  if (all(c("q2.5", "q97.5") %in% names(tab))) {
    tab$sig <- ifelse(tab$q2.5 > 0 | tab$q97.5 < 0, "*", "")
  }
  print(tab, row.names = FALSE)
  invisible(x)
}

#' @exportS3Method base::summary
summary.DSEMfit <- function(object, standardized = FALSE, r2 = FALSE,
                            point = c("mean", "median"), ...) {
  point <- match.arg(point)
  est <- object$estimates
  if (all(c("q2.5", "q97.5") %in% names(est))) {
    est$sig <- ifelse(est$q2.5 > 0 | est$q97.5 < 0, "*", "")
  }
  out <- list(
    call = object$call,
    estimates = est,
    point = point,
    fit = object$fit,
    diagnostics = object$diagnostics,
    standardized = if (isTRUE(standardized)) dsem_standardize(object, "stdyx") else NULL,
    r2 = if (isTRUE(r2) || isTRUE(standardized)) dsem_r2(object) else NULL,
    constraints = object$constraints,
    wald_test = object$wald_test,
    engine = object$engine,
    elapsed = object$elapsed,
    status = object$status,
    level_type = object$model$level_type
  )
  class(out) <- "summary.DSEMfit"
  out
}

#' @exportS3Method base::print
print.summary.DSEMfit <- function(x, digits = 3L, ...) {
  cat("DSEMr Bayesian Dynamic Structural Equation Model\n")
  cat("Hierarchy:", x$level_type, " Status:", x$status,
      " Engine:", x$engine, " Elapsed:", round(x$elapsed, 3), "s\n")
  if (!is.null(x$fit$dic) && is.finite(x$fit$dic)) {
    cat("Model Fit: DIC =", round(x$fit$dic, 3), "| pD =", round(x$fit$pd, 3))
    if (!is.null(x$fit$d_bar) && is.finite(x$fit$d_bar)) {
      cat(" | D_bar =", round(x$fit$d_bar, 3))
    }
    cat("\n")
  } else {
    cat("Model Fit: DIC = NA (categorical / probit outcomes)\n")
  }
  pt <- if (!is.null(x$point)) x$point else "mean"
  cat(sprintf("\nUnstandardized Posterior Summaries (Point = %s):\n", toupper(pt)))
  est <- x$estimates
  if (identical(pt, "median") && "q50" %in% names(est)) {
    first_cols <- intersect(c("parameter", "q50", "sd", "p_one_tailed", "q2.5", "q97.5", "sig", "rhat", "psr", "ess_bulk"), names(est))
    est <- est[, first_cols, drop = FALSE]
    names(est)[names(est) == "q50"] <- "estimate"
  }
  print(est, row.names = FALSE, digits = digits)
  if (!is.null(x$standardized)) {
    cat("\nStandardized Estimates (STDYX):\n")
    print(x$standardized, row.names = FALSE, digits = digits)
  }
  if (!is.null(x$r2) && nrow(x$r2) > 0L) {
    cat("\nBayesian R-Square:\n")
    print(x$r2, row.names = FALSE, digits = digits)
  }
  if (!is.null(x$constraints) && !is.null(x$constraints$summary)) {
    cat("\nDerived Parameters (MODEL CONSTRAINT):\n")
    print(x$constraints$summary, row.names = FALSE, digits = digits)
  }
  if (!is.null(x$wald_test) && !is.null(x$wald_test$wald)) {
    w <- x$wald_test$wald
    cat(sprintf("\nJoint Bayesian Wald Test (MODEL TEST): chi2(%d) = %.3f, p = %.4g\n",
                w$df, w$statistic, w$p_value))
  }
  invisible(x)
}

#' @exportS3Method stats::coef
coef.DSEMfit <- function(object, point = c("mean", "median"), ...) {
  point <- match.arg(point)
  col_nm <- if (point == "median" && "q50" %in% names(object$estimates)) "q50" else "mean"
  stats::setNames(object$estimates[[col_nm]], object$estimates$parameter)
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
    params <- params[!is.na(params)]
  }
  if (!length(params)) {
    empty <- matrix(numeric(0), nrow = 0L, ncol = 2L)
    colnames(empty) <- paste0(format(100 * c(alpha, 1 - alpha), trim = TRUE, digits = 3), " %")
    return(empty)
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
  regs <- object$model$terms[object$model$terms$op == "~" & object$model$terms$rhs != "1" & object$model$terms$level == "within", , drop = FALSE]
  out_nm <- if (nrow(regs)) unique(regs$lhs)[1L] else ""
  if (isTRUE(object$model$continuous_time)) {
    dr_idx <- grep("^(drift_|mean_drift_|fixed_drift_)", names(cf))
    phi <- if (length(dr_idx)) exp(cf[dr_idx[1L]]) else 0
  } else {
    ar_idx <- grep("^(lag1_|ar1_|mean_s|mean_lag1_|fixed_s|fixed_lag1_)", names(cf))
    phi <- if (length(ar_idx)) cf[ar_idx[1L]] else 0
  }
  int_candidates <- c("(Intercept)", paste0("intercept_", out_nm), paste0("mean_", out_nm), paste0("l3_mean_", out_nm))
  int_hit <- intersect(int_candidates, names(cf))
  if (length(int_hit)) {
    mu <- cf[int_hit[1L]]
  } else {
    int_idx <- grep("^(\\(Intercept\\)|intercept_)", names(cf))
    mu <- if (length(int_idx)) cf[int_idx[1L]] else 0
  }

  if (horizon > 0L) {
    last_y <- if (is.matrix(object$observed)) {
      utils::tail(object$observed[, 1L][!is.na(object$observed[, 1L])], 1L)
    } else {
      utils::tail(object$observed[!is.na(object$observed)], 1L)
    }
    fc <- numeric(horizon)
    cur <- last_y
    is_lat_cent <- any(grepl("^(mean_|fixed_)", names(cf)))
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
  is_lat_cent <- any(grepl("^(mean_|fixed_)", names(cf)))
  pred <- if (is_lat_cent) mu + phi * (yv[-n_new] - mu) else mu + phi * yv[-n_new]
  c(mu, pred)
}

#' @exportS3Method graphics::plot
plot.DSEMfit <- function(x, type = c("trace", "density", "acf", "intervals", "latent", "diagram"),
                         parameter = NULL, ...) {
  type <- match.arg(type)
  if (type == "diagram") {
    return(dsem_diagram(x, format = "plot", ...))
  }
  if (type == "latent") {
    if (is.null(x$latent_states)) .dsem_abort("Latent state summaries are not available for this model.")
    ls_df <- x$latent_states
    graphics::plot(ls_df$time, ls_df$mean, type = "l", lwd = 2,
                   ylim = range(c(ls_df$q2.5, ls_df$q97.5), na.rm = TRUE),
                   xlab = "Time", ylab = "Latent factor score", main = "Posterior Latent Trajectory", ...)
    graphics::polygon(
      c(ls_df$time, rev(ls_df$time)),
      c(ls_df$q2.5, rev(ls_df$q97.5)),
      col = grDevices::adjustcolor("#2563eb", alpha.f = 0.18),
      border = NA
    )
    graphics::lines(ls_df$time, ls_df$mean, lwd = 2, col = "#1d4ed8")
    graphics::lines(ls_df$time, ls_df$q2.5, lty = 2, col = "#64748b")
    graphics::lines(ls_df$time, ls_df$q97.5, lty = 2, col = "#64748b")
    return(invisible(x))
  }
  if (type == "intervals") {
    est <- x$estimates
    k <- nrow(est)
    old_par <- graphics::par(no.readonly = TRUE)
    on.exit(graphics::par(old_par), add = TRUE)
    max_chars <- max(nchar(est$parameter), 6L)
    left_mar <- min(max(4.5, max_chars * 0.48 + 1.2), 14)
    graphics::par(mar = c(4.5, left_mar, 3, 1.5))
    sig <- est$q2.5 > 0 | est$q97.5 < 0
    cols <- ifelse(sig, "#1d4ed8", "#64748b")
    graphics::plot(est$mean, seq_len(k), xlim = range(c(0, est$q2.5, est$q97.5), na.rm = TRUE),
                   yaxt = "n", xlab = "Posterior Estimate (95% CrI)", ylab = "",
                   main = "Posterior Parameter Intervals", pch = 19, col = cols, ...)
    graphics::axis(2, at = seq_len(k), labels = est$parameter, las = 1, cex.axis = 0.85)
    graphics::segments(est$q2.5, seq_len(k), est$q97.5, seq_len(k), lwd = 2, col = cols)
    graphics::abline(v = 0, lty = 3, col = "#94a3b8")
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
