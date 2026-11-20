.dsem_suppress_ess_cap <- function(expr) {
  withCallingHandlers(
    expr,
    warning = function(w) {
      if (grepl("ESS has been capped", conditionMessage(w), fixed = TRUE)) {
        invokeRestart("muffleWarning")
      }
    }
  )
}

.dsem_compute_diagnostics <- function(draws) {
  if (!is.numeric(draws) || length(dim(draws)) != 3L || any(dim(draws) == 0L)) {
    .dsem_abort("`draws` must be a numeric iteration-by-parameter-by-chain array.")
  }
  params <- dimnames(draws)[[2L]]
  if (is.null(params)) params <- paste0("parameter", seq_len(dim(draws)[2L]))
  if (anyNA(params) || any(!nzchar(params)) || anyDuplicated(params)) {
    .dsem_abort("Parameter names must be non-empty and unique.")
  }
  n_iter <- dim(draws)[1L]
  n_chains <- dim(draws)[3L]

  rows <- lapply(seq_along(params), function(j) {
    mat <- matrix(draws[, j, ], nrow = n_iter, ncol = n_chains)
    vec <- as.numeric(mat)
    finite_all <- all(is.finite(vec))
    non_const <- finite_all && length(unique(vec)) > 1L && n_iter >= 4L
    psr_val <- if (non_const && n_chains >= 2L) .dsem_mplus_psr(mat) else NA_real_
    p_one <- if (finite_all) {
      p_pos <- mean(vec > 0)
      p_neg <- mean(vec < 0)
      min(p_pos, p_neg)
    } else {
      NA_real_
    }
    mcse_med <- if (non_const) {
      tryCatch(
        .dsem_suppress_ess_cap(as.numeric(posterior::mcse_quantile(mat, probs = 0.5))),
        error = function(e) NA_real_
      )
    } else {
      NA_real_
    }
    data.frame(
      parameter = params[j],
      rhat = if (non_const) posterior::rhat(mat) else NA_real_,
      psr = psr_val,
      ess_bulk = if (non_const) .dsem_suppress_ess_cap(posterior::ess_bulk(mat)) else NA_real_,
      ess_tail = if (non_const) .dsem_suppress_ess_cap(posterior::ess_tail(mat)) else NA_real_,
      mcse_mean = if (non_const) .dsem_suppress_ess_cap(posterior::mcse_mean(mat)) else NA_real_,
      mcse_median = mcse_med,
      p_one_tailed = p_one,
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  out$ess <- out$ess_bulk
  out$mcse <- out$mcse_mean
  out
}

.dsem_mplus_psr <- function(mat) {
  n <- nrow(mat)
  m <- ncol(mat)
  if (m < 2L || n < 2L) return(NA_real_)
  chain_means <- colMeans(mat)
  chain_vars <- apply(mat, 2L, stats::var)
  w <- mean(chain_vars)
  if (!is.finite(w) || w <= 0) return(NA_real_)
  b <- n * stats::var(chain_means)
  v_hat <- ((n - 1) / n) * w + (1 / n) * b
  sqrt(max(v_hat / w, 1))
}

#' Compute covariate Variance Inflation Factors (VIF) for DSEM models
#'
#' Evaluates predictor collinearity across within-level and between-level
#' regression blocks, reporting \eqn{\text{VIF}_j = 1 / (1 - R^2_j)}.
#'
#' @param object Either a fitted `DSEMfit` object or a compiled `DSEMmodel`.
#' @param data Optional data frame (required when `object` is a `DSEMmodel`, or
#'   used to augment fitted objects).
#' @return A data frame with columns `level`, `outcome`, `predictor`, `r_squared`,
#'   and `vif`.
#' @examples
#' set.seed(12)
#' d <- simulate_dsem(n = 80, seed = 12)
#' d$x1 <- rnorm(80)
#' d$x2 <- 0.6 * d$x1 + rnorm(80, sd = 0.8)
#' m <- dsem_model("y ~ lag(y, 1) + x1 + x2", time = "time")
#' dsem_vif(m, data = d)
#' @export
dsem_vif <- function(object, data = NULL) {
  if (inherits(object, "DSEMfit")) {
    model <- object$model
    if (is.null(data)) data <- object$data
  } else if (inherits(object, "DSEMmodel")) {
    model <- object
  } else {
    .dsem_abort("`object` must be a `DSEMmodel` or `DSEMfit`.")
  }
  if (is.null(data) || !is.data.frame(data)) {
    .dsem_abort("Supply `data` as a data frame to compute covariate VIFs.")
  }

  regs <- model$terms[model$terms$op == "~" & model$terms$rhs != "1", , drop = FALSE]
  if (!nrow(regs)) {
    return(data.frame(
      level = character(), outcome = character(), predictor = character(),
      r_squared = numeric(), vif = numeric(), stringsAsFactors = FALSE
    ))
  }

  out_rows <- list()
  split_keys <- paste(regs$level, regs$lhs, sep = "::")
  for (key in unique(split_keys)) {
    sub <- regs[split_keys == key, , drop = FALSE]
    lvl <- sub$level[1L]
    lhs <- sub$lhs[1L]
    cols <- list()
    for (i in seq_len(nrow(sub))) {
      v <- sub$rhs[i]
      lg <- sub$lag[i]
      if (!v %in% names(data)) next
      vec <- as.numeric(data[[v]])
      if (lg > 0L) {
        vec <- c(rep(NA_real_, lg), utils::head(vec, length(vec) - lg))
      }
      label <- if (lg > 0L) paste0("lag", lg, "_", v) else v
      cols[[label]] <- vec
    }
    if (!length(cols)) next
    xmat <- as.data.frame(cols, check.names = FALSE)
    xmat <- xmat[stats::complete.cases(xmat), , drop = FALSE]
    p <- ncol(xmat)
    if (p == 1L) {
      out_rows[[length(out_rows) + 1L]] <- data.frame(
        level = lvl, outcome = lhs, predictor = names(xmat)[1L],
        r_squared = 0, vif = 1, stringsAsFactors = FALSE
      )
    } else if (nrow(xmat) > p + 1L) {
      for (j in seq_len(p)) {
        yj <- xmat[[j]]
        xj <- as.matrix(xmat[, -j, drop = FALSE])
        s_tot <- sum((yj - mean(yj))^2)
        if (s_tot <= 1e-12) {
          r2 <- NA_real_; vif <- NA_real_
        } else {
          fit_lm <- stats::lm.fit(cbind(1, xj), yj)
          rss <- sum(fit_lm$residuals^2)
          r2 <- min(max(1 - rss / s_tot, 0), 0.999999)
          vif <- 1 / (1 - r2)
        }
        out_rows[[length(out_rows) + 1L]] <- data.frame(
          level = lvl, outcome = lhs, predictor = names(xmat)[j],
          r_squared = r2, vif = vif, stringsAsFactors = FALSE
        )
      }
    }
  }
  if (!length(out_rows)) {
    return(data.frame(
      level = character(), outcome = character(), predictor = character(),
      r_squared = numeric(), vif = numeric(), stringsAsFactors = FALSE
    ))
  }
  res <- do.call(rbind, out_rows)
  rownames(res) <- NULL
  res
}

#' Extract Bayesian Deviance Information Criterion (DIC) and fit indices
#'
#' Returns the Deviance Information Criterion (`DIC`), effective number of
#' parameters (`pD`), posterior mean deviance (`D_bar`), and log-likelihood
#' summary from a fitted `DSEMfit` object.
#'
#' @param object A fitted `DSEMfit` object.
#' @return Named numeric vector containing `dic`, `pd`, `d_bar`, and `log_lik`.
#' @examples
#' dat <- simulate_dsem(n = 60, seed = 4)
#' fit <- dsem("y ~ lag(y, 1)", dat, chains = 2, iter = 200, warmup = 100, seed = 4)
#' dsem_dic(fit)
#' @export
dsem_dic <- function(object) {
  if (!inherits(object, "DSEMfit")) .dsem_abort("`object` must be a `DSEMfit`.")
  f <- object$fit
  c(
    dic = if (!is.null(f$dic)) f$dic else NA_real_,
    pd = if (!is.null(f$pd)) f$pd else NA_real_,
    d_bar = if (!is.null(f$d_bar)) f$d_bar else NA_real_,
    log_lik = if (!is.null(f$log_lik)) f$log_lik else NA_real_
  )
}
