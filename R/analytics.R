#' Posterior partial correlations for DSEM covariance blocks
#'
#' Computes the full posterior distribution of partial correlations (as well as
#' zero-order correlations) by inverting each retained MCMC draw of the
#' model's between-level random-effect covariance matrix and/or within-level
#' multivariate innovation covariance matrix (Asparouhov & Muthen, 2025).
#'
#' @param object A fitted `DSEMfit` object.
#' @param level Covariance block(s) to summarize: `"all"`, `"between"`,
#'   `"within"`, `"level2"`, `"level3"`, or `"time"`.
#' @param prob Credible interval probability (default `0.95`).
#' @return A data frame of pairwise zero-order and partial correlation summaries
#'   (`level`, `var1`, `var2`, `cor_mean`, `pcor_mean`, `pcor_sd`,
#'   `pcor_q_lower`, `pcor_q50`, `pcor_q_upper`, `p_one_tailed`, `sig`).
#' @examples
#' set.seed(42)
#' n <- 80
#' y1 <- numeric(n); y2 <- numeric(n)
#' for (t in 2:n) {
#'   e1 <- rnorm(1, sd = 0.7)
#'   y1[t] <- 0.4 * y1[t - 1] + e1
#'   y2[t] <- 0.4 * y2[t - 1] + 0.5 * e1 + rnorm(1, sd = 0.5)
#' }
#' fit <- dsem("y1 ~ lag(y1, 1) + lag(y2, 1)\ny2 ~ lag(y1, 1) + lag(y2, 1)",
#'             data.frame(time = seq_len(n), y1 = y1, y2 = y2),
#'             time = "time", chains = 2, iter = 160, warmup = 80, seed = 42)
#' dsem_partial_cor(fit)
#' @export
dsem_partial_cor <- function(object,
                             level = c("all", "between", "within", "level2", "level3", "time"),
                             prob = 0.95) {
  if (!inherits(object, "DSEMfit")) .dsem_abort("`object` must be a `DSEMfit`.")
  level <- match.arg(level)
  if (!is.numeric(prob) || length(prob) != 1L || !is.finite(prob) || prob <= 0 || prob >= 1) {
    .dsem_abort("`prob` must be strictly between 0 and 1.")
  }
  alpha <- (1 - prob) / 2
  draws <- object$draws
  params <- dimnames(draws)[[2L]]
  n_tot <- dim(draws)[1L] * dim(draws)[3L]
  flat <- matrix(aperm(draws, c(1L, 3L, 2L)), nrow = n_tot, ncol = length(params))
  colnames(flat) <- params

  # Helper to locate a covariance matrix block given var prefix and cov prefix
  extract_block <- function(lvl_name, var_prefix, cov_prefix, exclude_vars = character()) {
    v_cols <- params[startsWith(params, var_prefix)]
    v_names <- sub(paste0("^", var_prefix), "", v_cols)
    keep <- !v_names %in% exclude_vars
    v_cols <- v_cols[keep]; v_names <- v_names[keep]
    # Keep only variables that participate in at least one covariance column
    p <- length(v_names)
    if (p < 2L) return(NULL)
    cov_map <- matrix(NA_character_, p, p, dimnames = list(v_names, v_names))
    has_any_cov <- FALSE
    for (i in seq_len(p)) {
      cov_map[i, i] <- v_cols[i]
      if (i < p) {
        for (j in (i + 1L):p) {
          c1 <- paste0(cov_prefix, v_names[j], "_", v_names[i])
          c2 <- paste0(cov_prefix, v_names[i], "_", v_names[j])
          hit <- if (c1 %in% params) c1 else if (c2 %in% params) c2 else NA_character_
          if (!is.na(hit)) {
            cov_map[i, j] <- hit
            cov_map[j, i] <- hit
            has_any_cov <- TRUE
          }
        }
      }
    }
    if (!has_any_cov) return(NULL)
    # Restrict to connected subset of variables that have covariances
    connected <- vapply(seq_len(p), function(i) any(!is.na(cov_map[i, -i])), logical(1))
    if (sum(connected) < 2L) return(NULL)
    v_names <- v_names[connected]
    cov_map <- cov_map[connected, connected, drop = FALSE]
    p <- length(v_names)

    n_pairs <- p * (p - 1L) / 2L
    cor_draws <- matrix(NA_real_, n_tot, n_pairs)
    pcor_draws <- matrix(NA_real_, n_tot, n_pairs)
    pair_i <- integer(n_pairs); pair_j <- integer(n_pairs)
    idx_p <- 0L
    for (i in seq_len(p - 1L)) {
      for (j in (i + 1L):p) {
        idx_p <- idx_p + 1L
        pair_i[idx_p] <- i; pair_j[idx_p] <- j
      }
    }

    mat_s <- matrix(0, p, p)
    for (s in seq_len(n_tot)) {
      for (i in seq_len(p)) {
        mat_s[i, i] <- max(flat[s, cov_map[i, i]], 1e-10)
        if (i < p) {
          for (j in (i + 1L):p) {
            v <- if (!is.na(cov_map[i, j])) flat[s, cov_map[i, j]] else 0
            mat_s[i, j] <- v; mat_s[j, i] <- v
          }
        }
      }
      prec_s <- .dsem_spd_solve(mat_s)
      for (k in seq_len(n_pairs)) {
        ii <- pair_i[k]; jj <- pair_j[k]
        cor_draws[s, k] <- mat_s[ii, jj] / sqrt(mat_s[ii, ii] * mat_s[jj, jj])
        pcor_draws[s, k] <- -prec_s[ii, jj] / sqrt(max(prec_s[ii, ii] * prec_s[jj, jj], 1e-16))
      }
    }

    rows <- lapply(seq_len(n_pairs), function(k) {
      pc <- pcor_draws[, k]
      q <- stats::quantile(pc, probs = c(alpha, 0.5, 1 - alpha), names = FALSE, na.rm = TRUE)
      data.frame(
        level = lvl_name,
        var1 = v_names[pair_i[k]],
        var2 = v_names[pair_j[k]],
        cor_mean = mean(cor_draws[, k], na.rm = TRUE),
        pcor_mean = mean(pc, na.rm = TRUE),
        pcor_sd = stats::sd(pc, na.rm = TRUE),
        pcor_q_lower = q[1L],
        pcor_q50 = q[2L],
        pcor_q_upper = q[3L],
        p_one_tailed = min(mean(pc <= 0, na.rm = TRUE), mean(pc >= 0, na.rm = TRUE)),
        sig = ifelse(q[1L] > 0 || q[3L] < 0, "*", ""),
        stringsAsFactors = FALSE
      )
    })
    do.call(rbind, rows)
  }

  # Exclude distal outcome residual variances from between random-effect Omega_B block
  b_regs <- object$model$terms[object$model$terms$op == "~" & object$model$terms$level == "between", , drop = FALSE]
  distal_excl <- setdiff(unique(b_regs$lhs), c(
    unique(object$model$terms$lhs[object$model$terms$level == "within"]),
    object$model$terms$label[object$model$terms$level == "within"]
  ))

  blocks <- list()
  want <- function(lbl) level == "all" || level == lbl
  if (want("between")) {
    blocks[[length(blocks) + 1L]] <- extract_block("Between", "var_", "cov_", distal_excl)
    blocks[[length(blocks) + 1L]] <- extract_block("Between", "between_var_", "between_cov_")
    blocks[[length(blocks) + 1L]] <- extract_block("Between (Subject)", "id_var_", "id_cov_")
  }
  if (want("level3")) {
    blocks[[length(blocks) + 1L]] <- extract_block("Level 3", "l3_var_", "l3_cov_")
  }
  if (want("level2")) {
    blocks[[length(blocks) + 1L]] <- extract_block("Level 2", "l2_var_", "l2_cov_")
  }
  if (want("time")) {
    blocks[[length(blocks) + 1L]] <- extract_block("Occasion (Time)", "time_var_", "time_cov_")
  }
  if (want("within")) {
    if (identical(object$model$level_type, "single")) {
      blocks[[length(blocks) + 1L]] <- extract_block("Within", "sigma2_", "cov_")
    }
    blocks[[length(blocks) + 1L]] <- extract_block("Within", "sigma2_", "within_cov_")
    blocks[[length(blocks) + 1L]] <- extract_block("Within (Factors)", "factor_variance_", "factor_cov_")
  }

  blocks <- Filter(Negate(is.null), blocks)
  if (!length(blocks)) {
    return(data.frame(
      level = character(), var1 = character(), var2 = character(),
      cor_mean = numeric(), pcor_mean = numeric(), pcor_sd = numeric(),
      pcor_q_lower = numeric(), pcor_q50 = numeric(), pcor_q_upper = numeric(),
      p_one_tailed = numeric(), sig = character(), stringsAsFactors = FALSE
    ))
  }
  out <- do.call(rbind, blocks)
  rownames(out) <- NULL
  out
}

#' Bayesian derived parameters, contrast tests, and joint Wald tests
#'
#' Evaluates one or more parameter expressions (such as indirect effects,
#' cross-lagged differences, simple slopes, or variance ratios) across every
#' posterior MCMC draw of a fitted `DSEMfit` object, computing posterior
#' summaries, directional exceedance probabilities, and an optional joint
#' multivariate Bayesian Wald chi-square test.
#'
#' @param object A fitted `DSEMfit` object.
#' @param ... Named or unnamed character strings specifying expressions over
#'   parameter names in `names(coef(object))` (e.g., `diff = "s12 - s21"`,
#'   `indirect = "y_on_w * z_on_y"`, or directional comparisons `"s12 > s21"`
#'   which are evaluated as `s12 - s21`).
#' @param hypotheses Optional character vector of expressions (combined with `...`).
#' @param prob Credible interval probability (default `0.95`).
#' @param joint_test Logical; if `TRUE`, computes a joint multivariate Bayesian
#'   Wald chi-square statistic testing that all evaluated quantities equal zero.
#' @return A `DSEMhypothesis` object with `summary` data frame, posterior
#'   `draws` matrix, and optional `wald` list (`statistic`, `df`, `p_value`).
#' @examples
#' dat <- simulate_dsem(n = 80, ar = 0.5, sigma = 0.8, seed = 11)
#' fit <- dsem("y ~ lag(y, 1)", dat, chains = 2, iter = 160, warmup = 80, seed = 11)
#' dsem_hypothesis(fit, stationary_var = "sigma2 / (1 - lag1_y^2)",
#'                 positive_ar = "lag1_y > 0")
#' @export
dsem_hypothesis <- function(object, ..., hypotheses = NULL, prob = 0.95,
                            joint_test = TRUE) {
  if (!inherits(object, "DSEMfit")) .dsem_abort("`object` must be a `DSEMfit`.")
  if (!is.numeric(prob) || length(prob) != 1L || !is.finite(prob) || prob <= 0 || prob >= 1) {
    .dsem_abort("`prob` must be strictly between 0 and 1.")
  }
  dots <- list(...)
  exprs <- character()
  nms <- character()
  if (length(dots)) {
    dot_nms <- names(dots)
    if (is.null(dot_nms)) dot_nms <- rep("", length(dots))
    for (i in seq_along(dots)) {
      v <- dots[[i]]
      if (!is.character(v) || !length(v)) .dsem_abort("Hypothesis expressions must be character strings.")
      v_nms <- if (length(v) == 1L && nzchar(dot_nms[i])) dot_nms[i] else if (!is.null(names(v))) names(v) else rep("", length(v))
      exprs <- c(exprs, unname(v))
      nms <- c(nms, v_nms)
    }
  }
  if (!is.null(hypotheses)) {
    if (!is.character(hypotheses)) .dsem_abort("`hypotheses` must be a character vector.")
    h_nms <- if (!is.null(names(hypotheses))) names(hypotheses) else rep("", length(hypotheses))
    exprs <- c(exprs, unname(hypotheses))
    nms <- c(nms, h_nms)
  }
  if (!length(exprs)) .dsem_abort("Supply at least one hypothesis or derived-parameter expression.")
  for (i in seq_along(nms)) {
    if (!nzchar(nms[i])) nms[i] <- exprs[i]
  }

  draws <- object$draws
  params <- dimnames(draws)[[2L]]
  n_tot <- dim(draws)[1L] * dim(draws)[3L]
  flat <- matrix(aperm(draws, c(1L, 3L, 2L)), nrow = n_tot, ncol = length(params))
  colnames(flat) <- params
  env <- list2env(as.data.frame(flat, check.names = FALSE), parent = baseenv())
  # Expose common math functions in a safe evaluation environment
  for (fn in c("exp", "log", "sqrt", "abs", "pnorm", "qnorm", "plogis", "qlogis")) {
    assign(fn, get(fn, envir = asNamespace(if (fn %in% c("pnorm", "qnorm", "plogis", "qlogis")) "stats" else "base")), envir = env)
  }

  alpha <- (1 - prob) / 2
  k <- length(exprs)
  eval_mat <- matrix(NA_real_, nrow = n_tot, ncol = k, dimnames = list(NULL, nms))
  rows <- vector("list", k)

  for (i in seq_len(k)) {
    raw_e <- trimws(exprs[i])
    eval_str <- raw_e
    if (grepl(">=", raw_e, fixed = TRUE)) {
      sp <- strsplit(raw_e, ">=", fixed = TRUE)[[1L]]
      eval_str <- paste0("(", sp[1L], ") - (", sp[2L], ")")
    } else if (grepl("<=", raw_e, fixed = TRUE)) {
      sp <- strsplit(raw_e, "<=", fixed = TRUE)[[1L]]
      eval_str <- paste0("(", sp[2L], ") - (", sp[1L], ")")
    } else if (grepl(">", raw_e, fixed = TRUE)) {
      sp <- strsplit(raw_e, ">", fixed = TRUE)[[1L]]
      eval_str <- paste0("(", sp[1L], ") - (", sp[2L], ")")
    } else if (grepl("<", raw_e, fixed = TRUE)) {
      sp <- strsplit(raw_e, "<", fixed = TRUE)[[1L]]
      eval_str <- paste0("(", sp[2L], ") - (", sp[1L], ")")
    } else if (grepl("=", raw_e, fixed = TRUE) && !grepl("==", raw_e, fixed = TRUE)) {
      sp <- strsplit(raw_e, "=", fixed = TRUE)[[1L]]
      eval_str <- paste0("(", sp[1L], ") - (", sp[2L], ")")
    }

    parsed <- tryCatch(parse(text = eval_str), error = function(e) {
      .dsem_abort("Could not parse expression `%s`: %s", raw_e, conditionMessage(e))
    })
    vals <- tryCatch(eval(parsed, envir = env), error = function(e) {
      .dsem_abort("Failed evaluating `%s` (available parameters: %s): %s",
                  raw_e, paste(utils::head(params, 12L), collapse = ", "), conditionMessage(e))
    })
    if (!is.numeric(vals) || length(vals) != n_tot) {
      if (is.numeric(vals) && length(vals) == 1L) vals <- rep(vals, n_tot)
      else .dsem_abort("Expression `%s` did not evaluate to a numeric vector of length %d.", raw_e, n_tot)
    }
    eval_mat[, i] <- as.double(vals)
    q <- stats::quantile(vals, probs = c(alpha, 0.5, 1 - alpha), names = FALSE, na.rm = TRUE)
    p_gt <- mean(vals > 0, na.rm = TRUE)
    p_lt <- mean(vals < 0, na.rm = TRUE)
    rows[[i]] <- data.frame(
      name = nms[i],
      expression = raw_e,
      mean = mean(vals, na.rm = TRUE),
      sd = stats::sd(vals, na.rm = TRUE),
      q_lower = q[1L],
      q50 = q[2L],
      q_upper = q[3L],
      prob_gt_0 = p_gt,
      prob_lt_0 = p_lt,
      p_one_tailed = min(mean(vals <= 0, na.rm = TRUE), mean(vals >= 0, na.rm = TRUE)),
      sig = ifelse(q[1L] > 0 || q[3L] < 0, "*", ""),
      stringsAsFactors = FALSE
    )
  }

  wald <- NULL
  if (isTRUE(joint_test) && k >= 1L) {
    mu_d <- colMeans(eval_mat, na.rm = TRUE)
    cov_d <- stats::cov(eval_mat, use = "pairwise.complete.obs")
    if (all(is.finite(mu_d)) && all(is.finite(cov_d))) {
      w_stat <- as.numeric(t(mu_d) %*% .dsem_spd_solve(cov_d, mu_d))
      if (is.finite(w_stat) && w_stat >= 0) {
        wald <- list(
          statistic = w_stat,
          df = k,
          p_value = stats::pchisq(w_stat, df = k, lower.tail = FALSE)
        )
      }
    }
  }

  out <- list(
    summary = do.call(rbind, rows),
    draws = eval_mat,
    wald = wald,
    prob = prob
  )
  class(out) <- c("DSEMhypothesis", "list")
  out
}

#' @exportS3Method base::print
print.DSEMhypothesis <- function(x, digits = 3L, ...) {
  cat("DSEMr Bayesian Hypothesis & Derived Parameter Summary\n")
  print(x$summary, row.names = FALSE, digits = digits)
  if (!is.null(x$wald)) {
    cat(sprintf("\nJoint Bayesian Wald Test (H0: all = 0): chi2(%d) = %.3f, p = %.4g\n",
                x$wald$df, x$wald$statistic, x$wald$p_value))
  }
  invisible(x)
}

#' Posterior comparison and ranking of cluster random effects
#'
#' Summarizes, ranks, and compares cluster-specific posterior random-effect
#' distributions (such as person-specific intercepts, autoregressive slopes, or
#' log innovation variances), including posterior probabilities of exceeding
#' the population average and optional pairwise exceedance probabilities
#' \eqn{P(\theta_i > \theta_j \mid y)}.
#'
#' @param object A fitted multilevel `DSEMfit` object.
#' @param parameter Name of the random effect to compare (defaults to the first
#'   stored random effect).
#' @param level Which random-effect level to inspect: `"between"` (Level-2 in
#'   two-level / Level-3 in three-level / subject in cross-classified),
#'   `"level2"` (Level-2 in three-level DSEM), or `"time"` (occasion effects in
#'   cross-classified DSEM).
#' @param prob Credible interval probability (default `0.95`).
#' @param pairwise Logical; if `TRUE`, also computes the \eqn{G \times G}
#'   pairwise exceedance probability matrix \eqn{P(\theta_i > \theta_j \mid y)}.
#' @return A `DSEMfscomparison` object with elements `ranking` (data frame),
#'   `parameter`, `level`, `prob`, and optional `pairwise` matrix.
#' @examples
#' dat <- simulate_dsem(n = 20, clusters = 8, ar = 0.4, seed = 15)
#' fit <- dsem("level: 1\n y ~ lag(y, 1)\nlevel: 2\n y ~~ y", dat,
#'             id = "id", time = "time", chains = 2, iter = 160, warmup = 80, seed = 15)
#' cmp <- dsem_fscomparison(fit)
#' head(cmp$ranking)
#' @export
dsem_fscomparison <- function(object, parameter = NULL,
                              level = c("between", "level2", "time"),
                              prob = 0.95, pairwise = FALSE) {
  if (!inherits(object, "DSEMfit")) .dsem_abort("`object` must be a `DSEMfit`.")
  level <- match.arg(level)
  if (!is.numeric(prob) || length(prob) != 1L || !is.finite(prob) || prob <= 0 || prob >= 1) {
    .dsem_abort("`prob` must be strictly between 0 and 1.")
  }
  arr <- switch(
    level,
    between = object$random_effect_draws,
    level2 = object$l2_random_effect_draws,
    time = object$time_effect_draws
  )
  if (is.null(arr)) {
    .dsem_abort("Random-effect draws for level `%s` are not available on this fit.", level)
  }
  re_params <- dimnames(arr)[[3L]]
  if (is.null(parameter)) parameter <- re_params[1L]
  if (!parameter %in% re_params) {
    .dsem_abort("Unknown random-effect parameter `%s` (available: %s).",
                parameter, paste(re_params, collapse = ", "))
  }

  p_idx <- match(parameter, re_params)
  clusters <- dimnames(arr)[[2L]]
  g_cnt <- length(clusters)
  n_keep <- dim(arr)[1L]
  n_chains <- dim(arr)[4L]
  n_tot <- n_keep * n_chains

  # Flatten draws to n_tot x g_cnt matrix
  sub <- arr[, , p_idx, , drop = FALSE]
  mat <- matrix(aperm(sub, c(1L, 4L, 2L, 3L)), nrow = n_tot, ncol = g_cnt)
  colnames(mat) <- clusters

  draw_means <- rowMeans(mat)
  ranks_mat <- t(apply(mat, 1L, function(r) rank(-r, ties.method = "average")))
  alpha <- (1 - prob) / 2

  rows <- lapply(seq_len(g_cnt), function(g) {
    vg <- mat[, g]
    q <- stats::quantile(vg, probs = c(alpha, 0.5, 1 - alpha), names = FALSE)
    rq <- stats::quantile(ranks_mat[, g], probs = c(alpha, 1 - alpha), names = FALSE)
    p_above <- mean(vg > draw_means)
    data.frame(
      cluster = clusters[g],
      mean = mean(vg),
      sd = stats::sd(vg),
      q_lower = q[1L],
      q50 = q[2L],
      q_upper = q[3L],
      prob_above_mean = p_above,
      prob_gt_0 = mean(vg > 0),
      mean_rank = mean(ranks_mat[, g]),
      rank_lower = rq[1L],
      rank_upper = rq[2L],
      stringsAsFactors = FALSE
    )
  })
  ranking <- do.call(rbind, rows)
  ord <- order(-ranking$mean)
  ranking <- ranking[ord, , drop = FALSE]
  rownames(ranking) <- NULL

  pw_mat <- NULL
  if (isTRUE(pairwise)) {
    mat_ord <- mat[, ord, drop = FALSE]
    pw_mat <- matrix(0.5, g_cnt, g_cnt, dimnames = list(clusters[ord], clusters[ord]))
    for (i in seq_len(g_cnt)) {
      for (j in seq_len(g_cnt)) {
        if (i != j) pw_mat[i, j] <- mean(mat_ord[, i] > mat_ord[, j])
      }
    }
  }

  out <- list(
    ranking = ranking,
    parameter = parameter,
    level = level,
    prob = prob,
    pairwise = pw_mat
  )
  class(out) <- c("DSEMfscomparison", "list")
  out
}

#' @exportS3Method base::print
print.DSEMfscomparison <- function(x, digits = 3L, n = 15L, ...) {
  cat(sprintf("DSEMr Cluster Random-Effect Comparison (%s | level: %s)\n", x$parameter, x$level))
  print(utils::head(x$ranking, n), row.names = FALSE, digits = digits)
  if (nrow(x$ranking) > n) {
    cat(sprintf("... (%d additional clusters omitted)\n", nrow(x$ranking) - n))
  }
  invisible(x)
}

#' @exportS3Method graphics::plot
plot.DSEMfscomparison <- function(x, ...) {
  df <- x$ranking[rev(seq_len(nrow(x$ranking))), , drop = FALSE]
  k <- nrow(df)
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(old_par), add = TRUE)
  graphics::par(mar = c(4.5, 5.5, 3, 1.5))
  avg <- mean(df$mean)
  cols <- ifelse(df$prob_above_mean > 0.95, "#1d4ed8",
                 ifelse(df$prob_above_mean < 0.05, "#dc2626", "#64748b"))
  graphics::plot(
    df$mean, seq_len(k),
    xlim = range(c(df$q_lower, df$q_upper, avg), na.rm = TRUE),
    yaxt = "n", pch = 19, col = cols,
    xlab = sprintf("%s (%.0f%% CrI)", x$parameter, 100 * x$prob),
    ylab = "Cluster",
    main = sprintf("Caterpillar Ranking: %s", x$parameter),
    ...
  )
  graphics::axis(2, at = seq_len(k), labels = df$cluster, las = 1, cex.axis = 0.75)
  graphics::segments(df$q_lower, seq_len(k), df$q_upper, seq_len(k), col = cols, lwd = 1.6)
  graphics::abline(v = avg, lty = 2, col = "#475569")
  invisible(x)
}

#' Bayesian Impulse Response Functions (IRFs) for DSEM
#'
#' Computes dynamic impulse response functions across horizons \eqn{h = 0, \dots, H}
#' with full posterior credible bands for univariate or multivariate VAR(p),
#' two-level multivariate DSEM, and multi-factor dynamic SEM fits.
#'
#' @param object A fitted `DSEMfit` object.
#' @param horizon Maximum forecast/impulse horizon \eqn{H \ge 1} (default `10L`).
#' @param orthogonal Logical; if `TRUE` (default), applies a 1-SD Cholesky
#'   orthogonalized shock \eqn{L} (\eqn{LL^\top = \Sigma}); if `FALSE`, computes
#'   Pesaran-Shin generalized impulse responses \eqn{\Sigma e_j / \sqrt{\Sigma_{jj}}}.
#' @param prob Credible interval probability (default `0.95`).
#' @return A `DSEMirf` object containing a tidy `data` frame (`impulse`,
#'   `response`, `horizon`, `mean`, `sd`, `q_lower`, `q50`, `q_upper`),
#'   `variables`, `horizon`, `orthogonal`, and `prob`.
#' @examples
#' set.seed(42)
#' n <- 80
#' y1 <- numeric(n); y2 <- numeric(n)
#' for (t in 2:n) {
#'   y1[t] <- 0.5 * y1[t - 1] + rnorm(1, sd = 0.7)
#'   y2[t] <- 0.4 * y2[t - 1] + 0.3 * y1[t - 1] + rnorm(1, sd = 0.7)
#' }
#' fit <- dsem("y1 ~ lag(y1, 1) + lag(y2, 1)\ny2 ~ lag(y1, 1) + lag(y2, 1)",
#'             data.frame(time = seq_len(n), y1 = y1, y2 = y2),
#'             time = "time", chains = 2, iter = 160, warmup = 80, seed = 42)
#' irf <- dsem_irf(fit, horizon = 6)
#' head(irf$data)
#' @export
dsem_irf <- function(object, horizon = 10L, orthogonal = TRUE, prob = 0.95) {
  if (!inherits(object, "DSEMfit")) .dsem_abort("`object` must be a `DSEMfit`.")
  horizon <- .dsem_integer(horizon, "horizon", 1)
  if (!is.numeric(prob) || length(prob) != 1L || !is.finite(prob) || prob <= 0 || prob >= 1) {
    .dsem_abort("`prob` must be strictly between 0 and 1.")
  }

  draws <- object$draws
  params <- dimnames(draws)[[2L]]
  n_tot <- dim(draws)[1L] * dim(draws)[3L]
  flat <- matrix(aperm(draws, c(1L, 3L, 2L)), nrow = n_tot, ncol = length(params))
  colnames(flat) <- params

  terms <- object$model$terms
  w_regs <- .dsem_active_regs(terms, object$model$term_meta, "within")
  has_latent <- any(terms$op == "=~" & terms$level == "within")
  if (has_latent) {
    vars <- unique(terms$lhs[terms$op == "=~" & terms$level == "within"])
    dyn_regs <- w_regs[w_regs$lhs %in% vars & w_regs$rhs %in% vars & w_regs$lag > 0L, , drop = FALSE]
  } else {
    vars <- unique(w_regs$lhs)
    dyn_regs <- w_regs[w_regs$rhs %in% vars & w_regs$lag > 0L, , drop = FALSE]
  }
  d <- length(vars)
  if (d < 1L || !nrow(dyn_regs)) {
    .dsem_abort("Impulse response functions require at least one lagged dynamic equation.")
  }
  max_lag <- max(dyn_regs$lag)
  p_comp <- d * max_lag

  # Map (outcome_i, predictor_j, lag_l) -> parameter column name in `flat`
  find_phi_col <- function(out_nm, in_nm, lg) {
    hit <- which(dyn_regs$lhs == out_nm & dyn_regs$rhs == in_nm & dyn_regs$lag == lg)
    lbl <- if (length(hit) && nzchar(dyn_regs$label[hit[1L]])) dyn_regs$label[hit[1L]] else ""
    cands <- unique(c(
      if (nzchar(lbl)) c(paste0("mean_", lbl), paste0("fixed_", lbl), lbl) else character(),
      paste0("mean_", out_nm, "_on_lag", lg, "_", in_nm),
      paste0("fixed_", out_nm, "_on_lag", lg, "_", in_nm),
      paste0(out_nm, "_on_lag", lg, "_", in_nm),
      if (out_nm == in_nm && lg == 1L) c(paste0("ar1_", out_nm), paste0("mean_ar1_", out_nm)) else character(),
      paste0("mean_lag", lg, "_", in_nm),
      paste0("fixed_lag", lg, "_", in_nm),
      paste0("lag", lg, "_", in_nm)
    ))
    found <- intersect(cands, params)
    if (length(found)) found[1L] else NA_character_
  }

  phi_map <- array(NA_character_, dim = c(d, d, max_lag))
  for (i in seq_len(d)) {
    for (j in seq_len(d)) {
      for (lg in seq_len(max_lag)) {
        phi_map[i, j, lg] <- find_phi_col(vars[i], vars[j], lg)
      }
    }
  }

  # Map innovation covariance matrix Sigma (d x d)
  sig_map <- matrix(NA_character_, d, d)
  sig_is_log <- rep(FALSE, d)
  for (i in seq_len(d)) {
    v_i <- vars[i]
    lv_lbl <- terms$label[terms$op == "~~" & terms$lhs == v_i & terms$level == "within" & terms$random]
    cands_v <- unique(c(
      paste0("sigma2_", v_i), if (d == 1L) "sigma2" else character(),
      paste0("factor_variance_", v_i), if (d == 1L) "factor_variance" else character()
    ))
    hit_v <- intersect(cands_v, params)
    if (length(hit_v)) {
      sig_map[i, i] <- hit_v[1L]
    } else {
      cands_lv <- unique(c(
        if (length(lv_lbl) && nzchar(lv_lbl[1L])) c(paste0("mean_", lv_lbl[1L]), paste0("fixed_", lv_lbl[1L])) else character(),
        paste0("mean_logv_", v_i), paste0("fixed_logv_", v_i), "mean_logv", "fixed_logv"
      ))
      hit_lv <- intersect(cands_lv, params)
      if (length(hit_lv)) {
        sig_map[i, i] <- hit_lv[1L]
        sig_is_log[i] <- TRUE
      }
    }
    if (i < d) {
      for (j in (i + 1L):d) {
        v_j <- vars[j]
        cands_c <- c(
          paste0("within_cov_", v_j, "_", v_i), paste0("within_cov_", v_i, "_", v_j),
          if (identical(object$model$level_type, "single")) c(paste0("cov_", v_j, "_", v_i), paste0("cov_", v_i, "_", v_j)) else character(),
          paste0("factor_cov_", v_j, "_", v_i), paste0("factor_cov_", v_i, "_", v_j)
        )
        hit_c <- intersect(cands_c, params)
        if (length(hit_c)) {
          sig_map[i, j] <- hit_c[1L]
          sig_map[j, i] <- hit_c[1L]
        }
      }
    }
  }

  # Thin draw set cleanly if very large to keep computation snappy
  use_idx <- if (n_tot > 1000L) unique(round(seq(1, n_tot, length.out = 1000L))) else seq_len(n_tot)
  n_use <- length(use_idx)
  irf_arr <- array(0, dim = c(n_use, d, d, horizon + 1L)) # draw x response x impulse x (h+1)

  for (si in seq_len(n_use)) {
    s <- use_idx[si]
    # Build companion matrix F (p_comp x p_comp)
    f_mat <- matrix(0, p_comp, p_comp)
    if (max_lag > 1L) {
      f_mat[(d + 1L):p_comp, seq_len(p_comp - d)] <- diag(1, p_comp - d)
    }
    for (lg in seq_len(max_lag)) {
      col_off <- (lg - 1L) * d
      for (i in seq_len(d)) {
        for (j in seq_len(d)) {
          nm <- phi_map[i, j, lg]
          if (!is.na(nm)) f_mat[i, col_off + j] <- flat[s, nm]
        }
      }
    }
    # Build innovation covariance Sigma
    sig <- diag(1, d)
    for (i in seq_len(d)) {
      if (!is.na(sig_map[i, i])) {
        v <- flat[s, sig_map[i, i]]
        sig[i, i] <- if (sig_is_log[i]) exp(min(max(v, -10), 10)) else max(v, 1e-8)
      }
      if (i < d) {
        for (j in (i + 1L):d) {
          if (!is.na(sig_map[i, j])) {
            sig[i, j] <- flat[s, sig_map[i, j]]
            sig[j, i] <- sig[i, j]
          }
        }
      }
    }
    # Shock matrix B0 (d x d): column j is initial shock to variable j
    b0 <- if (isTRUE(orthogonal)) {
      ch <- try(chol(sig), silent = TRUE)
      if (inherits(ch, "try-error")) ch <- chol(sig + diag(1e-6, d))
      t(ch)
    } else {
      sweep(sig, 2L, sqrt(pmax(diag(sig), 1e-8)), "/")
    }

    # Propagate companion powers F^h
    pow_f <- diag(1, p_comp)
    for (h in 0:horizon) {
      psi_h <- pow_f[seq_len(d), seq_len(d), drop = FALSE] %*% b0
      irf_arr[si, , , h + 1L] <- psi_h
      if (h < horizon) pow_f <- pow_f %*% f_mat
    }
  }

  alpha <- (1 - prob) / 2
  rows <- list()
  for (imp in seq_len(d)) {
    for (resp in seq_len(d)) {
      for (h in 0:horizon) {
        vals <- irf_arr[, resp, imp, h + 1L]
        q <- stats::quantile(vals, probs = c(alpha, 0.5, 1 - alpha), names = FALSE)
        rows[[length(rows) + 1L]] <- data.frame(
          impulse = vars[imp],
          response = vars[resp],
          horizon = h,
          mean = mean(vals),
          sd = stats::sd(vals),
          q_lower = q[1L],
          q50 = q[2L],
          q_upper = q[3L],
          stringsAsFactors = FALSE
        )
      }
    }
  }

  out <- list(
    data = do.call(rbind, rows),
    variables = vars,
    horizon = horizon,
    orthogonal = isTRUE(orthogonal),
    prob = prob
  )
  class(out) <- c("DSEMirf", "list")
  out
}

#' @exportS3Method base::print
print.DSEMirf <- function(x, digits = 3L, ...) {
  cat(sprintf("DSEMr Bayesian %s Impulse Response Function (horizon = 0..%d)\n",
              if (x$orthogonal) "Orthogonalized" else "Generalized", x$horizon))
  print(utils::head(x$data, 20L), row.names = FALSE, digits = digits)
  if (nrow(x$data) > 20L) {
    cat(sprintf("... (%d additional rows; inspect `$data` or call `plot()`)\n", nrow(x$data) - 20L))
  }
  invisible(x)
}

#' @exportS3Method graphics::plot
plot.DSEMirf <- function(x, ...) {
  vars <- x$variables
  d <- length(vars)
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(old_par), add = TRUE)
  graphics::par(mfrow = c(d, d), mar = c(3.8, 3.8, 2.6, 1.0))
  for (resp in vars) {
    for (imp in vars) {
      sub <- x$data[x$data$impulse == imp & x$data$response == resp, , drop = FALSE]
      yr <- range(c(0, sub$q_lower, sub$q_upper), na.rm = TRUE)
      graphics::plot(
        sub$horizon, sub$mean, type = "n", ylim = yr,
        xlab = "Horizon", ylab = "Response",
        main = sprintf("%s -> %s", imp, resp), ...
      )
      graphics::polygon(
        c(sub$horizon, rev(sub$horizon)),
        c(sub$q_lower, rev(sub$q_upper)),
        col = grDevices::adjustcolor("#2563eb", alpha.f = 0.18),
        border = NA
      )
      graphics::abline(h = 0, lty = 2, col = "#94a3b8")
      graphics::lines(sub$horizon, sub$mean, lwd = 2, col = "#1d4ed8")
      graphics::points(sub$horizon, sub$mean, pch = 19, cex = 0.7, col = "#1d4ed8")
    }
  }
  invisible(x)
}
