#' Compute standardized DSEM estimates (STDYX, STDY, STD)
#'
#' Computes posterior summaries of standardized model parameters (`STDYX`,
#' `STDY`, and `STD`). For multilevel (`twolevel`, `threelevel`, and
#' `crossclassified`) models:
#' - Within-level slopes (including labeled random slopes such as `s`, `sy`,
#'   `sx`, and cross-lagged slopes `s12`, `s21`) are standardized within each
#'   cluster using cluster-specific model-implied standard deviations and then
#'   averaged across clusters at each posterior draw following Schuurman,
#'   Ferrer, de Boer-Sonnenschein, and Hamaker (2016).
#' - Setting `cluster = TRUE` returns individual cluster-specific standardized
#'   within-level estimates for idiographic interpretation.
#' - Between-level regressions, covariances (standardized to residual
#'   correlations), residual variances (`1 - R^2`), and intercepts are
#'   standardized using draw-specific model-implied between-level standard
#'   deviations.
#'
#' @param object A fitted `DSEMfit` object.
#' @param type Standardization convention: `"stdyx"` (standardizes by both
#'   outcome and predictor SDs), `"stdy"` (standardizes by outcome SD only), or
#'   `"std"` (standardizes latent variables only).
#' @param prob Credible interval probability (default `0.95`).
#' @param cluster Logical; if `TRUE` on a multilevel model, returns
#'   cluster-specific standardized within-person slopes (`cluster`, `parameter`,
#'   `type`, `mean`, `sd`, `q_lower`, `q50`, `q_upper`, `p_one_tailed`) instead
#'   of population-level summaries.
#' @return A data frame of standardized parameter summaries (`level`,
#'   `parameter`, `type`, `mean`, `sd`, `q_lower`, `q50`, `q_upper`,
#'   `p_one_tailed`), or per-cluster summaries when `cluster = TRUE`.
#' @references
#' Schuurman, N. K., Ferrer, E., de Boer-Sonnenschein, M., & Hamaker, E. L.
#' (2016). How to compare cross-lagged associations in a multilevel
#' autoregressive model. *Psychological Methods*, 21(2), 206-221.
#' \doi{10.1037/met0000062}
#' @examples
#' dat <- simulate_dsem(n = 80, ar = 0.5, sigma = 0.8, seed = 10)
#' fit <- dsem("y ~ lag(y, 1)", dat, chains = 2, iter = 200, warmup = 100, seed = 10)
#' dsem_standardize(fit, type = "stdyx")
#' @export
dsem_standardize <- function(object, type = c("stdyx", "stdy", "std"), prob = 0.95,
                             cluster = FALSE) {
  if (!inherits(object, "DSEMfit")) .dsem_abort("`object` must be a `DSEMfit`.")
  type <- match.arg(type)
  if (!is.numeric(prob) || length(prob) != 1L || !is.finite(prob) || prob <= 0 || prob >= 1) {
    .dsem_abort("`prob` must be strictly between 0 and 1.")
  }
  if (!is.logical(cluster) || length(cluster) != 1L || is.na(cluster)) {
    .dsem_abort("`cluster` must be TRUE or FALSE.")
  }
  alpha <- (1 - prob) / 2
  draws <- object$draws
  params <- dimnames(draws)[[2L]]
  n_keep <- dim(draws)[1L]
  n_chains <- dim(draws)[3L]
  n_tot <- n_keep * n_chains
  flat <- matrix(aperm(draws, c(1L, 3L, 2L)), nrow = n_tot, ncol = length(params))
  colnames(flat) <- params

  lvl_type <- object$model$level_type
  is_multilevel <- lvl_type %in% c("twolevel", "threelevel", "crossclassified")

  if (isTRUE(cluster) && (!is_multilevel || is.null(object$random_effect_draws))) {
    .dsem_abort("`cluster = TRUE` requires a multilevel `DSEMfit` with stored cluster random effects.")
  }

  # Collect observed data statistics (overall & cluster-level)
  obs_y <- object$observed
  obs_sd_map <- .dsem_obs_sd_map(object)
  between_cov_w <- .dsem_between_predictor_cov(object)

  std_entries <- list()
  cluster_entries <- list()
  add_entry <- function(level_lbl, nm, vals) {
    std_entries[[length(std_entries) + 1L]] <<- list(level = level_lbl, parameter = nm, vals = vals)
  }

  # 1. Within-level cluster-averaged (and optional per-cluster) standardization (Schuurman et al., 2016)
  if (is_multilevel && !is.null(object$random_effect_draws)) {
    re <- object$random_effect_draws # iter x cluster x param x chain
    re_params <- dimnames(re)[[3L]]
    terms <- object$model$terms
    w_regs <- terms[terms$op == "~" & terms$rhs != "1" & terms$level == "within", , drop = FALSE]
    w_outcomes <- unique(w_regs$lhs)
    id_col <- if (length(object$model$id)) object$model$id[length(object$model$id)] else NULL
    cluster_ids <- dimnames(re)[[2L]]
    g_cnt <- dim(re)[2L]

    # Helper to compute cluster-specific model-implied within SD for any outcome `var_nm` at draw (it, ch, pos)
    outcome_sd_for_var <- function(var_nm, it, ch, pos) {
      lv_c <- unique(c(
        terms$label[terms$op == "~~" & terms$lhs == var_nm & terms$level == "within" & terms$random],
        paste0("logv_", var_nm), if (length(w_outcomes) == 1L) c("logv", "logvf") else character()
      ))
      lv_c <- lv_c[nzchar(lv_c)]
      lv_h <- intersect(lv_c, re_params)
      sig_c <- intersect(c(paste0("sigma2_", var_nm), if (length(w_outcomes) == 1L) "sigma2" else character(),
                           paste0("residual_variance_", var_nm), "factor_variance"), params)
      sd_e <- if (length(lv_h)) {
        exp(0.5 * pmin(pmax(re[it, , match(lv_h[1L], re_params), ch], -10), 10))
      } else if (length(sig_c)) {
        rep(sqrt(pmax(flat[pos, sig_c[1L]], 1e-8)), g_cnt)
      } else {
        rep(1, g_cnt)
      }
      ar_c <- unique(c(
        w_regs$label[w_regs$lhs == var_nm & w_regs$rhs == var_nm & w_regs$lag == 1L],
        paste0("lag1_", var_nm), paste0(var_nm, "_on_lag1_", var_nm)
      ))
      ar_c <- ar_c[nzchar(ar_c)]
      ar_h <- intersect(ar_c, re_params)
      phi_c <- if (length(ar_h)) {
        pmin(abs(re[it, , match(ar_h[1L], re_params), ch]), 0.96)
      } else {
        rep(0, g_cnt)
      }
      sd_e / sqrt(pmax(1 - phi_c^2, 0.04))
    }

    # Precompute cluster-specific within SDs for any observed within predictors
    within_x_sd_by_cluster <- list()
    if (!is.null(id_col) && is.data.frame(object$data) && id_col %in% names(object$data)) {
      w_rhs_vars <- unique(w_regs$rhs)
      for (xv in w_rhs_vars) {
        if (xv %in% names(object$data)) {
          sds <- vapply(cluster_ids, function(cid) {
            vals_c <- object$data[[xv]][as.character(object$data[[id_col]]) == cid]
            s <- stats::sd(vals_c, na.rm = TRUE)
            if (is.finite(s) && s > 1e-8) s else 1
          }, numeric(1))
          within_x_sd_by_cluster[[xv]] <- sds
        }
      }
    }

    matched_re_slopes <- character()
    for (r_i in seq_len(nrow(w_regs))) {
      lhs <- w_regs$lhs[r_i]
      rhs <- w_regs$rhs[r_i]
      lg <- w_regs$lag[r_i]
      lbl <- w_regs$label[r_i]
      cands <- unique(c(
        if (nzchar(lbl)) lbl else character(),
        if (lg > 0L) c(paste0("lag", lg, "_", rhs), paste0(lhs, "_on_lag", lg, "_", rhs)) else c(rhs, paste0(lhs, "_on_", rhs))
      ))
      hit <- intersect(cands, re_params)
      if (!length(hit)) next
      p_re <- hit[1L]
      if (p_re %in% matched_re_slopes) next
      matched_re_slopes <- c(matched_re_slopes, p_re)
      idx_p <- match(p_re, re_params)

      vals <- numeric(n_tot)
      cl_mat <- if (isTRUE(cluster)) matrix(NA_real_, nrow = n_tot, ncol = g_cnt) else NULL
      pos <- 0L
      is_self_lag <- (lg >= 1L && identical(lhs, rhs))
      is_cross_lag <- (lg >= 1L && !identical(lhs, rhs) && rhs %in% w_outcomes)
      x_sd_vec <- if (!is_self_lag && !is_cross_lag && rhs %in% names(within_x_sd_by_cluster)) {
        within_x_sd_by_cluster[[rhs]]
      } else {
        rep(1, g_cnt)
      }

      for (ch in seq_len(n_chains)) {
        for (it in seq_len(n_keep)) {
          pos <- pos + 1L
          slopes_c <- re[it, , idx_p, ch]
          if (type == "stdyx" && is_self_lag) {
            std_c <- slopes_c
          } else if (type == "std") {
            std_c <- slopes_c
          } else {
            sd_y <- outcome_sd_for_var(lhs, it, ch, pos)
            mult_x <- if (type == "stdyx") {
              if (is_self_lag) {
                sd_y
              } else if (is_cross_lag) {
                outcome_sd_for_var(rhs, it, ch, pos)
              } else {
                x_sd_vec
              }
            } else {
              rep(1, g_cnt)
            }
            std_c <- slopes_c * mult_x / pmax(sd_y, 1e-6)
          }
          vals[pos] <- mean(std_c)
          if (isTRUE(cluster)) cl_mat[pos, ] <- std_c
        }
      }
      add_entry("Within", paste0("within_std_", p_re), vals)
      if (isTRUE(cluster)) {
        for (g in seq_len(g_cnt)) {
          vg <- cl_mat[, g]
          q <- stats::quantile(vg, probs = c(alpha, 0.5, 1 - alpha), names = FALSE)
          cluster_entries[[length(cluster_entries) + 1L]] <- data.frame(
            cluster = cluster_ids[g],
            parameter = p_re,
            type = type,
            mean = mean(vg),
            sd = stats::sd(vg),
            q_lower = q[1L],
            q50 = q[2L],
            q_upper = q[3L],
            p_one_tailed = min(mean(vg <= 0), mean(vg >= 0)),
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }

  if (isTRUE(cluster)) {
    if (!length(cluster_entries)) {
      return(data.frame(
        cluster = character(), parameter = character(), type = character(),
        mean = numeric(), sd = numeric(), q_lower = numeric(), q50 = numeric(),
        q_upper = numeric(), p_one_tailed = numeric(), stringsAsFactors = FALSE
      ))
    }
    res_cl <- do.call(rbind, cluster_entries)
    rownames(res_cl) <- NULL
    return(res_cl)
  }

  # 2. Between-level model-implied variance computation for multilevel models
  between_var_draws <- list()
  if (is_multilevel) {
    between_var_draws <- .dsem_between_total_vars(object, flat, between_cov_w)
  }

  # 3. Standardize population / structural / measurement parameters in `flat`
  for (p_name in params) {
    v <- flat[, p_name]

    # 3a. Factor loadings (Single-level or Multilevel Dynamic CFA/SEM)
    if (grepl("^(loading_|between_loading_)", p_name)) {
      is_b_load <- grepl("^between_loading_", p_name)
      lvl_lbl <- if (is_b_load) "Between" else "Within"
      item <- sub("^(between_loading_|loading_)", "", p_name)
      # Handle multi-factor names like loading_f1_y12 -> item = y12
      if (!paste0("residual_variance_", item) %in% params && grepl("_", item)) {
        parts <- strsplit(item, "_", fixed = TRUE)[[1L]]
        cand_item <- parts[length(parts)]
        if (paste0("residual_variance_", cand_item) %in% params ||
            paste0("between_residual_variance_", cand_item) %in% params) {
          item <- cand_item
        }
      }
      if (is_b_load) {
        rvar_col <- paste0("between_residual_variance_", item)
        fb_name <- sub(paste0("_", item, "$"), "", sub("^between_loading_", "", p_name))
        fvar_col <- intersect(c(paste0("between_var_", fb_name), "between_var_fb"), params)
        sd_eta <- if (length(fvar_col)) sqrt(pmax(flat[, fvar_col[1L]], 1e-8)) else rep(1, n_tot)
        rvar <- if (rvar_col %in% params) pmax(flat[, rvar_col], 1e-8) else rep(0.5, n_tot)
        sd_item <- sqrt(v^2 * sd_eta^2 + rvar)
      } else {
        rvar_col <- paste0("residual_variance_", item)
        fvar_col <- intersect(c("factor_variance", grep("^factor_variance_", params, value = TRUE)), params)
        ar_col <- grep("^(ar1_|mean_s)", params, value = TRUE)
        sd_eta <- if (length(fvar_col)) {
          phi_sq <- if (length(ar_col)) pmin(flat[, ar_col[1L]]^2, 0.96) else 0
          sqrt(pmax(flat[, fvar_col[1L]], 1e-8) / pmax(1 - phi_sq, 0.04))
        } else {
          rep(1, n_tot)
        }
        rvar <- if (rvar_col %in% params) pmax(flat[, rvar_col], 1e-8) else rep(1, n_tot)
        sd_item <- sqrt(v^2 * sd_eta^2 + rvar)
      }
      std_v <- switch(
        type,
        stdyx = v * sd_eta / pmax(sd_item, 1e-8),
        stdy = v / pmax(sd_item, 1e-8),
        std = v * sd_eta
      )
      add_entry(lvl_lbl, p_name, std_v)
      next
    }

    # 3b. Multilevel Between-level regressions (e.g., y_on_w, s_on_w, z_on_y, l3_y_on_w3)
    if (is_multilevel && grepl("_on_", p_name) && !grepl("_on_lag[0-9]+_", p_name)) {
      clean_nm <- sub("^(l3_|l2_)", "", p_name)
      lvl_lbl <- if (grepl("^l3_", p_name)) "Between_L3" else if (grepl("^l2_", p_name)) "Between_L2" else "Between"
      pieces <- strsplit(clean_nm, "_on_", fixed = TRUE)[[1L]]
      lhs_nm <- pieces[1L]
      rhs_nm <- paste(pieces[-1L], collapse = "_on_")
      sd_y_b <- if (lhs_nm %in% names(between_var_draws)) {
        sqrt(pmax(between_var_draws[[lhs_nm]], 1e-8))
      } else {
        rep(.dsem_lookup_between_sd(object, lhs_nm, between_cov_w), n_tot)
      }
      sd_x_b <- if (rhs_nm %in% names(between_var_draws)) {
        sqrt(pmax(between_var_draws[[rhs_nm]], 1e-8))
      } else {
        rep(.dsem_lookup_between_sd(object, rhs_nm, between_cov_w), n_tot)
      }
      std_v <- switch(
        type,
        stdyx = v * sd_x_b / pmax(sd_y_b, 1e-8),
        stdy = v / pmax(sd_y_b, 1e-8),
        std = if (rhs_nm %in% names(between_var_draws)) v * sd_x_b / pmax(sd_y_b, 1e-8) else v / pmax(sd_y_b, 1e-8)
      )
      add_entry(lvl_lbl, p_name, std_v)
      next
    }

    # 3c. Multilevel Between-level covariances -> residual correlations (STDYX)
    if (is_multilevel && grepl("^(cov_|l3_cov_|l2_cov_|id_cov_|between_cov_)", p_name)) {
      prefix <- sub("^((?:l3_|l2_|id_|between_)?cov_).*$", "\\1", p_name)
      var_prefix <- sub("cov_$", "var_", prefix)
      lvl_lbl <- if (grepl("^l3_", prefix)) "Between_L3" else if (grepl("^l2_", prefix)) "Between_L2" else "Between"
      rest <- sub(paste0("^", prefix), "", p_name)
      # Find matching pair of var_a and var_b in params
      pair <- .dsem_split_cov_pair(rest, var_prefix, params)
      if (!is.null(pair)) {
        va <- pmax(flat[, pair[1L]], 1e-10)
        vb <- pmax(flat[, pair[2L]], 1e-10)
        add_entry(lvl_lbl, p_name, pmin(pmax(v / sqrt(va * vb), -1), 1))
      }
      next
    }

    # 3d. Multilevel Between-level residual variances -> 1 - R^2 (STDYX)
    if (is_multilevel && grepl("^(var_|l3_var_|l2_var_|id_var_|between_var_)", p_name)) {
      prefix <- sub("^((?:l3_|l2_|id_|between_)?var_).*$", "\\1", p_name)
      lvl_lbl <- if (grepl("^l3_", prefix)) "Between_L3" else if (grepl("^l2_", prefix)) "Between_L2" else "Between"
      nm_var <- sub(paste0("^", prefix), "", p_name)
      if (nm_var %in% names(between_var_draws)) {
        tot_v <- pmax(between_var_draws[[nm_var]], 1e-10)
        add_entry(lvl_lbl, p_name, pmin(pmax(v / tot_v, 0), 1))
      }
      next
    }

    # 3e. Multilevel Between-level means / intercepts -> mean / SD_B (STDYX)
    if (is_multilevel && grepl("^(mean_|fixed_|intercept_)", p_name)) {
      nm_var <- sub("^(mean_|fixed_|intercept_)", "", p_name)
      lvl_lbl <- if (grepl("^fixed_", p_name)) "Between_L3" else "Between"
      if (nm_var %in% names(between_var_draws)) {
        sd_b <- sqrt(pmax(between_var_draws[[nm_var]], 1e-8))
        add_entry(lvl_lbl, p_name, if (type == "std") v else v / sd_b)
      }
      next
    }

    # 3f. Single-level AR(1) / VAR(p) / cross-lagged & exogenous slopes
    if (!is_multilevel) {
      if (grepl("^lag[0-9]+_", p_name) || grepl("^ar1_", p_name)) {
        std_v <- if (type == "stdy") {
          sig_col <- intersect(c("sigma2", "factor_variance"), params)
          if (length(sig_col)) {
            sd_y <- sqrt(pmax(flat[, sig_col[1L]], 1e-8) / pmax(1 - pmin(v^2, 0.98), 0.02))
            v / sd_y
          } else v
        } else {
          v
        }
        add_entry("Within", p_name, std_v)
      } else if (grepl("_on_lag[0-9]+_", p_name)) {
        parts <- strsplit(p_name, "_on_lag[0-9]+_")[[1L]]
        sd_y <- obs_sd_map[[parts[1L]]]
        sd_x <- obs_sd_map[[parts[2L]]]
        if (is.null(sd_y) || !is.finite(sd_y) || sd_y <= 0) sd_y <- 1
        if (is.null(sd_x) || !is.finite(sd_x) || sd_x <= 0) sd_x <- 1
        mult <- switch(type, stdyx = sd_x / sd_y, stdy = 1 / sd_y, std = 1)
        add_entry("Within", p_name, v * mult)
      } else if (grepl("^lagged_factor_effect_", p_name)) {
        item <- sub("^lagged_factor_effect_", "", p_name)
        fvar_col <- intersect(c("factor_variance"), params)
        sd_eta <- if (length(fvar_col)) sqrt(pmax(flat[, fvar_col[1L]], 1e-8)) else rep(1, n_tot)
        sd_y <- if (!is.null(obs_sd_map[[item]])) obs_sd_map[[item]] else 1
        mult <- switch(type, stdyx = sd_eta / sd_y, stdy = 1 / sd_y, std = sd_eta)
        add_entry("Within", p_name, v * mult)
      } else if (!grepl("^(sigma2|tau_|cov_|omega_|residual_variance_|factor_variance|\\(Intercept\\)|intercept_|threshold_)", p_name)) {
        # Exogenous covariate slope
        if (grepl("_on_", p_name)) {
          parts <- strsplit(p_name, "_on_", fixed = TRUE)[[1L]]
          lhs_nm <- parts[1L]; rhs_nm <- parts[2L]
        } else {
          lhs_nm <- names(obs_sd_map)[1L]; rhs_nm <- p_name
        }
        sd_y <- if (!is.null(lhs_nm) && lhs_nm %in% names(obs_sd_map)) obs_sd_map[[lhs_nm]] else 1
        sd_x <- if (!is.null(object$data) && rhs_nm %in% names(object$data)) {
          s <- stats::sd(object$data[[rhs_nm]], na.rm = TRUE)
          if (is.finite(s) && s > 0) s else 1
        } else 1
        mult <- switch(type, stdyx = sd_x / sd_y, stdy = 1 / sd_y, std = 1)
        add_entry("Within", p_name, v * mult)
      }
    }
  }

  if (!length(std_entries)) {
    for (p_name in params) add_entry("Within", p_name, flat[, p_name])
  }

  rows <- lapply(std_entries, function(entry) {
    vals <- entry$vals
    p_one <- min(mean(vals > 0, na.rm = TRUE), mean(vals < 0, na.rm = TRUE))
    data.frame(
      level = entry$level,
      parameter = entry$parameter,
      type = toupper(type),
      mean = mean(vals, na.rm = TRUE),
      sd = stats::sd(vals, na.rm = TRUE),
      q_lower = stats::quantile(vals, alpha, names = FALSE, na.rm = TRUE),
      q50 = stats::quantile(vals, 0.5, names = FALSE, na.rm = TRUE),
      q_upper = stats::quantile(vals, 1 - alpha, names = FALSE, na.rm = TRUE),
      p_one_tailed = p_one,
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

#' Compute Within- and Between-Level Bayesian R-squared
#'
#' Computes posterior summaries of explained variance (\eqn{R^2}) across
#' within- and between-level equations. For multilevel models, within-level
#' \eqn{R^2} is averaged across clusters at each posterior draw (Schuurman,
#' Ferrer, de Boer-Sonnenschein, & Hamaker, 2016) and between-level \eqn{R^2}
#' is computed from model-implied between structural variance for every
#' predicted random effect and distal outcome.
#'
#' @param object A fitted `DSEMfit` object.
#' @param prob Credible interval probability (default `0.95`).
#' @return A data frame with columns `level`, `variable`, `mean`, `sd`,
#'   `q_lower`, `q50`, `q_upper`.
#' @examples
#' dat <- simulate_dsem(n = 80, ar = 0.5, sigma = 0.7, seed = 11)
#' fit <- dsem("y ~ lag(y, 1)", dat, chains = 2, iter = 200, warmup = 100, seed = 11)
#' dsem_r2(fit)
#' @export
dsem_r2 <- function(object, prob = 0.95) {
  if (!inherits(object, "DSEMfit")) .dsem_abort("`object` must be a `DSEMfit`.")
  if (!is.numeric(prob) || length(prob) != 1L || !is.finite(prob) || prob <= 0 || prob >= 1) {
    .dsem_abort("`prob` must be strictly between 0 and 1.")
  }
  alpha <- (1 - prob) / 2
  draws <- object$draws
  params <- dimnames(draws)[[2L]]
  n_keep <- dim(draws)[1L]
  n_chains <- dim(draws)[3L]
  n_tot <- n_keep * n_chains
  flat <- matrix(aperm(draws, c(1L, 3L, 2L)), nrow = n_tot, ncol = length(params))
  colnames(flat) <- params
  lvl_type <- object$model$level_type
  is_multilevel <- lvl_type %in% c("twolevel", "threelevel", "crossclassified")

  r2_entries <- list()
  add_r2 <- function(lvl, var_nm, vals) {
    vals <- pmin(pmax(vals, 0), 1)
    r2_entries[[length(r2_entries) + 1L]] <<- data.frame(
      level = lvl,
      variable = var_nm,
      mean = mean(vals, na.rm = TRUE),
      sd = stats::sd(vals, na.rm = TRUE),
      q_lower = stats::quantile(vals, alpha, names = FALSE, na.rm = TRUE),
      q50 = stats::quantile(vals, 0.5, names = FALSE, na.rm = TRUE),
      q_upper = stats::quantile(vals, 1 - alpha, names = FALSE, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }

  # 1. Within-level R^2
  if (is_multilevel && !is.null(object$random_effect_draws)) {
    re <- object$random_effect_draws
    re_params <- dimnames(re)[[3L]]
    terms <- object$model$terms
    w_regs <- terms[terms$op == "~" & terms$rhs != "1" & terms$level == "within", , drop = FALSE]
    outcomes <- unique(w_regs$lhs)
    for (out_nm in outcomes) {
      # Check self-AR(1) random slope
      ar_cands <- unique(c(
        w_regs$label[w_regs$lhs == out_nm & w_regs$rhs == out_nm & w_regs$lag == 1L],
        paste0("lag1_", out_nm), paste0(out_nm, "_on_lag1_", out_nm)
      ))
      ar_cands <- ar_cands[nzchar(ar_cands)]
      ar_hit <- intersect(ar_cands, re_params)
      if (length(ar_hit)) {
        idx_ar <- match(ar_hit[1L], re_params)
        vals <- numeric(n_tot)
        pos <- 0L
        for (ch in seq_len(n_chains)) {
          for (it in seq_len(n_keep)) {
            pos <- pos + 1L
            phi_c <- pmin(abs(re[it, , idx_ar, ch]), 0.99)
            vals[pos] <- mean(phi_c^2)
          }
        }
        add_r2("Within (Cluster Avg)", out_nm, vals)
      }
    }
  } else if (any(grepl("^loading_", params))) {
    # Dynamic CFA: latent AR(1) R^2 and item R^2
    ar_cols <- grep("^ar1_", params, value = TRUE)
    for (ac in ar_cols) {
      add_r2("Within (Latent)", sub("^ar1_", "", ac), pmin(flat[, ac]^2, 0.999))
    }
    load_cols <- grep("^loading_", params, value = TRUE)
    fvar_col <- intersect(c("factor_variance", grep("^factor_variance_", params, value = TRUE)), params)
    for (lc in load_cols) {
      item <- sub("^loading_", "", lc)
      if (!paste0("residual_variance_", item) %in% params && grepl("_", item)) {
        parts <- strsplit(item, "_", fixed = TRUE)[[1L]]
        item <- parts[length(parts)]
      }
      rvar_col <- paste0("residual_variance_", item)
      if (rvar_col %in% params && length(fvar_col)) {
        phi_sq <- if (length(ar_cols)) pmin(flat[, ar_cols[1L]]^2, 0.96) else 0
        var_eta <- pmax(flat[, fvar_col[1L]], 1e-8) / pmax(1 - phi_sq, 0.04)
        exp_v <- flat[, lc]^2 * var_eta
        add_r2("Within (Item)", item, exp_v / (exp_v + pmax(flat[, rvar_col], 1e-8)))
      }
    }
  } else {
    # Single-level N=1 observed regression / AR(p) / VAR(p)
    obs_y <- object$observed
    if (is.matrix(obs_y)) {
      for (nm in colnames(obs_y)) {
        sc <- intersect(c(paste0("sigma2_", nm), "sigma2"), params)
        vy <- stats::var(obs_y[, nm], na.rm = TRUE)
        if (length(sc) && is.finite(vy) && vy > 0) {
          add_r2("Within", nm, 1 - flat[, sc[1L]] / vy)
        }
      }
    } else {
      vy <- stats::var(obs_y, na.rm = TRUE)
      if ("sigma2" %in% params && is.finite(vy) && vy > 0) {
        add_r2("Within", "y", 1 - flat[, "sigma2"] / vy)
      }
    }
  }

  # 2. Between-level R^2 for every predicted random effect & distal outcome
  if (is_multilevel) {
    between_cov_w <- .dsem_between_predictor_cov(object)
    tot_vars <- .dsem_between_total_vars(object, flat, between_cov_w)
    for (v_nm in names(tot_vars)) {
      # Only report Between R^2 when variable actually has at least one between predictor
      has_pred <- any(grepl(paste0("^(?:l3_|l2_)?", v_nm, "_on_"), params))
      if (!has_pred) next
      res_col <- intersect(c(paste0("var_", v_nm), paste0("l3_var_", v_nm), paste0("l2_var_", v_nm)), params)
      if (length(res_col)) {
        r2_b <- 1 - pmax(flat[, res_col[1L]], 0) / pmax(tot_vars[[v_nm]], 1e-10)
        add_r2("Between", v_nm, r2_b)
      }
    }
  }

  if (!length(r2_entries)) {
    return(data.frame(
      level = character(), variable = character(), mean = numeric(), sd = numeric(),
      q_lower = numeric(), q50 = numeric(), q_upper = numeric(),
      stringsAsFactors = FALSE
    ))
  }
  out <- do.call(rbind, r2_entries)
  rownames(out) <- NULL
  out
}

# ---- Internal helpers for within/between standardization & Bayesian R^2 ----

.dsem_obs_sd_map <- function(object) {
  obs_y <- object$observed
  if (is.matrix(obs_y)) {
    sds <- apply(obs_y, 2L, stats::sd, na.rm = TRUE)
    if (is.null(names(sds))) names(sds) <- paste0("y", seq_along(sds))
    return(as.list(sds))
  }
  regs <- object$model$terms[object$model$terms$op == "~" & object$model$terms$level == "within", , drop = FALSE]
  nm <- if (nrow(regs)) regs$lhs[1L] else "y"
  s <- stats::sd(obs_y, na.rm = TRUE)
  stats::setNames(list(if (is.finite(s) && s > 0) s else 1), nm)
}

.dsem_between_predictor_cov <- function(object) {
  id_col <- if (length(object$model$id)) object$model$id[length(object$model$id)] else NULL
  dat <- object$data
  if (is.null(id_col) || !is.data.frame(dat) || !id_col %in% names(dat)) {
    return(list(cov = matrix(0, 0, 0), sd = list()))
  }
  terms <- object$model$terms
  b_terms <- terms[terms$level %in% c("between", "between_2", "between_3") & terms$op == "~", , drop = FALSE]
  cand_vars <- intersect(unique(c(b_terms$rhs, b_terms$lhs)), names(dat))
  if (!length(cand_vars)) return(list(cov = matrix(0, 0, 0), sd = list()))
  ids <- unique(dat[[id_col]])
  mat <- matrix(NA_real_, length(ids), length(cand_vars), dimnames = list(NULL, cand_vars))
  for (v in cand_vars) {
    if (!is.numeric(dat[[v]])) next
    for (g in seq_along(ids)) {
      vals <- dat[[v]][dat[[id_col]] == ids[g]]
      vals <- vals[!is.na(vals)]
      if (length(vals)) mat[g, v] <- vals[1L]
    }
  }
  keep_cols <- colSums(is.finite(mat)) >= 2L
  mat <- mat[, keep_cols, drop = FALSE]
  if (!ncol(mat)) return(list(cov = matrix(0, 0, 0), sd = list()))
  cov_m <- stats::cov(mat, use = "pairwise.complete.obs")
  cov_m[!is.finite(cov_m)] <- 0
  sd_list <- as.list(sqrt(pmax(diag(cov_m), 1e-12)))
  names(sd_list) <- colnames(mat)
  list(cov = cov_m, sd = sd_list)
}

.dsem_lookup_between_sd <- function(object, var_nm, between_cov_w) {
  if (var_nm %in% names(between_cov_w$sd)) return(between_cov_w$sd[[var_nm]])
  if (is.data.frame(object$data) && var_nm %in% names(object$data)) {
    s <- stats::sd(object$data[[var_nm]], na.rm = TRUE)
    if (is.finite(s) && s > 1e-8) return(s)
  }
  1
}

.dsem_split_cov_pair <- function(rest, var_prefix, params) {
  var_params <- params[startsWith(params, var_prefix)]
  var_names <- sub(paste0("^", var_prefix), "", var_params)
  for (a in var_names) {
    for (b in var_names) {
      if (identical(paste0(a, "_", b), rest)) {
        return(c(paste0(var_prefix, a), paste0(var_prefix, b)))
      }
    }
  }
  NULL
}

.dsem_between_total_vars <- function(object, flat, between_cov_w) {
  params <- colnames(flat)
  n_tot <- nrow(flat)
  res <- list()

  # Identify all between-level variance parameters (var_*, l3_var_*, l2_var_*, id_var_*, between_var_*)
  var_cols <- grep("^(?:l3_|l2_|id_|between_)?var_", params, value = TRUE)
  if (!length(var_cols)) return(res)

  # Separate primary random effects vs distal outcomes (those regressed on random effects)
  var_meta <- lapply(var_cols, function(vc) {
    nm <- sub("^(?:l3_|l2_|id_|between_)?var_", "", vc)
    pred_cols <- grep(paste0("^(?:l3_|l2_)?", nm, "_on_"), params, value = TRUE)
    preds <- sub(paste0("^(?:l3_|l2_)?", nm, "_on_"), "", pred_cols)
    list(var_col = vc, name = nm, pred_cols = pred_cols, preds = preds)
  })
  all_re_names <- vapply(var_meta, `[[`, character(1), "name")

  # Pass 1: Variables whose predictors are observed between covariates W (not other latent random effects)
  for (vm in var_meta) {
    if (any(vm$preds %in% all_re_names)) next
    tot_v <- pmax(flat[, vm$var_col], 1e-10)
    if (length(vm$preds)) {
      obs_preds <- intersect(vm$preds, colnames(between_cov_w$cov))
      if (length(obs_preds)) {
        sub_cov <- between_cov_w$cov[obs_preds, obs_preds, drop = FALSE]
        b_mat <- flat[, vm$pred_cols[match(obs_preds, vm$preds)], drop = FALSE]
        # Row-wise quadratic form b_i^T Sigma_W b_i
        exp_v <- rowSums((b_mat %*% sub_cov) * b_mat)
        tot_v <- tot_v + pmax(exp_v, 0)
      }
    }
    res[[vm$name]] <- tot_v
  }

  # Pass 2: Distal outcomes (regressed on random effects + optional W)
  for (vm in var_meta) {
    if (vm$name %in% names(res)) next
    tot_v <- pmax(flat[, vm$var_col], 1e-10)
    preds <- vm$preds
    p_k <- length(preds)
    if (p_k > 0L) {
      # Build draw-by-draw predictor covariance matrix among `preds`
      b_mat <- flat[, vm$pred_cols, drop = FALSE]
      exp_v <- numeric(n_tot)
      for (i in seq_len(p_k)) {
        vi <- if (preds[i] %in% names(res)) res[[preds[i]]] else rep(.dsem_lookup_between_sd(object, preds[i], between_cov_w)^2, n_tot)
        exp_v <- exp_v + b_mat[, i]^2 * vi
        if (i < p_k) {
          for (j in (i + 1L):p_k) {
            cov_ij <- rep(0, n_tot)
            # Check residual covariance + shared W-induced covariance between two random effects
            c1 <- paste0("cov_", preds[i], "_", preds[j])
            c2 <- paste0("cov_", preds[j], "_", preds[i])
            if (c1 %in% params) cov_ij <- cov_ij + flat[, c1]
            else if (c2 %in% params) cov_ij <- cov_ij + flat[, c2]
            # Shared W-induced covariance
            w_shared <- intersect(
              sub(paste0("^", preds[i], "_on_"), "", grep(paste0("^", preds[i], "_on_"), params, value = TRUE)),
              sub(paste0("^", preds[j], "_on_"), "", grep(paste0("^", preds[j], "_on_"), params, value = TRUE))
            )
            w_shared <- intersect(w_shared, colnames(between_cov_w$cov))
            for (wn in w_shared) {
              bi_w <- flat[, paste0(preds[i], "_on_", wn)]
              bj_w <- flat[, paste0(preds[j], "_on_", wn)]
              cov_ij <- cov_ij + bi_w * bj_w * between_cov_w$cov[wn, wn]
            }
            if (preds[i] %in% colnames(between_cov_w$cov) && preds[j] %in% colnames(between_cov_w$cov)) {
              cov_ij <- rep(between_cov_w$cov[preds[i], preds[j]], n_tot)
            }
            exp_v <- exp_v + 2 * b_mat[, i] * b_mat[, j] * cov_ij
          }
        }
      }
      tot_v <- tot_v + pmax(exp_v, 0)
    }
    res[[vm$name]] <- tot_v
  }

  res
}
