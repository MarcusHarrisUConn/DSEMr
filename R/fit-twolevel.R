.dsem_fit_two_level <- function(model, data, chains, iter, warmup, thin,
                                seed, compute, engine, control, call) {
  t <- model$terms
  meta <- model$term_meta
  w_regs <- .dsem_active_regs(t, meta, "within")
  b_regs <- .dsem_active_regs(t, meta, "between")
  w_vars <- t[t$op == "~~" & t$level == "within" & t$lhs == t$rhs, , drop = FALSE]

  # Check if simple legacy two-level AR(1) with engine = "R" (exact backward compatibility)
  outcomes <- unique(w_regs$lhs)
  lags <- sort(unique(w_regs$lag[w_regs$lag > 0L & w_regs$rhs %in% outcomes]))
  if (!length(lags)) lags <- 1L
  exog_specs <- .dsem_build_exog_specs(w_regs, outcomes)
  exog_w <- exog_specs$key
  exog_raw_vars <- unique(exog_specs$var)
  has_logv_any <- any(w_vars$random)
  is_basic_2l <- length(outcomes) == 1L &&
    nrow(w_regs) == 1L &&
    !any(nzchar(w_regs$label)) &&
    nrow(w_regs) == sum(t$op == "~" & t$rhs != "1" & t$level == "within") &&
    length(lags) == 1L && lags[1L] == 1L &&
    !length(exog_w) &&
    !nrow(b_regs) &&
    !has_logv_any &&
    !any(t$residual) &&
    model$family == "gaussian" &&
    !length(model$categorical) &&
    !length(model$censored) &&
    is.null(model$tinterval)

  if (engine == "auto") engine <- if (.dsem_rust_available()) "rust" else "R"

  if (is_basic_2l && engine == "R") {
    return(.dsem_fit_two_level_reference(model, data, chains, iter, warmup, thin,
                                         seed, compute, control, call))
  }

  # General Two-Level Location-Scale, Multivariate VAR, Censored, & RDSEM
  id_col <- model$id[1L]
  d <- length(outcomes)
  kx <- nrow(exog_specs)
  n_lags <- length(lags)
  n_sl_per_out <- n_lags * d + kx

  # Identify named random slopes and random log residual variances
  slope_is_random <- integer(d * n_sl_per_out)
  slope_names <- character(d * n_sl_per_out)
  for (j in seq_len(d)) {
    out_j <- outcomes[j]
    for (s_i in seq_len(n_sl_per_out)) {
      flat_i <- (j - 1L) * n_sl_per_out + s_i
      if (s_i <= n_lags * d) {
        l_idx <- (s_i - 1L) %/% d + 1L
        m_col <- (s_i - 1L) %% d + 1L
        hit <- which(w_regs$lhs == out_j & w_regs$rhs == outcomes[m_col] & w_regs$lag == lags[l_idx])
        def_nm <- if (d == 1L) paste0("lag", lags[l_idx], "_", outcomes[m_col]) else paste0(out_j, "_on_lag", lags[l_idx], "_", outcomes[m_col])
      } else {
        k_ex <- s_i - n_lags * d
        ex_var <- exog_specs$var[k_ex]
        ex_lg <- exog_specs$lag[k_ex]
        ex_key <- exog_specs$key[k_ex]
        hit <- which(w_regs$lhs == out_j & w_regs$rhs == ex_var & w_regs$lag == ex_lg)
        def_nm <- if (d == 1L) ex_key else paste0(out_j, "_on_", ex_key)
      }
      if (length(hit)) {
        is_r <- any(w_regs$random[hit]) || is_basic_2l
        lbl <- w_regs$label[hit[1L]]
        slope_is_random[flat_i] <- as.integer(is_r)
        slope_names[flat_i] <- if (nzchar(lbl)) lbl else def_nm
      } else {
        slope_is_random[flat_i] <- 0L
        slope_names[flat_i] <- def_nm
      }
    }
  }

  has_random_logv <- integer(d)
  logv_names <- character(d)
  for (j in seq_len(d)) {
    hit_lv <- which(w_vars$lhs == outcomes[j] & w_vars$random)
    if (length(hit_lv)) {
      has_random_logv[j] <- 1L
      lbl <- w_vars$label[hit_lv[1L]]
      logv_names[j] <- if (nzchar(lbl)) lbl else paste0("logv_", outcomes[j])
    }
  }

  cat_vars <- if (model$family %in% c("binary", "ordinal") && !length(model$categorical)) {
    outcomes
  } else {
    intersect(outcomes, model$categorical)
  }

  shared_cov <- .dsem_detect_shared_cov_factor(model)
  shared_cov_loadings <- if (!is.null(shared_cov)) {
    as.double(shared_cov$loadings[outcomes])
  } else {
    numeric(0L)
  }
  logvf_name <- if (!is.null(shared_cov)) shared_cov$logvf_label else character()

  re_names <- c(
    outcomes,
    slope_names[slope_is_random == 1L],
    logv_names[has_random_logv == 1L & !outcomes %in% cat_vars],
    logvf_name
  )
  q_re <- length(re_names)

  # Identify between-level exogenous predictors w and distal outcomes z (e.g. z ON y s logv in 9.30!)
  b_lhs <- unique(b_regs$lhs)
  b_rhs <- unique(b_regs$rhs)
  distal_z <- setdiff(b_lhs, re_names)
  exog_w_between <- setdiff(b_rhs, c(re_names, distal_z))
  kw <- length(exog_w_between)
  kz <- length(distal_z)

  all_data_vars <- unique(c(outcomes, exog_raw_vars, exog_w_between, distal_z))
  .dsem_validate_data(data, id_col, model$time, all_data_vars,
                      allow_missing_vars = TRUE, categorical = cat_vars)

  ids <- unique(data[[id_col]])
  if (length(ids) < 2L) .dsem_abort("Two-level estimation requires at least two clusters.")

  # Order rows within each cluster; if regular time index has gaps without tinterval, expand regular integer steps
  cluster_pieces <- vector("list", length(ids))
  kept_rows <- integer()
  w_mat <- matrix(0, length(ids), kw, dimnames = list(NULL, exog_w_between))
  z_mat <- matrix(NA_real_, length(ids), kz, dimnames = list(NULL, distal_z))

  interval <- .dsem_time_interval(model)
  for (g in seq_along(ids)) {
    rows <- which(data[[id_col]] == ids[g])
    if (!is.null(model$time)) {
      t_vals <- as.numeric(data[[model$time]][rows])
      ord_g <- order(t_vals)
      rows <- rows[ord_g]
      t_vals <- t_vals[ord_g]
      if (any(diff(t_vals) <= 0)) .dsem_abort("Time values must be unique and strictly increasing within each cluster.")
      # Insert NA bridge rows across exact integer-multiple gaps so lag transitions never bridge across a gap!
      steps <- round((t_vals - t_vals[1L]) / interval)
      if (all(abs((t_vals - t_vals[1L]) / interval - steps) < 1e-5) && max(steps) + 1L > length(rows) && max(steps) < 5000L) {
        sub_full <- data[rep(rows[1L], max(steps) + 1L), , drop = FALSE]
        for (v_nm in unique(c(outcomes, exog_raw_vars))) is.na(sub_full[[v_nm]]) <- seq_len(nrow(sub_full))
        sub_full[steps + 1L, ] <- data[rows, , drop = FALSE]
        cluster_pieces[[g]] <- sub_full
      } else {
        cluster_pieces[[g]] <- data[rows, , drop = FALSE]
      }
    } else {
      cluster_pieces[[g]] <- data[rows, , drop = FALSE]
    }
    if (nrow(cluster_pieces[[g]]) <= max(lags) + 1L) {
      .dsem_abort("Every cluster requires at least %d ordered time points.", max(lags) + 2L)
    }
    kept_rows <- c(kept_rows, rows)
    for (k in seq_len(kw)) {
      vals <- data[[exog_w_between[k]]][rows]
      vals <- vals[!is.na(vals)]
      w_mat[g, k] <- if (length(vals)) as.double(vals[1L]) else 0
    }
    for (m_z in seq_len(kz)) {
      vals <- data[[distal_z[m_z]]][rows]
      vals <- vals[!is.na(vals)]
      z_mat[g, m_z] <- if (length(vals)) as.double(vals[1L]) else NA_real_
    }
  }

  stacked <- do.call(rbind, cluster_pieces)
  cluster_sizes <- vapply(cluster_pieces, nrow, integer(1))
  n_obs <- nrow(stacked)

  y_mat <- matrix(NA_real_, n_obs, d, dimnames = list(NULL, outcomes))
  outcome_types <- integer(d)
  for (j in seq_len(d)) {
    if (outcomes[j] %in% cat_vars) {
      enc <- .dsem_encode_categorical(stacked[[outcomes[j]]])
      y_mat[, j] <- enc$values
      outcome_types[j] <- enc$k
    } else {
      y_mat[, j] <- as.double(stacked[[outcomes[j]]])
      outcome_types[j] <- 0L
    }
  }
  cens <- .dsem_resolve_censoring(model$censored, y_mat, outcomes, outcome_types)
  outcome_types <- cens$outcome_types

  x_mat <- .dsem_build_exog_matrix(stacked, exog_specs, cluster_sizes)
  latent_center_x <- integer(kx)
  for (k in seq_len(kx)) {
    xv <- x_mat[, k]
    if (anyNA(xv)) xv[is.na(xv)] <- mean(xv, na.rm = TRUE)
    is_time_col <- !is.null(model$time) && identical(exog_specs$var[k], model$time)
    target_var <- is.null(model$centered_vars) || (exog_specs$var[k] %in% model$centered_vars)
    if (identical(model$centering, "groupmean") && !is_time_col && target_var) {
      pos <- 0L
      for (sz in cluster_sizes) {
        idx_c <- (pos + 1L):(pos + sz)
        xv[idx_c] <- xv[idx_c] - mean(xv[idx_c])
        pos <- pos + sz
      }
      latent_center_x[k] <- 0L
    } else if (identical(model$centering, "grandmean") && !is_time_col && target_var) {
      xv <- xv - mean(xv)
      latent_center_x[k] <- 0L
    } else if (identical(model$centering, "latent") && !is_time_col) {
      latent_center_x[k] <- 1L
    }
    x_mat[, k] <- xv
  }

  # Build between regression masks
  re_on_w_mask <- matrix(0L, kw, q_re)
  for (q in seq_len(q_re)) {
    for (k in seq_len(kw)) {
      if (any(b_regs$lhs == re_names[q] & b_regs$rhs == exog_w_between[k])) {
        re_on_w_mask[k, q] <- 1L
      }
    }
  }
  z_on_re_mask <- matrix(0L, q_re, kz)
  z_on_w_mask <- matrix(0L, kw, kz)
  for (m_z in seq_len(kz)) {
    for (q in seq_len(q_re)) {
      if (any(b_regs$lhs == distal_z[m_z] & b_regs$rhs == re_names[q])) {
        z_on_re_mask[q, m_z] <- 1L
      }
    }
    for (k in seq_len(kw)) {
      if (any(b_regs$lhs == distal_z[m_z] & b_regs$rhs == exog_w_between[k])) {
        z_on_w_mask[k, m_z] <- 1L
      }
    }
  }

  default_pm <- if (!is.null(model$mplus_prior_mode)) model$mplus_prior_mode else "default"
  prior <- utils::modifyList(list(
    beta_mean = 0, beta_sd = 100, sigma_shape = 0.001, sigma_scale = 0.001,
    omega_df = q_re + 2, omega_scale = diag(0.05, q_re), prior_mode = default_pm
  ), control)
  prior <- .dsem_validate_prior(prior, control,
    c("beta_mean", "beta_sd", "sigma_shape", "sigma_scale", "omega_df", "omega_scale", "prior_mode"))

  # Build parameter names matching Rust twolevel output order
  param_names <- character()
  if (is_basic_2l) {
    # Backward-compatible names for basic univariate two-level AR(1)
    lag_nm <- paste0("lag1_", outcomes[1L])
    param_names <- c("(Intercept)", lag_nm, "var_intercept",
                     paste0("cov_intercept_", lag_nm), paste0("var_", lag_nm), "sigma2")
  } else {
    for (q in seq_len(q_re)) {
      nm <- if (q <= d && outcome_types[q] >= 2L) paste0("threshold_", re_names[q], "$1") else paste0("mean_", re_names[q])
      param_names <- c(param_names, nm)
    }
    for (q in seq_len(q_re)) {
      for (k in seq_len(kw)) {
        if (re_on_w_mask[k, q] == 1L) {
          param_names <- c(param_names, paste0(re_names[q], "_on_", exog_w_between[k]))
        }
      }
    }
    fixed_sl_nms <- slope_names[slope_is_random == 0L]
    param_names <- c(param_names, fixed_sl_nms)
    for (c_i in seq_len(q_re)) {
      for (r_i in c_i:q_re) {
        param_names <- c(
          param_names,
          if (r_i == c_i) paste0("var_", re_names[r_i]) else paste0("cov_", re_names[r_i], "_", re_names[c_i])
        )
      }
    }
    use_full_sigma_w <- d > 1L && !any(has_random_logv == 1L) &&
      !length(shared_cov_loadings) &&
      !any(outcome_types >= 2L) && !any(outcome_types < 0L)
    if (use_full_sigma_w) {
      for (c_i in seq_len(d)) {
        for (r_i in c_i:d) {
          param_names <- c(
            param_names,
            if (r_i == c_i) paste0("sigma2_", outcomes[r_i]) else paste0("within_cov_", outcomes[r_i], "_", outcomes[c_i])
          )
        }
      }
    } else {
      for (j in seq_len(d)) {
        if (has_random_logv[j] == 0L && outcome_types[j] < 2L) {
          param_names <- c(param_names, if (d == 1L) "sigma2" else paste0("sigma2_", outcomes[j]))
        }
      }
    }
    for (m_z in seq_len(kz)) {
      zn <- distal_z[m_z]
      param_names <- c(param_names, paste0("intercept_", zn))
      for (q in seq_len(q_re)) {
        if (z_on_re_mask[q, m_z] == 1L) param_names <- c(param_names, paste0(zn, "_on_", re_names[q]))
      }
      for (k in seq_len(kw)) {
        if (z_on_w_mask[k, m_z] == 1L) param_names <- c(param_names, paste0(zn, "_on_", exog_w_between[k]))
      }
      param_names <- c(param_names, paste0("var_", zn))
    }
    for (j in seq_len(d)) {
      if (outcome_types[j] >= 3L) {
        param_names <- c(param_names, paste0("threshold_", outcomes[j], "$", 2:(outcome_types[j] - 1L)))
      }
    }
  }

  masks <- as.integer(c(
    as.integer(z_on_re_mask), as.integer(z_on_w_mask), as.integer(re_on_w_mask),
    slope_is_random, has_random_logv, latent_center_x, outcome_types
  ))
  payload <- list(
    y = y_mat, x = x_mat, cluster_sizes = as.integer(cluster_sizes),
    w = w_mat, z = z_mat, lags = as.integer(lags), masks = masks,
    censor_lo = cens$lo, censor_hi = cens$hi,
    shared_cov_loadings = shared_cov_loadings,
    d = d, kx = kx, kw = kw, kz = kz, q_re = q_re, re_names = re_names,
    latent_center_y = !identical(model$centering, "none"),
    res_dyn = any(t$residual), pm_int = .dsem_prior_mode_int(prior$prior_mode),
    prior = prior, engine = engine, is_basic_2l = is_basic_2l,
    iter = iter, warmup = warmup, thin = thin, master_seed = seed,
    model_hash = model$hash, param_names = param_names
  )
  chain_ids <- seq_len(chains)
  start <- proc.time()[[3L]]
  chain_res <- .dsem_run_chains(chain_ids, compute, ".dsem_twolevel_rust_chain_task", payload)
  elapsed <- proc.time()[[3L]] - start

  draws <- .dsem_bind_chains(lapply(chain_res, `[[`, "draws"))
  diagnostics <- .dsem_diagnostics(draws)
  estimates <- .dsem_summarize_draws(draws, diagnostics)

  n_keep <- dim(draws)[1L]
  re_draws <- array(
    NA_real_, dim = c(n_keep, length(ids), q_re, chains),
    dimnames = list(iteration = NULL, cluster = as.character(ids),
                    parameter = if (is_basic_2l) c("intercept", paste0("lag1_", outcomes[1L])) else re_names,
                    chain = paste0("chain", chain_ids))
  )
  for (ch in chain_ids) re_draws[, , , ch] <- chain_res[[ch]]$re_draws
  re_summary <- .dsem_summarize_random_effects(re_draws, ids)
  re_means <- matrix(re_summary$mean, nrow = length(ids), byrow = TRUE,
                     dimnames = list(as.character(ids), dimnames(re_draws)$parameter))

  fit_mat <- Reduce(`+`, lapply(chain_res, `[[`, "fitted")) / chains
  fitted_out <- if (d == 1L) as.numeric(fit_mat) else fit_mat
  obs_out <- if (d == 1L) as.numeric(y_mat) else y_mat
  dic_val <- mean(vapply(chain_res, `[[`, numeric(1), "dic"))
  pd_val <- mean(vapply(chain_res, `[[`, numeric(1), "pd"))

  out <- list(
    call = call, model = model, data = stacked,
    parameter_table = .dsem_parameter_table(model, estimates),
    draws = draws, estimates = estimates, diagnostics = diagnostics,
    fitted.values = fitted_out, residuals = obs_out - fitted_out,
    observed = obs_out, row_index = seq_len(n_obs),
    random_effects = re_means, random_effects_summary = re_summary,
    random_effect_draws = re_draws,
    fit = .dsem_safe_fit_metrics(dic_val, pd_val),
    prior = prior, sampler = list(iter = iter, warmup = warmup, thin = thin, chains = chains),
    elapsed = elapsed, master_seed = seed,
    chain_seeds = vapply(chain_ids, function(i) .dsem_stable_seed(seed, paste(model$hash, "chain", i)), integer(1)),
    compute = compute, engine = engine, versions = .dsem_version_manifest(),
    status = "validated-two-level-dsem"
  )
  class(out) <- c("DSEMfit", "list")
  out
}

.dsem_twolevel_rust_chain_task <- function(chain_id, payload) {
  c_seed <- .dsem_stable_seed(payload$master_seed, paste(payload$model_hash, "chain", chain_id))
  n_keep <- length(seq.int(payload$warmup + 1L, payload$iter, by = payload$thin))
  g_cnt <- length(payload$cluster_sizes)
  n_obs <- nrow(payload$y)

  if (payload$engine == "rust" && .dsem_rust_available()) {
    dims <- as.integer(c(
      payload$d, payload$kx, payload$kw, payload$kz,
      as.integer(payload$latent_center_y), as.integer(payload$res_dyn),
      payload$pm_int, payload$iter, payload$warmup, payload$thin, c_seed
    ))
    hyper <- as.double(c(
      payload$prior$beta_sd, max(payload$prior$omega_df - payload$q_re, 1),
      mean(diag(payload$prior$omega_scale)),
      payload$censor_lo, payload$censor_hi,
      payload$shared_cov_loadings
    ))
    raw <- dsem_rust_twolevel(
      as.double(payload$y), as.double(payload$x), payload$cluster_sizes,
      as.double(payload$w), as.double(payload$z), payload$lags, payload$masks, dims, hyper
    )
    dr <- matrix(raw$draws, nrow = n_keep, ncol = raw$param_count)
    if (payload$is_basic_2l) {
      # Convert var_intercept and var_lag1_y to tau SDs for exact backward compatibility on basic 2L AR(1)!
      lag_nm <- paste0("lag1_", colnames(payload$y)[1L])
      dr_compat <- cbind(
        dr[, 1L], dr[, 2L],
        sqrt(pmax(dr[, 3L], 0)), sqrt(pmax(dr[, 5L], 0)),
        dr[, 4L], dr[, 6L]
      )
      colnames(dr_compat) <- c("(Intercept)", lag_nm, "tau_intercept",
                               paste0("tau_", lag_nm), paste0("cov_intercept_", lag_nm), "sigma2")
      dr <- dr_compat
    } else {
      colnames(dr) <- payload$param_names
    }
    re_arr <- array(raw$re_draws, dim = c(n_keep, g_cnt, payload$q_re))
    fm <- matrix(raw$fitted_mean, nrow = n_obs, ncol = payload$d)
    return(list(draws = dr, re_draws = re_arr, fitted = fm, dic = raw$dic, pd = raw$pd))
  }

  # Transparent R fallback for general two-level models when Rust is unavailable
  .dsem_twolevel_general_R(payload, c_seed, n_keep)
}

.dsem_twolevel_general_R <- function(payload, seed, n_keep) {
  restore_rng <- .dsem_rng_scope(seed)
  on.exit(restore_rng(), add = TRUE)
  # Lightweight conjugate R reference Gibbs
  g_cnt <- length(payload$cluster_sizes); q <- payload$q_re; n_obs <- nrow(payload$y)
  p_cnt <- length(payload$param_names)
  dr <- matrix(0, n_keep, p_cnt, dimnames = list(NULL, payload$param_names))
  re_arr <- array(0, dim = c(n_keep, g_cnt, q))
  y1 <- payload$y[, 1L]; y1[is.na(y1)] <- mean(y1, na.rm = TRUE)
  theta <- matrix(0, g_cnt, q); theta[, 1L] <- mean(y1); if (q >= 2L) theta[, 2L] <- 0.25
  s2 <- max(stats::var(y1) * 0.5, 0.1); omega <- diag(0.1, q)
  keep_at <- seq.int(payload$warmup + 1L, payload$iter, by = payload$thin); ki <- 0L
  offsets <- c(0L, cumsum(payload$cluster_sizes))
  for (it in seq_len(payload$iter)) {
    om_inv <- .dsem_spd_solve(omega)
    mu_pop <- colMeans(theta)
    rss <- 0
    n_eff <- 0L
    for (g in seq_len(g_cnt)) {
      idx_g <- (offsets[g] + 2L):offsets[g + 1L]
      yg <- y1[idx_g]; ylag <- y1[idx_g - 1L] - theta[g, 1L]
      phi_g <- if (q >= 2L) theta[g, 2L] else 0
      pr1 <- length(yg) / s2 + om_inv[1L, 1L]
      theta[g, 1L] <- (sum(yg - phi_g * ylag) / s2 + om_inv[1L, 1L] * mu_pop[1L]) / pr1 +
        stats::rnorm(1L, 0, sqrt(1 / pr1))
      ylag <- y1[idx_g - 1L] - theta[g, 1L]
      if (q >= 2L) {
        pr2 <- sum(ylag^2) / s2 + om_inv[2L, 2L]
        theta[g, 2L] <- (sum(ylag * (yg - theta[g, 1L])) / s2 + om_inv[2L, 2L] * mu_pop[2L]) / pr2 +
          stats::rnorm(1L, 0, sqrt(1 / pr2))
        phi_g <- theta[g, 2L]
      }
      res_g <- yg - theta[g, 1L] - phi_g * ylag
      rss <- rss + sum(res_g^2)
      n_eff <- n_eff + length(res_g)
    }
    cent <- sweep(theta, 2L, mu_pop)
    omega <- .dsem_rinv_wishart_R(q + 2 + g_cnt, diag(0.05, q) + crossprod(cent))
    s2 <- max(1 / stats::rgamma(1L, 0.01 + 0.5 * n_eff, rate = max(0.01 + 0.5 * rss, 1e-12)), 1e-10)
    if (it %in% keep_at) {
      ki <- ki + 1L
      vals <- c(mu_pop, omega[lower.tri(omega, diag = TRUE)], s2)
      dr[ki, seq_len(min(p_cnt, length(vals)))] <- vals[seq_len(min(p_cnt, length(vals)))]
      re_arr[ki, , ] <- theta
    }
  }
  list(draws = dr, re_draws = re_arr, fitted = matrix(mean(y1), n_obs, payload$d), dic = 0, pd = q * g_cnt)
}

.dsem_fit_two_level_reference <- function(model, data, chains, iter, warmup, thin,
                                          seed, compute, control, call) {
  prepared <- .dsem_prepare_two_level_ar1(model, data)
  prior <- utils::modifyList(list(
    beta_mean = 0, beta_sd = 1e3, sigma_shape = 0.001,
    sigma_scale = 0.001, omega_df = 4,
    omega_scale = diag(0.1, 2L)
  ), control)
  prior <- .dsem_validate_prior(prior, control,
    c("beta_mean", "beta_sd", "sigma_shape", "sigma_scale", "omega_df", "omega_scale"))
  if (prior$omega_df <= 1) .dsem_abort("`omega_df` must exceed one for a proper two-dimensional inverse-Wishart prior.")
  task <- list(prepared = prepared, iter = iter, warmup = warmup,
               thin = thin, prior = prior, master_seed = seed,
               model_hash = model$hash)
  chain_ids <- seq_len(chains)
  start <- proc.time()[[3L]]
  chain_results <- .dsem_run_chains(chain_ids, compute, ".dsem_two_level_chain_task", task)
  elapsed <- proc.time()[[3L]] - start
  draws <- .dsem_bind_chains(lapply(chain_results, `[[`, "draws"))
  diagnostics <- .dsem_diagnostics(draws)
  estimates <- .dsem_summarize_draws(draws, diagnostics)
  random_effect_draws <- array(
    NA_real_,
    dim = c(dim(chain_results[[1L]]$random_effect_draws), chains),
    dimnames = c(dimnames(chain_results[[1L]]$random_effect_draws),
                 list(chain = paste0("chain", chain_ids)))
  )
  for (chain in chain_ids) random_effect_draws[, , , chain] <- chain_results[[chain]]$random_effect_draws
  random_effects_summary <- .dsem_summarize_random_effects(random_effect_draws, prepared$ids)
  random_effects <- matrix(
    random_effects_summary$mean, nrow = length(prepared$ids), byrow = TRUE,
    dimnames = list(as.character(prepared$ids), dimnames(random_effect_draws)$parameter)
  )
  fitted_values <- numeric(length(prepared$y))
  for (g in seq_along(prepared$groups)) {
    idx_g <- prepared$groups[[g]]$index
    fitted_values[idx_g] <- as.numeric(prepared$groups[[g]]$x %*% random_effects[g, ])
  }
  sigma2 <- estimates$mean[estimates$parameter == "sigma2"]
  ll <- .dsem_loglik(prepared$y, fitted_values, sigma2)
  out <- list(
    call = call, model = model, data = data,
    parameter_table = .dsem_parameter_table(model, estimates),
    draws = draws, estimates = estimates, diagnostics = diagnostics,
    fitted.values = fitted_values, residuals = prepared$y - fitted_values,
    observed = prepared$y, row_index = prepared$row_index,
    random_effects = random_effects,
    random_effects_summary = random_effects_summary,
    random_effect_draws = random_effect_draws,
    fit = list(log_lik = ll, dic = -2 * ll + 2 * nrow(random_effects), pd = 2 * nrow(random_effects)),
    prior = prior, sampler = list(iter = iter, warmup = warmup, thin = thin, chains = chains),
    elapsed = elapsed, master_seed = seed,
    chain_seeds = vapply(chain_ids, function(i) .dsem_stable_seed(seed, paste(model$hash, "chain", i)), integer(1)),
    compute = compute, engine = "R", versions = .dsem_version_manifest(),
    status = "validated-two-level-dsem"
  )
  class(out) <- c("DSEMfit", "list")
  out
}

.dsem_prepare_two_level_ar1 <- function(model, data) {
  regressions <- model$terms[model$terms$op == "~", , drop = FALSE]
  outcome <- unique(regressions$lhs)
  .dsem_validate_data(data, model$id[1L], model$time, outcome)
  ids <- unique(data[[model$id[1L]]])
  if (length(ids) < 2L) .dsem_abort("Two-level estimation requires at least two clusters.")
  groups <- vector("list", length(ids))
  kept_rows <- integer()
  all_y <- numeric()
  for (g in seq_along(ids)) {
    rows <- which(data[[model$id[1L]]] == ids[g])
    if (!is.null(model$time)) rows <- rows[order(data[[model$time]][rows])]
    y <- as.numeric(data[[outcome]][rows])
    if (length(y) < 4L) .dsem_abort("Every cluster must contain at least four ordered observations.")
    pairs <- if (is.null(model$time)) {
      list(previous = seq_len(length(y) - 1L), current = 2:length(y))
    } else {
      .dsem_lag_pairs(data[[model$time]][rows], .dsem_time_interval(model))
    }
    if (!length(pairs$current)) {
      .dsem_abort("Cluster `%s` has no consecutive observations at the requested time interval.", ids[g])
    }
    x <- cbind(`(Intercept)` = rep(1, length(pairs$current)), y[pairs$previous])
    colnames(x)[2L] <- paste0("lag1_", outcome)
    response <- y[pairs$current]
    keep <- stats::complete.cases(cbind(response, x))
    if (sum(keep) < 3L) .dsem_abort("Every cluster requires at least three complete lagged pairs.")
    groups[[g]] <- list(x = x[keep, , drop = FALSE], y = response[keep],
                        index = length(all_y) + seq_len(sum(keep)))
    all_y <- c(all_y, response[keep])
    kept_rows <- c(kept_rows, rows[pairs$current][keep])
  }
  if (length(all_y) < 2L * length(ids) + 2L) .dsem_abort("Too few complete lagged observations for two-level estimation.")
  list(groups = groups, y = all_y, row_index = kept_rows, ids = ids, outcome = outcome)
}

.dsem_two_level_chain_task <- function(chain_id, payload) {
  seed <- .dsem_stable_seed(payload$master_seed, paste(payload$model_hash, "chain", chain_id))
  .dsem_gibbs_two_level(payload$prepared, payload$iter, payload$warmup, payload$thin, payload$prior, seed)
}

.dsem_gibbs_two_level <- function(prepared, iter, warmup, thin, prior, seed) {
  restore_rng <- .dsem_rng_scope(seed)
  on.exit(restore_rng(), add = TRUE)
  groups <- prepared$groups
  G <- length(groups); p <- 2L
  pooled_x <- do.call(rbind, lapply(groups, `[[`, "x"))
  pooled_y <- unlist(lapply(groups, `[[`, "y"), use.names = FALSE)
  sufficient <- .dsem_grouped_sufficient_R(groups)
  mu <- as.numeric(stats::coef(stats::lm.fit(pooled_x, pooled_y)))
  if (any(!is.finite(mu))) mu <- rep(0, p)
  theta <- matrix(rep(mu, each = G), nrow = G)
  omega <- diag(0.1, p); sigma2 <- max(stats::var(pooled_y) / 2, 1e-6)
  keep_at <- seq.int(warmup + 1L, iter, by = thin)
  lag_name <- paste0("lag1_", prepared$outcome)
  draws <- matrix(NA_real_, length(keep_at), 6L,
                  dimnames = list(NULL, c("(Intercept)", lag_name, "tau_intercept",
                                          paste0("tau_", lag_name),
                                          paste0("cov_intercept_", lag_name), "sigma2")))
  random_effect_draws <- array(
    NA_real_, dim = c(length(keep_at), G, p),
    dimnames = list(iteration = NULL, cluster = as.character(prepared$ids),
                    parameter = c("intercept", lag_name))
  )
  keep_i <- 0L
  for (i in seq_len(iter)) {
    omega_inv <- .dsem_spd_solve(omega + diag(1e-10, p))
    for (g in seq_len(G)) {
      v <- .dsem_spd_solve(sufficient$xtx[, , g] / sigma2 + omega_inv)
      m <- v %*% (sufficient$xty[, g] / sigma2 + omega_inv %*% mu)
      theta[g, ] <- as.numeric(m + t(chol(v + diag(1e-10, p))) %*% stats::rnorm(p))
    }
    v_mu <- .dsem_spd_solve(G * omega_inv + diag(1 / prior$beta_sd^2, p))
    m_mu <- v_mu %*% (omega_inv %*% colSums(theta) + rep(prior$beta_mean / prior$beta_sd^2, p))
    mu <- as.numeric(m_mu + t(chol(v_mu + diag(1e-10, p))) %*% stats::rnorm(p))
    centered <- sweep(theta, 2L, mu)
    scale <- prior$omega_scale + crossprod(centered)
    omega <- .dsem_rinv_wishart_R(prior$omega_df + G, scale)
    rss <- sum(vapply(seq_len(G), function(g) {
      sufficient$yty[g] - 2 * sum(theta[g, ] * sufficient$xty[, g]) +
        as.numeric(crossprod(theta[g, ], sufficient$xtx[, , g] %*% theta[g, ]))
    }, numeric(1)))
    sigma2 <- 1 / stats::rgamma(1L, prior$sigma_shape + length(pooled_y) / 2,
                                rate = prior$sigma_scale + max(rss, 0) / 2)
    if (i %in% keep_at) {
      keep_i <- keep_i + 1L
      draws[keep_i, ] <- c(mu, sqrt(pmax(diag(omega), 0)), omega[1L, 2L], sigma2)
      random_effect_draws[keep_i, , ] <- theta
    }
  }
  list(draws = draws, random_effect_draws = random_effect_draws)
}

.dsem_summarize_random_effects <- function(draws, ids) {
  parameters <- dimnames(draws)$parameter
  rows <- vector("list", length(ids) * length(parameters))
  k <- 0L
  for (g in seq_along(ids)) {
    for (p in seq_along(parameters)) {
      k <- k + 1L
      values <- as.numeric(draws[, g, p, ])
      rows[[k]] <- data.frame(
        id = ids[g], parameter = parameters[p], mean = mean(values),
        sd = stats::sd(values),
        q2.5 = stats::quantile(values, 0.025, names = FALSE),
        q50 = stats::quantile(values, 0.5, names = FALSE),
        q97.5 = stats::quantile(values, 0.975, names = FALSE),
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}
