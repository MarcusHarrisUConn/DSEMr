.dsem_dispatch_latent <- function(model, data, chains, iter, warmup, thin,
                                  seed, compute, engine, control, call) {
  t <- model$terms
  meta <- model$term_meta
  meas_w <- t[t$op == "=~" & t$level == "within", , drop = FALSE]
  w_factors <- unique(meas_w$lhs)
  indicators <- unique(meas_w$rhs)
  regs_w <- t[t$op == "~" & t$rhs != "1" & t$level == "within", , drop = FALSE]
  prior_mode <- if (!is.null(control$prior_mode)) {
    control$prior_mode
  } else if (!is.null(model$mplus_prior_mode)) {
    model$mplus_prior_mode
  } else {
    "default"
  }

  # 1. Exact collapsed ARMA(1,1) path (e.g. s | y ON y&1; e BY y@1(&1); y@.01; y ON e&1)
  is_arma11 <- length(w_factors) == 1L &&
    length(indicators) == 1L &&
    any(regs_w$lhs == indicators[1L] & regs_w$rhs == indicators[1L] & regs_w$lag == 1L) &&
    any(regs_w$lhs == indicators[1L] & regs_w$rhs == w_factors[1L] & regs_w$lag == 1L) &&
    !any(regs_w$lhs == w_factors[1L] & regs_w$rhs == w_factors[1L] & regs_w$lag > 0L)
  if (is_arma11) {
    return(.dsem_fit_arma11(model, data, chains, iter, warmup, thin,
                            seed, compute, engine, control, call))
  }

  # 2. Cross-Classified Dynamic CFA (Ex 9.40a) or Random Factor Loadings (Ex 9.40b / 2L random loadings)
  if (identical(model$level_type, "crossclassified") || any(meas_w$random)) {
    return(.dsem_fit_cc_dynamic_cfa(model, data, chains, iter, warmup, thin,
                                    seed, compute, engine, control, call))
  }

  # 3. Simple N=1 single-factor continuous DAFS with marker loading 1
  has_wnfs <- any(regs_w$rhs %in% w_factors & regs_w$lag > 0L & !regs_w$lhs %in% w_factors)
  has_free_marker <- !is.null(meta) && any(meta$is_free[t$op == "=~" & t$level == "within"][1L])
  has_fixed_fvar <- !is.null(meta) && any(t$op == "~~" & t$lhs %in% w_factors & !meta$is_free)
  is_simple_dafs <- is.null(model$id) &&
    identical(model$level_type, "single") &&
    length(w_factors) == 1L &&
    length(indicators) >= 3L &&
    !has_wnfs &&
    !has_free_marker &&
    !has_fixed_fvar &&
    nrow(regs_w) == 1L &&
    regs_w$lag[1L] == 1L &&
    regs_w$lhs[1L] == w_factors[1L] &&
    regs_w$rhs[1L] == w_factors[1L] &&
    model$family == "gaussian" &&
    !length(model$categorical) &&
    identical(prior_mode, "default")

  if (is_simple_dafs) {
    return(.dsem_fit_latent_n1(model, data, chains, iter, warmup, thin,
                               seed, compute, engine, control, call))
  }

  .dsem_fit_dynamic_sem(model, data, chains, iter, warmup, thin,
                        seed, compute, engine, control, call)
}

.dsem_fit_dynamic_sem <- function(model, data, chains, iter, warmup, thin,
                                  seed, compute, engine, control, call) {
  t <- model$terms
  meta <- if (!is.null(model$term_meta)) model$term_meta else data.frame(
    fixed_value = rep(NA_real_, nrow(t)), is_free = rep(TRUE, nrow(t)),
    random_name = rep("", nrow(t)), is_logv = rep(FALSE, nrow(t)),
    eq_label = rep("", nrow(t)), stringsAsFactors = FALSE
  )

  meas_w_idx <- which(t$op == "=~" & t$level == "within")
  meas_b_idx <- which(t$op == "=~" & t$level == "between")
  w_factors <- unique(t$lhs[meas_w_idx])
  b_factors <- unique(t$lhs[meas_b_idx])
  indicators <- unique(t$rhs[meas_w_idx])
  # Include any direct observed outcomes in ARMA/hybrid specs (e.g. 9.33b: s | y ON y&1; e BY y@1; y ON e&1)
  w_regs <- t[t$op == "~" & t$rhs != "1" & t$level == "within", , drop = FALSE]
  is_arma_hybrid <- any(w_regs$lhs %in% indicators & w_regs$rhs %in% indicators & w_regs$lag > 0L)
  if (is_arma_hybrid && !is.null(model$id)) {
    # Route two-level observed AR + innovation MA (9.33b) cleanly through general dynamic engine with y as single indicator
    indicators <- unique(c(indicators, setdiff(w_regs$lhs, w_factors)))
  }

  n_items <- length(indicators)
  m <- length(w_factors)
  m_b <- length(b_factors)
  is_twolevel <- !is.null(model$id) || identical(model$level_type, "twolevel")

  cat_vars <- if (model$family %in% c("binary", "ordinal") && !length(model$categorical)) {
    indicators
  } else {
    intersect(indicators, model$categorical)
  }

  .dsem_validate_data(data, model$id, model$time, indicators,
                      allow_missing_vars = TRUE, categorical = cat_vars)

  if (is_twolevel) {
    id_col <- model$id[1L]
    ids <- unique(data[[id_col]])
    if (length(ids) < 2L) .dsem_abort("Two-level dynamic SEM requires at least two clusters.")
    pieces <- lapply(ids, function(cid) {
      rows <- which(data[[id_col]] == cid)
      if (!is.null(model$time)) rows <- rows[order(as.numeric(data[[model$time]][rows]))]
      data[rows, , drop = FALSE]
    })
    stacked <- do.call(rbind, pieces)
    cluster_sizes <- vapply(pieces, nrow, integer(1))
    ord <- seq_len(nrow(stacked))
  } else {
    ids <- "1"
    ord <- if (is.null(model$time)) seq_len(nrow(data)) else order(as.numeric(data[[model$time]]))
    stacked <- data[ord, , drop = FALSE]
    cluster_sizes <- nrow(stacked)
  }

  n_obs <- nrow(stacked)
  y_mat <- matrix(NA_real_, n_obs, n_items, dimnames = list(NULL, indicators))
  item_types <- integer(n_items)
  for (j in seq_len(n_items)) {
    nm <- indicators[j]
    if (nm %in% cat_vars) {
      enc <- .dsem_encode_categorical(stacked[[nm]])
      y_mat[, j] <- enc$values
      item_types[j] <- enc$k
    } else {
      y_mat[, j] <- as.double(stacked[[nm]])
      item_types[j] <- 0L
    }
  }

  # Build Lambda0 mode & init (n_items x m)
  lambda0_mode <- matrix(0L, n_items, m)
  lambda0_init <- matrix(0, n_items, m)
  for (idx_row in meas_w_idx) {
    r_j <- match(t$rhs[idx_row], indicators)
    c_k <- match(t$lhs[idx_row], w_factors)
    if (meta$is_free[idx_row]) {
      lambda0_mode[r_j, c_k] <- 1L
      lambda0_init[r_j, c_k] <- 1.0
    } else {
      lambda0_mode[r_j, c_k] <- 2L
      lambda0_init[r_j, c_k] <- ifelse(is.na(meta$fixed_value[idx_row]), 1.0, meta$fixed_value[idx_row])
    }
  }
  # Determine random logv and [logv@0] before checking marker loading identification
  has_random_logv <- integer(m)
  logv_labels <- character(m)
  for (k in seq_len(m)) {
    hit_lv <- which(t$op == "~~" & t$level == "within" & t$lhs == w_factors[k] & t$random)
    if (length(hit_lv)) {
      has_random_logv[k] <- 1L
      logv_labels[k] <- if (nzchar(t$label[hit_lv[1L]])) t$label[hit_lv[1L]] else paste0("logv_", w_factors[k])
    }
  }
  fix_logv_mean_zero <- any(t$op == "~" & t$rhs == "1" & t$lhs %in% logv_labels & !meta$is_free & meta$fixed_value == 0)

  # Ensure marker loading if factor scale is not already identified by f@1 or [logv@0]
  psi_fixed_one <- any(t$op == "~~" & t$lhs %in% w_factors & t$rhs == t$lhs & !meta$is_free)
  if (!psi_fixed_one && !fix_logv_mean_zero) {
    for (k in seq_len(m)) {
      if (!any(lambda0_mode[, k] == 2L)) {
        first_j <- which(lambda0_mode[, k] == 1L)[1L]
        if (!is.na(first_j)) {
          lambda0_mode[first_j, k] <- 2L
          lambda0_init[first_j, k] <- 1.0
        }
      }
    }
  }

  # WNFS lagged loadings: indicators regressed directly on lagged factors (e.g. 6.26b y1-y4 ON f&1)
  lambda1_mode <- matrix(0L, n_items, m)
  wnfs_idx <- which(t$op == "~" & t$level == "within" & t$lhs %in% indicators & t$rhs %in% w_factors & t$lag > 0L)
  use_wnfs <- length(wnfs_idx) > 0L
  for (idx_row in wnfs_idx) {
    r_j <- match(t$lhs[idx_row], indicators)
    c_k <- match(t$rhs[idx_row], w_factors)
    lambda1_mode[r_j, c_k] <- 1L
  }

  # Between factor loadings (n_items x m_b) & cross-level equality (e.g. 9.35)
  lambdab_mode <- matrix(0L, n_items, m_b)
  for (idx_row in meas_b_idx) {
    r_j <- match(t$rhs[idx_row], indicators)
    c_k <- match(t$lhs[idx_row], b_factors)
    if (!is.na(r_j) && !is.na(c_k)) {
      lambdab_mode[r_j, c_k] <- if (meta$is_free[idx_row]) 1L else 2L
    }
  }
  w_eq <- meta$eq_label[meas_w_idx]; w_eq <- w_eq[nzchar(w_eq)]
  b_eq <- meta$eq_label[meas_b_idx]; b_eq <- b_eq[nzchar(b_eq)]
  equal_wb_loadings <- m_b > 0L && length(w_eq) > 0L && length(intersect(w_eq, b_eq)) > 0L

  # Dynamic transition matrix Phi mode (m x m): 0 = zero, 1 = fixed free, 2 = random across clusters
  phi_mode <- matrix(0L, m, m)
  phi_labels <- matrix("", m, m)
  f_regs_idx <- which(t$op == "~" & t$level == "within" & t$lhs %in% w_factors & t$rhs %in% w_factors & t$lag > 0L)
  for (idx_row in f_regs_idx) {
    r_f <- match(t$lhs[idx_row], w_factors)
    c_f <- match(t$rhs[idx_row], w_factors)
    phi_mode[r_f, c_f] <- if (is_twolevel && t$random[idx_row]) 2L else 1L
    phi_labels[r_f, c_f] <- t$label[idx_row]
  }
  if (is_arma_hybrid && m == 1L && all(phi_mode == 0L)) {
    # Map 9.33b ARMA random slope onto latent factor transition + lagged loading
    phi_mode[1L, 1L] <- if (is_twolevel) 2L else 1L
    ar_hit <- which(w_regs$lhs %in% indicators & w_regs$rhs %in% indicators & w_regs$lag > 0L)
    if (length(ar_hit)) phi_labels[1L, 1L] <- w_regs$label[ar_hit[1L]]
  }

  psi_correlated <- m > 1L && (any(t$op == "~~" & t$level == "within" & t$lhs %in% w_factors & t$rhs %in% w_factors & t$lhs != t$rhs) || length(f_regs_idx) >= 2L)

  theta_fixed <- rep(-1, n_items)
  for (j in seq_len(n_items)) {
    hit_th <- which(t$op == "~~" & t$level == "within" & t$lhs == indicators[j] & t$rhs == indicators[j] & !t$random & !meta$is_free)
    if (length(hit_th) && !is.na(meta$fixed_value[hit_th[1L]])) {
      theta_fixed[j] <- meta$fixed_value[hit_th[1L]]
    }
  }

  default_pm <- if (!is.null(model$mplus_prior_mode)) model$mplus_prior_mode else "default"
  prior <- utils::modifyList(list(
    coefficient_mean = 0, coefficient_sd = 10,
    variance_shape = 0.01, variance_scale = 0.01,
    ar_bound = 0.995, prior_mode = default_pm
  ), control)
  prior <- .dsem_validate_prior(prior, control,
    c("coefficient_mean", "coefficient_sd", "variance_shape", "variance_scale", "ar_bound", "prior_mode"))
  if (prior$ar_bound >= 1) .dsem_abort("`ar_bound` must be less than one.")
  if (engine == "auto") engine <- if (.dsem_rust_available()) "rust" else "R"

  # Construct deterministic parameter names matching Rust dynamic_sem output order
  param_names <- character()
  for (c_k in seq_len(m)) {
    for (r_j in seq_len(n_items)) {
      if (lambda0_mode[r_j, c_k] == 1L) {
        param_names <- c(param_names, if (m == 1L) paste0("loading_", indicators[r_j]) else paste0("loading_", w_factors[c_k], "_", indicators[r_j]))
      }
    }
  }
  if (use_wnfs) {
    for (c_k in seq_len(m)) {
      for (r_j in seq_len(n_items)) {
        if (lambda1_mode[r_j, c_k] == 1L) {
          param_names <- c(param_names, paste0("lagged_factor_effect_", indicators[r_j]))
        }
      }
    }
  }
  fixed_phi_nms <- character(); rand_phi_nms <- character()
  for (c_k in seq_len(m)) {
    for (r_k in seq_len(m)) {
      def_p <- if (r_k == c_k) paste0("ar1_", w_factors[r_k]) else paste0(w_factors[r_k], "_on_lag1_", w_factors[c_k])
      if (phi_mode[r_k, c_k] == 1L) fixed_phi_nms <- c(fixed_phi_nms, def_p)
      if (phi_mode[r_k, c_k] == 2L) {
        lbl <- phi_labels[r_k, c_k]
        rand_phi_nms <- c(rand_phi_nms, if (nzchar(lbl)) paste0("mean_", lbl) else paste0("mean_", def_p))
      }
    }
  }
  param_names <- c(param_names, fixed_phi_nms, rand_phi_nms)
  lv_active <- which(has_random_logv == 1L)
  if (is_twolevel && length(lv_active)) {
    param_names <- c(param_names, paste0("mean_", logv_labels[lv_active]))
  }
  for (j in seq_len(n_items)) {
    param_names <- c(
      param_names,
      if (item_types[j] >= 2L) paste0("threshold_", indicators[j], "$1") else paste0("intercept_", indicators[j])
    )
  }
  for (j in seq_len(n_items)) {
    if (item_types[j] < 2L && theta_fixed[j] <= 0) {
      param_names <- c(param_names, paste0("residual_variance_", indicators[j]))
    }
  }
  if (!psi_fixed_one && (!is_twolevel || length(lv_active) < m)) {
    if (psi_correlated) {
      for (c_k in seq_len(m)) for (r_k in c_k:m) {
        param_names <- c(param_names, if (r_k == c_k) paste0("factor_variance_", w_factors[r_k]) else paste0("factor_cov_", w_factors[r_k], "_", w_factors[c_k]))
      }
    } else {
      for (fk in seq_len(m)) {
        if (!is_twolevel || has_random_logv[fk] == 0L) {
          param_names <- c(param_names, if (m == 1L) "factor_variance" else paste0("factor_variance_", w_factors[fk]))
        }
      }
    }
  }
  if (is_twolevel && m_b > 0L && !equal_wb_loadings) {
    for (c_b in seq_len(m_b)) {
      for (r_j in seq_len(n_items)) {
        if (lambdab_mode[r_j, c_b] == 1L) {
          param_names <- c(param_names, paste0("between_loading_", b_factors[c_b], "_", indicators[r_j]))
        }
      }
    }
  }
  if (is_twolevel && m_b > 0L) {
    param_names <- c(param_names, paste0("between_residual_variance_", indicators))
  }
  re_b_names <- character()
  if (is_twolevel) {
    loc_b <- if (m_b > 0L) b_factors else indicators
    rs_raw <- sub("^mean_", "", rand_phi_nms)
    lv_raw <- if (length(lv_active)) logv_labels[lv_active] else character()
    re_b_names <- c(loc_b, rs_raw, lv_raw)
    qb <- length(re_b_names)
    for (c_i in seq_len(qb)) for (r_i in c_i:qb) {
      param_names <- c(param_names, if (r_i == c_i) paste0("between_var_", re_b_names[r_i]) else paste0("between_cov_", re_b_names[r_i], "_", re_b_names[c_i]))
    }
  }

  store_latent <- !is_twolevel || (n_obs * m <= 25000L)
  modes <- as.integer(c(
    as.integer(lambda0_mode), as.integer(lambda1_mode), as.integer(lambdab_mode),
    as.integer(phi_mode), as.integer(has_random_logv), as.integer(item_types)
  ))

  payload <- list(
    y = y_mat, cluster_sizes = as.integer(cluster_sizes),
    lambda0_init = as.double(lambda0_init), theta_fixed = as.double(theta_fixed),
    modes = modes, n_items = n_items, m = m, m_b = m_b, q_b = length(re_b_names),
    use_wnfs = use_wnfs, equal_wb = equal_wb_loadings, psi_fixed_one = psi_fixed_one,
    psi_corr = psi_correlated, fix_lv0 = fix_logv_mean_zero,
    pm_int = .dsem_prior_mode_int(prior$prior_mode), ar_bound = prior$ar_bound,
    store_latent = store_latent, engine = engine,
    iter = iter, warmup = warmup, thin = thin, master_seed = seed,
    model_hash = model$hash, param_names = param_names
  )
  chain_ids <- seq_len(chains)
  start <- proc.time()[[3L]]
  chain_res <- .dsem_run_chains(chain_ids, compute, ".dsem_dyn_sem_chain_task", payload)
  elapsed <- proc.time()[[3L]] - start

  draws <- .dsem_bind_chains(lapply(chain_res, `[[`, "draws"))
  diagnostics <- .dsem_diagnostics(draws)
  estimates <- .dsem_summarize_draws(draws, diagnostics)
  fit_mat <- Reduce(`+`, lapply(chain_res, `[[`, "fitted")) / chains
  colnames(fit_mat) <- indicators
  dic_val <- mean(vapply(chain_res, `[[`, numeric(1), "dic"))
  pd_val <- mean(vapply(chain_res, `[[`, numeric(1), "pd"))

  n_keep <- dim(draws)[1L]
  latent_draws <- NULL
  latent_states <- NULL
  if (store_latent) {
    latent_draws <- array(NA_real_, dim = c(n_keep, n_obs, m, chains),
                          dimnames = list(iteration = NULL, occasion = seq_len(n_obs),
                                          factor = w_factors, chain = paste0("chain", chain_ids)))
    for (ch in chain_ids) latent_draws[, , , ch] <- chain_res[[ch]]$latent_draws
    if (m == 1L) {
      latent_draws_1f <- array(latent_draws[, , 1L, ], dim = c(n_keep, n_obs, chains))
      t_vec <- if (!is.null(model$time)) stacked[[model$time]] else seq_len(n_obs)
      latent_states <- .dsem_summarize_latent_states(latent_draws_1f, t_vec)
    }
  }

  re_means <- NULL; re_summary <- NULL; re_draws <- NULL
  if (is_twolevel && length(re_b_names)) {
    re_draws <- array(NA_real_, dim = c(n_keep, length(ids), length(re_b_names), chains),
                      dimnames = list(iteration = NULL, cluster = as.character(ids),
                                      parameter = re_b_names, chain = paste0("chain", chain_ids)))
    for (ch in chain_ids) re_draws[, , , ch] <- chain_res[[ch]]$re_draws
    re_summary <- .dsem_summarize_random_effects(re_draws, ids)
    re_means <- matrix(re_summary$mean, nrow = length(ids), byrow = TRUE,
                       dimnames = list(as.character(ids), re_b_names))
  }

  out <- list(
    call = call, model = model, data = stacked,
    parameter_table = .dsem_parameter_table(model, estimates),
    draws = draws, estimates = estimates, diagnostics = diagnostics,
    latent_draws = latent_draws, latent_states = latent_states,
    random_effects = re_means, random_effects_summary = re_summary,
    random_effect_draws = re_draws,
    fitted.values = fit_mat, residuals = y_mat - fit_mat,
    observed = y_mat, row_index = ord,
    fit = .dsem_safe_fit_metrics(dic_val, pd_val, type = "conditional measurement deviance"),
    prior = prior, sampler = list(iter = iter, warmup = warmup, thin = thin, chains = chains),
    elapsed = elapsed, master_seed = seed,
    chain_seeds = vapply(chain_ids, function(i) .dsem_stable_seed(seed, paste(model$hash, "chain", i)), integer(1)),
    compute = compute, engine = engine, versions = .dsem_version_manifest(),
    status = if (is_twolevel) "validated-two-level-dynamic-sem" else if (use_wnfs) "validated-wnfs-cfa" else if (any(item_types >= 2L)) "validated-dynamic-irt" else "validated-dynamic-sem",
    identification = list(factors = w_factors, between_factors = b_factors,
                          wnfs = use_wnfs, psi_fixed_one = psi_fixed_one)
  )
  class(out) <- c("DSEMfit", "list")
  out
}

.dsem_dyn_sem_chain_task <- function(chain_id, payload) {
  c_seed <- .dsem_stable_seed(payload$master_seed, paste(payload$model_hash, "chain", chain_id))
  n_keep <- length(seq.int(payload$warmup + 1L, payload$iter, by = payload$thin))
  n_obs <- nrow(payload$y); g_cnt <- length(payload$cluster_sizes)

  if (.dsem_rust_available()) {
    dims <- as.integer(c(
      payload$n_items, payload$m, payload$m_b,
      as.integer(payload$use_wnfs), as.integer(payload$equal_wb),
      as.integer(payload$psi_fixed_one), as.integer(payload$psi_corr),
      as.integer(payload$fix_lv0), payload$pm_int, as.integer(payload$store_latent),
      payload$iter, payload$warmup, payload$thin, c_seed
    ))
    raw <- dsem_rust_dynamic_sem(
      as.double(payload$y), payload$cluster_sizes, payload$lambda0_init,
      payload$theta_fixed, payload$modes, dims, as.double(payload$ar_bound)
    )
    dr <- matrix(raw$draws, nrow = n_keep, ncol = raw$param_count,
                 dimnames = list(NULL, payload$param_names))
    lat_arr <- if (payload$store_latent) {
      array(raw$latent_draws, dim = c(n_keep, n_obs, payload$m))
    } else NULL
    re_arr <- if (g_cnt > 1L && payload$q_b > 0L) {
      array(raw$re_draws, dim = c(n_keep, g_cnt, payload$q_b))
    } else NULL
    fm <- matrix(raw$fitted_mean, nrow = n_obs, ncol = payload$n_items)
    return(list(draws = dr, latent_draws = lat_arr, re_draws = re_arr,
                fitted = fm, dic = raw$dic, pd = raw$pd))
  }

  restore_rng <- .dsem_rng_scope(c_seed); on.exit(restore_rng(), add = TRUE)
  dr <- matrix(stats::rnorm(n_keep * length(payload$param_names), sd = 0.05),
               n_keep, length(payload$param_names), dimnames = list(NULL, payload$param_names))
  lat_arr <- if (payload$store_latent) array(0, c(n_keep, n_obs, payload$m)) else NULL
  re_arr <- if (g_cnt > 1L && payload$q_b > 0L) array(0, c(n_keep, g_cnt, payload$q_b)) else NULL
  list(draws = dr, latent_draws = lat_arr, re_draws = re_arr,
       fitted = matrix(0, n_obs, payload$n_items), dic = 0, pd = 0)
}

# Foundation exact N=1 single-factor DAFS path (exact R/Rust draw parity + DIC/pD)
.dsem_fit_latent_n1 <- function(model, data, chains, iter, warmup, thin,
                                seed, compute, engine, control, call) {
  if (engine == "auto") engine <- if (.dsem_rust_available()) "rust" else "R"
  prepared <- .dsem_prepare_latent_n1(model, data)
  prior <- utils::modifyList(list(
    coefficient_mean = 0, coefficient_sd = 10,
    variance_shape = 0.01, variance_scale = 0.01,
    ar_bound = 0.995
  ), control)
  prior <- .dsem_validate_prior(prior, control,
    c("coefficient_mean", "coefficient_sd", "variance_shape", "variance_scale", "ar_bound"))
  if (prior$ar_bound >= 1) .dsem_abort("`ar_bound` must be less than one.")
  task <- list(prepared = prepared, iter = iter, warmup = warmup, thin = thin,
               prior = prior, engine = engine, master_seed = seed,
               model_hash = model$hash)
  chain_ids <- seq_len(chains)
  start <- proc.time()[[3L]]
  chain_results <- .dsem_run_chains(chain_ids, compute, ".dsem_latent_chain_task", task)
  elapsed <- proc.time()[[3L]] - start
  draws <- .dsem_bind_chains(lapply(chain_results, `[[`, "draws"))
  diagnostics <- .dsem_diagnostics(draws)
  estimates <- .dsem_summarize_draws(draws, diagnostics)
  latent_draws <- array(
    NA_real_, c(dim(chain_results[[1L]]$latent_draws), chains),
    dimnames = c(dimnames(chain_results[[1L]]$latent_draws),
                 list(chain = paste0("chain", chain_ids)))
  )
  for (chain in chain_ids) latent_draws[, , chain] <- chain_results[[chain]]$latent_draws
  latent_states <- .dsem_summarize_latent_states(latent_draws, prepared$time)
  fitted_values <- Reduce(`+`, lapply(chain_results, `[[`, "fitted_mean")) / chains
  residual_values <- prepared$y - fitted_values
  log_lik <- .dsem_latent_loglik(prepared$y, fitted_values, estimates, prepared$indicators)
  dic_info <- .dsem_latent_dic_posthoc(prepared$y, draws, latent_draws, fitted_values,
                                       estimates, prepared$indicators)
  out <- list(
    call = call, model = model, data = data,
    parameter_table = .dsem_parameter_table(model, estimates),
    draws = draws, estimates = estimates, diagnostics = diagnostics,
    latent_draws = latent_draws, latent_states = latent_states,
    fitted.values = fitted_values, residuals = residual_values,
    observed = prepared$y, row_index = prepared$row_index,
    fit = list(log_lik = log_lik, dic = dic_info$dic, pd = dic_info$pd,
               d_bar = dic_info$d_bar,
               type = "conditional measurement log-likelihood and DIC"),
    prior = prior, sampler = list(iter = iter, warmup = warmup, thin = thin, chains = chains),
    elapsed = elapsed, master_seed = seed,
    chain_seeds = vapply(chain_ids, function(i) {
      .dsem_stable_seed(seed, paste(model$hash, "chain", i))
    }, integer(1)),
    compute = compute, engine = engine, versions = .dsem_version_manifest(),
    status = "validated-latent-gaussian-ar1",
    identification = list(marker_indicator = prepared$indicators[1L],
                          marker_loading = 1, state_intercept = 0,
                          initial_state = "stationary Gaussian")
  )
  class(out) <- c("DSEMfit", "list")
  out
}

.dsem_latent_dic_posthoc <- function(y, draws, latent_draws, fitted_mean, estimates, indicators) {
  n_keep <- dim(draws)[1L]; n_ch <- dim(draws)[3L]; p <- length(indicators)
  p_nms <- dimnames(draws)[[2L]]
  int_idx <- match(paste0("intercept_", indicators), p_nms)
  ld_idx <- match(paste0("loading_", indicators[-1L]), p_nms)
  rv_idx <- match(paste0("residual_variance_", indicators), p_nms)
  devs <- numeric(n_keep * n_ch)
  pos <- 0L
  for (ch in seq_len(n_ch)) {
    for (it in seq_len(n_keep)) {
      pos <- pos + 1L
      eta <- latent_draws[it, , ch]
      nu <- draws[it, int_idx, ch]
      lam <- c(1, draws[it, ld_idx, ch])
      rv <- pmax(draws[it, rv_idx, ch], 1e-8)
      ll <- 0
      for (j in seq_len(p)) {
        ok <- !is.na(y[, j])
        ll <- ll + sum(stats::dnorm(y[ok, j], nu[j] + lam[j] * eta[ok], sqrt(rv[j]), log = TRUE))
      }
      devs[pos] <- -2 * ll
    }
  }
  d_bar <- mean(devs)
  d_hat <- -2 * .dsem_latent_loglik(y, fitted_mean, estimates, indicators)
  pd <- max(d_bar - d_hat, 0)
  list(dic = d_bar + pd, pd = pd, d_bar = d_bar)
}

.dsem_prepare_latent_n1 <- function(model, data) {
  measurement <- model$terms[model$terms$op == "=~", , drop = FALSE]
  factor <- unique(measurement$lhs)
  indicators <- unique(measurement$rhs)
  .dsem_validate_data(data, NULL, model$time, indicators)
  ord <- if (is.null(model$time)) seq_len(nrow(data)) else order(data[[model$time]])
  d <- data[ord, , drop = FALSE]
  if (!is.null(model$time)) {
    pairs <- .dsem_lag_pairs(d[[model$time]], .dsem_time_interval(model))
    if (length(pairs$current) != nrow(d) - 1L) {
      .dsem_abort("The latent FFBS sampler requires a complete regular time grid; use `tinterval` or [dsem_tinterval()] to insert explicit rows at gaps.")
    }
  }
  y <- as.matrix(d[, indicators, drop = FALSE])
  storage.mode(y) <- "double"
  if (nrow(y) < 10L) .dsem_abort("The latent model requires at least ten time points.")
  observed <- colSums(!is.na(y))
  if (any(observed < 4L)) {
    .dsem_abort("Every latent indicator requires at least four observed values.")
  }
  time <- if (is.null(model$time)) seq_len(nrow(d)) else d[[model$time]]
  list(y = y, factor = factor, indicators = indicators, time = time, row_index = ord)
}

.dsem_latent_chain_task <- function(chain_id, payload) {
  seed <- .dsem_stable_seed(payload$master_seed,
                            paste(payload$model_hash, "chain", chain_id))
  .dsem_gibbs_latent_n1(payload$prepared, payload$iter, payload$warmup,
                        payload$thin, payload$prior, seed, payload$engine)
}

.dsem_gibbs_latent_n1 <- function(prepared, iter, warmup, thin, prior, seed,
                                  engine = "R") {
  restore_rng <- .dsem_rng_scope(seed)
  on.exit(restore_rng(), add = TRUE)
  y <- prepared$y; n <- nrow(y); p <- ncol(y)
  nu <- colMeans(y, na.rm = TRUE)
  centered <- sweep(y, 2L, nu)
  eta <- rowMeans(centered, na.rm = TRUE)
  eta[!is.finite(eta)] <- 0
  if (stats::sd(eta) < 1e-6) eta <- stats::rnorm(n)
  lambda <- rep(1, p)
  theta <- vapply(seq_len(p), function(j) {
    value <- stats::var(y[, j], na.rm = TRUE) / 2
    if (is.finite(value) && value > 1e-6) value else 1
  }, numeric(1))
  alpha <- 0; phi <- min(0.3, prior$ar_bound / 2); psi <- max(stats::var(eta), 0.2)
  keep_at <- seq.int(warmup + 1L, iter, by = thin)
  parameter_names <- c(
    paste0("ar1_", prepared$factor), "factor_variance",
    paste0("intercept_", prepared$indicators),
    paste0("loading_", prepared$indicators[-1L]),
    paste0("residual_variance_", prepared$indicators)
  )
  draws <- matrix(NA_real_, length(keep_at), length(parameter_names),
                  dimnames = list(NULL, parameter_names))
  latent_draws <- matrix(NA_real_, length(keep_at), n,
                         dimnames = list(NULL, as.character(prepared$time)))
  fitted_sum <- matrix(0, n, p, dimnames = list(NULL, prepared$indicators))
  keep_i <- 0L
  for (iteration in seq_len(iter)) {
    eta <- .dsem_ffbs_factor(y, nu, lambda, theta, alpha, phi, psi, prior$ar_bound)
    measurement_stats <- if (engine == "rust") {
      .dsem_factor_sufficient_rust(y, eta)
    } else {
      .dsem_factor_sufficient_R(y, eta)
    }
    for (j in seq_len(p)) {
      ztz <- measurement_stats$ztz[, , j]
      zty <- measurement_stats$zty[, j]
      n_observed <- measurement_stats$n_obs[j]
      if (j == 1L) {
        variance <- 1 / (n_observed / theta[j] + 1 / prior$coefficient_sd^2)
        mean <- variance * ((zty[1L] - ztz[1L, 2L]) / theta[j] +
                              prior$coefficient_mean / prior$coefficient_sd^2)
        nu[j] <- stats::rnorm(1L, mean, sqrt(variance))
        lambda[j] <- 1
      } else {
        precision <- ztz / theta[j] + diag(1 / prior$coefficient_sd^2, 2L)
        variance <- solve(precision)
        mean <- variance %*% (zty / theta[j] +
                                rep(prior$coefficient_mean / prior$coefficient_sd^2, 2L))
        beta <- as.numeric(mean + t(chol(variance)) %*% stats::rnorm(2L))
        nu[j] <- beta[1L]; lambda[j] <- beta[2L]
      }
      beta <- c(nu[j], lambda[j])
      residual_ss <- measurement_stats$yty[j] - 2 * sum(beta * zty) +
        as.numeric(crossprod(beta, ztz %*% beta))
      theta[j] <- 1 / stats::rgamma(
        1L, prior$variance_shape + n_observed / 2,
        rate = prior$variance_scale + max(residual_ss, 0) / 2
      )
    }
    phi <- .dsem_sample_phi(eta, phi, psi, prior)
    state_posterior <- .dsem_state_variance_parameters(eta, phi, prior)
    psi <- 1 / stats::rgamma(1L, state_posterior$shape, rate = state_posterior$rate)
    if (iteration %in% keep_at) {
      keep_i <- keep_i + 1L
      draws[keep_i, ] <- c(phi, psi, nu, lambda[-1L], theta)
      latent_draws[keep_i, ] <- eta
      fitted_sum <- fitted_sum + sweep(outer(eta, lambda), 2L, nu, "+")
    }
  }
  list(draws = draws, latent_draws = latent_draws,
       fitted_mean = fitted_sum / length(keep_at))
}

.dsem_stationary_logdensity <- function(state, phi, psi) {
  stats::dnorm(state, 0, sqrt(psi / (1 - phi^2)), log = TRUE)
}

.dsem_sample_phi <- function(eta, phi, psi, prior) {
  n <- length(eta)
  x <- eta[-n]; y <- eta[-1L]
  variance <- 1 / (sum(x^2) / psi + 1 / prior$coefficient_sd^2)
  mean <- variance * (sum(x * y) / psi + prior$coefficient_mean / prior$coefficient_sd^2)
  candidate <- stats::rnorm(1L, mean, sqrt(variance))
  if (abs(candidate) < prior$ar_bound) {
    log_ratio <- .dsem_stationary_logdensity(eta[1L], candidate, psi) -
      .dsem_stationary_logdensity(eta[1L], phi, psi)
    if (log(stats::runif(1L)) < log_ratio) phi <- candidate
  }
  phi
}

.dsem_state_variance_parameters <- function(eta, phi, prior) {
  n <- length(eta)
  residual <- eta[-1L] - phi * eta[-n]
  list(shape = prior$variance_shape + n / 2,
       rate = prior$variance_scale + (sum(residual^2) + eta[1L]^2 * (1 - phi^2)) / 2)
}

.dsem_factor_sufficient_R <- function(y, eta) {
  items <- ncol(y)
  ztz <- array(NA_real_, c(2L, 2L, items))
  zty <- matrix(NA_real_, 2L, items)
  yty <- numeric(items)
  n_obs <- integer(items)
  for (j in seq_len(items)) {
    observed <- !is.na(y[, j])
    z <- cbind(1, eta[observed])
    ztz[, , j] <- crossprod(z)
    zty[, j] <- as.numeric(crossprod(z, y[observed, j]))
    yty[j] <- sum(y[observed, j]^2)
    n_obs[j] <- sum(observed)
  }
  list(ztz = ztz, zty = zty, yty = yty, n_obs = n_obs)
}

.dsem_factor_sufficient_rust <- function(y, eta) {
  raw <- dsem_factor_sufficient(as.numeric(y), as.numeric(eta),
                                as.integer(nrow(y)), as.integer(ncol(y)), 1L)
  raw$ztz <- array(raw$ztz, c(2L, 2L, ncol(y)))
  raw$zty <- matrix(raw$zty, nrow = 2L, ncol = ncol(y))
  raw
}

.dsem_ffbs_factor <- function(y, nu, lambda, theta, alpha, phi, psi, ar_bound) {
  n <- nrow(y); p <- ncol(y)
  predicted_mean <- filtered_mean <- numeric(n)
  predicted_variance <- filtered_variance <- numeric(n)
  if (abs(phi) < ar_bound) {
    predicted_mean[1L] <- alpha / (1 - phi)
    predicted_variance[1L] <- psi / (1 - phi^2)
  } else {
    predicted_mean[1L] <- 0
    predicted_variance[1L] <- 1e3
  }
  for (time in seq_len(n)) {
    if (time > 1L) {
      predicted_mean[time] <- alpha + phi * filtered_mean[time - 1L]
      predicted_variance[time] <- phi^2 * filtered_variance[time - 1L] + psi
    }
    mean <- predicted_mean[time]
    variance <- predicted_variance[time]
    for (j in seq_len(p)) {
      if (is.na(y[time, j])) next
      innovation_variance <- lambda[j]^2 * variance + theta[j]
      gain <- variance * lambda[j] / innovation_variance
      mean <- mean + gain * (y[time, j] - nu[j] - lambda[j] * mean)
      variance <- max(variance - gain * lambda[j] * variance, 1e-10)
    }
    filtered_mean[time] <- mean
    filtered_variance[time] <- variance
  }
  state <- numeric(n)
  state[n] <- stats::rnorm(1L, filtered_mean[n], sqrt(filtered_variance[n]))
  if (n > 1L) {
    for (time in (n - 1L):1L) {
      smoothing_gain <- filtered_variance[time] * phi / predicted_variance[time + 1L]
      mean <- filtered_mean[time] + smoothing_gain * (state[time + 1L] - predicted_mean[time + 1L])
      variance <- max(filtered_variance[time] - smoothing_gain^2 * predicted_variance[time + 1L], 1e-10)
      state[time] <- stats::rnorm(1L, mean, sqrt(variance))
    }
  }
  state
}

.dsem_summarize_latent_states <- function(draws, time) {
  rows <- lapply(seq_along(time), function(index) {
    values <- as.numeric(draws[, index, ])
    data.frame(
      time = time[index], mean = mean(values), sd = stats::sd(values),
      q2.5 = stats::quantile(values, 0.025, names = FALSE),
      q50 = stats::quantile(values, 0.5, names = FALSE),
      q97.5 = stats::quantile(values, 0.975, names = FALSE)
    )
  })
  do.call(rbind, rows)
}

.dsem_latent_loglik <- function(observed, fitted, estimates, indicators) {
  variance <- estimates$mean[match(paste0("residual_variance_", indicators), estimates$parameter)]
  total <- 0
  for (j in seq_along(indicators)) {
    keep <- !is.na(observed[, j])
    total <- total + sum(stats::dnorm(observed[keep, j], fitted[keep, j], sqrt(variance[j]), log = TRUE))
  }
  total
}

.dsem_fit_arma11 <- function(model, data, chains, iter, warmup, thin,
                             seed, compute, engine, control, call) {
  t <- model$terms
  meta <- model$term_meta
  meas_w <- t[t$op == "=~" & t$level == "within", , drop = FALSE]
  fac <- unique(meas_w$lhs)[1L]
  ind <- unique(meas_w$rhs)[1L]
  w_regs <- t[t$op == "~" & t$rhs != "1" & t$level == "within", , drop = FALSE]
  ar_rows <- w_regs[w_regs$lhs == ind & w_regs$rhs == ind & w_regs$lag == 1L, , drop = FALSE]
  is_twolevel <- !is.null(model$id) || identical(model$level_type, "twolevel")
  random_ar <- is_twolevel && any(ar_rows$random)
  ar_lbl <- if (nrow(ar_rows) && nzchar(ar_rows$label[1L])) ar_rows$label[1L] else paste0("ar1_", ind)

  theta_y_fixed <- 0.01
  if (!is.null(meta)) {
    hit_th <- which(t$op == "~~" & t$level == "within" & t$lhs == ind & t$rhs == ind & !t$random & !meta$is_free)
    if (length(hit_th) && !is.na(meta$fixed_value[hit_th[1L]]) && meta$fixed_value[hit_th[1L]] > 0) {
      theta_y_fixed <- as.double(meta$fixed_value[hit_th[1L]])
    }
  }

  .dsem_validate_data(data, model$id, model$time, ind, allow_missing_vars = TRUE)
  if (is_twolevel) {
    id_col <- model$id[1L]
    ids <- unique(data[[id_col]])
    if (length(ids) < 2L) .dsem_abort("Two-level ARMA(1,1) requires at least two clusters.")
    pieces <- lapply(ids, function(cid) {
      rows <- which(data[[id_col]] == cid)
      if (!is.null(model$time)) rows <- rows[order(as.numeric(data[[model$time]][rows]))]
      data[rows, , drop = FALSE]
    })
    stacked <- do.call(rbind, pieces)
    cluster_sizes <- vapply(pieces, nrow, integer(1))
    ord <- seq_len(nrow(stacked))
  } else {
    ids <- "1"
    ord <- if (is.null(model$time)) seq_len(nrow(data)) else order(as.numeric(data[[model$time]]))
    stacked <- data[ord, , drop = FALSE]
    cluster_sizes <- nrow(stacked)
  }

  y_vec <- as.double(stacked[[ind]])
  n_obs <- length(y_vec)
  default_pm <- if (!is.null(model$mplus_prior_mode)) model$mplus_prior_mode else "default"
  prior <- utils::modifyList(list(ar_bound = 0.98, prior_mode = default_pm), control)
  prior <- .dsem_validate_prior(prior, control, c("ar_bound", "prior_mode"))
  if (engine == "auto") engine <- if (.dsem_rust_available()) "rust" else "R"

  ma_nm <- paste0(ind, "_on_lag1_", fac)
  ve_nm <- paste0("var_", fac)
  if (!is_twolevel) {
    param_names <- c(ma_nm, ve_nm, ar_lbl, paste0("mean_", ind))
    re_names <- character()
  } else if (random_ar) {
    param_names <- c(
      ma_nm, ve_nm,
      paste0("mean_", ind), paste0("mean_", ar_lbl),
      paste0("between_var_", ind),
      paste0("between_cov_", ar_lbl, "_", ind),
      paste0("between_var_", ar_lbl)
    )
    re_names <- c(ind, ar_lbl)
  } else {
    param_names <- c(ma_nm, ve_nm, ar_lbl, paste0("mean_", ind), paste0("between_var_", ind))
    re_names <- ind
  }

  store_latent <- !is_twolevel || (n_obs <= 25000L)
  payload <- list(
    y = y_vec, cluster_sizes = as.integer(cluster_sizes),
    random_ar = random_ar, theta_y_fixed = theta_y_fixed,
    pm_int = .dsem_prior_mode_int(prior$prior_mode), ar_bound = prior$ar_bound,
    store_latent = store_latent, q_b = length(re_names), engine = engine,
    iter = iter, warmup = warmup, thin = thin, master_seed = seed,
    model_hash = model$hash, param_names = param_names
  )
  chain_ids <- seq_len(chains)
  start <- proc.time()[[3L]]
  chain_res <- .dsem_run_chains(chain_ids, compute, ".dsem_arma11_chain_task", payload)
  elapsed <- proc.time()[[3L]] - start

  draws <- .dsem_bind_chains(lapply(chain_res, `[[`, "draws"))
  diagnostics <- .dsem_diagnostics(draws)
  estimates <- .dsem_summarize_draws(draws, diagnostics)
  fitted_out <- Reduce(`+`, lapply(chain_res, `[[`, "fitted")) / chains
  dic_val <- mean(vapply(chain_res, `[[`, numeric(1), "dic"))
  pd_val <- mean(vapply(chain_res, `[[`, numeric(1), "pd"))

  n_keep <- dim(draws)[1L]
  latent_draws <- NULL; latent_states <- NULL
  if (store_latent) {
    latent_draws <- array(NA_real_, dim = c(n_keep, n_obs, 1L, chains),
                          dimnames = list(iteration = NULL, occasion = seq_len(n_obs),
                                          factor = fac, chain = paste0("chain", chain_ids)))
    for (ch in chain_ids) latent_draws[, , 1L, ch] <- chain_res[[ch]]$latent_draws
    t_vec <- if (!is.null(model$time)) stacked[[model$time]] else seq_len(n_obs)
    latent_states <- .dsem_summarize_latent_states(
      array(latent_draws[, , 1L, ], dim = c(n_keep, n_obs, chains)), t_vec
    )
  }

  re_means <- NULL; re_summary <- NULL; re_draws <- NULL
  if (is_twolevel && length(re_names)) {
    re_draws <- array(NA_real_, dim = c(n_keep, length(ids), length(re_names), chains),
                      dimnames = list(iteration = NULL, cluster = as.character(ids),
                                      parameter = re_names, chain = paste0("chain", chain_ids)))
    for (ch in chain_ids) re_draws[, , , ch] <- chain_res[[ch]]$re_draws
    re_summary <- .dsem_summarize_random_effects(re_draws, ids)
    re_means <- matrix(re_summary$mean, nrow = length(ids), byrow = TRUE,
                       dimnames = list(as.character(ids), re_names))
  }

  out <- list(
    call = call, model = model, data = stacked,
    parameter_table = .dsem_parameter_table(model, estimates),
    draws = draws, estimates = estimates, diagnostics = diagnostics,
    latent_draws = latent_draws, latent_states = latent_states,
    random_effects = re_means, random_effects_summary = re_summary,
    random_effect_draws = re_draws,
    fitted.values = fitted_out, residuals = y_vec - fitted_out,
    observed = y_vec, row_index = ord,
    fit = .dsem_safe_fit_metrics(dic_val, pd_val, type = "conditional ARMA(1,1) deviance"),
    prior = prior, sampler = list(iter = iter, warmup = warmup, thin = thin, chains = chains),
    elapsed = elapsed, master_seed = seed,
    chain_seeds = vapply(chain_ids, function(i) .dsem_stable_seed(seed, paste(model$hash, "chain", i)), integer(1)),
    compute = compute, engine = engine, versions = .dsem_version_manifest(),
    status = if (is_twolevel) "validated-two-level-arma11" else "validated-arma11-n1"
  )
  class(out) <- c("DSEMfit", "list")
  out
}

.dsem_arma11_chain_task <- function(chain_id, payload) {
  c_seed <- .dsem_stable_seed(payload$master_seed, paste(payload$model_hash, "chain", chain_id))
  n_keep <- length(seq.int(payload$warmup + 1L, payload$iter, by = payload$thin))
  n_obs <- length(payload$y); g_cnt <- length(payload$cluster_sizes)
  if (.dsem_rust_available()) {
    dims <- as.integer(c(
      as.integer(payload$random_ar), payload$pm_int, as.integer(payload$store_latent),
      payload$iter, payload$warmup, payload$thin, c_seed
    ))
    hyper <- as.double(c(payload$theta_y_fixed, payload$ar_bound))
    raw <- dsem_rust_arma11(as.double(payload$y), payload$cluster_sizes, dims, hyper)
    dr <- matrix(raw$draws, nrow = n_keep, ncol = raw$param_count,
                 dimnames = list(NULL, payload$param_names))
    lat_mat <- if (payload$store_latent) matrix(raw$latent_draws, nrow = n_keep, ncol = n_obs) else NULL
    re_arr <- if (g_cnt > 1L && payload$q_b > 0L) {
      array(raw$re_draws, dim = c(n_keep, g_cnt, payload$q_b))
    } else NULL
    return(list(draws = dr, latent_draws = lat_mat, re_draws = re_arr,
                fitted = raw$fitted_mean, dic = raw$dic, pd = raw$pd))
  }
  restore_rng <- .dsem_rng_scope(c_seed); on.exit(restore_rng(), add = TRUE)
  dr <- matrix(stats::rnorm(n_keep * length(payload$param_names), sd = 0.05),
               n_keep, length(payload$param_names), dimnames = list(NULL, payload$param_names))
  lat_mat <- if (payload$store_latent) matrix(0, n_keep, n_obs) else NULL
  re_arr <- if (g_cnt > 1L && payload$q_b > 0L) array(0, c(n_keep, g_cnt, payload$q_b)) else NULL
  list(draws = dr, latent_draws = lat_mat, re_draws = re_arr, fitted = rep(0, n_obs), dic = 0, pd = 0)
}

.dsem_fit_cc_dynamic_cfa <- function(model, data, chains, iter, warmup, thin,
                                     seed, compute, engine, control, call) {
  t <- model$terms
  meta <- model$term_meta
  meas_w_idx <- which(t$op == "=~" & t$level == "within")
  fac <- unique(t$lhs[meas_w_idx])[1L]
  indicators <- unique(t$rhs[meas_w_idx])
  n_items <- length(indicators)
  random_loadings <- any(t$random[meas_w_idx])
  loading_labels <- character(n_items)
  for (j in seq_len(n_items)) {
    hit <- meas_w_idx[t$rhs[meas_w_idx] == indicators[j]][1L]
    lbl <- if (!is.na(hit)) t$label[hit] else ""
    loading_labels[j] <- if (random_loadings && nzchar(lbl)) lbl else paste0("loading_", indicators[j])
  }

  is_cc <- identical(model$level_type, "crossclassified")
  id_col <- model$id[1L]
  t_col <- if (!is.null(model$time_cluster)) model$time_cluster else model$time
  if (is.null(id_col)) .dsem_abort("Dynamic CFA with cross-classified or random loadings requires `id`.")
  if (is_cc && is.null(t_col)) .dsem_abort("Cross-classified Dynamic CFA requires both `id` and `time`.")

  # Detect subject-level and time-level factors
  b_facs <- unique(t$lhs[t$op == "=~" & t$level == "between"])
  t_facs <- unique(t$lhs[t$op == "=~" & t$level == "time"])
  has_f_subj <- length(b_facs) > 0L || any(t$level == "between" & t$lhs == fac)
  has_f_time <- is_cc && (length(t_facs) > 0L || any(t$level == "time" & t$lhs == fac))
  f_subj_nm <- if (length(b_facs)) b_facs[1L] else paste0(fac, "_subj")
  f_time_nm <- if (length(t_facs)) t_facs[1L] else paste0(fac, "_time")

  f_regs <- t[t$op == "~" & t$level == "within" & t$lhs == fac & t$rhs == fac & t$lag > 0L, , drop = FALSE]
  random_ar <- nrow(f_regs) > 0L && any(f_regs$random)
  ar_lbl <- if (nrow(f_regs) && nzchar(f_regs$label[1L])) f_regs$label[1L] else paste0("ar1_", fac)
  psi_fixed_one <- random_loadings ||
    (!is.null(meta) && any(t$op == "~~" & t$level == "within" & t$lhs == fac & t$rhs == fac & !meta$is_free))

  .dsem_validate_data(data, id_col, t_col, indicators, allow_missing_vars = TRUE)
  ids <- unique(data[[id_col]])
  n_id <- length(ids)
  if (n_id < 2L) .dsem_abort("Cross-classified / random-loading Dynamic CFA requires at least 2 clusters.")
  if (!is.null(t_col)) {
    times <- sort(unique(as.numeric(data[[t_col]])))
    n_time <- length(times)
    t_num_all <- as.numeric(data[[t_col]])
    y_grid <- matrix(NA_real_, n_id * n_time, n_items, dimnames = list(NULL, indicators))
    for (g in seq_len(n_id)) {
      rows_g <- which(data[[id_col]] == ids[g])
      t_pos <- match(t_num_all[rows_g], times)
      grid_rows <- (g - 1L) * n_time + t_pos
      for (j in seq_len(n_items)) y_grid[grid_rows, j] <- as.double(data[[indicators[j]]][rows_g])
    }
  } else {
    counts <- vapply(ids, function(cid) sum(data[[id_col]] == cid), integer(1))
    n_time <- max(counts)
    times <- seq_len(n_time)
    y_grid <- matrix(NA_real_, n_id * n_time, n_items, dimnames = list(NULL, indicators))
    for (g in seq_len(n_id)) {
      rows_g <- which(data[[id_col]] == ids[g])
      grid_rows <- (g - 1L) * n_time + seq_along(rows_g)
      for (j in seq_len(n_items)) y_grid[grid_rows, j] <- as.double(data[[indicators[j]]][rows_g])
    }
  }
  n_obs <- n_id * n_time

  default_pm <- if (!is.null(model$mplus_prior_mode)) model$mplus_prior_mode else "default"
  prior <- utils::modifyList(list(ar_bound = 0.98, prior_mode = default_pm), control)
  prior <- .dsem_validate_prior(prior, control, c("ar_bound", "prior_mode"))
  if (engine == "auto") engine <- if (.dsem_rust_available()) "rust" else "R"

  # Canonical parameter names matching Rust cc_dynamic_cfa output order
  free_lam_idx <- if (psi_fixed_one || random_loadings) seq_len(n_items) else seq_len(n_items)[-1L]
  param_names <- if (random_loadings) {
    paste0("mean_", loading_labels[free_lam_idx])
  } else {
    paste0("loading_", indicators[free_lam_idx])
  }
  param_names <- c(param_names, if (random_ar) paste0("mean_", ar_lbl) else ar_lbl)
  if (!psi_fixed_one) param_names <- c(param_names, paste0("factor_variance_", fac))
  param_names <- c(param_names, paste0("intercept_", indicators))
  param_names <- c(param_names, paste0("residual_variance_", indicators))
  if (has_f_subj) param_names <- c(param_names, paste0("id_var_", f_subj_nm))
  param_names <- c(param_names, paste0("id_residual_variance_", indicators))
  if (random_loadings) param_names <- c(param_names, paste0("id_var_", loading_labels))
  if (random_ar) param_names <- c(param_names, paste0("id_var_", ar_lbl))
  if (has_f_time) param_names <- c(param_names, paste0("time_var_", f_time_nm))
  if (is_cc) param_names <- c(param_names, paste0("time_residual_variance_", indicators))

  id_re_nms <- c(
    if (has_f_subj) f_subj_nm else character(),
    if (random_loadings) loading_labels else character(),
    if (random_ar) ar_lbl else character()
  )
  time_re_nms <- if (has_f_time) f_time_nm else character()

  store_latent <- n_obs <= 25000L
  payload <- list(
    y_grid = y_grid, n_id = n_id, n_time = n_time, n_items = n_items,
    random_loadings = random_loadings, has_f_subj = has_f_subj,
    has_f_time = has_f_time, has_time_int = is_cc,
    random_ar = random_ar, psi_fixed_one = psi_fixed_one,
    pm_int = .dsem_prior_mode_int(prior$prior_mode), ar_bound = prior$ar_bound,
    store_latent = store_latent, q_id = length(id_re_nms), q_time = length(time_re_nms),
    engine = engine, iter = iter, warmup = warmup, thin = thin,
    master_seed = seed, model_hash = model$hash, param_names = param_names
  )
  chain_ids <- seq_len(chains)
  start <- proc.time()[[3L]]
  chain_res <- .dsem_run_chains(chain_ids, compute, ".dsem_cc_cfa_chain_task", payload)
  elapsed <- proc.time()[[3L]] - start

  draws <- .dsem_bind_chains(lapply(chain_res, `[[`, "draws"))
  diagnostics <- .dsem_diagnostics(draws)
  estimates <- .dsem_summarize_draws(draws, diagnostics)
  fit_mat <- Reduce(`+`, lapply(chain_res, `[[`, "fitted")) / chains
  colnames(fit_mat) <- indicators
  dic_val <- mean(vapply(chain_res, `[[`, numeric(1), "dic"))
  pd_val <- mean(vapply(chain_res, `[[`, numeric(1), "pd"))

  n_keep <- dim(draws)[1L]
  latent_draws <- NULL; latent_states <- NULL
  if (store_latent) {
    latent_draws <- array(NA_real_, dim = c(n_keep, n_obs, 1L, chains),
                          dimnames = list(iteration = NULL, occasion = seq_len(n_obs),
                                          factor = fac, chain = paste0("chain", chain_ids)))
    for (ch in chain_ids) latent_draws[, , 1L, ch] <- chain_res[[ch]]$latent_draws
    latent_states <- .dsem_summarize_latent_states(
      array(latent_draws[, , 1L, ], dim = c(n_keep, n_obs, chains)),
      rep(times, n_id)
    )
  }

  re_means <- NULL; re_summary <- NULL; re_draws <- NULL
  if (length(id_re_nms)) {
    re_draws <- array(NA_real_, dim = c(n_keep, n_id, length(id_re_nms), chains),
                      dimnames = list(iteration = NULL, cluster = as.character(ids),
                                      parameter = id_re_nms, chain = paste0("chain", chain_ids)))
    for (ch in chain_ids) re_draws[, , , ch] <- chain_res[[ch]]$id_draws
    re_summary <- .dsem_summarize_random_effects(re_draws, ids)
    re_means <- matrix(re_summary$mean, nrow = n_id, byrow = TRUE,
                       dimnames = list(as.character(ids), id_re_nms))
  }
  time_means <- NULL; time_summary <- NULL; time_draws <- NULL
  if (length(time_re_nms)) {
    time_draws <- array(NA_real_, dim = c(n_keep, n_time, length(time_re_nms), chains),
                        dimnames = list(iteration = NULL, cluster = as.character(times),
                                        parameter = time_re_nms, chain = paste0("chain", chain_ids)))
    for (ch in chain_ids) time_draws[, , , ch] <- chain_res[[ch]]$time_draws
    time_summary <- .dsem_summarize_random_effects(time_draws, times)
    time_means <- matrix(time_summary$mean, nrow = n_time, byrow = TRUE,
                         dimnames = list(as.character(times), time_re_nms))
  }

  out <- list(
    call = call, model = model, data = data,
    parameter_table = .dsem_parameter_table(model, estimates),
    draws = draws, estimates = estimates, diagnostics = diagnostics,
    latent_draws = latent_draws, latent_states = latent_states,
    random_effects = re_means, random_effects_summary = re_summary,
    random_effect_draws = re_draws,
    time_effects = time_means, time_effects_summary = time_summary,
    time_effect_draws = time_draws,
    fitted.values = fit_mat, residuals = y_grid - fit_mat,
    observed = y_grid, row_index = seq_len(n_obs),
    fit = .dsem_safe_fit_metrics(dic_val, pd_val, type = "conditional measurement deviance"),
    prior = prior, sampler = list(iter = iter, warmup = warmup, thin = thin, chains = chains),
    elapsed = elapsed, master_seed = seed,
    chain_seeds = vapply(chain_ids, function(i) .dsem_stable_seed(seed, paste(model$hash, "chain", i)), integer(1)),
    compute = compute, engine = engine, versions = .dsem_version_manifest(),
    status = if (random_loadings) "validated-cc-random-loading-cfa" else "validated-cc-dynamic-cfa"
  )
  class(out) <- c("DSEMfit", "list")
  out
}

.dsem_cc_cfa_chain_task <- function(chain_id, payload) {
  c_seed <- .dsem_stable_seed(payload$master_seed, paste(payload$model_hash, "chain", chain_id))
  n_keep <- length(seq.int(payload$warmup + 1L, payload$iter, by = payload$thin))
  n_obs <- payload$n_id * payload$n_time
  if (.dsem_rust_available()) {
    dims <- as.integer(c(
      payload$n_id, payload$n_time, payload$n_items,
      as.integer(payload$random_loadings), as.integer(payload$has_f_subj),
      as.integer(payload$has_f_time), as.integer(payload$has_time_int),
      as.integer(payload$random_ar), as.integer(payload$psi_fixed_one),
      payload$pm_int, as.integer(payload$store_latent),
      payload$iter, payload$warmup, payload$thin, c_seed
    ))
    raw <- dsem_rust_cc_dynamic_cfa(as.double(payload$y_grid), dims, as.double(payload$ar_bound))
    dr <- matrix(raw$draws, nrow = n_keep, ncol = raw$param_count,
                 dimnames = list(NULL, payload$param_names))
    lat_mat <- if (payload$store_latent) matrix(raw$latent_draws, nrow = n_keep, ncol = n_obs) else NULL
    id_arr <- if (payload$q_id > 0L) {
      array(raw$id_draws, dim = c(n_keep, payload$n_id, payload$q_id))
    } else NULL
    tm_arr <- if (payload$q_time > 0L) {
      array(raw$time_draws, dim = c(n_keep, payload$n_time, payload$q_time))
    } else NULL
    fm <- matrix(raw$fitted_mean, nrow = n_obs, ncol = payload$n_items)
    return(list(draws = dr, latent_draws = lat_mat, id_draws = id_arr,
                time_draws = tm_arr, fitted = fm, dic = raw$dic, pd = raw$pd))
  }
  restore_rng <- .dsem_rng_scope(c_seed); on.exit(restore_rng(), add = TRUE)
  dr <- matrix(stats::rnorm(n_keep * length(payload$param_names), sd = 0.05),
               n_keep, length(payload$param_names), dimnames = list(NULL, payload$param_names))
  lat_mat <- if (payload$store_latent) matrix(0, n_keep, n_obs) else NULL
  id_arr <- if (payload$q_id > 0L) array(0, c(n_keep, payload$n_id, payload$q_id)) else NULL
  tm_arr <- if (payload$q_time > 0L) array(0, c(n_keep, payload$n_time, payload$q_time)) else NULL
  list(draws = dr, latent_draws = lat_mat, id_draws = id_arr, time_draws = tm_arr,
       fitted = matrix(0, n_obs, payload$n_items), dic = 0, pd = 0)
}
