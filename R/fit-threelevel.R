.dsem_fit_three_level <- function(model, data, chains, iter, warmup, thin,
                                  seed, compute, engine, control, call) {
  if (length(model$id) != 2L) {
    .dsem_abort("Three-level DSEM requires `id = c(level2_col, level3_col)`.")
  }
  l2_col <- model$id[1L]
  l3_col <- model$id[2L]
  t <- model$terms
  meta <- model$term_meta
  w_regs <- .dsem_active_regs(t, meta, "within")
  b2_regs <- .dsem_active_regs(t, meta, "between_2")
  b3_regs <- .dsem_active_regs(t, meta, "between_3")
  w_vars <- t[t$op == "~~" & t$level == "within" & t$lhs == t$rhs, , drop = FALSE]

  outcomes <- unique(w_regs$lhs)
  d <- length(outcomes)
  lags <- sort(unique(w_regs$lag[w_regs$lag > 0L & w_regs$rhs %in% outcomes]))
  if (!length(lags)) lags <- 1L
  n_lags <- length(lags)
  exog_specs <- .dsem_build_exog_specs(w_regs, outcomes)
  exog_raw_vars <- unique(exog_specs$var)
  kx <- nrow(exog_specs)
  n_sl <- n_lags * d + kx

  shared_cov <- .dsem_detect_shared_cov_factor(model)
  shared_cov_loadings <- if (!is.null(shared_cov)) {
    as.double(shared_cov$loadings[outcomes])
  } else {
    numeric(0L)
  }
  has_logvf <- length(shared_cov_loadings) > 0L

  # Slot layout in Rust threelevel kernel: [d intercepts, d*n_sl slopes, d logv, optional logvf]
  q_tot <- d + d * n_sl + d + as.integer(has_logvf)
  slot_names <- character(q_tot)
  slot_default_names <- character(q_tot)
  slot_is_rand <- logical(q_tot)
  slot_names[seq_len(d)] <- outcomes
  slot_default_names[seq_len(d)] <- outcomes
  for (j in seq_len(d)) {
    for (s_i in seq_len(n_sl)) {
      pos <- d + (j - 1L) * n_sl + s_i
      if (s_i <= n_lags * d) {
        li <- (s_i - 1L) %/% d + 1L; mc <- (s_i - 1L) %% d + 1L
        hit <- which(w_regs$lhs == outcomes[j] & w_regs$rhs == outcomes[mc] & w_regs$lag == lags[li])
        def_nm <- if (d == 1L) paste0("lag", lags[li], "_", outcomes[mc]) else paste0(outcomes[j], "_on_lag", lags[li], "_", outcomes[mc])
      } else {
        k_ex <- s_i - n_lags * d
        ex_var <- exog_specs$var[k_ex]
        ex_lg <- exog_specs$lag[k_ex]
        ex_key <- exog_specs$key[k_ex]
        hit <- which(w_regs$lhs == outcomes[j] & w_regs$rhs == ex_var & w_regs$lag == ex_lg)
        def_nm <- if (d == 1L) ex_key else paste0(outcomes[j], "_on_", ex_key)
      }
      lbl <- if (length(hit)) w_regs$label[hit[1L]] else ""
      slot_names[pos] <- if (nzchar(lbl)) lbl else def_nm
      slot_default_names[pos] <- def_nm
      slot_is_rand[pos] <- length(hit) > 0L && any(w_regs$random[hit])
    }
    hit_lv <- which(w_vars$lhs == outcomes[j] & w_vars$random)
    lbl_lv <- if (length(hit_lv)) w_vars$label[hit_lv[1L]] else ""
    slot_names[d + d * n_sl + j] <- if (nzchar(lbl_lv)) lbl_lv else paste0("logv_", outcomes[j])
    slot_default_names[d + d * n_sl + j] <- paste0("logv_", outcomes[j])
    slot_is_rand[d + d * n_sl + j] <- length(hit_lv) > 0L
  }
  if (has_logvf) {
    slot_names[q_tot] <- shared_cov$logvf_label
    slot_default_names[q_tot] <- shared_cov$logvf_label
    slot_is_rand[q_tot] <- TRUE
  }

  # Determine which parameters vary at L2 and L3 (Asparouhov & Muthen, 2026b):
  # intercepts vary at both L2 and L3 by default; random slopes/logv/logvf vary at L3 by default,
  # and at L2 if declared in between_2 (and not fixed to 0).
  b2_mentioned <- unique(c(t$lhs[t$level == "between_2"], t$rhs[t$level == "between_2" & t$op == "~~"]))
  b3_mentioned <- unique(c(t$lhs[t$level == "between_3"], t$rhs[t$level == "between_3" & t$op == "~~"]))
  b2_fixed_zero <- character()
  b3_fixed_zero <- character()
  if (!is.null(meta) && nrow(meta) == nrow(t)) {
    fz2 <- which(t$level == "between_2" & t$op == "~~" & t$lhs == t$rhs & !meta$is_free & !is.na(meta$fixed_value) & abs(meta$fixed_value) < 1e-12)
    b2_fixed_zero <- unique(t$lhs[fz2])
    fz3 <- which(t$level == "between_3" & t$op == "~~" & t$lhs == t$rhs & !meta$is_free & !is.na(meta$fixed_value) & abs(meta$fixed_value) < 1e-12)
    b3_fixed_zero <- unique(t$lhs[fz3])
  }
  has_l2_re <- integer(q_tot)
  has_l3_re <- integer(q_tot)
  for (j in seq_len(d)) {
    if (!outcomes[j] %in% b2_fixed_zero) has_l2_re[j] <- 1L
    if (!outcomes[j] %in% b3_fixed_zero) has_l3_re[j] <- 1L
  }
  for (s_pos in (d + 1L):q_tot) {
    cands <- unique(c(slot_names[s_pos], slot_default_names[s_pos]))
    if (any(cands %in% b2_mentioned) && !any(cands %in% b2_fixed_zero)) has_l2_re[s_pos] <- 1L
    if ((slot_is_rand[s_pos] || any(cands %in% b3_mentioned)) && !any(cands %in% b3_fixed_zero)) {
      has_l3_re[s_pos] <- 1L
    }
  }

  # Detect non-random Level-2 AR(1) terms: e.g. `%BETWEEN day% y ON y&1;` or `level: 2\n y ~ lag(y, 1)`
  l2_ar_mask <- integer(d)
  for (j in seq_len(d)) {
    if (has_l2_re[j] == 1L &&
        any(b2_regs$lhs == outcomes[j] & b2_regs$rhs == outcomes[j] & b2_regs$lag > 0L)) {
      l2_ar_mask[j] <- 1L
    }
  }
  b2_exog_regs <- b2_regs[!(b2_regs$lag > 0L & b2_regs$rhs %in% outcomes), , drop = FALSE]
  exog_w2 <- setdiff(unique(b2_exog_regs$rhs), c(slot_names, slot_default_names))
  exog_w3 <- setdiff(unique(b3_regs$rhs), c(slot_names, slot_default_names))
  kw2 <- length(exog_w2); kw3 <- length(exog_w3)

  .dsem_validate_data(data, c(l2_col, l3_col), model$time,
                      unique(c(outcomes, exog_raw_vars, exog_w2, exog_w3)))

  l3_ids <- unique(data[[l3_col]])
  if (length(l3_ids) < 2L) .dsem_abort("Three-level DSEM requires at least two Level-3 clusters.")

  # Build contiguous L2 units nested within L3 clusters (sorted chronologically by L2 id when L2 AR is active)
  l2_keys <- paste(data[[l3_col]], data[[l2_col]], sep = "\r")
  u_l2_keys <- unique(l2_keys)
  if (any(l2_ar_mask == 1L)) {
    first_idx <- match(u_l2_keys, l2_keys)
    l3_ord_key <- match(data[[l3_col]][first_idx], l3_ids)
    l2_raw_key <- data[[l2_col]][first_idx]
    l2_ord_key <- if (is.numeric(l2_raw_key)) as.numeric(l2_raw_key) else as.character(l2_raw_key)
    u_l2_keys <- u_l2_keys[order(l3_ord_key, l2_ord_key)]
  }
  u_l2_labels <- vapply(u_l2_keys, function(k) sub("\r", ":", k, fixed = TRUE), character(1))
  if (length(u_l2_keys) <= length(l3_ids)) {
    .dsem_abort("Three-level DSEM requires multiple Level-2 units nested within Level-3 clusters.")
  }

  pieces <- vector("list", length(u_l2_keys))
  l3_of_l2 <- integer(length(u_l2_keys))
  w2_mat <- matrix(0, length(u_l2_keys), kw2, dimnames = list(NULL, exog_w2))
  for (c_i in seq_along(u_l2_keys)) {
    rows <- which(l2_keys == u_l2_keys[c_i])
    if (!is.null(model$time)) rows <- rows[order(as.numeric(data[[model$time]][rows]))]
    if (length(rows) <= max(lags) + 1L) {
      .dsem_abort("Every Level-2 unit requires at least %d ordered observations.", max(lags) + 2L)
    }
    pieces[[c_i]] <- data[rows, , drop = FALSE]
    l3_val <- data[[l3_col]][rows[1L]]
    l3_of_l2[c_i] <- match(l3_val, l3_ids) - 1L
    for (k in seq_len(kw2)) {
      v <- data[[exog_w2[k]]][rows]; v <- v[!is.na(v)]
      w2_mat[c_i, k] <- if (length(v)) as.double(v[1L]) else 0
    }
  }
  w3_mat <- matrix(0, length(l3_ids), kw3, dimnames = list(NULL, exog_w3))
  for (g_i in seq_along(l3_ids)) {
    rows <- which(data[[l3_col]] == l3_ids[g_i])
    for (k in seq_len(kw3)) {
      v <- data[[exog_w3[k]]][rows]; v <- v[!is.na(v)]
      w3_mat[g_i, k] <- if (length(v)) as.double(v[1L]) else 0
    }
  }

  stacked <- do.call(rbind, pieces)
  l2_sizes <- vapply(pieces, nrow, integer(1))
  n_obs <- nrow(stacked)
  y_mat <- as.matrix(stacked[, outcomes, drop = FALSE]); storage.mode(y_mat) <- "double"
  x_mat <- if (kx > 0L) {
    xm <- .dsem_build_exog_matrix(stacked, exog_specs, l2_sizes)
    for (k in seq_len(kx)) if (anyNA(xm[, k])) xm[is.na(xm[, k]), k] <- mean(xm[, k], na.rm = TRUE)
    xm
  } else matrix(0, n_obs, 0L)
  lc_x <- if (kx == 0L) {
    integer(0L)
  } else if (identical(model$centering, "latent")) {
    t_nm <- if (is.null(model$time)) "" else model$time
    as.integer(exog_specs$var != t_nm)
  } else {
    rep(0L, kx)
  }

  default_pm <- if (!is.null(model$mplus_prior_mode)) model$mplus_prior_mode else "default"
  prior <- utils::modifyList(list(beta_sd = 100, prior_mode = default_pm), control)
  prior <- .dsem_validate_prior(prior, control, c("beta_sd", "prior_mode"))
  if (engine == "auto") engine <- if (.dsem_rust_available()) "rust" else "R"

  l2_idx <- which(has_l2_re == 1L)
  l3_idx <- which(has_l3_re == 1L)
  l2_nms <- slot_names[l2_idx]
  l3_nms <- slot_names[l3_idx]
  re_l3_on_w3_mask <- matrix(0L, kw3, length(l3_nms))
  for (p3 in seq_along(l3_nms)) {
    cands <- unique(c(slot_names[l3_idx[p3]], slot_default_names[l3_idx[p3]]))
    for (k in seq_len(kw3)) {
      if (any(b3_regs$lhs %in% cands & b3_regs$rhs == exog_w3[k])) {
        re_l3_on_w3_mask[k, p3] <- 1L
      }
    }
  }
  re_l2_on_w2_mask <- matrix(0L, kw2, length(l2_nms))
  for (p2 in seq_along(l2_nms)) {
    cands <- unique(c(slot_names[l2_idx[p2]], slot_default_names[l2_idx[p2]]))
    for (k in seq_len(kw2)) {
      if (any(b2_exog_regs$lhs %in% cands & b2_exog_regs$rhs == exog_w2[k])) {
        re_l2_on_w2_mask[k, p2] <- 1L
      }
    }
  }

  param_names <- paste0("fixed_", slot_names)
  for (j in seq_len(d)) {
    if (l2_ar_mask[j] == 1L) param_names <- c(param_names, paste0("l2_ar1_", outcomes[j]))
  }
  for (p3 in seq_along(l3_nms)) {
    for (k in seq_len(kw3)) {
      if (re_l3_on_w3_mask[k, p3] == 1L) {
        param_names <- c(param_names, paste0("l3_", l3_nms[p3], "_on_", exog_w3[k]))
      }
    }
  }
  for (p2 in seq_along(l2_nms)) {
    for (k in seq_len(kw2)) {
      if (re_l2_on_w2_mask[k, p2] == 1L) {
        param_names <- c(param_names, paste0("l2_", l2_nms[p2], "_on_", exog_w2[k]))
      }
    }
  }
  if (length(l3_nms)) {
    for (c_i in seq_along(l3_nms)) for (r_i in c_i:length(l3_nms)) {
      param_names <- c(param_names, if (r_i == c_i) paste0("l3_var_", l3_nms[r_i]) else paste0("l3_cov_", l3_nms[r_i], "_", l3_nms[c_i]))
    }
  }
  if (length(l2_nms)) {
    for (c_i in seq_along(l2_nms)) for (r_i in c_i:length(l2_nms)) {
      param_names <- c(param_names, if (r_i == c_i) paste0("l2_var_", l2_nms[r_i]) else paste0("l2_cov_", l2_nms[r_i], "_", l2_nms[c_i]))
    }
  }
  any_rand_logv <- any(has_l2_re[(d + d * n_sl + 1L):(d + d * n_sl + d)] == 1L |
                         has_l3_re[(d + d * n_sl + 1L):(d + d * n_sl + d)] == 1L)
  use_full_sigma_w <- d > 1L && !any_rand_logv && !has_logvf
  if (use_full_sigma_w) {
    for (c_i in seq_len(d)) {
      if (c_i < d) {
        for (r_i in (c_i + 1L):d) {
          param_names <- c(param_names, paste0("within_cov_", outcomes[r_i], "_", outcomes[c_i]))
        }
      }
    }
  }

  masks <- as.integer(c(
    has_l2_re, has_l3_re, lc_x, l2_ar_mask,
    as.integer(re_l3_on_w3_mask), as.integer(re_l2_on_w2_mask)
  ))
  payload <- list(
    y = y_mat, x = x_mat, l2_sizes = as.integer(l2_sizes), l3_of_l2 = as.integer(l3_of_l2),
    g3_cnt = length(l3_ids), w2 = w2_mat, w3 = w3_mat, lags = as.integer(lags),
    masks = masks, shared_cov_loadings = shared_cov_loadings,
    d = d, kx = kx, kw2 = kw2, kw3 = kw3,
    q_l2 = length(l2_nms), q_l3 = length(l3_nms),
    res_dyn = any(t$residual), pm_int = .dsem_prior_mode_int(prior$prior_mode),
    beta_sd = prior$beta_sd, engine = engine,
    iter = iter, warmup = warmup, thin = thin, master_seed = seed,
    model_hash = model$hash, param_names = param_names
  )
  chain_ids <- seq_len(chains)
  start <- proc.time()[[3L]]
  chain_res <- .dsem_run_chains(chain_ids, compute, ".dsem_threelevel_chain_task", payload)
  elapsed <- proc.time()[[3L]] - start

  draws <- .dsem_bind_chains(lapply(chain_res, `[[`, "draws"))
  diagnostics <- .dsem_diagnostics(draws)
  estimates <- .dsem_summarize_draws(draws, diagnostics)
  n_keep <- dim(draws)[1L]
  l3_draws <- array(NA_real_, dim = c(n_keep, length(l3_ids), max(1L, length(l3_nms)), chains),
                    dimnames = list(iteration = NULL, cluster = as.character(l3_ids),
                                    parameter = if (length(l3_nms)) l3_nms else "none",
                                    chain = paste0("chain", chain_ids)))
  l2_draws <- array(NA_real_, dim = c(n_keep, length(u_l2_keys), max(1L, length(l2_nms)), chains),
                    dimnames = list(iteration = NULL, cluster = u_l2_labels,
                                    parameter = if (length(l2_nms)) l2_nms else "none",
                                    chain = paste0("chain", chain_ids)))
  for (ch in chain_ids) {
    l3_draws[, , , ch] <- chain_res[[ch]]$l3_draws
    l2_draws[, , , ch] <- chain_res[[ch]]$l2_draws
  }
  re_summary <- .dsem_summarize_random_effects(l3_draws, l3_ids)
  re_means <- matrix(re_summary$mean, nrow = length(l3_ids), byrow = TRUE,
                     dimnames = list(as.character(l3_ids), dimnames(l3_draws)$parameter))
  l2_summary <- .dsem_summarize_random_effects(l2_draws, u_l2_labels)
  l2_means <- matrix(l2_summary$mean, nrow = length(u_l2_labels), byrow = TRUE,
                     dimnames = list(u_l2_labels, dimnames(l2_draws)$parameter))

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
    random_effect_draws = l3_draws,
    l2_random_effects = l2_means, l2_random_effects_summary = l2_summary,
    l2_random_effect_draws = l2_draws,
    fit = .dsem_safe_fit_metrics(dic_val, pd_val),
    prior = prior, sampler = list(iter = iter, warmup = warmup, thin = thin, chains = chains),
    elapsed = elapsed, master_seed = seed,
    chain_seeds = vapply(chain_ids, function(i) .dsem_stable_seed(seed, paste(model$hash, "chain", i)), integer(1)),
    compute = compute, engine = engine, versions = .dsem_version_manifest(),
    status = "validated-three-level-dsem"
  )
  class(out) <- c("DSEMfit", "list")
  out
}

.dsem_threelevel_chain_task <- function(chain_id, payload) {
  c_seed <- .dsem_stable_seed(payload$master_seed, paste(payload$model_hash, "chain", chain_id))
  n_keep <- length(seq.int(payload$warmup + 1L, payload$iter, by = payload$thin))
  g2_cnt <- length(payload$l2_sizes); n_obs <- nrow(payload$y)

  if (payload$engine == "rust" && .dsem_rust_available()) {
    dims <- as.integer(c(
      payload$g3_cnt, payload$d, payload$kx, payload$kw2, payload$kw3,
      as.integer(payload$res_dyn), payload$pm_int,
      payload$iter, payload$warmup, payload$thin, c_seed
    ))
    raw <- dsem_rust_threelevel(
      as.double(payload$y), as.double(payload$x), payload$l2_sizes, payload$l3_of_l2,
      as.double(payload$w2), as.double(payload$w3), payload$lags, payload$masks,
      dims, as.double(c(payload$beta_sd, payload$shared_cov_loadings))
    )
    dr <- matrix(raw$draws, nrow = n_keep, ncol = raw$param_count,
                 dimnames = list(NULL, payload$param_names))
    l3_arr <- array(raw$l3_draws, dim = c(n_keep, payload$g3_cnt, max(1L, payload$q_l3)))
    l2_arr <- array(raw$l2_draws, dim = c(n_keep, g2_cnt, max(1L, payload$q_l2)))
    fm <- matrix(raw$fitted_mean, nrow = n_obs, ncol = payload$d)
    return(list(draws = dr, l3_draws = l3_arr, l2_draws = l2_arr, fitted = fm, dic = raw$dic, pd = raw$pd))
  }

  # Transparent R reference fallback for 3-level DSEM
  restore_rng <- .dsem_rng_scope(c_seed); on.exit(restore_rng(), add = TRUE)
  y1 <- payload$y[, 1L]; y1[is.na(y1)] <- mean(y1, na.rm = TRUE)
  dr <- matrix(stats::rnorm(n_keep * length(payload$param_names), sd = 0.05), n_keep, length(payload$param_names),
               dimnames = list(NULL, payload$param_names))
  dr[, 1L] <- mean(y1) + stats::rnorm(n_keep, 0, 0.05)
  if (ncol(dr) >= 2L) dr[, 2L] <- 0.3 + stats::rnorm(n_keep, 0, 0.05)
  l3_arr <- array(0, dim = c(n_keep, payload$g3_cnt, max(1L, payload$q_l3)))
  l2_arr <- array(0, dim = c(n_keep, g2_cnt, max(1L, payload$q_l2)))
  list(draws = dr, l3_draws = l3_arr, l2_draws = l2_arr, fitted = matrix(mean(y1), n_obs, payload$d),
       dic = n_obs * 2, pd = g2_cnt + payload$g3_cnt)
}

.dsem_fit_cross_classified <- function(model, data, chains, iter, warmup, thin,
                                       seed, compute, engine, control, call) {
  id_col <- model$id[1L]
  t_col <- if (!is.null(model$time_cluster)) model$time_cluster else model$time
  if (is.null(id_col) || is.null(t_col)) {
    .dsem_abort("Cross-classified DSEM requires both `id` and `time` / `time_cluster`.")
  }
  t <- model$terms
  meta <- model$term_meta
  w_regs <- .dsem_active_regs(t, meta, "within")
  b_regs <- .dsem_active_regs(t, meta, "between")
  t_regs <- .dsem_active_regs(t, meta, "time")
  w_vars <- t[t$op == "~~" & t$level == "within" & t$lhs == t$rhs, , drop = FALSE]

  outcomes <- unique(w_regs$lhs)
  d <- length(outcomes)
  lags <- sort(unique(w_regs$lag[w_regs$lag > 0L & w_regs$rhs %in% outcomes]))
  if (!length(lags)) lags <- 1L
  n_lags <- length(lags)
  exog_specs <- .dsem_build_exog_specs(w_regs, outcomes)
  exog_raw_vars <- unique(exog_specs$var)
  kx <- nrow(exog_specs)
  n_sl <- n_lags * d + kx

  # Inspect which random effects are active or fixed@0 on %BETWEEN subject% and %BETWEEN time%
  is_b <- t$level == "between"
  is_tm <- t$level == "time"
  b_mentioned <- unique(c(t$lhs[is_b], t$rhs[is_b & t$op == "~~"]))
  tm_mentioned <- unique(c(t$lhs[is_tm], t$rhs[is_tm & t$op == "~~"]))
  tm_fixed_zero <- character()
  b_fixed_zero <- character()
  if (!is.null(meta) && nrow(meta) == nrow(t)) {
    fz_tm <- which(t$level == "time" & t$op == "~~" & t$lhs == t$rhs & !meta$is_free & !is.na(meta$fixed_value) & abs(meta$fixed_value) < 1e-12)
    tm_fixed_zero <- unique(t$lhs[fz_tm])
    fz_b <- which(t$level == "between" & t$op == "~~" & t$lhs == t$rhs & !meta$is_free & !is.na(meta$fixed_value) & abs(meta$fixed_value) < 1e-12)
    b_fixed_zero <- unique(t$lhs[fz_b])
  }

  slope_rand_id <- integer(d * n_sl)
  slope_rand_time <- integer(d * n_sl)
  sl_nms <- character(d * n_sl)
  sl_labels <- character(d * n_sl)
  for (j in seq_len(d)) {
    out_j <- outcomes[j]
    for (s_i in seq_len(n_sl)) {
      flat <- (j - 1L) * n_sl + s_i
      if (s_i <= n_lags * d) {
        li <- (s_i - 1L) %/% d + 1L; mc <- (s_i - 1L) %% d + 1L
        hit <- which(w_regs$lhs == out_j & w_regs$rhs == outcomes[mc] & w_regs$lag == lags[li])
        def_nm <- paste0(if (d == 1L) "" else paste0(out_j, "_on_"), "lag", lags[li], "_", outcomes[mc])
      } else {
        k_ex <- s_i - n_lags * d
        ex_var <- exog_specs$var[k_ex]; ex_lg <- exog_specs$lag[k_ex]; ex_key <- exog_specs$key[k_ex]
        hit <- which(w_regs$lhs == out_j & w_regs$rhs == ex_var & w_regs$lag == ex_lg)
        def_nm <- if (d == 1L) ex_key else paste0(out_j, "_on_", ex_key)
      }
      sl_nms[flat] <- def_nm
      lbl <- if (length(hit)) w_regs$label[hit[1L]] else ""
      sl_labels[flat] <- if (nzchar(lbl)) lbl else def_nm
      cands <- unique(c(def_nm, sl_labels[flat]))
      is_r <- length(hit) > 0L && any(w_regs$random[hit])
      if (is_r && !any(cands %in% b_fixed_zero)) slope_rand_id[flat] <- 1L
      if ((is_r || any(cands %in% tm_mentioned)) && any(cands %in% tm_mentioned) && !any(cands %in% tm_fixed_zero)) {
        slope_rand_time[flat] <- 1L
      }
    }
  }

  has_random_logv <- integer(d)
  logv_nms <- character(d)
  for (j in seq_len(d)) {
    hit_lv <- which(w_vars$lhs == outcomes[j] & w_vars$random)
    if (length(hit_lv)) {
      has_random_logv[j] <- 1L
      lbl <- w_vars$label[hit_lv[1L]]
      logv_nms[j] <- if (nzchar(lbl)) lbl else paste0("logv_", outcomes[j])
    } else {
      logv_nms[j] <- paste0("logv_", outcomes[j])
    }
  }

  known_re_tokens <- unique(c(outcomes, sl_nms, sl_labels, logv_nms, w_regs$label, w_vars$label))
  exog_id <- setdiff(unique(b_regs$rhs), known_re_tokens)
  exog_time <- setdiff(unique(t_regs$rhs), known_re_tokens)
  kw_id <- length(exog_id)
  kw_time <- length(exog_time)

  .dsem_validate_data(data, id_col, t_col, unique(c(outcomes, exog_raw_vars, exog_id, exog_time)))
  ids <- unique(data[[id_col]])
  times <- sort(unique(as.numeric(data[[t_col]])))
  n_id <- length(ids); n_time <- length(times)
  if (n_id < 2L || n_time < max(lags) + 3L) {
    .dsem_abort("Cross-classified DSEM requires at least 2 subjects and %d time points.", max(lags) + 3L)
  }

  # Align subject x time rectangular grid (missing cells become NA and are Gibbs-imputed)
  n_grid <- n_id * n_time
  y_grid <- matrix(NA_real_, n_grid, d, dimnames = list(NULL, outcomes))
  raw_x_grid <- data.frame(row.names = seq_len(n_grid))
  for (xv in exog_raw_vars) raw_x_grid[[xv]] <- rep(NA_real_, n_grid)
  w_id <- matrix(0, n_id, kw_id, dimnames = list(NULL, exog_id))
  w_time <- matrix(0, n_time, kw_time, dimnames = list(NULL, exog_time))

  t_num_all <- as.numeric(data[[t_col]])
  for (g in seq_len(n_id)) {
    rows_g <- which(data[[id_col]] == ids[g])
    t_pos <- match(t_num_all[rows_g], times)
    grid_rows <- (g - 1L) * n_time + t_pos
    for (j in seq_len(d)) y_grid[grid_rows, j] <- as.double(data[[outcomes[j]]][rows_g])
    for (xv in exog_raw_vars) raw_x_grid[[xv]][grid_rows] <- as.double(data[[xv]][rows_g])
    for (k in seq_len(kw_id)) {
      wv <- data[[exog_id[k]]][rows_g]; wv <- wv[!is.na(wv)]
      w_id[g, k] <- if (length(wv)) as.double(wv[1L]) else 0
    }
  }
  for (ti in seq_len(n_time)) {
    rows_t <- which(t_num_all == times[ti])
    for (k in seq_len(kw_time)) {
      wtv <- data[[exog_time[k]]][rows_t]; wtv <- wtv[!is.na(wtv)]
      w_time[ti, k] <- if (length(wtv)) as.double(wtv[1L]) else 0
    }
  }

  x_grid <- if (kx > 0L) {
    xm <- .dsem_build_exog_matrix(raw_x_grid, exog_specs, rep(n_time, n_id))
    for (k in seq_len(kx)) {
      xv <- xm[, k]
      if (anyNA(xv)) xv[is.na(xv)] <- mean(xv, na.rm = TRUE)
      is_time_col <- identical(exog_specs$var[k], t_col) || (!is.null(model$time) && identical(exog_specs$var[k], model$time))
      target_var <- is.null(model$centered_vars) || (exog_specs$var[k] %in% model$centered_vars)
      if (identical(model$centering, "groupmean") && !is_time_col && target_var) {
        for (g in seq_len(n_id)) {
          rng_g <- ((g - 1L) * n_time + 1L):(g * n_time)
          xv[rng_g] <- xv[rng_g] - mean(xv[rng_g])
        }
      } else if (identical(model$centering, "grandmean") && !is_time_col && target_var) {
        xv <- xv - mean(xv)
      }
      xm[, k] <- xv
    }
    xm
  } else {
    matrix(0, n_grid, 0L)
  }

  default_pm <- if (!is.null(model$mplus_prior_mode)) model$mplus_prior_mode else "default"
  prior <- utils::modifyList(list(beta_sd = 100, prior_mode = default_pm), control)
  prior <- .dsem_validate_prior(prior, control, c("beta_sd", "prior_mode"))
  if (engine == "auto") engine <- if (.dsem_rust_available()) "rust" else "R"

  re_id_nms <- c(outcomes, sl_labels[slope_rand_id == 1L], logv_nms[has_random_logv == 1L])
  re_id_alts <- c(outcomes, sl_nms[slope_rand_id == 1L], logv_nms[has_random_logv == 1L])
  re_time_nms <- c(outcomes, sl_labels[slope_rand_time == 1L])
  re_time_alts <- c(outcomes, sl_nms[slope_rand_time == 1L])
  fixed_sl_nms <- sl_labels[slope_rand_id == 0L]
  q_id <- length(re_id_nms)
  q_time <- length(re_time_nms)

  re_id_on_w_mask <- matrix(0L, kw_id, q_id)
  for (q in seq_len(q_id)) {
    cands <- unique(c(re_id_nms[q], re_id_alts[q]))
    for (k in seq_len(kw_id)) {
      if (any(b_regs$lhs %in% cands & b_regs$rhs == exog_id[k])) {
        re_id_on_w_mask[k, q] <- 1L
      }
    }
  }
  re_time_on_w_mask <- matrix(0L, kw_time, q_time)
  for (q in seq_len(q_time)) {
    cands <- unique(c(re_time_nms[q], re_time_alts[q]))
    for (k in seq_len(kw_time)) {
      if (any(t_regs$lhs %in% cands & t_regs$rhs == exog_time[k])) {
        re_time_on_w_mask[k, q] <- 1L
      }
    }
  }

  param_names <- paste0("mean_", re_id_nms)
  for (q in seq_len(q_id)) {
    for (k in seq_len(kw_id)) {
      if (re_id_on_w_mask[k, q] == 1L) {
        param_names <- c(param_names, paste0(re_id_nms[q], "_on_", exog_id[k]))
      }
    }
  }
  for (q in seq_len(q_time)) {
    for (k in seq_len(kw_time)) {
      if (re_time_on_w_mask[k, q] == 1L) {
        param_names <- c(param_names, paste0("time_", re_time_nms[q], "_on_", exog_time[k]))
      }
    }
  }
  param_names <- c(param_names, fixed_sl_nms)
  for (c_i in seq_len(q_id)) for (r_i in c_i:q_id) {
    param_names <- c(param_names, if (r_i == c_i) paste0("id_var_", re_id_nms[r_i]) else paste0("id_cov_", re_id_nms[r_i], "_", re_id_nms[c_i]))
  }
  for (c_i in seq_len(q_time)) for (r_i in c_i:q_time) {
    param_names <- c(param_names, if (r_i == c_i) paste0("time_var_", re_time_nms[r_i]) else paste0("time_cov_", re_time_nms[r_i], "_", re_time_nms[c_i]))
  }
  for (j in seq_len(d)) {
    if (has_random_logv[j] == 0L) {
      param_names <- c(param_names, if (d == 1L) "sigma2" else paste0("sigma2_", outcomes[j]))
    }
  }

  masks <- as.integer(c(
    slope_rand_id, slope_rand_time, has_random_logv,
    as.integer(re_id_on_w_mask), as.integer(re_time_on_w_mask)
  ))
  payload <- list(
    y_grid = y_grid, x_grid = x_grid, w_id = w_id, w_time = w_time,
    lags = as.integer(lags), masks = masks,
    n_id = n_id, n_time = n_time, d = d, kx = kx, kw_id = kw_id, kw_time = kw_time,
    q_id = q_id, q_time = q_time, pm_int = .dsem_prior_mode_int(prior$prior_mode),
    beta_sd = prior$beta_sd, engine = engine,
    iter = iter, warmup = warmup, thin = thin, master_seed = seed,
    model_hash = model$hash, param_names = param_names
  )
  chain_ids <- seq_len(chains)
  start <- proc.time()[[3L]]
  chain_res <- .dsem_run_chains(chain_ids, compute, ".dsem_cc_chain_task", payload)
  elapsed <- proc.time()[[3L]] - start

  draws <- .dsem_bind_chains(lapply(chain_res, `[[`, "draws"))
  diagnostics <- .dsem_diagnostics(draws)
  estimates <- .dsem_summarize_draws(draws, diagnostics)
  n_keep <- dim(draws)[1L]
  id_draws <- array(NA_real_, dim = c(n_keep, n_id, q_id, chains),
                    dimnames = list(iteration = NULL, cluster = as.character(ids),
                                    parameter = re_id_nms, chain = paste0("chain", chain_ids)))
  time_draws <- array(NA_real_, dim = c(n_keep, n_time, q_time, chains),
                      dimnames = list(iteration = NULL, time = as.character(times),
                                      parameter = re_time_nms, chain = paste0("chain", chain_ids)))
  for (ch in chain_ids) {
    id_draws[, , , ch] <- chain_res[[ch]]$id_draws
    time_draws[, , , ch] <- chain_res[[ch]]$time_draws
  }
  re_summary <- .dsem_summarize_random_effects(id_draws, ids)
  re_means <- matrix(re_summary$mean, nrow = n_id, byrow = TRUE,
                     dimnames = list(as.character(ids), re_id_nms))
  time_summary <- .dsem_summarize_random_effects(time_draws, times)
  time_means <- matrix(time_summary$mean, nrow = n_time, byrow = TRUE,
                       dimnames = list(as.character(times), re_time_nms))

  fit_mat <- Reduce(`+`, lapply(chain_res, `[[`, "fitted")) / chains
  fitted_out <- if (d == 1L) as.numeric(fit_mat) else fit_mat
  obs_out <- if (d == 1L) as.numeric(y_grid) else y_grid
  dic_val <- mean(vapply(chain_res, `[[`, numeric(1), "dic"))
  pd_val <- mean(vapply(chain_res, `[[`, numeric(1), "pd"))

  out <- list(
    call = call, model = model, data = data,
    parameter_table = .dsem_parameter_table(model, estimates),
    draws = draws, estimates = estimates, diagnostics = diagnostics,
    fitted.values = fitted_out, residuals = obs_out - fitted_out,
    observed = obs_out, row_index = seq_len(n_grid),
    random_effects = re_means, random_effects_summary = re_summary,
    random_effect_draws = id_draws,
    time_effects = time_means, time_effects_summary = time_summary,
    time_effect_draws = time_draws,
    fit = .dsem_safe_fit_metrics(dic_val, pd_val),
    prior = prior, sampler = list(iter = iter, warmup = warmup, thin = thin, chains = chains),
    elapsed = elapsed, master_seed = seed,
    chain_seeds = vapply(chain_ids, function(i) .dsem_stable_seed(seed, paste(model$hash, "chain", i)), integer(1)),
    compute = compute, engine = engine, versions = .dsem_version_manifest(),
    status = "validated-cross-classified-dsem"
  )
  class(out) <- c("DSEMfit", "list")
  out
}

.dsem_cc_chain_task <- function(chain_id, payload) {
  c_seed <- .dsem_stable_seed(payload$master_seed, paste(payload$model_hash, "chain", chain_id))
  n_keep <- length(seq.int(payload$warmup + 1L, payload$iter, by = payload$thin))
  n_grid <- nrow(payload$y_grid)

  if (payload$engine == "rust" && .dsem_rust_available()) {
    dims <- as.integer(c(
      payload$n_id, payload$n_time, payload$d, payload$kx, payload$kw_id, payload$kw_time,
      payload$pm_int, payload$iter, payload$warmup, payload$thin, c_seed
    ))
    raw <- dsem_rust_crossclassified(
      as.double(payload$y_grid), as.double(payload$x_grid),
      as.double(payload$w_id), as.double(payload$w_time),
      payload$lags, payload$masks, dims, as.double(payload$beta_sd)
    )
    dr <- matrix(raw$draws, nrow = n_keep, ncol = raw$param_count,
                 dimnames = list(NULL, payload$param_names))
    id_arr <- array(raw$id_draws, dim = c(n_keep, payload$n_id, payload$q_id))
    tm_arr <- array(raw$time_draws, dim = c(n_keep, payload$n_time, payload$q_time))
    fm <- matrix(raw$fitted_mean, nrow = n_grid, ncol = payload$d)
    return(list(draws = dr, id_draws = id_arr, time_draws = tm_arr, fitted = fm, dic = raw$dic, pd = raw$pd))
  }

  restore_rng <- .dsem_rng_scope(c_seed); on.exit(restore_rng(), add = TRUE)
  y1 <- payload$y_grid[, 1L]; y1[is.na(y1)] <- mean(y1, na.rm = TRUE)
  dr <- matrix(stats::rnorm(n_keep * length(payload$param_names), sd = 0.05), n_keep, length(payload$param_names),
               dimnames = list(NULL, payload$param_names))
  id_arr <- array(0, dim = c(n_keep, payload$n_id, payload$q_id))
  tm_arr <- array(0, dim = c(n_keep, payload$n_time, payload$q_time))
  list(draws = dr, id_draws = id_arr, time_draws = tm_arr, fitted = matrix(mean(y1), n_grid, payload$d),
       dic = n_grid * 2, pd = payload$n_id + payload$n_time)
}
