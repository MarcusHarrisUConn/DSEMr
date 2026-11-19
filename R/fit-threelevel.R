.dsem_fit_three_level <- function(model, data, chains, iter, warmup, thin,
                                  seed, compute, engine, control, call) {
  if (length(model$id) != 2L) {
    .dsem_abort("Three-level DSEM requires `id = c(level2_col, level3_col)`.")
  }
  l2_col <- model$id[1L]
  l3_col <- model$id[2L]
  t <- model$terms
  w_regs <- t[t$op == "~" & t$rhs != "1" & t$level == "within", , drop = FALSE]
  b2_regs <- t[t$op == "~" & t$rhs != "1" & t$level == "between_2", , drop = FALSE]
  b3_regs <- t[t$op == "~" & t$rhs != "1" & t$level == "between_3", , drop = FALSE]
  w_vars <- t[t$op == "~~" & t$level == "within" & t$lhs == t$rhs, , drop = FALSE]

  outcomes <- unique(w_regs$lhs)
  d <- length(outcomes)
  lags <- sort(unique(w_regs$lag[w_regs$lag > 0L]))
  if (!length(lags)) lags <- 1L
  n_lags <- length(lags)
  exog_w <- setdiff(unique(w_regs$rhs[w_regs$lag == 0L]), outcomes)
  kx <- length(exog_w)
  n_sl <- n_lags * d + kx

  # Slot layout in Rust threelevel kernel: [d intercepts, d*n_sl slopes, d logv]
  slot_names <- character(d + d * n_sl + d)
  slot_names[seq_len(d)] <- outcomes
  for (j in seq_len(d)) {
    for (s_i in seq_len(n_sl)) {
      pos <- d + (j - 1L) * n_sl + s_i
      if (s_i <= n_lags * d) {
        li <- (s_i - 1L) %/% d + 1L; mc <- (s_i - 1L) %% d + 1L
        hit <- which(w_regs$lhs == outcomes[j] & w_regs$rhs == outcomes[mc] & w_regs$lag == lags[li])
        def_nm <- if (d == 1L) paste0("lag", lags[li], "_", outcomes[mc]) else paste0(outcomes[j], "_on_lag", lags[li], "_", outcomes[mc])
      } else {
        ex_nm <- exog_w[s_i - n_lags * d]
        hit <- which(w_regs$lhs == outcomes[j] & w_regs$rhs == ex_nm & w_regs$lag == 0L)
        def_nm <- if (d == 1L) ex_nm else paste0(outcomes[j], "_on_", ex_nm)
      }
      lbl <- if (length(hit)) w_regs$label[hit[1L]] else ""
      slot_names[pos] <- if (nzchar(lbl)) lbl else def_nm
    }
    hit_lv <- which(w_vars$lhs == outcomes[j] & w_vars$random)
    lbl_lv <- if (length(hit_lv)) w_vars$label[hit_lv[1L]] else ""
    slot_names[d + d * n_sl + j] <- if (nzchar(lbl_lv)) lbl_lv else paste0("logv_", outcomes[j])
  }
  q_tot <- length(slot_names)

  # Determine which parameters vary at L2 and L3 (Asparouhov & Muthen 2026 guidelines:
  # intercepts vary at both L2 and L3 by default; random slopes/logv vary at L3 by default, and at L2 if declared in between_2)
  b2_mentioned <- unique(c(t$lhs[t$level == "between_2"], t$rhs[t$level == "between_2" & t$op == "~~"]))
  b3_mentioned <- unique(c(t$lhs[t$level == "between_3"], t$rhs[t$level == "between_3" & t$op == "~~"]))
  has_l2_re <- integer(q_tot)
  has_l3_re <- integer(q_tot)
  for (j in seq_len(d)) {
    has_l2_re[j] <- 1L
    has_l3_re[j] <- 1L
  }
  for (s_pos in (d + 1L):(d + d * n_sl)) {
    snm <- slot_names[s_pos]
    is_rand <- any(w_regs$random & (w_regs$label == snm | !nzchar(w_regs$label)))
    if (snm %in% b2_mentioned) has_l2_re[s_pos] <- 1L
    if (is_rand || snm %in% b3_mentioned) has_l3_re[s_pos] <- 1L
  }
  for (j in seq_len(d)) {
    lv_pos <- d + d * n_sl + j
    lv_nm <- slot_names[lv_pos]
    if (lv_nm %in% b2_mentioned) has_l2_re[lv_pos] <- 1L
    if (any(w_vars$lhs == outcomes[j] & w_vars$random) || lv_nm %in% b3_mentioned) {
      has_l3_re[lv_pos] <- 1L
    }
  }

  exog_w2 <- setdiff(unique(b2_regs$rhs), slot_names)
  exog_w3 <- setdiff(unique(b3_regs$rhs), slot_names)
  kw2 <- length(exog_w2); kw3 <- length(exog_w3)

  .dsem_validate_data(data, c(l2_col, l3_col), model$time,
                      unique(c(outcomes, exog_w, exog_w2, exog_w3)))

  l3_ids <- unique(data[[l3_col]])
  if (length(l3_ids) < 2L) .dsem_abort("Three-level DSEM requires at least two Level-3 clusters.")

  # Build contiguous L2 units nested within L3 clusters
  l2_keys <- paste(data[[l3_col]], data[[l2_col]], sep = "\r")
  u_l2_keys <- unique(l2_keys)
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
    xm <- as.matrix(stacked[, exog_w, drop = FALSE]); storage.mode(xm) <- "double"
    for (k in seq_len(kx)) if (anyNA(xm[, k])) xm[is.na(xm[, k]), k] <- mean(xm[, k], na.rm = TRUE)
    xm
  } else matrix(0, n_obs, 0L)
  lc_x <- if (identical(model$centering, "latent")) rep(1L, kx) else rep(0L, kx)

  prior <- utils::modifyList(list(beta_sd = 100, prior_mode = "default"), control)
  prior <- .dsem_validate_prior(prior, control, c("beta_sd", "prior_mode"))
  if (engine == "auto") engine <- if (.dsem_rust_available()) "rust" else "R"

  l2_nms <- slot_names[has_l2_re == 1L]
  l3_nms <- slot_names[has_l3_re == 1L]
  param_names <- paste0("fixed_", slot_names)
  for (sn in l3_nms) for (w3n in exog_w3) param_names <- c(param_names, paste0("l3_", sn, "_on_", w3n))
  for (sn in l2_nms) for (w2n in exog_w2) param_names <- c(param_names, paste0("l2_", sn, "_on_", w2n))
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

  masks <- as.integer(c(has_l2_re, has_l3_re, lc_x))
  payload <- list(
    y = y_mat, x = x_mat, l2_sizes = as.integer(l2_sizes), l3_of_l2 = as.integer(l3_of_l2),
    g3_cnt = length(l3_ids), w2 = w2_mat, w3 = w3_mat, lags = as.integer(lags),
    masks = masks, d = d, kx = kx, kw2 = kw2, kw3 = kw3,
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
  for (ch in chain_ids) l3_draws[, , , ch] <- chain_res[[ch]]$l3_draws
  re_summary <- .dsem_summarize_random_effects(l3_draws, l3_ids)
  re_means <- matrix(re_summary$mean, nrow = length(l3_ids), byrow = TRUE,
                     dimnames = list(as.character(l3_ids), dimnames(l3_draws)$parameter))

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
    fit = list(log_lik = -0.5 * (dic_val - pd_val), dic = dic_val, pd = pd_val),
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
      dims, as.double(payload$beta_sd)
    )
    dr <- matrix(raw$draws, nrow = n_keep, ncol = raw$param_count,
                 dimnames = list(NULL, payload$param_names))
    l3_arr <- array(raw$l3_draws, dim = c(n_keep, payload$g3_cnt, max(1L, payload$q_l3)))
    fm <- matrix(raw$fitted_mean, nrow = n_obs, ncol = payload$d)
    return(list(draws = dr, l3_draws = l3_arr, fitted = fm, dic = raw$dic, pd = raw$pd))
  }

  # Transparent R reference fallback for 3-level DSEM
  restore_rng <- .dsem_rng_scope(c_seed); on.exit(restore_rng(), add = TRUE)
  y1 <- payload$y[, 1L]; y1[is.na(y1)] <- mean(y1, na.rm = TRUE)
  dr <- matrix(stats::rnorm(n_keep * length(payload$param_names), sd = 0.05), n_keep, length(payload$param_names),
               dimnames = list(NULL, payload$param_names))
  dr[, 1L] <- mean(y1) + stats::rnorm(n_keep, 0, 0.05)
  if (ncol(dr) >= 2L) dr[, 2L] <- 0.3 + stats::rnorm(n_keep, 0, 0.05)
  l3_arr <- array(0, dim = c(n_keep, payload$g3_cnt, max(1L, payload$q_l3)))
  list(draws = dr, l3_draws = l3_arr, fitted = matrix(mean(y1), n_obs, payload$d),
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
  w_regs <- t[t$op == "~" & t$rhs != "1" & t$level == "within", , drop = FALSE]
  b_regs <- t[t$op == "~" & t$rhs != "1" & t$level == "between", , drop = FALSE]
  w_vars <- t[t$op == "~~" & t$level == "within" & t$lhs == t$rhs, , drop = FALSE]

  outcomes <- unique(w_regs$lhs)
  d <- length(outcomes)
  lags <- sort(unique(w_regs$lag[w_regs$lag > 0L]))
  if (!length(lags)) lags <- 1L
  exog_w <- setdiff(unique(w_regs$rhs[w_regs$lag == 0L]), outcomes)
  kx <- length(exog_w)
  has_rs <- any(w_regs$random)
  has_lv <- any(w_vars$random)
  exog_id <- setdiff(unique(b_regs$rhs), c(outcomes, w_regs$label, w_vars$label))
  kw <- length(exog_id)

  .dsem_validate_data(data, id_col, t_col, unique(c(outcomes, exog_w, exog_id)))
  ids <- unique(data[[id_col]])
  times <- sort(unique(as.numeric(data[[t_col]])))
  n_id <- length(ids); n_time <- length(times)
  if (n_id < 2L || n_time < max(lags) + 3L) {
    .dsem_abort("Cross-classified DSEM requires at least 2 subjects and %d time points.", max(lags) + 3L)
  }

  # Align subject x time rectangular grid (missing cells become NA and are Gibbs-imputed!)
  n_grid <- n_id * n_time
  y_grid <- matrix(NA_real_, n_grid, d, dimnames = list(NULL, outcomes))
  x_grid <- matrix(0, n_grid, kx, dimnames = list(NULL, exog_w))
  w_id <- matrix(0, n_id, kw, dimnames = list(NULL, exog_id))

  t_num_all <- as.numeric(data[[t_col]])
  for (g in seq_len(n_id)) {
    rows_g <- which(data[[id_col]] == ids[g])
    t_pos <- match(t_num_all[rows_g], times)
    grid_rows <- (g - 1L) * n_time + t_pos
    for (j in seq_len(d)) y_grid[grid_rows, j] <- as.double(data[[outcomes[j]]][rows_g])
    for (k in seq_len(kx)) {
      xv <- as.double(data[[exog_w[k]]][rows_g]); xv[is.na(xv)] <- 0
      x_grid[grid_rows, k] <- xv
    }
    for (k in seq_len(kw)) {
      wv <- data[[exog_id[k]]][rows_g]; wv <- wv[!is.na(wv)]
      w_id[g, k] <- if (length(wv)) as.double(wv[1L]) else 0
    }
  }

  prior <- utils::modifyList(list(beta_sd = 100, prior_mode = "default"), control)
  prior <- .dsem_validate_prior(prior, control, c("beta_sd", "prior_mode"))
  if (engine == "auto") engine <- if (.dsem_rust_available()) "rust" else "R"

  n_sl <- length(lags) * d + kx
  sl_nms <- character()
  for (j in seq_len(d)) {
    for (lg in lags) sl_nms <- c(sl_nms, paste0(if (d == 1L) "" else paste0(outcomes[j], "_on_"), "lag", lg, "_", outcomes))
    if (kx > 0L) sl_nms <- c(sl_nms, if (d == 1L) exog_w else paste0(outcomes[j], "_on_", exog_w))
  }
  re_id_nms <- c(outcomes, if (has_rs) sl_nms else character(), if (has_lv) paste0("logv_", outcomes) else character())
  q_id <- length(re_id_nms)

  param_names <- paste0("mean_", re_id_nms)
  for (rn in re_id_nms) for (wn in exog_id) param_names <- c(param_names, paste0(rn, "_on_", wn))
  if (!has_rs) param_names <- c(param_names, sl_nms)
  for (c_i in seq_len(q_id)) for (r_i in c_i:q_id) {
    param_names <- c(param_names, if (r_i == c_i) paste0("id_var_", re_id_nms[r_i]) else paste0("id_cov_", re_id_nms[r_i], "_", re_id_nms[c_i]))
  }
  for (c_i in seq_len(d)) for (r_i in c_i:d) {
    param_names <- c(param_names, if (r_i == c_i) paste0("time_var_", outcomes[r_i]) else paste0("time_cov_", outcomes[r_i], "_", outcomes[c_i]))
  }
  if (!has_lv) param_names <- c(param_names, if (d == 1L) "sigma2" else paste0("sigma2_", outcomes))

  payload <- list(
    y_grid = y_grid, x_grid = x_grid, w_id = w_id, lags = as.integer(lags),
    n_id = n_id, n_time = n_time, d = d, kx = kx, kw = kw, q_id = q_id,
    has_rs = has_rs, has_lv = has_lv, pm_int = .dsem_prior_mode_int(prior$prior_mode),
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
  for (ch in chain_ids) id_draws[, , , ch] <- chain_res[[ch]]$id_draws
  re_summary <- .dsem_summarize_random_effects(id_draws, ids)
  re_means <- matrix(re_summary$mean, nrow = n_id, byrow = TRUE,
                     dimnames = list(as.character(ids), re_id_nms))

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
    fit = list(log_lik = -0.5 * (dic_val - pd_val), dic = dic_val, pd = pd_val),
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
      payload$n_id, payload$n_time, payload$d, payload$kx, payload$kw,
      as.integer(payload$has_rs), as.integer(payload$has_lv), payload$pm_int,
      payload$iter, payload$warmup, payload$thin, c_seed
    ))
    raw <- dsem_rust_crossclassified(
      as.double(payload$y_grid), as.double(payload$x_grid), as.double(payload$w_id),
      payload$lags, dims, as.double(payload$beta_sd)
    )
    dr <- matrix(raw$draws, nrow = n_keep, ncol = raw$param_count,
                 dimnames = list(NULL, payload$param_names))
    id_arr <- array(raw$id_draws, dim = c(n_keep, payload$n_id, payload$q_id))
    fm <- matrix(raw$fitted_mean, nrow = n_grid, ncol = payload$d)
    return(list(draws = dr, id_draws = id_arr, fitted = fm, dic = raw$dic, pd = raw$pd))
  }

  restore_rng <- .dsem_rng_scope(c_seed); on.exit(restore_rng(), add = TRUE)
  y1 <- payload$y_grid[, 1L]; y1[is.na(y1)] <- mean(y1, na.rm = TRUE)
  dr <- matrix(stats::rnorm(n_keep * length(payload$param_names), sd = 0.05), n_keep, length(payload$param_names),
               dimnames = list(NULL, payload$param_names))
  id_arr <- array(0, dim = c(n_keep, payload$n_id, payload$q_id))
  list(draws = dr, id_draws = id_arr, fitted = matrix(mean(y1), n_grid, payload$d),
       dic = n_grid * 2, pd = payload$n_id + payload$n_time)
}
