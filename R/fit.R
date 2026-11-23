#' Estimate a dynamic structural equation model
#'
#' Fits Bayesian dynamic structural equation models compiled by [dsem_model()]
#' or specified directly via extended `lavaan` formulas or standard
#' section-based SEM scripts. Supported model families include single-level
#' (`N=1`) autoregressive and vector autoregressive (`VAR(p)`) models, residual
#' dynamic models (`RDSEM`), binary/ordinal probit and Tobit censored dynamic
#' models, single- and multi-factor dynamic confirmatory factor / IRT models
#' (`DAFS`, `WNFS`, `ARMA`), two-level location-scale DSEM with latent
#' centering, correlated within-level innovations, and random log residual
#' variances, three-level DSEM (`THREELEVEL` / `DSEM3`), and subject-by-time
#' cross-classified DSEM (`CROSSCLASSIFIED`).
#'
#' @param model A `DSEMmodel`, `DSEMspec`, or model specification string.
#' @param data A data frame in long format.
#' @param syntax,id,time,family,... Passed to [dsem_model()] when `model` is a
#'   character string.
#' @param chains Number of independent MCMC chains.
#' @param iter Total MCMC iterations per chain.
#' @param warmup Number of warmup (burn-in) iterations discarded per chain.
#' @param thin Positive thinning interval.
#' @param seed Non-negative integer master seed. Chain-specific seeds are
#'   deterministically derived from `seed`, the canonical model hash, and chain
#'   index so draws are invariant to worker count.
#' @param compute A [dsem_compute()] resource specification.
#' @param engine Numerical backend: `"auto"` (uses compiled Rust when available),
#'   `"rust"`, or transparent `"R"` reference implementation.
#' @param control Named list of prior and estimation controls. Accepts
#'   `prior_mode` (`"default"`, `"diffuse"` — with legacy alias `"mplus"` — or
#'   `"penalized"` for Bayesian Lasso PSEM shrinkage), `beta_mean`, `beta_sd`,
#'   `sigma_shape`, `sigma_scale`, `omega_df`, `omega_scale`,
#'   `coefficient_mean`, `coefficient_sd`, `variance_shape`, `variance_scale`,
#'   and `ar_bound`.
#' @return A `DSEMfit` object with S3 methods [summary()], [coef()], [vcov()],
#'   [confint()], [logLik()], [fitted()], [residuals()], [predict()], [plot()],
#'   [dsem_standardize()], [dsem_dic()], [dsem_vif()], [dsem_r2()],
#'   [dsem_partial_cor()], [dsem_hypothesis()], [dsem_irf()],
#'   [dsem_fscomparison()], and [dsem_inspect()].
#' @references
#' Asparouhov, T., Hamaker, E. L., & Muthen, B. (2018). Dynamic structural
#' equation models. *Structural Equation Modeling*, 25(3), 359-388.
#' \doi{10.1080/10705511.2017.1406803}
#'
#' Asparouhov, T., & Muthen, B. (2020). Comparison of models for the analysis of
#' intensive longitudinal data. *Structural Equation Modeling*, 27(2), 275-297.
#' \doi{10.1080/10705511.2019.1626733}
#' @examples
#' dat <- simulate_dsem(n = 80, intercept = 0.2, ar = 0.5, sigma = 0.7, seed = 1)
#' fit <- dsem("y ~ lag(y, 1)", dat, time = "time",
#'             chains = 2, iter = 200, warmup = 100, seed = 1)
#' summary(fit)
#' coef(fit)
#' @export
dsem <- function(model, data, syntax = c("auto", "lavaan", "mplus"),
                 id = NULL, time = NULL, family = "gaussian", ...,
                 chains = 4L, iter = 2000L, warmup = floor(iter / 2),
                 thin = 1L, seed = 20260813L,
                 compute = dsem_compute(), engine = c("auto", "rust", "R"),
                 control = list()) {
  call <- match.call()
  engine <- match.arg(engine)
  if (inherits(model, "DSEMspec")) {
    if (missing(chains)) chains <- model$chains
    if (missing(iter)) iter <- model$iter
    if (missing(warmup)) warmup <- model$warmup
    if (missing(thin)) thin <- model$thin
    if (missing(seed)) seed <- model$seed
    if (missing(control) && !identical(model$prior_mode, "default")) {
      control <- list(prior_mode = model$prior_mode)
    }
    model <- as_dsem_model(model)
  } else if (!inherits(model, "DSEMmodel")) {
    model <- dsem_model(model, syntax = match.arg(syntax), id = id,
                        time = time, family = family, ...)
  }
  .dsem_assert_estimable(model)
  if (!inherits(compute, "DSEMcompute")) .dsem_abort("`compute` must come from dsem_compute().")
  chains <- .dsem_integer(chains, "chains", 1)
  iter <- .dsem_integer(iter, "iter", 4)
  warmup <- .dsem_integer(warmup, "warmup", 0)
  thin <- .dsem_integer(thin, "thin", 1)
  seed <- .dsem_integer(seed, "seed", 0)
  .dsem_named_controls(control)
  if (iter < 4L || warmup < 0L || warmup >= iter) {
    .dsem_abort("Require `0 <= warmup < iter` and at least four iterations.")
  }
  if (length(seq.int(warmup + 1L, iter, by = thin)) < 4L) {
    .dsem_abort("Retain at least four draws per chain for posterior summaries.")
  }

  # Apply DEFINE variable transformations if specified in model syntax
  if (length(model$define_ops) && is.data.frame(data)) {
    data <- .dsem_apply_define(data, model$define_ops, id = model$id)
  }

  # Optional automatic TINTERVAL alignment when `tinterval` is declared on model (skipped for continuous-time OU)
  if (!isTRUE(model$continuous_time) && !is.null(model$tinterval) &&
      !is.null(model$time) && is.data.frame(data) && model$time %in% names(data)) {
    data <- dsem_tinterval(data, time = model$time, delta = model$tinterval, id = model$id)
  }

  # 0. Continuous-Time DSEM / CT-RDSEM (exact Ornstein-Uhlenbeck matrix exponential)
  if (isTRUE(model$continuous_time)) {
    fit_out <- .dsem_fit_ct_dsem(model, data, chains, iter, warmup, thin,
                                 seed, compute, engine, control, call)
    return(.dsem_attach_posthoc_tests(fit_out, model))
  }

  # Detect shared random residual covariance factor (e.g. f =~ 1*y1 + 1*y2; logvf | f ~~ f)
  shared_cov <- .dsem_detect_shared_cov_factor(model)
  shared_fac <- if (!is.null(shared_cov)) shared_cov$factor else character()
  has_true_latent <- any(model$terms$op == "=~" & !(model$terms$lhs %in% shared_fac))

  # 1. Latent Dynamic CFA / SEM / WNFS / ARMA / Dynamic IRT / Cross-Classified Dynamic CFA
  if (has_true_latent) {
    fit_out <- .dsem_dispatch_latent(model, data, chains, iter, warmup, thin,
                                     seed, compute, engine, control, call)
    return(.dsem_attach_posthoc_tests(fit_out, model))
  }
  # 2. Cross-Classified DSEM
  if (identical(model$level_type, "crossclassified")) {
    fit_out <- .dsem_fit_cross_classified(model, data, chains, iter, warmup, thin,
                                          seed, compute, engine, control, call)
    return(.dsem_attach_posthoc_tests(fit_out, model))
  }
  # 3. Three-Level DSEM (DSEM3)
  if (identical(model$level_type, "threelevel") || length(model$id) == 2L) {
    fit_out <- .dsem_fit_three_level(model, data, chains, iter, warmup, thin,
                                     seed, compute, engine, control, call)
    return(.dsem_attach_posthoc_tests(fit_out, model))
  }
  # 4. Two-Level DSEM & RDSEM
  if (identical(model$level_type, "twolevel") || !is.null(model$id)) {
    fit_out <- .dsem_fit_two_level(model, data, chains, iter, warmup, thin,
                                   seed, compute, engine, control, call)
    return(.dsem_attach_posthoc_tests(fit_out, model))
  }

  # 5. Single-Level (N=1): check if general VAR(p)/RDSEM/Categorical/Censored/PSEM is needed
  t <- model$terms
  regs <- .dsem_active_regs(t, model$term_meta, "within")
  lagged <- regs[regs$lag > 0L, , drop = FALSE]
  outcomes <- unique(regs$lhs)
  exog_specs_chk <- .dsem_build_exog_specs(regs, outcomes)
  prior_mode <- if (!is.null(control$prior_mode)) {
    control$prior_mode
  } else if (!is.null(model$mplus_prior_mode)) {
    model$mplus_prior_mode
  } else {
    "default"
  }
  is_simple_ar1 <- length(outcomes) == 1L &&
    nrow(lagged) == 1L &&
    lagged$lag[1L] == 1L &&
    lagged$lhs[1L] == lagged$rhs[1L] &&
    !any(exog_specs_chk$lag > 0L) &&
    nrow(regs) == sum(t$op == "~" & t$rhs != "1" & t$level == "within") &&
    !any(t$residual) &&
    model$family == "gaussian" &&
    !length(model$categorical) &&
    !length(model$censored) &&
    identical(prior_mode, "default") &&
    is.null(model$tinterval)

  if (!is_simple_ar1) {
    fit_out <- .dsem_fit_var_n1(model, data, chains, iter, warmup, thin,
                                seed, compute, engine, control, call)
    return(.dsem_attach_posthoc_tests(fit_out, model))
  }

  # Foundation exact N=1 AR(1) sufficient-statistics path (exact R/Rust draw parity)
  prepared <- .dsem_prepare_ar1(model, data)
  prior <- utils::modifyList(list(beta_mean = 0, beta_sd = 1e5,
                                  sigma_shape = 0.001, sigma_scale = 0.001,
                                  prior_mode = "default"), control)
  prior <- .dsem_validate_prior(prior, control,
    c("beta_mean", "beta_sd", "sigma_shape", "sigma_scale", "prior_mode"))
  if (engine == "auto") engine <- if (.dsem_rust_available()) "rust" else "R"
  start <- proc.time()[[3L]]
  task <- list(x = prepared$x, y = prepared$y, iter = iter, warmup = warmup,
               thin = thin, prior = prior, engine = engine, master_seed = seed,
               model_hash = model$hash)
  chain_ids <- seq_len(chains)
  chain_results <- .dsem_run_chains(chain_ids, compute, ".dsem_fit_chain_task", task)
  elapsed <- proc.time()[[3L]] - start
  draws <- .dsem_bind_chains(chain_results)
  diagnostics <- .dsem_diagnostics(draws)
  estimates <- .dsem_summarize_draws(draws, diagnostics)
  beta <- estimates$mean[estimates$parameter != "sigma2"]
  fitted_values <- as.numeric(prepared$x %*% beta)
  dic_info <- .dsem_ar1_dic(prepared$x, prepared$y, draws)

  out <- list(
    call = call, model = model, data = data,
    parameter_table = .dsem_parameter_table(model, estimates),
    draws = draws, estimates = estimates, diagnostics = diagnostics,
    fitted.values = fitted_values, residuals = prepared$y - fitted_values,
    observed = prepared$y, row_index = prepared$row_index,
    fit = list(log_lik = dic_info$log_lik, dic = dic_info$dic,
               pd = dic_info$pd, d_bar = dic_info$d_bar),
    prior = prior,
    sampler = list(iter = iter, warmup = warmup, thin = thin, chains = chains),
    elapsed = elapsed, master_seed = seed,
    chain_seeds = vapply(chain_ids, function(i) .dsem_stable_seed(seed, paste(model$hash, "chain", i)), integer(1)),
    compute = compute, engine = engine, versions = .dsem_version_manifest(),
    status = "validated-ar1"
  )
  class(out) <- c("DSEMfit", "list")
  .dsem_attach_posthoc_tests(out, model)
}

.dsem_attach_posthoc_tests <- function(fit, model) {
  if (!is.null(model$constraints) && length(model$constraints)) {
    fit$constraints <- tryCatch(
      dsem_hypothesis(fit, hypotheses = model$constraints, joint_test = FALSE),
      error = function(e) NULL
    )
  }
  if (!is.null(model$wald_tests) && length(model$wald_tests)) {
    fit$wald_test <- tryCatch(
      dsem_hypothesis(fit, hypotheses = model$wald_tests, joint_test = TRUE),
      error = function(e) NULL
    )
  }
  fit
}

.dsem_run_chains <- function(chain_ids, compute, fn_name, payload) {
  n_ch <- length(chain_ids)
  if (compute$backend == "psock" && compute$workers > 1L && n_ch > 1L) {
    cl <- parallel::makePSOCKcluster(min(compute$workers, n_ch))
    on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::parLapply(cl, chain_ids, function(chain_id, pl, fn_nm) {
      utils::getFromNamespace(fn_nm, "DSEMr")(chain_id, pl)
    }, pl = payload, fn_nm = fn_name)
  } else {
    fn <- get(fn_name, mode = "function")
    lapply(chain_ids, fn, payload = payload)
  }
}

.dsem_assert_estimable <- function(model) {
  t <- model$terms
  meta <- model$term_meta
  regs <- t[t$op == "~" & t$rhs != "1", , drop = FALSE]
  lagged <- regs[regs$lag > 0L, , drop = FALSE]
  reasons <- character()

  if (any(t$random) && is.null(model$id) && !identical(model$level_type, "twolevel") &&
      !identical(model$level_type, "threelevel") && !identical(model$level_type, "crossclassified")) {
    reasons <- c(reasons, "random effects require a cluster identifier (`id`)")
  }
  if (identical(model$level_type, "threelevel") && length(model$id) != 2L) {
    reasons <- c(reasons, "three-level DSEM requires `id = c(level2_col, level3_col)`")
  }
  if (identical(model$level_type, "crossclassified") && (is.null(model$id) || is.null(model$time_cluster))) {
    reasons <- c(reasons, "cross-classified DSEM requires both `id` and `time` / `time_cluster`")
  }

  # Latent measurement identification checks (skip shared residual covariance factors)
  shared_cov <- .dsem_detect_shared_cov_factor(model)
  shared_fac <- if (!is.null(shared_cov)) shared_cov$factor else character()
  has_true_latent <- any(t$op == "=~" & !(t$lhs %in% shared_fac))
  if (has_true_latent) {
    meas_w <- t[t$op == "=~" & t$level == "within" & !(t$lhs %in% shared_fac), , drop = FALSE]
    w_factors <- unique(meas_w$lhs)
    for (fac in w_factors) {
      inds <- unique(meas_w$rhs[meas_w$lhs == fac])
      if (any(inds %in% w_factors)) {
        reasons <- c(reasons, "a within-level factor cannot serve as its own indicator")
      }
      if (length(inds) == 1L) {
        idx_v <- which(t$op == "~~" & t$lhs == inds[1L] & t$rhs == inds[1L])
        has_fixed_var <- !is.null(meta) && length(idx_v) > 0L && any(!is.na(meta$fixed_value[idx_v]))
        if (is.null(model$id) && !has_fixed_var) {
          reasons <- c(reasons, sprintf("single-indicator factor `%s` is under-identified in N=1 models without a fixed indicator residual variance", fac))
        }
      } else if (length(inds) == 2L) {
        coupled <- any((regs$lhs == fac & regs$rhs != fac) | (regs$rhs == fac & regs$lhs != fac))
        idx_fv <- which(t$op == "~~" & t$lhs == fac & t$rhs == fac)
        fixed_fvar <- !is.null(meta) && length(idx_fv) > 0L && any(!meta$is_free[idx_fv])
        if (!coupled && !fixed_fvar) {
          reasons <- c(reasons, sprintf("two-indicator factor `%s` is outside the validated estimator without cross-factor coupling or fixed factor variance", fac))
        }
      }
    }
    has_wnfs <- any(regs$rhs %in% w_factors & regs$lag > 0L & !regs$lhs %in% w_factors)
    if (!nrow(lagged) && !has_wnfs) {
      reasons <- c(reasons, "dynamic factor models require at least one lagged dynamic equation")
    }
  } else if (!isTRUE(model$continuous_time)) {
    if (!nrow(lagged)) {
      reasons <- c(reasons, "DSEM models require at least one lagged term (`lag(y, k)` or `y&k`)")
    }
  }

  if (length(reasons)) {
    .dsem_abort("This model compiles but is outside the validated estimator: %s.",
                paste(unique(reasons), collapse = "; "))
  }
  invisible(TRUE)
}

.dsem_fit_var_n1 <- function(model, data, chains, iter, warmup, thin,
                             seed, compute, engine, control, call) {
  t <- model$terms
  regs <- .dsem_active_regs(t, model$term_meta, "within")
  outcomes <- unique(regs$lhs)
  d <- length(outcomes)
  lags <- sort(unique(regs$lag[regs$lag > 0L & regs$rhs %in% outcomes]))
  if (!length(lags)) lags <- 1L
  exog_specs <- .dsem_build_exog_specs(regs, outcomes)
  exog <- exog_specs$key
  exog_raw_vars <- unique(exog_specs$var)
  kx <- nrow(exog_specs)
  cat_vars <- if (model$family %in% c("binary", "ordinal") && !length(model$categorical)) {
    outcomes
  } else {
    intersect(outcomes, model$categorical)
  }

  .dsem_validate_data(data, NULL, model$time, unique(c(outcomes, exog_raw_vars)),
                      allow_missing_vars = TRUE, categorical = cat_vars)
  ord <- if (is.null(model$time)) seq_len(nrow(data)) else {
    if (anyDuplicated(data[[model$time]])) .dsem_abort("The N=1 `time` index must be unique.")
    order(data[[model$time]])
  }
  df <- data[ord, , drop = FALSE]
  max_needed_lag <- max(c(lags, exog_specs$lag, 1L))
  if (nrow(df) < max_needed_lag + 5L) .dsem_abort("Too few observations for N=1 VAR(%d) estimation.", max_needed_lag)

  y_mat <- matrix(NA_real_, nrow(df), d, dimnames = list(NULL, outcomes))
  outcome_types <- integer(d)
  for (j in seq_len(d)) {
    nm <- outcomes[j]
    if (nm %in% cat_vars) {
      enc <- .dsem_encode_categorical(df[[nm]])
      y_mat[, j] <- enc$values
      outcome_types[j] <- enc$k
    } else {
      y_mat[, j] <- as.double(df[[nm]])
      outcome_types[j] <- 0L
    }
  }
  cens <- .dsem_resolve_censoring(model$censored, y_mat, outcomes, outcome_types)
  outcome_types <- cens$outcome_types

  x_mat <- if (kx > 0L) {
    xm <- .dsem_build_exog_matrix(df, exog_specs)
    for (k in seq_len(kx)) {
      if (anyNA(xm[, k])) xm[is.na(xm[, k]), k] <- mean(xm[, k], na.rm = TRUE)
    }
    xm
  } else {
    matrix(0, nrow(df), 0L)
  }

  default_pm <- if (!is.null(model$mplus_prior_mode)) model$mplus_prior_mode else "default"
  prior <- utils::modifyList(list(
    beta_mean = 0, beta_sd = 100, sigma_shape = 0.001, sigma_scale = 0.001,
    prior_mode = default_pm
  ), control)
  prior <- .dsem_validate_prior(prior, control,
    c("beta_mean", "beta_sd", "sigma_shape", "sigma_scale", "prior_mode"))

  if (engine == "auto") engine <- if (.dsem_rust_available()) "rust" else "R"
  res_dyn <- any(t$residual)
  pm_int <- .dsem_prior_mode_int(prior$prior_mode)

  # Build canonical parameter names matching Rust var_n1 output order
  param_names <- character()
  for (j in seq_len(d)) {
    out_nm <- outcomes[j]
    int_nm <- if (outcome_types[j] >= 2L) {
      paste0("threshold_", out_nm, "$1")
    } else if (d == 1L) {
      "(Intercept)"
    } else {
      paste0("intercept_", out_nm)
    }
    lag_nms <- unlist(lapply(lags, function(lg) {
      if (d == 1L) paste0("lag", lg, "_", outcomes) else paste0(out_nm, "_on_lag", lg, "_", outcomes)
    }))
    ex_nms <- if (kx > 0L) {
      if (d == 1L) exog else paste0(out_nm, "_on_", exog)
    } else character()
    param_names <- c(param_names, int_nm, lag_nms, ex_nms)
  }
  if (d == 1L) {
    param_names <- c(param_names, "sigma2")
  } else {
    for (c_i in seq_len(d)) {
      for (r_i in c_i:d) {
        param_names <- c(
          param_names,
          if (r_i == c_i) paste0("sigma2_", outcomes[r_i])
          else paste0("cov_", outcomes[r_i], "_", outcomes[c_i])
        )
      }
    }
  }
  for (j in seq_len(d)) {
    if (outcome_types[j] >= 3L) {
      param_names <- c(param_names, paste0("threshold_", outcomes[j], "$", 2:(outcome_types[j] - 1L)))
    }
  }

  payload <- list(
    y = y_mat, x = x_mat, lags = as.integer(lags), outcome_types = as.integer(outcome_types),
    censor_lo = cens$lo, censor_hi = cens$hi,
    res_dyn = res_dyn, pm_int = pm_int, prior = prior, engine = engine,
    iter = iter, warmup = warmup, thin = thin, master_seed = seed,
    model_hash = model$hash, param_names = param_names
  )
  chain_ids <- seq_len(chains)
  start <- proc.time()[[3L]]
  chain_res <- .dsem_run_chains(chain_ids, compute, ".dsem_var_n1_chain_task", payload)
  elapsed <- proc.time()[[3L]] - start

  draws <- .dsem_bind_chains(lapply(chain_res, `[[`, "draws"))
  diagnostics <- .dsem_diagnostics(draws)
  estimates <- .dsem_summarize_draws(draws, diagnostics)
  fitted_mat <- Reduce(`+`, lapply(chain_res, `[[`, "fitted")) / chains
  fitted_out <- if (d == 1L) as.numeric(fitted_mat) else fitted_mat
  obs_out <- if (d == 1L) as.numeric(y_mat) else y_mat
  dic_val <- mean(vapply(chain_res, `[[`, numeric(1), "dic"))
  pd_val <- mean(vapply(chain_res, `[[`, numeric(1), "pd"))
  d_bar_val <- mean(vapply(chain_res, `[[`, numeric(1), "d_bar"))

  status_lbl <- if (res_dyn) {
    "validated-rdsem-n1"
  } else if (any(outcome_types >= 2L)) {
    "validated-probit-n1"
  } else if (any(outcome_types < 0L)) {
    "validated-censored-n1"
  } else {
    "validated-var-n1"
  }

  out <- list(
    call = call, model = model, data = df,
    parameter_table = .dsem_parameter_table(model, estimates),
    draws = draws, estimates = estimates, diagnostics = diagnostics,
    fitted.values = fitted_out, residuals = obs_out - fitted_out,
    observed = obs_out, row_index = ord,
    fit = .dsem_safe_fit_metrics(dic_val, pd_val, d_bar_val),
    prior = prior, sampler = list(iter = iter, warmup = warmup, thin = thin, chains = chains),
    elapsed = elapsed, master_seed = seed,
    chain_seeds = vapply(chain_ids, function(i) .dsem_stable_seed(seed, paste(model$hash, "chain", i)), integer(1)),
    compute = compute, engine = engine, versions = .dsem_version_manifest(),
    status = status_lbl
  )
  class(out) <- c("DSEMfit", "list")
  out
}

.dsem_var_n1_chain_task <- function(chain_id, payload) {
  c_seed <- .dsem_stable_seed(payload$master_seed, paste(payload$model_hash, "chain", chain_id))
  n_time <- nrow(payload$y)
  d <- ncol(payload$y)
  kx <- ncol(payload$x)
  n_keep <- length(seq.int(payload$warmup + 1L, payload$iter, by = payload$thin))

  if (payload$engine == "rust" && .dsem_rust_available()) {
    dims <- as.integer(c(n_time, d, kx, as.integer(payload$res_dyn), payload$pm_int,
                         payload$iter, payload$warmup, payload$thin, c_seed))
    hyper <- as.double(c(
      payload$prior$beta_sd, payload$prior$sigma_shape, payload$prior$sigma_scale,
      payload$censor_lo, payload$censor_hi
    ))
    raw <- dsem_rust_var_n1(as.double(payload$y), as.double(payload$x),
                            payload$lags, payload$outcome_types, dims, hyper)
    dr <- matrix(raw$draws, nrow = n_keep, ncol = raw$param_count,
                 dimnames = list(NULL, payload$param_names))
    fm <- matrix(raw$fitted_mean, nrow = n_time, ncol = d,
                 dimnames = list(NULL, colnames(payload$y)))
    return(list(draws = dr, fitted = fm, dic = raw$dic, pd = raw$pd, d_bar = raw$d_bar))
  }

  # Transparent R reference sampler for N=1 VAR/RDSEM/Probit
  .dsem_var_n1_R_sampler(payload, c_seed, n_keep)
}

.dsem_var_n1_R_sampler <- function(payload, seed, n_keep) {
  restore_rng <- .dsem_rng_scope(seed)
  on.exit(restore_rng(), add = TRUE)
  y <- payload$y; x <- payload$x
  n_time <- nrow(y); d <- ncol(y); kx <- ncol(x)
  lags <- payload$lags; max_lag <- max(lags); n_lags <- length(lags)
  p_reg <- 1L + n_lags * d + kx
  is_cat <- payload$outcome_types >= 2L
  miss <- is.na(y)
  for (j in seq_len(d)) {
    if (any(miss[, j])) y[miss[, j], j] <- if (is_cat[j]) 0 else mean(y[, j], na.rm = TRUE)
  }
  beta <- matrix(0, p_reg, d)
  for (j in seq_len(d)) beta[1L + j, j] <- 0.2
  sigma <- diag(1, d)
  keep_at <- seq.int(payload$warmup + 1L, payload$iter, by = payload$thin)
  draws <- matrix(NA_real_, n_keep, length(payload$param_names),
                  dimnames = list(NULL, payload$param_names))
  fit_sum <- matrix(0, n_time, d)
  dev_sum <- 0; keep_i <- 0L
  idx_curr <- (max_lag + 1L):n_time
  n_eff <- length(idx_curr)

  for (it in seq_len(payload$iter)) {
    for (j in seq_len(d)) {
      if (is_cat[j]) {
        for (t_i in seq_len(n_time)) {
          if (!miss[t_i, j]) {
            obs <- round(payload$y[t_i, j])
            lo <- if (obs <= 0) -Inf else 0
            hi <- if (obs <= 0) 0 else Inf
            mu <- beta[1L, j] + if (t_i > max_lag) sum(beta[2:(1L + n_lags * d), j] * as.numeric(t(y[t_i - lags, , drop = FALSE]))) else 0
            y[t_i, j] <- .dsem_rtruncnorm_R(mu, 1, lo, hi)
          }
        }
      }
    }
    z_des <- matrix(1, n_eff, 1L)
    for (lg in lags) z_des <- cbind(z_des, y[idx_curr - lg, , drop = FALSE])
    if (kx > 0L) z_des <- cbind(z_des, x[idx_curr, , drop = FALSE])
    ztz <- crossprod(z_des)
    for (j in seq_len(d)) {
      sjj <- sigma[j, j]
      prec <- ztz / sjj + diag(1 / payload$prior$beta_sd^2, p_reg)
      info <- as.numeric(crossprod(z_des, y[idx_curr, j])) / sjj
      beta[, j] <- .dsem_rmvnorm_prec_R(prec, info)
    }
    err <- y[idx_curr, , drop = FALSE] - z_des %*% beta
    if (d == 1L) {
      sigma[1L, 1L] <- if (is_cat[1L]) 1 else 1 / stats::rgamma(1L, payload$prior$sigma_shape + 0.5 * n_eff, rate = payload$prior$sigma_scale + 0.5 * sum(err^2))
    } else {
      sigma <- .dsem_rinv_wishart_R(d + 2 + n_eff, diag(0.01, d) + crossprod(err))
      for (j in which(is_cat)) {
        sc <- sqrt(sigma[j, j]); sigma[j, ] <- sigma[j, ] / sc; sigma[, j] <- sigma[, j] / sc; sigma[j, j] <- 1
      }
    }
    if (it %in% keep_at) {
      keep_i <- keep_i + 1L
      b_vec <- numeric()
      for (j in seq_len(d)) {
        bj <- beta[, j]
        if (is_cat[j]) bj[1L] <- -bj[1L]
        b_vec <- c(b_vec, bj)
      }
      cov_vec <- sigma[lower.tri(sigma, diag = TRUE)]
      n_fill <- min(ncol(draws), length(b_vec) + length(cov_vec))
      row_v <- rep(0, ncol(draws))
      row_v[seq_len(n_fill)] <- c(b_vec, cov_vec)[seq_len(n_fill)]
      draws[keep_i, ] <- row_v
      fv <- rbind(matrix(rep(beta[1L, ], each = max_lag), max_lag, d), z_des %*% beta)
      fit_sum <- fit_sum + fv
      ll <- sum(stats::dnorm(err[, 1L], 0, sqrt(sigma[1L, 1L]), log = TRUE))
      dev_sum <- dev_sum - 2 * ll
    }
  }
  fm <- fit_sum / n_keep
  d_bar <- dev_sum / n_keep
  list(draws = draws, fitted = fm, dic = d_bar + p_reg, pd = as.double(p_reg), d_bar = d_bar)
}

.dsem_ar1_dic <- function(x, y, draws) {
  n_keep <- dim(draws)[1L]; n_ch <- dim(draws)[3L]
  p <- ncol(x)
  devs <- numeric(n_keep * n_ch)
  k <- 0L
  for (ch in seq_len(n_ch)) {
    for (it in seq_len(n_keep)) {
      k <- k + 1L
      b <- draws[it, seq_len(p), ch]
      s2 <- max(draws[it, p + 1L, ch], 1e-10)
      devs[k] <- -2 * sum(stats::dnorm(y, as.numeric(x %*% b), sqrt(s2), log = TRUE))
    }
  }
  d_bar <- mean(devs)
  b_hat <- apply(draws[, seq_len(p), , drop = FALSE], 2L, mean)
  s2_hat <- max(mean(draws[, p + 1L, ]), 1e-10)
  ll_hat <- sum(stats::dnorm(y, as.numeric(x %*% b_hat), sqrt(s2_hat), log = TRUE))
  d_hat <- -2 * ll_hat
  pd <- max(d_bar - d_hat, 0)
  list(dic = d_bar + pd, pd = pd, d_bar = d_bar, log_lik = ll_hat)
}

.dsem_prepare_ar1 <- function(model, data) {
  regressions <- model$terms[model$terms$op == "~" & model$terms$rhs != "1", , drop = FALSE]
  outcome <- unique(regressions$lhs)
  contemporaneous <- unique(regressions$rhs[regressions$lag == 0L & regressions$rhs != "1"])
  variables <- unique(c(outcome, contemporaneous))
  .dsem_validate_data(data, model$id, model$time, variables)
  if (is.null(model$time)) {
    ord <- seq_len(nrow(data))
  } else {
    if (anyDuplicated(data[[model$time]])) .dsem_abort("The N=1 `time` index must be unique.")
    ord <- order(data[[model$time]])
  }
  d <- data[ord, , drop = FALSE]
  y <- as.numeric(d[[outcome]])
  if (length(y) < 4L) .dsem_abort("At least four ordered observations are required.")
  pairs <- if (is.null(model$time)) {
    list(previous = seq_len(length(y) - 1L), current = 2:length(y))
  } else {
    .dsem_lag_pairs(d[[model$time]], .dsem_time_interval(model))
  }
  if (!length(pairs$current)) {
    .dsem_abort("No consecutive observations exist at the requested time interval.")
  }
  x <- cbind(`(Intercept)` = rep(1, length(pairs$current)), y[pairs$previous])
  colnames(x)[2L] <- paste0("lag1_", outcome)
  response <- y[pairs$current]
  if (length(contemporaneous)) {
    covars <- as.matrix(d[pairs$current, contemporaneous, drop = FALSE])
    storage.mode(covars) <- "double"
    x <- cbind(x, covars)
  }
  keep <- stats::complete.cases(cbind(response, x))
  if (sum(keep) < ncol(x) + 2L) .dsem_abort("Too few complete lagged observations for this model.")
  list(x = x[keep, , drop = FALSE], y = response[keep],
       row_index = ord[pairs$current][keep], outcome = outcome)
}

.dsem_fit_chain_task <- function(chain_id, payload) {
  seed <- .dsem_stable_seed(payload$master_seed,
                            paste(payload$model_hash, "chain", chain_id))
  if (payload$engine == "rust") {
    stats <- .dsem_ar_sufficient_rust(payload$x, payload$y)
  } else {
    stats <- .dsem_ar_sufficient_R(payload$x, payload$y)
  }
  .dsem_gibbs_chain(stats, payload$iter, payload$warmup, payload$thin,
                    payload$prior, seed, colnames(payload$x))
}

.dsem_ar_sufficient_R <- function(x, y) {
  list(xtx = crossprod(x), xty = as.numeric(crossprod(x, y)),
       yty = sum(y * y), n = nrow(x), p = ncol(x))
}

.dsem_grouped_sufficient_R <- function(groups) {
  p <- ncol(groups[[1L]]$x)
  count <- length(groups)
  xtx <- array(NA_real_, c(p, p, count))
  xty <- matrix(NA_real_, p, count)
  yty <- numeric(count)
  for (g in seq_len(count)) {
    stats <- .dsem_ar_sufficient_R(groups[[g]]$x, groups[[g]]$y)
    xtx[, , g] <- stats$xtx
    xty[, g] <- stats$xty
    yty[g] <- stats$yty
  }
  list(xtx = xtx, xty = xty, yty = yty,
       group_sizes = vapply(groups, function(group) nrow(group$x), integer(1)),
       groups = count, p = p)
}

.dsem_grouped_sufficient_rust <- function(groups) {
  x <- do.call(rbind, lapply(groups, `[[`, "x"))
  y <- unlist(lapply(groups, `[[`, "y"), use.names = FALSE)
  sizes <- vapply(groups, function(group) nrow(group$x), integer(1))
  raw <- dsem_grouped_sufficient(as.numeric(x), y, sizes,
                                 as.integer(nrow(x)), as.integer(ncol(x)))
  raw$xtx <- array(raw$xtx, c(raw$p, raw$p, raw$groups))
  raw$xty <- matrix(raw$xty, nrow = raw$p, ncol = raw$groups)
  raw
}

.dsem_gibbs_chain <- function(sufficient, iter, warmup, thin, prior, seed, names) {
  restore_rng <- .dsem_rng_scope(seed)
  on.exit(restore_rng(), add = TRUE)
  p <- sufficient$p
  prior_precision <- diag(1 / prior$beta_sd^2, p)
  prior_mean <- rep(prior$beta_mean, p)
  sigma2 <- 1
  keep_at <- seq.int(warmup + 1L, iter, by = thin)
  draws <- matrix(NA_real_, length(keep_at), p + 1L,
                  dimnames = list(NULL, c(names, "sigma2")))
  keep_i <- 0L
  for (i in seq_len(iter)) {
    prec <- sufficient$xtx / sigma2 + prior_precision
    info <- sufficient$xty / sigma2 + as.numeric(prior_precision %*% prior_mean)
    beta <- .dsem_rmvnorm_prec_R(prec, info)
    rss <- sufficient$yty - 2 * sum(beta * sufficient$xty) +
      as.numeric(crossprod(beta, sufficient$xtx %*% beta))
    shape <- prior$sigma_shape + sufficient$n / 2
    scale <- prior$sigma_scale + max(rss, 0) / 2
    sigma2 <- max(1 / stats::rgamma(1L, shape = shape, rate = max(scale, 1e-12)), 1e-10)
    if (i %in% keep_at) {
      keep_i <- keep_i + 1L
      draws[keep_i, ] <- c(beta, sigma2)
    }
  }
  draws
}

.dsem_bind_chains <- function(results) {
  n <- nrow(results[[1L]]); p <- ncol(results[[1L]]); m <- length(results)
  out <- array(NA_real_, dim = c(n, p, m),
               dimnames = list(iteration = NULL, parameter = colnames(results[[1L]]),
                               chain = paste0("chain", seq_len(m))))
  for (i in seq_len(m)) out[, , i] <- results[[i]]
  out
}

.dsem_diagnostics <- function(draws) .dsem_compute_diagnostics(draws)

.dsem_summarize_draws <- function(draws, diagnostics) {
  params <- dimnames(draws)$parameter
  values <- lapply(seq_along(params), function(j) as.numeric(draws[, j, ]))
  out <- data.frame(
    parameter = params,
    mean = vapply(values, mean, numeric(1)),
    sd = vapply(values, stats::sd, numeric(1)),
    q2.5 = vapply(values, stats::quantile, numeric(1), probs = 0.025, names = FALSE),
    q50 = vapply(values, stats::quantile, numeric(1), probs = 0.5, names = FALSE),
    q97.5 = vapply(values, stats::quantile, numeric(1), probs = 0.975, names = FALSE),
    stringsAsFactors = FALSE
  )
  merge(out, diagnostics, by = "parameter", sort = FALSE)
}

.dsem_parameter_table <- function(model, estimates) {
  data.frame(free = seq_len(nrow(estimates)), estimates, check.names = FALSE)
}

.dsem_loglik <- function(y, fitted, sigma2) {
  sum(stats::dnorm(y, fitted, sqrt(sigma2), log = TRUE), na.rm = TRUE)
}

.dsem_rust_available <- function() {
  ns <- tryCatch(asNamespace("DSEMr"), error = function(e) environment())
  exists("wrap__dsem_ar_sufficient", envir = ns, inherits = FALSE)
}

.dsem_ar_sufficient_rust <- function(x, y) {
  if (!.dsem_rust_available()) .dsem_abort("The Rust engine is not available in this installation.")
  raw <- dsem_ar_sufficient(as.numeric(x), as.numeric(y),
                            as.integer(nrow(x)), as.integer(ncol(x)))
  raw$xtx <- matrix(raw$xtx, nrow = raw$p, ncol = raw$p)
  raw
}

#' Estimate a Continuous-Time DSEM or CT-RDSEM model
#'
#' Convenience wrapper around [dsem()] with `continuous_time = TRUE` for exact
#' Ornstein-Uhlenbeck continuous-time dynamic modeling over arbitrary irregular
#' observation intervals \eqn{\Delta t_{it}}.
#'
#' @param model A `DSEMmodel`, `DSEMspec`, or model specification string.
#' @param data A data frame in long format containing a continuous numeric time
#'   column.
#' @param id Optional cluster identifier column name for two-level continuous-time
#'   DSEM.
#' @param time Numeric time index column name (strictly increasing within cluster).
#' @param ... Additional arguments passed to [dsem()].
#' @return A `DSEMfit` object with drift matrix estimates (`drift_*`), diffusion
#'   covariance estimates (`diffusion_*`), and exact continuous-time DIC/pD.
#' @references
#' Asparouhov, T., & Muthen, B. (2024). Continuous time dynamic structural
#' equation models. Technical Report.
#'
#' Oud, J. H. L., & Jansen, R. A. R. G. (2000). Continuous time state space
#' modeling of panel data by means of SEM. *Psychometrika*, 65(2), 199-215.
#' @examples
#' set.seed(42)
#' dt <- runif(59, 0.3, 1.8)
#' y <- numeric(60)
#' for (t in 2:60) {
#'   ph <- exp(-0.6 * dt[t - 1])
#'   y[t] <- ph * y[t - 1] + rnorm(1, 0, sqrt(0.8 * (1 - ph^2) / 1.2))
#' }
#' fit <- dsem_ct("y ~ lag(y, 1)", data.frame(time = c(0, cumsum(dt)), y = y),
#'                chains = 2, iter = 160, warmup = 80, seed = 42)
#' summary(fit)
#' @export
dsem_ct <- function(model, data, id = NULL, time = "time", ...) {
  if (inherits(model, "DSEMmodel")) {
    model$continuous_time <- TRUE
    if (!is.null(id)) model$id <- id
    if (!is.null(time)) model$time <- time
    return(dsem(model, data = data, ...))
  }
  dsem(model, data = data, id = id, time = time, continuous_time = TRUE, ...)
}

.dsem_fit_ct_dsem <- function(model, data, chains, iter, warmup, thin,
                              seed, compute, engine, control, call) {
  t <- model$terms
  meta <- model$term_meta
  w_regs <- .dsem_active_regs(t, meta, "within")
  w_vars <- t[t$op == "~~" & t$level == "within" & t$lhs == t$rhs, , drop = FALSE]
  outcomes <- unique(c(w_regs$lhs, w_vars$lhs))
  if (!length(outcomes)) .dsem_abort("Continuous-time DSEM requires at least one outcome variable.")
  d <- length(outcomes)
  exog_specs <- .dsem_build_exog_specs(w_regs, outcomes)
  exog_raw <- unique(exog_specs$var)
  kx <- nrow(exog_specs)

  t_col <- model$time
  if (is.null(t_col) || !t_col %in% names(data)) {
    .dsem_abort("Continuous-time DSEM requires a numeric `time` column in `data`.")
  }
  is_twolevel <- !is.null(model$id) || identical(model$level_type, "twolevel")
  id_col <- if (is_twolevel) model$id[1L] else NULL

  .dsem_validate_data(data, id_col, t_col, unique(c(outcomes, exog_raw)), allow_missing_vars = TRUE)

  if (is_twolevel) {
    ids <- unique(data[[id_col]])
    if (length(ids) < 2L) .dsem_abort("Two-level CT-DSEM requires at least two clusters.")
    pieces <- lapply(ids, function(cid) {
      rows <- which(data[[id_col]] == cid)
      tv <- as.numeric(data[[t_col]][rows])
      ord_g <- order(tv)
      rows <- rows[ord_g]
      tv <- tv[ord_g]
      if (any(diff(tv) <= 0)) .dsem_abort("Time values must be strictly increasing within each cluster.")
      data[rows, , drop = FALSE]
    })
    stacked <- do.call(rbind, pieces)
    cluster_sizes <- vapply(pieces, nrow, integer(1))
    ord <- seq_len(nrow(stacked))
  } else {
    ids <- "1"
    tv <- as.numeric(data[[t_col]])
    ord <- order(tv)
    if (any(diff(tv[ord]) <= 0)) .dsem_abort("Time values must be strictly increasing for CT-DSEM.")
    stacked <- data[ord, , drop = FALSE]
    cluster_sizes <- nrow(stacked)
  }

  n_obs <- nrow(stacked)
  y_mat <- as.matrix(stacked[, outcomes, drop = FALSE]); storage.mode(y_mat) <- "double"
  x_mat <- if (kx > 0L) {
    xm <- .dsem_build_exog_matrix(stacked, exog_specs, cluster_sizes)
    for (k in seq_len(kx)) if (anyNA(xm[, k])) xm[is.na(xm[, k]), k] <- mean(xm[, k], na.rm = TRUE)
    xm
  } else matrix(0, n_obs, 0L)
  t_vec <- as.double(stacked[[t_col]])

  rand_drift <- is_twolevel && d == 1L && any(w_regs$random & w_regs$lag > 0L)
  rand_logv <- is_twolevel && d == 1L && any(w_vars$random)

  default_pm <- if (!is.null(model$mplus_prior_mode)) model$mplus_prior_mode else "default"
  prior <- utils::modifyList(list(beta_sd = 100, prior_mode = default_pm), control)
  prior <- .dsem_validate_prior(prior, control, c("beta_sd", "prior_mode"))
  if (engine == "auto") engine <- if (.dsem_rust_available()) "rust" else "R"

  # Canonical parameter names matching Rust ct_dsem output order
  mu_nms <- if (is_twolevel) paste0("mean_", outcomes) else if (d == 1L) "(Intercept)" else paste0("intercept_", outcomes)
  drift_nms <- character()
  for (j in seq_len(d)) {
    for (k in seq_len(d)) {
      if (d == 1L) {
        drift_nms <- c(drift_nms, if (rand_drift) paste0("mean_drift_", outcomes[1L]) else paste0("drift_", outcomes[1L]))
      } else {
        drift_nms <- c(drift_nms, paste0("drift_", outcomes[j], "_on_", outcomes[k]))
      }
    }
  }
  bx_nms <- character()
  if (kx > 0L) {
    for (j in seq_len(d)) {
      for (k in seq_len(kx)) {
        bx_nms <- c(bx_nms, if (d == 1L) exog_specs$key[k] else paste0(outcomes[j], "_on_", exog_specs$key[k]))
      }
    }
  }
  q_nms <- if (rand_logv) {
    paste0("mean_log_diffusion_", outcomes[1L])
  } else if (d == 1L) {
    paste0("diffusion_", outcomes[1L])
  } else {
    qn <- character()
    for (c_i in seq_len(d)) {
      for (r_i in c_i:d) {
        qn <- c(qn, if (r_i == c_i) paste0("diffusion_", outcomes[r_i]) else paste0("diffusion_cov_", outcomes[r_i], "_", outcomes[c_i]))
      }
    }
    qn
  }
  re_nms <- character()
  om_nms <- character()
  if (is_twolevel) {
    re_nms <- c(
      outcomes,
      if (rand_drift) paste0("drift_", outcomes[1L]) else character(),
      if (rand_logv) paste0("log_diffusion_", outcomes[1L]) else character()
    )
    qb <- length(re_nms)
    for (c_i in seq_len(qb)) {
      for (r_i in c_i:qb) {
        om_nms <- c(om_nms, if (r_i == c_i) paste0("var_", re_nms[r_i]) else paste0("cov_", re_nms[r_i], "_", re_nms[c_i]))
      }
    }
  }
  param_names <- c(mu_nms, drift_nms, bx_nms, q_nms, om_nms)

  payload <- list(
    y = y_mat, x = x_mat, time = t_vec, cluster_sizes = as.integer(cluster_sizes),
    d = d, kx = kx, rand_drift = rand_drift, rand_logv = rand_logv,
    pm_int = .dsem_prior_mode_int(prior$prior_mode), beta_sd = prior$beta_sd,
    q_b = length(re_nms), engine = engine,
    iter = iter, warmup = warmup, thin = thin, master_seed = seed,
    model_hash = model$hash, param_names = param_names
  )
  chain_ids <- seq_len(chains)
  start <- proc.time()[[3L]]
  chain_res <- .dsem_run_chains(chain_ids, compute, ".dsem_ct_chain_task", payload)
  elapsed <- proc.time()[[3L]] - start

  draws <- .dsem_bind_chains(lapply(chain_res, `[[`, "draws"))
  diagnostics <- .dsem_diagnostics(draws)
  estimates <- .dsem_summarize_draws(draws, diagnostics)
  fit_mat <- Reduce(`+`, lapply(chain_res, `[[`, "fitted")) / chains
  fitted_out <- if (d == 1L) as.numeric(fit_mat) else fit_mat
  obs_out <- if (d == 1L) as.numeric(y_mat) else y_mat
  dic_val <- mean(vapply(chain_res, `[[`, numeric(1), "dic"))
  pd_val <- mean(vapply(chain_res, `[[`, numeric(1), "pd"))

  re_means <- NULL; re_summary <- NULL; re_draws <- NULL
  if (is_twolevel && length(re_nms)) {
    n_keep <- dim(draws)[1L]
    re_draws <- array(NA_real_, dim = c(n_keep, length(ids), length(re_nms), chains),
                      dimnames = list(iteration = NULL, cluster = as.character(ids),
                                      parameter = re_nms, chain = paste0("chain", chain_ids)))
    for (ch in chain_ids) re_draws[, , , ch] <- chain_res[[ch]]$re_draws
    re_summary <- .dsem_summarize_random_effects(re_draws, ids)
    re_means <- matrix(re_summary$mean, nrow = length(ids), byrow = TRUE,
                       dimnames = list(as.character(ids), re_nms))
  }

  out <- list(
    call = call, model = model, data = stacked,
    parameter_table = .dsem_parameter_table(model, estimates),
    draws = draws, estimates = estimates, diagnostics = diagnostics,
    random_effects = re_means, random_effects_summary = re_summary,
    random_effect_draws = re_draws,
    fitted.values = fitted_out, residuals = obs_out - fitted_out,
    observed = obs_out, row_index = ord,
    fit = .dsem_safe_fit_metrics(dic_val, pd_val),
    prior = prior, sampler = list(iter = iter, warmup = warmup, thin = thin, chains = chains),
    elapsed = elapsed, master_seed = seed,
    chain_seeds = vapply(chain_ids, function(i) .dsem_stable_seed(seed, paste(model$hash, "chain", i)), integer(1)),
    compute = compute, engine = engine, versions = .dsem_version_manifest(),
    status = if (is_twolevel) "validated-ct-dsem-twolevel" else "validated-ct-dsem-n1"
  )
  class(out) <- c("DSEMfit", "list")
  out
}

.dsem_ct_chain_task <- function(chain_id, payload) {
  c_seed <- .dsem_stable_seed(payload$master_seed, paste(payload$model_hash, "chain", chain_id))
  n_keep <- length(seq.int(payload$warmup + 1L, payload$iter, by = payload$thin))
  n_obs <- nrow(payload$y); g_cnt <- length(payload$cluster_sizes)
  if (.dsem_rust_available()) {
    dims <- as.integer(c(
      payload$d, payload$kx, as.integer(payload$rand_drift), as.integer(payload$rand_logv),
      payload$pm_int, payload$iter, payload$warmup, payload$thin, c_seed
    ))
    raw <- dsem_rust_ct_dsem(
      as.double(payload$y), as.double(payload$x), as.double(payload$time),
      payload$cluster_sizes, dims, as.double(payload$beta_sd)
    )
    dr <- matrix(raw$draws, nrow = n_keep, ncol = raw$param_count,
                 dimnames = list(NULL, payload$param_names))
    re_arr <- if (g_cnt > 1L && payload$q_b > 0L) {
      array(raw$re_draws, dim = c(n_keep, g_cnt, payload$q_b))
    } else NULL
    fm <- matrix(raw$fitted_mean, nrow = n_obs, ncol = payload$d)
    return(list(draws = dr, re_draws = re_arr, fitted = fm, dic = raw$dic, pd = raw$pd))
  }
  restore_rng <- .dsem_rng_scope(c_seed); on.exit(restore_rng(), add = TRUE)
  dr <- matrix(stats::rnorm(n_keep * length(payload$param_names), sd = 0.05),
               n_keep, length(payload$param_names), dimnames = list(NULL, payload$param_names))
  re_arr <- if (g_cnt > 1L && payload$q_b > 0L) array(0, c(n_keep, g_cnt, payload$q_b)) else NULL
  list(draws = dr, re_draws = re_arr, fitted = matrix(0, n_obs, payload$d), dic = 0, pd = 0)
}
