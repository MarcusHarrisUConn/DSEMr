#' Simulate dynamic structural equation model data
#'
#' Generates synthetic intensive longitudinal datasets for single-level (`N=1`),
#' two-level (`clusters` of length 1), Mplus Version 9 three-level (`clusters`
#' of length 2: `c(n_level3, n_level2_per_level3)`), cross-classified
#' (`cross_classified = TRUE`), binary/ordinal probit (`family = "binary"` or
#' `"ordinal"`), and latent dynamic factor models.
#'
#' @param model Optional compiled `DSEMmodel` used to infer variable names,
#'   hierarchy, and measurement structure.
#' @param n Number of retained time points per series.
#' @param intercept,ar,sigma Baseline location, autoregressive slope, and
#'   innovation standard deviation.
#' @param burnin Number of initialization observations discarded before `n`.
#' @param seed Integer reproducibility seed (caller RNG state is preserved).
#' @param outcome Name of the generated outcome column.
#' @param time Name of the generated time index column.
#' @param clusters Optional integer specifying cluster counts: length 1 for
#'   two-level (`N` individuals), or length 2 `c(n_l3, n_l2_per_l3)` for
#'   three-level DSEM3 (`n_l3` Level-3 clusters, each containing
#'   `n_l2_per_l3` Level-2 units).
#' @param random_sd Numeric vector of random-effect SDs (`c(sd_intercept, sd_ar)`
#'   or `c(sd_intercept, sd_ar, sd_logv)` for location-scale DSEM).
#' @param random_cor Correlation between random intercept and random AR slope.
#' @param id Character vector of cluster column name(s) (`"id"` for two-level or
#'   `c("l2_id", "l3_id")` for three-level).
#' @param loadings,indicator_intercepts,residual_sd Measurement parameters when
#'   generating dynamic factor indicators.
#' @param family Outcome family: `"gaussian"`, `"binary"`, or `"ordinal"`.
#' @param cross_classified Logical; if `TRUE`, adds occasion-specific random
#'   shocks with SD `time_sd`.
#' @param time_sd Standard deviation of occasion shocks when `cross_classified = TRUE`.
#' @return A data frame in long format with generating parameter truth attached
#'   in `attr(out, "dsem_truth")`.
#' @examples
#' # N=1 Gaussian AR(1)
#' d1 <- simulate_dsem(n = 50, ar = 0.5, seed = 1)
#' head(d1)
#'
#' # Three-level DSEM3 (5 Level-3 clusters x 4 Level-2 bursts x 15 occasions)
#' d3 <- simulate_dsem(n = 15, clusters = c(5, 4), id = c("day", "person"),
#'                     ar = 0.4, seed = 2)
#' head(d3)
#' @export
simulate_dsem <- function(model = NULL, n = 200L, intercept = 0, ar = 0.5,
                          sigma = 1, burnin = 100L, seed = 1L,
                          outcome = "y", time = "time", clusters = NULL,
                          random_sd = c(0.3, 0.1), random_cor = 0,
                          id = "id", loadings = NULL,
                          indicator_intercepts = 0, residual_sd = 0.5,
                          family = c("gaussian", "binary", "ordinal"),
                          cross_classified = FALSE, time_sd = 0.25) {
  family <- match.arg(family)
  latent_indicators <- NULL
  latent_factor <- NULL
  if (!is.null(model)) {
    if (!inherits(model, "DSEMmodel")) .dsem_abort("`model` must be a DSEMmodel.")
    .dsem_assert_estimable(model)
    regressions <- model$terms[model$terms$op == "~" & model$terms$rhs != "1" & model$terms$level == "within", , drop = FALSE]
    if (nrow(regressions)) outcome <- regressions$lhs[1L]
    measurement <- model$terms[model$terms$op == "=~" & model$terms$level == "within", , drop = FALSE]
    if (nrow(measurement)) {
      latent_factor <- unique(measurement$lhs)[1L]
      latent_indicators <- unique(measurement$rhs[measurement$lhs == latent_factor])
      outcome <- latent_factor
    }
    if (!is.null(model$id)) {
      id <- model$id
      if (is.null(clusters)) clusters <- if (length(id) == 2L) c(10L, 4L) else 50L
    }
    if (!is.null(model$time)) time <- model$time
    if (identical(model$level_type, "crossclassified")) cross_classified <- TRUE
    if (model$family != "gaussian") family <- model$family
  }

  n <- .dsem_integer(n, "n", 4)
  burnin <- .dsem_integer(burnin, "burnin", 0)
  seed <- .dsem_integer(seed, "seed", 0)
  restore_rng <- .dsem_rng_scope(seed)
  on.exit(restore_rng(), add = TRUE)
  if (as.double(n) + burnin > .Machine$integer.max) .dsem_abort("`n + burnin` exceeds the supported length.")
  for (field in c("intercept", "ar", "sigma", "random_cor", "time_sd")) {
    value <- get(field)
    if (!is.numeric(value) || length(value) != 1L || !is.finite(value)) {
      .dsem_abort("`%s` must be one finite number.", field)
    }
  }
  if (abs(ar) >= 1) .dsem_abort("`ar` must be strictly between -1 and 1 for stationary simulation.")
  if (sigma <= 0) .dsem_abort("`sigma` must be positive.")

  columns <- c(time, if (is.null(latent_indicators)) outcome else latent_indicators,
               if (!is.null(clusters)) id)
  if (!is.character(columns) || anyNA(columns) || any(!nzchar(columns)) || anyDuplicated(columns)) {
    .dsem_abort("Simulation column names must be non-empty, distinct strings.")
  }

  discretize_vec <- function(v) {
    if (family == "binary") as.double(v > 0)
    else if (family == "ordinal") as.double(findInterval(v, c(-0.8, 0.8)))
    else v
  }

  # Three-Level DSEM3 simulation: clusters = c(n_l3, n_l2_per_l3)
  if (!is.null(clusters) && length(clusters) == 2L) {
    n_l3 <- .dsem_integer(clusters[1L], "clusters[1]", 2)
    n_l2 <- .dsem_integer(clusters[2L], "clusters[2]", 2)
    if (length(id) == 1L) id <- c(paste0(id, "_l2"), paste0(id, "_l3"))
    sd_int3 <- random_sd[1L]
    sd_ar3 <- if (length(random_sd) >= 2L) random_sd[2L] else 0.05
    sd_lv3 <- if (length(random_sd) >= 3L) random_sd[3L] else 0
    sd_int2 <- 0.7 * sd_int3

    set.seed(as.integer(seed))
    u3_int <- stats::rnorm(n_l3, 0, sd_int3)
    u3_ar <- pmin(pmax(ar + stats::rnorm(n_l3, 0, sd_ar3), -0.95), 0.95)
    u3_lv <- log(sigma^2) + stats::rnorm(n_l3, 0, sd_lv3)

    pieces <- list()
    k_idx <- 0L
    for (g in seq_len(n_l3)) {
      for (c_i in seq_len(n_l2)) {
        k_idx <- k_idx + 1L
        u2_int <- stats::rnorm(1L, 0, sd_int2)
        mu_gc <- intercept + u3_int[g] + u2_int
        sig_gc <- exp(0.5 * u3_lv[g])
        sub <- simulate_dsem(
          n = n, intercept = mu_gc * (1 - u3_ar[g]), ar = u3_ar[g],
          sigma = sig_gc, burnin = burnin,
          seed = .dsem_stable_seed(seed, paste("sim-3l", g, c_i)),
          outcome = outcome, time = time
        )
        sub[[outcome]] <- discretize_vec(sub[[outcome]])
        sub[[id[1L]]] <- c_i
        sub[[id[2L]]] <- g
        pieces[[k_idx]] <- sub[c(id[2L], id[1L], time, outcome)]
      }
    }
    out <- do.call(rbind, pieces)
    rownames(out) <- NULL
    attr(out, "dsem_truth") <- list(
      population = c(intercept = intercept, ar = ar, logv = log(sigma^2)),
      l3_sd = c(intercept = sd_int3, ar = sd_ar3, logv = sd_lv3),
      l2_sd = c(intercept = sd_int2)
    )
    return(out)
  }

  # Dynamic factor simulation (N=1 or Two-Level)
  if (!is.null(latent_indicators)) {
    indicators <- latent_indicators
    count <- length(indicators)
    if (intercept != 0) .dsem_abort("Latent simulation fixes the state intercept to zero; use `indicator_intercepts` for location.")
    if (is.null(loadings)) loadings <- c(1, rep(0.8, count - 1L))
    if (length(loadings) == 1L) loadings <- rep(loadings, count)
    if (length(indicator_intercepts) == 1L) indicator_intercepts <- rep(indicator_intercepts, count)
    if (length(residual_sd) == 1L) residual_sd <- rep(residual_sd, count)
    if (!is.numeric(loadings) || !is.numeric(indicator_intercepts) || !is.numeric(residual_sd) ||
        length(loadings) != count || length(indicator_intercepts) != count ||
        length(residual_sd) != count || any(!is.finite(c(loadings, indicator_intercepts, residual_sd))) ||
        any(residual_sd <= 0)) {
      .dsem_abort("Latent simulation parameters must be finite vectors matching the number of indicators, with positive `residual_sd`.")
    }
    if (family == "gaussian" && loadings[1L] != 1) {
      .dsem_abort("The first (marker) loading must be one for the fitted model's scale.")
    }
    if (!is.null(clusters)) {
      g_cnt <- .dsem_integer(clusters[1L], "clusters", 2)
      pieces <- lapply(seq_len(g_cnt), function(cl) {
        ar_c <- pmin(pmax(ar + stats::rnorm(1L, 0, random_sd[min(2L, length(random_sd))]), -0.95), 0.95)
        fb_c <- stats::rnorm(1L, 0, random_sd[1L])
        sub <- simulate_dsem(
          n = n, intercept = 0, ar = ar_c, sigma = sigma, burnin = burnin,
          seed = .dsem_stable_seed(seed, paste("lat-cl", cl)),
          outcome = latent_factor, time = time
        )
        eta <- sub[[latent_factor]]
        obs <- vapply(seq_len(count), function(j) {
          v <- indicator_intercepts[j] + 0.6 * fb_c + loadings[j] * eta + stats::rnorm(n, 0, residual_sd[j])
          discretize_vec(v)
        }, numeric(n))
        df_c <- data.frame(cl, sub[[time]], obs, check.names = FALSE)
        names(df_c) <- c(id[1L], time, indicators)
        df_c
      })
      out <- do.call(rbind, pieces); rownames(out) <- NULL
      return(out)
    }

    state_data <- simulate_dsem(n = n, intercept = intercept, ar = ar,
                                sigma = sigma, burnin = burnin, seed = seed,
                                outcome = latent_factor, time = time)
    state <- state_data[[latent_factor]]
    set.seed(.dsem_stable_seed(seed, "latent-indicators"))
    observed <- vapply(seq_len(count), function(j) {
      v <- indicator_intercepts[j] + loadings[j] * state + stats::rnorm(n, 0, residual_sd[j])
      discretize_vec(v)
    }, numeric(n))
    out <- data.frame(state_data[[time]], observed, check.names = FALSE)
    names(out) <- c(time, indicators)
    attr(out, "dsem_truth") <- list(
      state_intercept = intercept, ar = ar, factor_variance = sigma^2,
      factor = latent_factor, latent_state = state,
      loadings = stats::setNames(loadings, indicators),
      indicator_intercepts = stats::setNames(indicator_intercepts, indicators),
      residual_variances = stats::setNames(residual_sd^2, indicators)
    )
    return(out)
  }

  # Two-Level / Cross-Classified cluster simulation
  if (!is.null(clusters)) {
    clusters <- .dsem_integer(clusters[1L], "clusters", 2)
    if (!is.numeric(random_sd) || length(random_sd) < 2L ||
        any(!is.finite(random_sd)) || any(random_sd < 0)) {
      .dsem_abort("`random_sd` must contain at least two non-negative finite values.")
    }
    if (abs(random_cor) >= 1) .dsem_abort("`random_cor` must be strictly between -1 and 1.")
    set.seed(as.integer(seed))
    z1 <- stats::rnorm(clusters)
    z2 <- random_cor * z1 + sqrt(1 - random_cor^2) * stats::rnorm(clusters)
    cluster_intercept <- intercept + random_sd[1L] * z1
    cluster_ar <- ar + random_sd[2L] * z2
    if (any(abs(cluster_ar) >= 1)) {
      .dsem_abort("Generated cluster AR effects are nonstationary; reduce `ar` or `random_sd[2]`.")
    }
    cluster_sig <- if (length(random_sd) >= 3L && random_sd[3L] > 0) {
      exp(0.5 * (log(sigma^2) + random_sd[3L] * stats::rnorm(clusters)))
    } else {
      rep(sigma, clusters)
    }
    time_shocks <- if (cross_classified) stats::rnorm(n, 0, time_sd) else rep(0, n)

    pieces <- lapply(seq_len(clusters), function(cluster) {
      dat <- simulate_dsem(
        n = n, intercept = cluster_intercept[cluster], ar = cluster_ar[cluster],
        sigma = cluster_sig[cluster], burnin = burnin,
        seed = .dsem_stable_seed(seed, paste("simulate-cluster", cluster)),
        outcome = outcome, time = time
      )
      dat[[outcome]] <- discretize_vec(dat[[outcome]] + time_shocks)
      dat[[id[1L]]] <- cluster
      dat[c(id[1L], time, outcome)]
    })
    out <- do.call(rbind, pieces)
    rownames(out) <- NULL
    attr(out, "dsem_truth") <- list(
      population = c(intercept = intercept, ar = ar, sigma2 = sigma^2),
      random_sd = random_sd, random_cor = random_cor,
      cluster_intercept = cluster_intercept, cluster_ar = cluster_ar,
      time_shocks = if (cross_classified) time_shocks else NULL
    )
    return(out)
  }

  set.seed(as.integer(seed))
  total <- n + burnin
  y <- numeric(total)
  y[1L] <- stats::rnorm(1L, intercept / (1 - ar), sigma / sqrt(1 - ar^2))
  if (total > 1L) {
    for (i in 2:total) y[i] <- intercept + ar * y[i - 1L] + stats::rnorm(1L, 0, sigma)
  }
  y <- discretize_vec(utils::tail(y, n))
  out <- data.frame(seq_len(n), y, check.names = FALSE)
  names(out) <- c(time, outcome)
  out
}

#' Run a reproducible DSEM Monte Carlo study
#'
#' @param model A compiled model or model string accepted by [dsem()].
#' @param replications Number of simulation replications.
#' @param generator Function called as `generator(seed = <derived seed>)`.
#' @param seed Master simulation seed.
#' @param compute Compute specification from [dsem_compute()].
#' @param fit_args Named list passed to [dsem()].
#' @param truth Optional named vector of true parameter values.
#' @return A `DSEMmontecarlo` list with replication estimates and summaries.
#' @examples
#' mc <- dsem_monte_carlo(
#'   "y ~ lag(y, 1)", replications = 2,
#'   generator = function(seed) simulate_dsem(n = 50, ar = 0.5, seed = seed),
#'   fit_args = list(chains = 2, iter = 120, warmup = 60),
#'   truth = c(lag1_y = 0.5)
#' )
#' mc$summary
#' @export
dsem_monte_carlo <- function(model, replications = 100L,
                             generator = function(seed) simulate_dsem(seed = seed),
                             seed = 20260813L, compute = dsem_compute(),
                             fit_args = list(chains = 2L, iter = 1000L, warmup = 500L),
                             truth = NULL) {
  replications <- .dsem_integer(replications, "replications", 1)
  seed <- .dsem_integer(seed, "seed", 0)
  if (!inherits(compute, "DSEMcompute")) .dsem_abort("`compute` must come from dsem_compute().")
  if (!is.function(generator)) .dsem_abort("`generator` must be a function.")
  .dsem_named_controls(fit_args)
  if (any(names(fit_args) %in% c("model", "data", "seed", "compute"))) {
    .dsem_abort("`fit_args` cannot override model, data, seed, or compute; replications own these settings.")
  }
  if (!is.null(truth) && (!is.numeric(truth) || is.null(names(truth)) ||
      any(!is.finite(truth)) || anyDuplicated(names(truth)) || any(!nzchar(names(truth))))) {
    .dsem_abort("`truth` must be a uniquely named finite numeric vector.")
  }
  if (!inherits(model, "DSEMmodel")) model <- dsem_model(model)
  rep_seeds <- vapply(seq_len(replications), function(i) {
    .dsem_stable_seed(seed, paste(model$hash, "replication", i))
  }, integer(1))
  run_one <- function(i) {
    dat <- generator(seed = rep_seeds[i])
    args <- c(list(model = model, data = dat,
                   seed = .dsem_stable_seed(seed, paste("fit", i)),
                   compute = dsem_compute()), fit_args)
    fit <- do.call(dsem, args)
    data.frame(replication = i, seed = rep_seeds[i], fit$estimates,
               converged = dim(fit$draws)[3L] >= 2L &&
                 all(is.finite(fit$diagnostics$rhat) & fit$diagnostics$rhat < 1.01) &&
                 all(is.finite(fit$diagnostics$ess_bulk) & fit$diagnostics$ess_bulk >= 100) &&
                 all(is.finite(fit$diagnostics$ess_tail) & fit$diagnostics$ess_tail >= 100),
               stringsAsFactors = FALSE)
  }
  ids <- seq_len(replications)
  estimates <- if (compute$backend == "psock" && compute$workers > 1L && replications > 1L) {
    cl <- parallel::makePSOCKcluster(min(compute$workers, replications))
    on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::parLapply(cl, ids, run_one)
  } else {
    lapply(ids, run_one)
  }
  estimates <- do.call(rbind, estimates)
  summary <- stats::aggregate(cbind(mean, sd) ~ parameter, estimates, mean)
  names(summary)[names(summary) == "mean"] <- "mean_estimate"
  summary$convergence_rate <- vapply(summary$parameter, function(p) {
    mean(estimates$converged[estimates$parameter == p])
  }, numeric(1))
  if (!is.null(truth)) {
    summary$truth <- unname(truth[summary$parameter])
    summary$bias <- summary$mean_estimate - summary$truth
    summary$rmse <- vapply(seq_len(nrow(summary)), function(j) {
      sqrt(mean((estimates$mean[estimates$parameter == summary$parameter[j]] - summary$truth[j])^2))
    }, numeric(1))
    summary$coverage <- vapply(seq_len(nrow(summary)), function(j) {
      e <- estimates[estimates$parameter == summary$parameter[j], ]
      mean(e$q2.5 <= summary$truth[j] & e$q97.5 >= summary$truth[j])
    }, numeric(1))
  }
  out <- list(model = model, estimates = estimates, summary = summary,
              master_seed = seed, replication_seeds = rep_seeds,
              compute = compute)
  class(out) <- c("DSEMmontecarlo", "list")
  out
}
