#' Simulate a Gaussian autoregressive process
#'
#' @param model Optional compiled model used to identify the outcome name.
#' @param n Number of retained time points.
#' @param intercept,ar,sigma Data-generating parameters.
#' @param burnin Number of discarded initialization observations.
#' @param seed Reproducibility seed.
#' @param outcome Name of the generated outcome.
#' @param time Name of the generated time index.
#' @param clusters Optional number of independent individuals. When supplied,
#'   a two-level process with random intercepts and AR effects is generated.
#' @param random_sd Length-two vector containing the random-intercept and
#'   random-AR standard deviations.
#' @param random_cor Correlation between the random intercept and AR effect.
#' @param id Name of the generated cluster identifier.
#' @return A data frame in long format.
#' @export
simulate_dsem <- function(model = NULL, n = 200L, intercept = 0, ar = 0.5,
                          sigma = 1, burnin = 100L, seed = 1L,
                          outcome = "y", time = "time", clusters = NULL,
                          random_sd = c(0.3, 0.1), random_cor = 0,
                          id = "id") {
  if (!is.null(model)) {
    if (!inherits(model, "DSEMmodel")) .dsem_abort("`model` must be a DSEMmodel.")
    regressions <- model$terms[model$terms$op == "~", , drop = FALSE]
    if (nrow(regressions)) outcome <- regressions$lhs[1L]
    if (!is.null(model$id)) {
      id <- model$id
      if (is.null(clusters)) clusters <- 50L
    }
    if (!is.null(model$time)) time <- model$time
  }
  n <- as.integer(n); burnin <- as.integer(burnin)
  if (n < 4L || burnin < 0L) .dsem_abort("Require `n >= 4` and `burnin >= 0`.")
  if (!is.finite(ar) || abs(ar) >= 1) .dsem_abort("`ar` must be strictly between -1 and 1 for stationary simulation.")
  if (!is.finite(sigma) || sigma <= 0) .dsem_abort("`sigma` must be positive.")
  if (!is.null(clusters)) {
    clusters <- as.integer(clusters)
    if (length(clusters) != 1L || is.na(clusters) || clusters < 2L) {
      .dsem_abort("`clusters` must be one integer of at least two.")
    }
    if (!is.numeric(random_sd) || length(random_sd) != 2L ||
        any(!is.finite(random_sd)) || any(random_sd < 0)) {
      .dsem_abort("`random_sd` must contain two non-negative finite values.")
    }
    if (!is.finite(random_cor) || abs(random_cor) >= 1) {
      .dsem_abort("`random_cor` must be strictly between -1 and 1.")
    }
    set.seed(as.integer(seed))
    z1 <- stats::rnorm(clusters)
    z2 <- random_cor * z1 + sqrt(1 - random_cor^2) * stats::rnorm(clusters)
    cluster_intercept <- intercept + random_sd[1L] * z1
    cluster_ar <- ar + random_sd[2L] * z2
    if (any(abs(cluster_ar) >= 1)) {
      .dsem_abort("Generated cluster AR effects are nonstationary; reduce `ar` or `random_sd[2]`.")
    }
    pieces <- lapply(seq_len(clusters), function(cluster) {
      dat <- simulate_dsem(
        n = n, intercept = cluster_intercept[cluster], ar = cluster_ar[cluster],
        sigma = sigma, burnin = burnin,
        seed = .dsem_stable_seed(seed, paste("simulate-cluster", cluster)),
        outcome = outcome, time = time
      )
      dat[[id]] <- cluster
      dat[c(id, time, outcome)]
    })
    out <- do.call(rbind, pieces)
    rownames(out) <- NULL
    attr(out, "dsem_truth") <- list(
      population = c(intercept = intercept, ar = ar, sigma2 = sigma^2),
      random_sd = stats::setNames(random_sd, c("intercept", "ar")),
      random_cor = random_cor,
      cluster_intercept = cluster_intercept, cluster_ar = cluster_ar
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
  y <- utils::tail(y, n)
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
#' @param compute Compute specification. Independent replications can be
#'   dispatched across a PSOCK cluster; deterministic seeds make results
#'   invariant to worker count and job order.
#' @param fit_args Named list passed to [dsem()].
#' @param truth Optional named vector of true parameter values.
#' @return A `DSEMmontecarlo` list with replication estimates and summaries.
#' @export
dsem_monte_carlo <- function(model, replications = 100L,
                             generator = function(seed) simulate_dsem(seed = seed),
                             seed = 20260813L, compute = dsem_compute(),
                             fit_args = list(chains = 2L, iter = 1000L, warmup = 500L),
                             truth = NULL) {
  replications <- as.integer(replications)
  if (replications < 1L) .dsem_abort("`replications` must be positive.")
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
               converged = all(is.na(fit$diagnostics$rhat) | fit$diagnostics$rhat < 1.1),
               stringsAsFactors = FALSE)
  }
  ids <- seq_len(replications)
  estimates <- if (compute$backend == "psock" && compute$workers > 1L &&
                   replications > 1L) {
    cl <- parallel::makePSOCKcluster(min(compute$workers, replications))
    on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::parLapply(cl, ids, run_one)
  } else {
    lapply(ids, run_one)
  }
  estimates <- do.call(rbind, estimates)
  summary <- stats::aggregate(cbind(mean, sd) ~ parameter, estimates, mean)
  names(summary)[names(summary) == "mean"] <- "mean_estimate"
  if (!is.null(truth)) {
    summary$truth <- unname(truth[summary$parameter])
    summary$bias <- summary$mean_estimate - summary$truth
  }
  out <- list(model = model, estimates = estimates, summary = summary,
              master_seed = seed, replication_seeds = rep_seeds,
              compute = compute)
  class(out) <- c("DSEMmontecarlo", "list")
  out
}
