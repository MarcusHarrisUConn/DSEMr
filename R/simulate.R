#' Simulate an N=1 Gaussian autoregressive process
#'
#' @param model Optional compiled model used to identify the outcome name.
#' @param n Number of retained time points.
#' @param intercept,ar,sigma Data-generating parameters.
#' @param burnin Number of discarded initialization observations.
#' @param seed Reproducibility seed.
#' @param outcome Name of the generated outcome.
#' @param time Name of the generated time index.
#' @return A data frame in long format.
#' @export
simulate_dsem <- function(model = NULL, n = 200L, intercept = 0, ar = 0.5,
                          sigma = 1, burnin = 100L, seed = 1L,
                          outcome = "y", time = "time") {
  if (!is.null(model)) {
    if (!inherits(model, "DSEMmodel")) .dsem_abort("`model` must be a DSEMmodel.")
    regressions <- model$terms[model$terms$op == "~", , drop = FALSE]
    if (nrow(regressions)) outcome <- regressions$lhs[1L]
  }
  n <- as.integer(n); burnin <- as.integer(burnin)
  if (n < 4L || burnin < 0L) .dsem_abort("Require `n >= 4` and `burnin >= 0`.")
  if (!is.finite(ar) || abs(ar) >= 1) .dsem_abort("`ar` must be strictly between -1 and 1 for stationary simulation.")
  if (!is.finite(sigma) || sigma <= 0) .dsem_abort("`sigma` must be positive.")
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
