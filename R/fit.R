#' Estimate a dynamic structural equation model
#'
#' The current development estimator is deliberately narrow: it supports an
#' N=1 Gaussian AR(1) regression, optionally with contemporaneous covariates.
#' The compiler can represent broader DSEM families, but [dsem()] stops with a
#' capability error until each estimator has passed its parity gate.
#'
#' @param model A `DSEMmodel` or model string.
#' @param data Data frame in long format.
#' @param syntax,id,time,family Used when `model` is a string.
#' @param chains Number of MCMC chains.
#' @param iter Total iterations per chain.
#' @param warmup Discarded warmup iterations.
#' @param thin Positive thinning interval.
#' @param seed Master seed. Task-specific chain seeds are deterministically
#'   derived from it.
#' @param compute A [dsem_compute()] specification.
#' @param engine `"auto"`, `"rust"`, or transparent `"R"` reference engine.
#' @param control Named list of prior controls.
#' @return A `DSEMfit` object.
#' @export
dsem <- function(model, data, syntax = c("auto", "lavaan", "mplus"),
                 id = NULL, time = NULL, family = "gaussian",
                 chains = 4L, iter = 2000L, warmup = floor(iter / 2),
                 thin = 1L, seed = 20260813L,
                 compute = dsem_compute(), engine = c("auto", "rust", "R"),
                 control = list()) {
  call <- match.call()
  engine <- match.arg(engine)
  if (!inherits(model, "DSEMmodel")) {
    model <- dsem_model(model, syntax = match.arg(syntax), id = id,
                        time = time, family = family)
  }
  .dsem_assert_estimable(model)
  if (!inherits(compute, "DSEMcompute")) .dsem_abort("`compute` must come from dsem_compute().")
  chains <- as.integer(chains); iter <- as.integer(iter)
  warmup <- as.integer(warmup); thin <- as.integer(thin); seed <- as.integer(seed)
  if (chains < 1L) .dsem_abort("`chains` must be positive.")
  if (iter < 4L || warmup < 0L || warmup >= iter) .dsem_abort("Require `0 <= warmup < iter` and at least four iterations.")
  if (thin < 1L) .dsem_abort("`thin` must be positive.")
  prepared <- .dsem_prepare_ar1(model, data)
  prior <- utils::modifyList(list(beta_mean = 0, beta_sd = 1e5,
                                  sigma_shape = 0.001, sigma_scale = 0.001), control)
  if (engine == "auto") engine <- if (.dsem_rust_available()) "rust" else "R"
  start <- proc.time()[[3L]]
  task <- list(x = prepared$x, y = prepared$y, iter = iter, warmup = warmup,
               thin = thin, prior = prior, engine = engine, master_seed = seed,
               model_hash = model$hash)
  chain_ids <- seq_len(chains)
  chain_results <- if (compute$backend == "psock" && compute$workers > 1L && chains > 1L) {
    cl <- parallel::makePSOCKcluster(min(compute$workers, chains))
    on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::parLapply(cl, chain_ids, function(chain_id, payload) {
      utils::getFromNamespace(".dsem_fit_chain_task", "DSEMr")(chain_id, payload)
    }, payload = task)
  } else {
    lapply(chain_ids, .dsem_fit_chain_task, payload = task)
  }
  elapsed <- proc.time()[[3L]] - start
  draws <- .dsem_bind_chains(chain_results)
  diagnostics <- .dsem_diagnostics(draws)
  estimates <- .dsem_summarize_draws(draws, diagnostics)
  beta <- estimates$mean[estimates$parameter != "sigma2"]
  fitted_values <- as.numeric(prepared$x %*% beta)
  out <- list(
    call = call, model = model, parameter_table = .dsem_parameter_table(model, estimates),
    draws = draws, estimates = estimates, diagnostics = diagnostics,
    fitted.values = fitted_values, residuals = prepared$y - fitted_values,
    observed = prepared$y, row_index = prepared$row_index,
    fit = list(log_lik = .dsem_loglik(prepared$y, fitted_values,
                                     estimates$mean[estimates$parameter == "sigma2"])),
    elapsed = elapsed, master_seed = seed,
    chain_seeds = vapply(chain_ids, function(i) .dsem_stable_seed(seed, paste(model$hash, "chain", i)), integer(1)),
    compute = compute, engine = engine, versions = .dsem_version_manifest(),
    status = "experimental-foundation"
  )
  class(out) <- c("DSEMfit", "list")
  out
}

.dsem_assert_estimable <- function(model) {
  t <- model$terms
  regressions <- t[t$op == "~", , drop = FALSE]
  lagged <- regressions[regressions$lag > 0L, , drop = FALSE]
  reasons <- character()
  if (model$family != "gaussian") reasons <- c(reasons, "only Gaussian outcomes are currently estimable")
  if (!is.null(model$id)) reasons <- c(reasons, "two-level estimation is not yet enabled")
  if (any(t$op == "=~")) reasons <- c(reasons, "latent measurement models are not yet enabled")
  if (any(t$residual)) reasons <- c(reasons, "RDSEM terms are not yet enabled")
  if (nrow(lagged) != 1L || lagged$lag[1L] != 1L || lagged$lhs[1L] != lagged$rhs[1L]) {
    reasons <- c(reasons, "the current engine requires exactly one self-lagged AR(1) outcome")
  }
  if (length(unique(regressions$lhs)) > 1L) reasons <- c(reasons, "only one outcome equation is currently estimable")
  if (length(reasons)) {
    .dsem_abort("This model compiles but is outside the validated estimator: %s.", paste(unique(reasons), collapse = "; "))
  }
  invisible(TRUE)
}

.dsem_prepare_ar1 <- function(model, data) {
  regressions <- model$terms[model$terms$op == "~", , drop = FALSE]
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
  x <- cbind(`(Intercept)` = rep(1, length(y) - 1L), y[-length(y)])
  colnames(x)[2L] <- paste0("lag1_", outcome)
  response <- y[-1L]
  if (length(contemporaneous)) {
    covars <- as.matrix(d[-1L, contemporaneous, drop = FALSE])
    storage.mode(covars) <- "double"
    x <- cbind(x, covars)
  }
  keep <- stats::complete.cases(cbind(response, x))
  if (sum(keep) < ncol(x) + 2L) .dsem_abort("Too few complete lagged observations for this model.")
  list(x = x[keep, , drop = FALSE], y = response[keep],
       row_index = ord[-1L][keep], outcome = outcome)
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

.dsem_gibbs_chain <- function(sufficient, iter, warmup, thin, prior, seed, names) {
  set.seed(seed)
  p <- sufficient$p
  prior_precision <- diag(1 / prior$beta_sd^2, p)
  precision <- sufficient$xtx + prior_precision
  inv_precision <- solve(precision)
  prior_mean <- rep(prior$beta_mean, p)
  post_mean <- as.numeric(inv_precision %*% (sufficient$xty + prior_precision %*% prior_mean))
  sigma2 <- 1
  keep_at <- seq.int(warmup + 1L, iter, by = thin)
  draws <- matrix(NA_real_, length(keep_at), p + 1L,
                  dimnames = list(NULL, c(names, "sigma2")))
  keep_i <- 0L
  chol_inv <- chol(inv_precision)
  for (i in seq_len(iter)) {
    beta <- post_mean + as.numeric(t(chol_inv) %*% stats::rnorm(p)) * sqrt(sigma2)
    rss <- sufficient$yty - 2 * sum(beta * sufficient$xty) +
      as.numeric(crossprod(beta, sufficient$xtx %*% beta))
    shape <- prior$sigma_shape + sufficient$n / 2
    scale <- prior$sigma_scale + max(rss, 0) / 2
    sigma2 <- 1 / stats::rgamma(1L, shape = shape, rate = scale)
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
               dimnames = list(iteration = NULL, parameter = colnames(results[[1L]]), chain = paste0("chain", seq_len(m))))
  for (i in seq_len(m)) out[, , i] <- results[[i]]
  out
}

.dsem_diagnostics <- function(draws) {
  params <- dimnames(draws)$parameter
  rhat <- stats::setNames(rep(NA_real_, length(params)), params)
  ess <- stats::setNames(rep(NA_real_, length(params)), params)
  mcse <- stats::setNames(rep(NA_real_, length(params)), params)
  for (j in seq_along(params)) {
    x <- draws[, j, , drop = FALSE]
    mat <- matrix(x, nrow = dim(draws)[1L], ncol = dim(draws)[3L])
    n <- nrow(mat); m <- ncol(mat)
    chain_means <- colMeans(mat)
    W <- mean(apply(mat, 2L, stats::var))
    B <- n * stats::var(chain_means)
    rhat[j] <- if (m > 1L && is.finite(W) && W > 0) sqrt(((n - 1) / n * W + B / n) / W) else NA_real_
    ac1 <- mean(apply(mat, 2L, function(z) {
      if (length(z) < 3L || stats::sd(z) == 0) return(0)
      stats::cor(z[-length(z)], z[-1L])
    }))
    ess[j] <- max(1, min(n * m, n * m * (1 - ac1) / (1 + ac1)))
    mcse[j] <- stats::sd(as.numeric(mat)) / sqrt(ess[j])
  }
  data.frame(parameter = params, rhat = unname(rhat), ess = unname(ess),
             mcse = unname(mcse), row.names = NULL)
}

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
  sum(stats::dnorm(y, fitted, sqrt(sigma2), log = TRUE))
}

.dsem_rust_available <- function() {
  exists("wrap__dsem_ar_sufficient", envir = asNamespace("DSEMr"), inherits = FALSE)
}

.dsem_ar_sufficient_rust <- function(x, y) {
  if (!.dsem_rust_available()) .dsem_abort("The Rust engine is not available in this installation.")
  raw <- dsem_ar_sufficient(as.numeric(x), as.numeric(y),
                            as.integer(nrow(x)), as.integer(ncol(x)))
  raw$xtx <- matrix(raw$xtx, nrow = raw$p, ncol = raw$p)
  raw
}
