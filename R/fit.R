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
  if (!is.null(model$id)) {
    return(.dsem_fit_two_level(model, data, chains, iter, warmup, thin,
                               seed, compute, engine, control, call))
  }
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

.dsem_fit_two_level <- function(model, data, chains, iter, warmup, thin,
                                seed, compute, engine, control, call) {
  if (engine == "rust") {
    .dsem_abort("The validated two-level sampler currently uses the R reference engine; use `engine = \"R\"` or `\"auto\"`.")
  }
  prepared <- .dsem_prepare_two_level_ar1(model, data)
  prior <- utils::modifyList(list(
    beta_mean = 0, beta_sd = 1e3, sigma_shape = 0.001,
    sigma_scale = 0.001, omega_df = 4,
    omega_scale = diag(0.1, 2L)
  ), control)
  task <- list(prepared = prepared, iter = iter, warmup = warmup,
               thin = thin, prior = prior, master_seed = seed,
               model_hash = model$hash)
  chain_ids <- seq_len(chains)
  start <- proc.time()[[3L]]
  chain_results <- if (compute$backend == "psock" && compute$workers > 1L && chains > 1L) {
    cl <- parallel::makePSOCKcluster(min(compute$workers, chains))
    on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::parLapply(cl, chain_ids, function(chain_id, payload) {
      utils::getFromNamespace(".dsem_two_level_chain_task", "DSEMr")(chain_id, payload)
    }, payload = task)
  } else {
    lapply(chain_ids, .dsem_two_level_chain_task, payload = task)
  }
  elapsed <- proc.time()[[3L]] - start
  draw_matrices <- lapply(chain_results, `[[`, "draws")
  draws <- .dsem_bind_chains(draw_matrices)
  diagnostics <- .dsem_diagnostics(draws)
  estimates <- .dsem_summarize_draws(draws, diagnostics)
  random_effects <- Reduce(`+`, lapply(chain_results, `[[`, "random_effects")) / chains
  fitted_values <- numeric(length(prepared$y))
  for (g in seq_along(prepared$groups)) {
    idx <- prepared$groups[[g]]$index
    fitted_values[idx] <- as.numeric(prepared$groups[[g]]$x %*% random_effects[g, ])
  }
  sigma2 <- estimates$mean[estimates$parameter == "sigma2"]
  out <- list(
    call = call, model = model,
    parameter_table = .dsem_parameter_table(model, estimates),
    draws = draws, estimates = estimates, diagnostics = diagnostics,
    fitted.values = fitted_values, residuals = prepared$y - fitted_values,
    observed = prepared$y, row_index = prepared$row_index,
    random_effects = random_effects,
    fit = list(log_lik = .dsem_loglik(prepared$y, fitted_values, sigma2)),
    elapsed = elapsed, master_seed = seed,
    chain_seeds = vapply(chain_ids, function(i) .dsem_stable_seed(seed, paste(model$hash, "chain", i)), integer(1)),
    compute = compute, engine = "R", versions = .dsem_version_manifest(),
    status = "experimental-two-level-ar1"
  )
  class(out) <- c("DSEMfit", "list")
  out
}

.dsem_prepare_two_level_ar1 <- function(model, data) {
  regressions <- model$terms[model$terms$op == "~", , drop = FALSE]
  outcome <- unique(regressions$lhs)
  .dsem_validate_data(data, model$id, model$time, outcome)
  ids <- unique(data[[model$id]])
  groups <- vector("list", length(ids))
  kept_rows <- integer()
  all_y <- numeric()
  for (g in seq_along(ids)) {
    rows <- which(data[[model$id]] == ids[g])
    if (!is.null(model$time)) rows <- rows[order(data[[model$time]][rows])]
    y <- as.numeric(data[[outcome]][rows])
    if (length(y) < 4L) .dsem_abort("Every cluster must contain at least four ordered observations.")
    x <- cbind(`(Intercept)` = rep(1, length(y) - 1L), y[-length(y)])
    colnames(x)[2L] <- paste0("lag1_", outcome)
    response <- y[-1L]
    keep <- stats::complete.cases(cbind(response, x))
    groups[[g]] <- list(x = x[keep, , drop = FALSE], y = response[keep],
                        index = length(all_y) + seq_len(sum(keep)))
    all_y <- c(all_y, response[keep])
    kept_rows <- c(kept_rows, rows[-1L][keep])
  }
  if (length(all_y) < 2L * length(ids) + 2L) .dsem_abort("Too few complete lagged observations for two-level estimation.")
  list(groups = groups, y = all_y, row_index = kept_rows, ids = ids,
       outcome = outcome)
}

.dsem_two_level_chain_task <- function(chain_id, payload) {
  seed <- .dsem_stable_seed(payload$master_seed,
                            paste(payload$model_hash, "chain", chain_id))
  .dsem_gibbs_two_level(payload$prepared, payload$iter, payload$warmup,
                        payload$thin, payload$prior, seed)
}

.dsem_gibbs_two_level <- function(prepared, iter, warmup, thin, prior, seed) {
  set.seed(seed)
  groups <- prepared$groups
  G <- length(groups); p <- 2L
  pooled_x <- do.call(rbind, lapply(groups, `[[`, "x"))
  pooled_y <- unlist(lapply(groups, `[[`, "y"), use.names = FALSE)
  mu <- as.numeric(stats::coef(stats::lm.fit(pooled_x, pooled_y)))
  if (any(!is.finite(mu))) mu <- rep(0, p)
  theta <- matrix(rep(mu, each = G), nrow = G)
  omega <- diag(0.1, p); sigma2 <- stats::var(pooled_y) / 2
  keep_at <- seq.int(warmup + 1L, iter, by = thin)
  lag_name <- paste0("lag1_", prepared$outcome)
  draws <- matrix(NA_real_, length(keep_at), 6L,
                  dimnames = list(NULL, c("(Intercept)", lag_name, "tau_intercept",
                                          paste0("tau_", lag_name),
                                          paste0("cov_intercept_", lag_name), "sigma2")))
  keep_i <- 0L
  for (i in seq_len(iter)) {
    omega_inv <- solve(omega + diag(1e-10, p))
    for (g in seq_len(G)) {
      x <- groups[[g]]$x; y <- groups[[g]]$y
      v <- solve(crossprod(x) / sigma2 + omega_inv)
      m <- v %*% (crossprod(x, y) / sigma2 + omega_inv %*% mu)
      theta[g, ] <- as.numeric(m + t(chol(v)) %*% stats::rnorm(p))
    }
    v_mu <- solve(G * omega_inv + diag(1 / prior$beta_sd^2, p))
    m_mu <- v_mu %*% (omega_inv %*% colSums(theta) + rep(prior$beta_mean / prior$beta_sd^2, p))
    mu <- as.numeric(m_mu + t(chol(v_mu)) %*% stats::rnorm(p))
    centered <- sweep(theta, 2L, mu)
    scale <- prior$omega_scale + crossprod(centered)
    omega <- solve(stats::rWishart(1L, prior$omega_df + G, solve(scale))[, , 1L])
    rss <- sum(vapply(seq_len(G), function(g) {
      sum((groups[[g]]$y - groups[[g]]$x %*% theta[g, ])^2)
    }, numeric(1)))
    sigma2 <- 1 / stats::rgamma(1L, prior$sigma_shape + length(pooled_y) / 2,
                                rate = prior$sigma_scale + rss / 2)
    if (i %in% keep_at) {
      keep_i <- keep_i + 1L
      draws[keep_i, ] <- c(mu, sqrt(diag(omega)), omega[1L, 2L], sigma2)
    }
  }
  rownames(theta) <- as.character(prepared$ids)
  colnames(theta) <- c("intercept", paste0("lag1_", prepared$outcome))
  list(draws = draws, random_effects = theta)
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
