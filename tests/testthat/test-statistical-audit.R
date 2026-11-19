test_that("independent normal coefficient prior agrees with numerical integration", {
  x <- matrix(1, 6, 1)
  y <- c(-2, 0, -1, -2, 1, -1)
  prior <- list(beta_mean = 2, beta_sd = 0.5, sigma_shape = 2, sigma_scale = 1)
  density <- function(beta) {
    vapply(beta, function(b) dnorm(b, 2, 0.5) *
      (1 + sum((y - b)^2) / 2)^(-5), numeric(1))
  }
  norm <- integrate(density, -8, 8, rel.tol = 1e-10)$value
  expected <- integrate(function(b) b * density(b), -8, 8, rel.tol = 1e-10)$value / norm
  draws <- DSEMr:::.dsem_gibbs_chain(DSEMr:::.dsem_ar_sufficient_R(x, y),
    12000L, 2000L, 1L, prior, 984L, "intercept")
  expect_equal(mean(draws[, 1]), expected, tolerance = 0.025)
})

test_that("FFBS matches a dense Gaussian posterior including the initial state", {
  n <- 6L; phi <- 0.85; psi <- 0.4
  lambda <- c(1, 0.7); theta <- c(0.3, 0.6); nu <- c(0.2, -0.1)
  y <- cbind(c(1, 0.4, NA, -0.3, 0.2, 1), c(NA, 0, 0.2, NA, 0.5, 0.1))
  covariance <- outer(1:n, 1:n, function(a, b) psi / (1 - phi^2) * phi^abs(a - b))
  precision <- solve(covariance); score <- numeric(n)
  for (t in 1:n) for (j in 1:2) if (!is.na(y[t, j])) {
    precision[t, t] <- precision[t, t] + lambda[j]^2 / theta[j]
    score[t] <- score[t] + lambda[j] * (y[t, j] - nu[j]) / theta[j]
  }
  target_cov <- solve(precision); target_mean <- as.numeric(target_cov %*% score)
  set.seed(612)
  states <- replicate(3000, DSEMr:::.dsem_ffbs_factor(y, nu, lambda, theta, 0, phi, psi, 0.995))
  expect_true(all(abs(rowMeans(states) - target_mean) < 5 * sqrt(diag(target_cov) / 3000)))
  covariance_mcse <- sqrt((outer(diag(target_cov), diag(target_cov)) + target_cov^2) / 2999)
  expect_true(all(abs(cov(t(states)) - target_cov) < 5 * covariance_mcse))
  expect_equal(DSEMr:::.dsem_stationary_logdensity(2, phi, psi),
               dnorm(2, 0, sqrt(psi / (1 - phi^2)), log = TRUE))
})

test_that("stationary AR update agrees with a numerically integrated conditional", {
  eta <- c(3, 2.3, 1.5, 1, 0.5); psi <- 0.8
  prior <- list(coefficient_mean = 0, coefficient_sd = 10, ar_bound = 0.995,
                variance_shape = 2, variance_scale = 1)
  density <- function(phi) vapply(phi, function(p) {
    exp(sum(dnorm(eta[-1], p * eta[-length(eta)], sqrt(psi), log = TRUE)) +
      dnorm(eta[1], 0, sqrt(psi / (1 - p^2)), log = TRUE) + dnorm(p, 0, 10, log = TRUE))
  }, numeric(1))
  norm <- integrate(density, -0.995, 0.995)$value
  target <- integrate(function(p) p * density(p), -0.995, 0.995)$value / norm
  set.seed(881)
  phi <- 0; draws <- numeric(12000)
  for (i in seq_along(draws)) {
    phi <- DSEMr:::.dsem_sample_phi(eta, phi, psi, prior)
    draws[i] <- phi
  }
  expect_lt(abs(mean(draws[-(1:2000)]) - target), 0.02)
  expect_true(all(abs(draws) < 0.995))
  pars <- DSEMr:::.dsem_state_variance_parameters(eta, 0.7, prior)
  log_joint <- function(v) sum(dnorm(eta[-1], 0.7 * eta[-length(eta)], sqrt(v), log = TRUE)) +
    dnorm(eta[1], 0, sqrt(v / (1 - 0.7^2)), log = TRUE) - 3 * log(v) - 1 / v
  log_conditional <- function(v) -(pars$shape + 1) * log(v) - pars$rate / v
  expect_equal(log_joint(0.8) - log_joint(1.5),
               log_conditional(0.8) - log_conditional(1.5))
})

test_that("unsupported terms and malformed controls cannot be silently fitted", {
  d <- simulate_dsem(n = 40, seed = 2)
  expect_error(dsem("y ~ lag(y, 1)", d, chains = NA), "chains")
  expect_error(dsem("y ~ lag(y, 1)", d, iter = 30.5), "iter")
  expect_error(dsem("y ~ lag(y, 1)", d, control = list(typo = 1)), "Unsupported control")
  expect_error(dsem_model("f =~ badmod*y + x\nf ~ lag(f, 1)"), "modifier")
  expect_error(dsem_model("ANALYSIS: TYPE = BOGUS; MODEL: y ON y&1;", syntax = "mplus"), "Unsupported Mplus")
  expect_error(dsem("y ~ lag(y, 1)", transform(d, y = factor(y))), "numeric")
  expect_error(dsem("y ~ lag(y, 1)", transform(d, y = Inf)), "finite")
})

test_that("Monte Carlo summaries report coverage and never call one chain converged", {
  mc <- dsem_monte_carlo("y ~ lag(y, 1)", replications = 2,
    generator = function(seed) simulate_dsem(n = 40, seed = seed),
    fit_args = list(chains = 1, iter = 400, warmup = 200, engine = "R"),
    truth = c(lag1_y = 0.5))
  e <- subset(mc$estimates, parameter == "lag1_y")
  s <- subset(mc$summary, parameter == "lag1_y")
  expect_equal(s$coverage, mean(e$q2.5 <= 0.5 & e$q97.5 >= 0.5))
  expect_equal(s$rmse, sqrt(mean((e$mean - 0.5)^2)))
  expect_false(any(mc$estimates$converged))
  expect_error(dsem_monte_carlo("y ~ lag(y, 1)", fit_args = list(seed = 3)), "override")
})

test_that("fitting and simulation preserve caller RNG state and fix their RNG algorithm", {
  old_kind <- RNGkind()
  on.exit(do.call(RNGkind, as.list(old_kind)), add = TRUE)
  set.seed(65)
  old_seed <- .Random.seed
  a <- simulate_dsem(n = 40, seed = 2)
  fit_a <- dsem("y ~ lag(y, 1)", a, chains = 2, iter = 200, warmup = 100, seed = 18)
  expect_identical(.Random.seed, old_seed)
  RNGkind("L'Ecuyer-CMRG")
  old_seed <- .Random.seed
  b <- simulate_dsem(n = 40, seed = 2)
  fit_b <- dsem("y ~ lag(y, 1)", b, chains = 2, iter = 200, warmup = 100, seed = 18)
  expect_identical(.Random.seed, old_seed)
  expect_identical(a, b)
  expect_identical(fit_a$draws, fit_b$draws)
})
