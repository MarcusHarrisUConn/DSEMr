test_that("dsem handles constant outcomes, out-of-bounds confint, CT predict, and RNG preservation", {
  # 1. Constant single-level outcome never hits singular solve()
  d_const <- data.frame(time = seq_len(30), y = rep(5.0, 30))
  f_const <- dsem("y ~ lag(y, 1)", d_const, chains = 2, iter = 100, warmup = 50, seed = 10)
  expect_s3_class(f_const, "DSEMfit")
  expect_true(all(is.finite(f_const$estimates$mean)))

  # 2. Out-of-bounds numeric confint parm returns clean 0-row matrix
  ci_empty <- confint(f_const, parm = 999L)
  expect_true(is.matrix(ci_empty))
  expect_equal(nrow(ci_empty), 0L)
  expect_equal(ncol(ci_empty), 2L)

  # 3. Caller .Random.seed is preserved across simulate_dsem() and dsem()
  set.seed(12345)
  seed_before <- .Random.seed
  d_rng <- simulate_dsem(n = 30, ar = 0.4, seed = 7)
  expect_identical(.Random.seed, seed_before)
  f_rng <- dsem("y ~ lag(y, 1)", d_rng, chains = 2, iter = 80, warmup = 40, seed = 7)
  expect_identical(.Random.seed, seed_before)

  # 4. Continuous-time OU predict() uses exp(drift) dynamics
  set.seed(88)
  dt <- runif(59, 0.3, 1.5)
  y <- numeric(60); y[1] <- 2.5
  for (t in 2:60) {
    ph <- exp(-0.5 * dt[t - 1])
    y[t] <- 0.5 + ph * (y[t - 1] - 0.5) + rnorm(1, 0, sqrt(0.5 * (1 - ph^2)))
  }
  f_ct <- dsem_ct("y ~ lag(y, 1)", data.frame(time = c(0, cumsum(dt)), y = y),
                  chains = 2, iter = 140, warmup = 70, seed = 88)
  fc <- predict(f_ct, horizon = 4L)
  expect_equal(length(fc), 4L)
  expect_true(all(is.finite(fc)))
  expect_true(abs(fc[1L] - fc[4L]) > 1e-4)
})

test_that("MODEL CONSTRAINT and MODEL TEST resolve short random-slope aliases and chained definitions", {
  d <- simulate_dsem(n = 20, clusters = 10, ar = 0.45, seed = 22)
  mp <- "
    VARIABLE: NAMES = id time y; CLUSTER = id; LAGGED = y(1);
    ANALYSIS: TYPE = TWOLEVEL RANDOM; ESTIMATOR = BAYES;
    MODEL:
      %WITHIN%
      phi | y ON y&1;
      %BETWEEN%
      y phi;
    MODEL CONSTRAINT:
      NEW(persist half_life);
      persist = abs(phi);
      half_life = -log(2) / log(persist + 1e-6);
    MODEL TEST:
      phi = 0;
  "
  m <- dsem_model(mp, syntax = "mplus")
  fit <- dsem(m, d, chains = 2, iter = 160, warmup = 80, seed = 22)

  expect_s3_class(fit$constraints, "DSEMhypothesis")
  expect_equal(fit$constraints$summary$name, c("persist", "half_life"))
  expect_true(all(is.finite(fit$constraints$summary$mean)))

  expect_s3_class(fit$wald_test, "DSEMhypothesis")
  expect_true(!is.null(fit$wald_test$wald))
  expect_true(is.finite(fit$wald_test$wald$statistic))

  expect_identical(dsem_inspect(fit, "constraints"), fit$constraints)
  expect_identical(dsem_inspect(fit, "wald_test"), fit$wald_test)

  out_txt <- capture.output(print(summary(fit)))
  expect_true(any(grepl("MODEL CONSTRAINT", out_txt, fixed = TRUE)))
  expect_true(any(grepl("MODEL TEST", out_txt, fixed = TRUE)))
})

test_that("dsem_monte_carlo works with PSOCK workers and user-defined generators calling simulate_dsem", {
  gen <- function(seed) simulate_dsem(n = 35, ar = 0.4, seed = seed)
  mc <- dsem_monte_carlo(
    "y ~ lag(y, 1)",
    replications = 2L,
    generator = gen,
    compute = dsem_compute(workers = 2L, backend = "psock"),
    fit_args = list(chains = 2L, iter = 100L, warmup = 50L),
    truth = c(lag1_y = 0.4)
  )
  expect_s3_class(mc, "DSEMmontecarlo")
  expect_equal(mc$n_completed, 2L)
  expect_true(all(is.finite(mc$summary$mean_estimate)))
})
