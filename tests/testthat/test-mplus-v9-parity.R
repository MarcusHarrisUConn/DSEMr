test_that("Mplus Version 9 three-level DSEM (DSEM3) compiles and recovers parameters", {
  dat3 <- simulate_dsem(
    n = 18, clusters = c(10, 3), id = c("day", "person"),
    intercept = 1.2, ar = 0.45, sigma = 0.7,
    random_sd = c(0.3, 0.06, 0.12), seed = 9001
  )
  m_mplus <- dsem_model("
    VARIABLE: CLUSTER = person day;
    ANALYSIS: TYPE = THREELEVEL RANDOM; ESTIMATOR = BAYES;
    MODEL:
      %WITHIN%
        s | y ON y&1;
        logv | y;
      %BETWEEN day%
        y;
      %BETWEEN person%
        y s logv WITH y s logv;
  ", syntax = "mplus", time = "time")
  expect_equal(m_mplus$level_type, "threelevel")
  expect_equal(m_mplus$id, c("day", "person"))

  fit3 <- dsem(m_mplus, dat3, chains = 2, iter = 300, warmup = 150, seed = 9002)
  expect_s3_class(fit3, "DSEMfit")
  expect_equal(fit3$status, "validated-three-level-dsem")
  expect_equal(unname(coef(fit3)["fixed_s"]), 0.45, tolerance = 0.2)
  expect_equal(unname(coef(fit3)["fixed_logv"]), log(0.49), tolerance = 0.35)
  expect_true(is.finite(dsem_dic(fit3)[["dic"]]))
})

test_that("cross-classified DSEM recovers AR slope, time shock variance, and sigma2", {
  dcc <- simulate_dsem(
    n = 18, clusters = 12, intercept = 0.4, ar = 0.4, sigma = 0.6,
    cross_classified = TRUE, time_sd = 0.35, seed = 940
  )
  mcc <- dsem_model("
    VARIABLE: CLUSTER = time id;
    ANALYSIS: TYPE = CROSSCLASSIFIED RANDOM; ESTIMATOR = BAYES;
    MODEL:
      %WITHIN%
        s | y ON y&1;
      %BETWEEN id%
        y s;
      %BETWEEN time%
        y;
  ", syntax = "mplus", time = "time")
  expect_equal(mcc$level_type, "crossclassified")
  fcc <- dsem(mcc, dcc, chains = 2, iter = 260, warmup = 130, seed = 941)
  expect_equal(fcc$status, "validated-cross-classified-dsem")
  expect_equal(unname(coef(fcc)["mean_s"]), 0.4, tolerance = 0.2)
  expect_equal(unname(coef(fcc)["sigma2"]), 0.36, tolerance = 0.2)
  expect_gt(unname(coef(fcc)["time_var_y"]), 0.01)
})

test_that("two-level location-scale DSEM supports random logv, w predictors, and distal z", {
  set.seed(930)
  g_n <- 18; t_n <- 25
  w <- rnorm(g_n)
  mu_i <- 0.4 * w + rnorm(g_n, sd = 0.25)
  phi_i <- pmin(pmax(0.35 + 0.1 * w + rnorm(g_n, sd = 0.05), -0.8), 0.8)
  lv_i <- 0.25 * w + rnorm(g_n, sd = 0.1)
  z_i <- 0.5 * mu_i + 0.4 * lv_i + rnorm(g_n, sd = 0.2)

  rows <- lapply(seq_len(g_n), function(g) {
    y <- numeric(t_n)
    sd_g <- exp(0.5 * lv_i[g])
    y[1] <- rnorm(1, 0, sd_g)
    for (t in 2:t_n) y[t] <- phi_i[g] * y[t - 1] + rnorm(1, 0, sd_g)
    data.frame(subject = g, time = seq_len(t_n), y = mu_i[g] + y, w = w[g], z = z_i[g])
  })
  dat <- do.call(rbind, rows)
  m930 <- dsem_model("
    VARIABLE: CLUSTER = subject; BETWEEN = z w;
    ANALYSIS: TYPE = TWOLEVEL RANDOM; ESTIMATOR = BAYES;
    MODEL:
      %WITHIN%
        s | y ON y&1;
        logv | y;
      %BETWEEN%
        y s logv ON w;
        y s logv WITH y s logv;
        z ON y s logv;
  ", syntax = "mplus", time = "time")
  fit <- dsem(m930, dat, chains = 2, iter = 300, warmup = 150, seed = 930)
  cf <- coef(fit)
  expect_equal(unname(cf["mean_s"]), 0.35, tolerance = 0.2)
  expect_gt(unname(cf["logv_on_w"]), 0.05)
  expect_gt(unname(cf["z_on_y"]), 0.1)
  expect_true(is.finite(dsem_dic(fit)[["dic"]]))
})

test_that("RDSEM separates structural slope from residual autocorrelation", {
  set.seed(2024)
  n <- 250
  x <- rnorm(n)
  u <- numeric(n)
  u[1] <- rnorm(1)
  for (t in 2:n) u[t] <- 0.55 * u[t - 1] + rnorm(1, sd = 0.6)
  dat <- data.frame(time = seq_len(n), x = x, y = 0.5 + 1.2 * x + u)
  oracle_ar <- unname(stats::coef(stats::lm(u[-1] ~ u[-n]))[2])

  m_lav <- dsem_model("y ~ residual(lag(y, 1)) + x", time = "time")
  m_mpl <- mplus_to_dsem("ANALYSIS: ESTIMATOR = BAYES; MODEL: y^ ON y^1; y ON x;", time = "time")
  expect_identical(m_lav$terms, m_mpl$terms)

  fit <- dsem(m_lav, dat, chains = 2, iter = 300, warmup = 150, seed = 2024)
  expect_equal(fit$status, "validated-rdsem-n1")
  expect_equal(unname(coef(fit)["lag1_y"]), oracle_ar, tolerance = 0.06)
  expect_equal(unname(coef(fit)["lag1_y"]), 0.55, tolerance = 0.15)
  expect_equal(unname(coef(fit)["x"]), 1.2, tolerance = 0.12)
})

test_that("binary/ordinal probit, WNFS, PSEM priors, TINTERVAL, VIF, and STDYX work cleanly", {
  # Binary probit N=1
  dbin <- simulate_dsem(n = 100, intercept = 0.2, ar = 0.5, sigma = 1, family = "binary", seed = 77)
  fbin <- dsem("y ~ lag(y, 1)", dbin, family = "binary", time = "time",
               chains = 2, iter = 260, warmup = 130, seed = 77)
  expect_equal(fbin$status, "validated-probit-n1")
  expect_equal(unname(coef(fbin)["lag1_y"]), 0.5, tolerance = 0.35)

  # WNFS N=1 dynamic CFA (Mplus 6.26b structure)
  m_wnfs <- mplus_to_dsem("
    ANALYSIS: ESTIMATOR = BAYES;
    MODEL:
      f BY y1-y4 (&1);
      y1-y4 ON f&1;
  ", time = "time")
  d_lat <- simulate_dsem(dsem_model("f =~ y1 + y2 + y3 + y4\nf ~ lag(f, 1)", time = "time"),
                         n = 80, ar = 0.45, seed = 626)
  f_wnfs <- dsem(m_wnfs, d_lat, chains = 2, iter = 200, warmup = 100, seed = 626)
  expect_equal(f_wnfs$status, "validated-wnfs-cfa")
  expect_true(all(paste0("lagged_factor_effect_y", 1:4) %in% names(coef(f_wnfs))))

  # Multivariate VAR(1) with Mplus Version 9 PSEM penalized shrinkage prior
  set.seed(44)
  n <- 80
  y1 <- numeric(n); y2 <- numeric(n)
  for (t in 2:n) {
    y1[t] <- 0.5 * y1[t - 1] + rnorm(1, sd = 0.7)
    y2[t] <- 0.4 * y2[t - 1] + 0.3 * y1[t - 1] + rnorm(1, sd = 0.7)
  }
  dvar <- data.frame(time = seq_len(n), y1 = y1, y2 = y2)
  f_psem <- dsem("y1 ~ lag(y1, 1) + lag(y2, 1)\ny2 ~ lag(y1, 1) + lag(y2, 1)",
                 dvar, time = "time", chains = 2, iter = 220, warmup = 110,
                 seed = 44, control = list(prior_mode = "penalized"))
  expect_true("y2_on_lag1_y1" %in% names(coef(f_psem)))

  # TINTERVAL alignment, VIF, STDYX, confint, logLik, and predict
  irreg <- data.frame(id = c(1, 1, 1), time = c(0, 1.1, 3.05), y = c(0.1, 0.5, 0.9))
  aligned <- dsem_tinterval(irreg, time = "time", delta = 1, id = "id")
  expect_equal(nrow(aligned), 4L)
  expect_true(is.na(aligned$y[3L]))

  d1 <- simulate_dsem(n = 60, ar = 0.5, seed = 9)
  d1$x1 <- rnorm(60); d1$x2 <- 0.7 * d1$x1 + rnorm(60, sd = 0.7)
  m1 <- dsem_model("y ~ lag(y, 1) + x1 + x2", time = "time")
  vifs <- dsem_vif(m1, data = d1)
  expect_true(all(vifs$vif >= 1))

  f1 <- dsem(m1, d1, chains = 2, iter = 200, warmup = 100, seed = 9)
  std <- dsem_standardize(f1, type = "stdyx")
  expect_true(nrow(std) >= 2L)
  ci <- confint(f1)
  expect_equal(ncol(ci), 2L)
  expect_s3_class(logLik(f1), "logLik")
  expect_equal(length(predict(f1, horizon = 3L)), 3L)
  expect_equal(length(predict(f1, newdata = d1[1:5, ])), 5L)
})
