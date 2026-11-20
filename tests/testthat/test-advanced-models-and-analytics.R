test_that("lagged exogenous covariates and sparse fixed-zero lags work in N=1 DSEM", {
  # Example 6.24 pattern: y ON y&1 x x&1
  set.seed(624)
  n <- 160
  x <- rnorm(n)
  y <- numeric(n)
  y[1] <- rnorm(1)
  for (t in 2:n) {
    y[t] <- 0.2 + 0.45 * y[t - 1] + 0.7 * x[t] - 0.35 * x[t - 1] + rnorm(1, sd = 0.5)
  }
  dat <- data.frame(time = seq_len(n), y = y, x = x)

  m_lav <- dsem_model("y ~ lag(y, 1) + x + lag(x, 1)", time = "time")
  m_scr <- mplus_to_dsem("
    VARIABLE: NAMES = y x; LAGGED = y(1) x(1);
    ANALYSIS: ESTIMATOR = BAYES;
    MODEL: y ON y&1 x x&1;
  ", time = "time")
  expect_identical(m_lav$terms, m_scr$terms)

  fit <- dsem(m_scr, dat, chains = 2, iter = 300, warmup = 150, seed = 624)
  cf <- coef(fit)
  expect_true(all(c("lag1_y", "x", "lag1_x") %in% names(cf)))
  expect_equal(unname(cf["lag1_y"]), 0.45, tolerance = 0.15)
  expect_equal(unname(cf["x"]), 0.70, tolerance = 0.15)
  expect_equal(unname(cf["lag1_x"]), -0.35, tolerance = 0.15)

  # Sparse fixed-zero lag: y ON y&1@0 y&2 (AR(2) with phi_1 = 0)
  set.seed(6232)
  y2 <- numeric(n)
  y2[1:2] <- rnorm(2)
  for (t in 3:n) y2[t] <- 0.5 * y2[t - 2] + rnorm(1, sd = 0.6)
  dat2 <- data.frame(time = seq_len(n), y = y2)
  m_sparse <- mplus_to_dsem("
    VARIABLE: LAGGED = y(2);
    ANALYSIS: ESTIMATOR = BAYES;
    MODEL: y ON y&1@0 y&2;
  ", time = "time")
  fit_sp <- dsem(m_sparse, dat2, chains = 2, iter = 260, warmup = 130, seed = 6232)
  cf_sp <- coef(fit_sp)
  expect_false("lag1_y" %in% names(cf_sp))
  expect_true("lag2_y" %in% names(cf_sp))
  expect_equal(unname(cf_sp["lag2_y"]), 0.5, tolerance = 0.18)
})

test_that("Tobit censored DSEM recovers parameters in N=1 and two-level models", {
  # N=1 left-censored at floor 0
  dat1 <- simulate_dsem(n = 180, intercept = 0.3, ar = 0.5, sigma = 0.7,
                        censored = c(0, Inf), seed = 701)
  expect_true(mean(dat1$y <= 1e-12) > 0.15)

  m_c1 <- dsem_model("y ~ lag(y, 1)", time = "time", censored = list(y = c(0, Inf)))
  m_c1_scr <- mplus_to_dsem("
    VARIABLE: CENSORED = y (b); LAGGED = y(1);
    ANALYSIS: ESTIMATOR = BAYES;
    MODEL: y ON y&1;
  ", time = "time")
  expect_true(m_c1$capabilities[["censored"]])
  expect_true(m_c1_scr$capabilities[["censored"]])

  fit1 <- dsem(m_c1, dat1, chains = 2, iter = 300, warmup = 150, seed = 702)
  expect_equal(fit1$status, "validated-censored-n1")
  expect_equal(unname(coef(fit1)["lag1_y"]), 0.5, tolerance = 0.2)
  expect_equal(unname(coef(fit1)["sigma2"]), 0.49, tolerance = 0.25)
  expect_true(is.finite(dsem_dic(fit1)[["dic"]]))

  # Two-level floor-censored AR(1)
  dat2 <- simulate_dsem(n = 20, clusters = 15, intercept = 0.5, ar = 0.45, sigma = 0.7,
                        random_sd = c(0.25, 0.06), censored = c(0, Inf), seed = 703)
  m_c2 <- dsem_model("
    level: 1
      s | y ~ lag(y, 1)
    level: 2
      y ~~ s
  ", id = "id", time = "time", censored = list(y = "below"))
  fit2 <- dsem(m_c2, dat2, chains = 2, iter = 260, warmup = 130, seed = 704)
  expect_lt(abs(unname(coef(fit2)["mean_s"]) - 0.45), 0.2)
  expect_true(is.finite(dsem_dic(fit2)[["dic"]]))
})

test_that("bivariate two-level VAR estimates correlated within innovations and cluster STDYX", {
  set.seed(932)
  g_n <- 15; t_n <- 25
  rows <- lapply(seq_len(g_n), function(g) {
    mu1 <- rnorm(1, 0, 0.4); mu2 <- rnorm(1, 0, 0.4)
    y1 <- numeric(t_n); y2 <- numeric(t_n)
    for (t in 2:t_n) {
      e1 <- rnorm(1, 0, 0.7)
      e2 <- 0.65 * e1 + rnorm(1, 0, 0.55)
      y1[t] <- 0.4 * y1[t - 1] + 0.2 * y2[t - 1] + e1
      y2[t] <- 0.35 * y2[t - 1] + 0.15 * y1[t - 1] + e2
    }
    data.frame(subject = g, time = seq_len(t_n), y1 = mu1 + y1, y2 = mu2 + y2)
  })
  dat <- do.call(rbind, rows)
  m932 <- mplus_to_dsem("
    VARIABLE: CLUSTER = subject; LAGGED = y1(1) y2(1);
    ANALYSIS: TYPE = TWOLEVEL RANDOM; ESTIMATOR = BAYES;
    MODEL:
      %WITHIN%
        s1 | y1 ON y1&1;
        s2 | y2 ON y2&1;
        s12 | y1 ON y2&1;
        s21 | y2 ON y1&1;
      %BETWEEN%
        y1 y2 s1-s21 WITH y1 y2 s1-s21;
  ", time = "time")
  fit <- dsem(m932, dat, chains = 2, iter = 260, warmup = 130, seed = 932)
  cf <- coef(fit)
  expect_true("within_cov_y2_y1" %in% names(cf))
  expect_gt(unname(cf["within_cov_y2_y1"]), 0.15)
  expect_equal(unname(cf["mean_s1"]), 0.4, tolerance = 0.2)
  expect_equal(unname(cf["mean_s12"]), 0.2, tolerance = 0.2)

  # Cluster-averaged and cluster-specific STDYX standardization
  std_pop <- dsem_standardize(fit, type = "stdyx")
  expect_true(all(c("within_std_s1", "within_std_s12") %in% std_pop$parameter))
  std_cl <- dsem_standardize(fit, type = "stdyx", cluster = TRUE)
  expect_equal(nrow(std_cl), g_n * 4L)
  expect_true(all(is.finite(std_cl$mean)))

  # Within & Between partial correlations
  pcor <- dsem_partial_cor(fit)
  expect_true(all(c("Between", "Within") %in% pcor$level))
  w_row <- pcor[pcor$level == "Within", , drop = FALSE]
  expect_gt(w_row$pcor_mean[1L], 0.25)

  # Impulse Response Functions (orthogonalized & generalized)
  irf_o <- dsem_irf(fit, horizon = 5L, orthogonal = TRUE)
  irf_g <- dsem_irf(fit, horizon = 5L, orthogonal = FALSE)
  expect_s3_class(irf_o, "DSEMirf")
  expect_equal(nrow(irf_o$data), 2L * 2L * 6L)
  expect_equal(nrow(irf_g$data), 2L * 2L * 6L)
  expect_true(all(is.finite(irf_o$data$mean)))
})

test_that("cross-classified DSEM handles CLUSTER = subject time, GROUPMEAN, time-varying slopes, and w_time", {
  set.seed(939)
  n_id <- 10; n_t <- 18
  timet <- scale(seq_len(n_t))[, 1]
  u_y <- rnorm(n_id, 0, 0.35); u_sx <- rnorm(n_id, 0, 0.15)
  v_y <- rnorm(n_t, 0, 0.25); v_sx <- 0.25 * timet + rnorm(n_t, 0, 0.1)
  rows <- lapply(seq_len(n_id), function(g) {
    x <- rnorm(n_t)
    y <- numeric(n_t)
    y[1] <- u_y[g] + v_y[1] + rnorm(1, 0, 0.5)
    for (t in 2:n_t) {
      sx_gt <- 0.5 + u_sx[g] + v_sx[t]
      y[t] <- u_y[g] + v_y[t] + 0.35 * (y[t - 1] - u_y[g] - v_y[t - 1]) + sx_gt * x[t] + rnorm(1, 0, 0.5)
    }
    data.frame(subject = g, time = seq_len(n_t), y = y, x = x, timet = timet)
  })
  dat <- do.call(rbind, rows)

  m_cc <- mplus_to_dsem("
    VARIABLE:
      NAMES = subject time y x timet;
      CLUSTER = subject time;
      WITHIN = x;
      BETWEEN = (time) timet;
      LAGGED = y(1);
    DEFINE:
      CENTER x (GROUPMEAN subject);
    ANALYSIS:
      TYPE = CROSSCLASSIFIED RANDOM;
      ESTIMATOR = BAYES;
    MODEL:
      %WITHIN%
        sx | y ON x;
        y ON y&1;
      %BETWEEN subject%
        y WITH sx;
      %BETWEEN time%
        sx ON timet;
        y WITH sx;
  ")
  expect_equal(m_cc$id, "subject")
  expect_equal(m_cc$time_cluster, "time")
  expect_equal(m_cc$centering, "groupmean")

  fit_cc <- dsem(m_cc, dat, chains = 2, iter = 260, warmup = 130, seed = 939)
  cf <- coef(fit_cc)
  expect_true(all(c("mean_x", "lag1_y", "time_x_on_timet", "id_var_x", "time_var_x") %in% names(cf)))
  expect_lt(abs(unname(cf["lag1_y"]) - 0.35), 0.18)
  expect_lt(abs(unname(cf["mean_x"]) - 0.50), 0.18)
  expect_gt(unname(cf["time_x_on_timet"]), 0.05)

  te <- dsem_inspect(fit_cc, "time_effects")
  expect_equal(nrow(te), n_t)
  expect_equal(ncol(te), 2L)
})

test_that("DSEM3 exposes both Level-2 and Level-3 random effects and short-T AR(1) recovers well", {
  # Short-T two-level AR(1) (T = 10, N = 60)
  dat_short <- simulate_dsem(n = 10, clusters = 60, intercept = 0.2, ar = 0.5,
                             sigma = 0.6, random_sd = c(0.3, 0.08), seed = 2026)
  fit_short <- dsem("level: 1\n s | y ~ lag(y, 1)\nlevel: 2\n y ~~ s",
                    dat_short, id = "id", time = "time",
                    chains = 2, iter = 400, warmup = 200, seed = 2026)
  expect_lt(abs(unname(coef(fit_short)["mean_s"]) - 0.5), 0.14)

  # Three-level simultaneous block Gibbs + L2/L3 exposure
  dat3 <- simulate_dsem(n = 14, clusters = c(8, 3), id = c("day", "person"),
                        intercept = 1.0, ar = 0.45, sigma = 0.65,
                        random_sd = c(0.35, 0.06, 0.1), seed = 333)
  m3 <- dsem_model("
    level: 1
      s | y ~ lag(y, 1)
    level: 2
      y ~~ y
    level: 3
      y ~~ s
  ", id = c("day", "person"), time = "time", level_type = "threelevel")
  fit3 <- dsem(m3, dat3, chains = 2, iter = 260, warmup = 130, seed = 334)
  l3_re <- dsem_inspect(fit3, "random_effects")
  l2_re <- dsem_inspect(fit3, "l2_random_effects")
  expect_equal(nrow(l3_re), 8L)
  expect_equal(nrow(l2_re), 24L)
  expect_lt(abs(unname(coef(fit3)["fixed_s"]) - 0.45), 0.2)
})

test_that("dsem_hypothesis, dsem_fscomparison, and dsem_irf plotting and summaries work cleanly", {
  dat <- simulate_dsem(n = 20, clusters = 10, intercept = 0.3, ar = 0.45,
                       sigma = 0.7, random_sd = c(0.4, 0.08), seed = 501)
  fit <- dsem("level: 1\n s | y ~ lag(y, 1)\nlevel: 2\n y ~~ s",
              dat, id = "id", time = "time",
              chains = 2, iter = 220, warmup = 110, seed = 502)

  hyp <- dsem_hypothesis(
    fit,
    icc_approx = "var_y / (var_y + sigma2)",
    ar_pos = "mean_s > 0",
    ar_vs_zero = "mean_s - 0.1"
  )
  expect_s3_class(hyp, "DSEMhypothesis")
  expect_equal(nrow(hyp$summary), 3L)
  expect_gt(hyp$summary$prob_gt_0[2L], 0.9)
  expect_true(!is.null(hyp$wald) && is.finite(hyp$wald$statistic))
  expect_output(print(hyp), "Joint Bayesian Wald Test")

  fsc <- dsem_fscomparison(fit, parameter = "y", pairwise = TRUE)
  expect_s3_class(fsc, "DSEMfscomparison")
  expect_equal(nrow(fsc$ranking), 10L)
  expect_equal(dim(fsc$pairwise), c(10L, 10L))
  expect_equal(unname(diag(fsc$pairwise)), rep(0.5, 10L))
  expect_output(print(fsc), "Cluster Random-Effect Comparison")

  irf <- dsem_irf(fit, horizon = 6L)
  expect_s3_class(irf, "DSEMirf")
  expect_equal(nrow(irf$data), 7L)
  expect_true(irf$data$mean[1L] > irf$data$mean[7L])

  tf <- tempfile(fileext = ".pdf")
  on.exit(unlink(tf), add = TRUE)
  grDevices::pdf(tf)
  plot(fsc)
  plot(irf)
  grDevices::dev.off()
  expect_true(file.exists(tf) && file.info(tf)$size > 0)
})
