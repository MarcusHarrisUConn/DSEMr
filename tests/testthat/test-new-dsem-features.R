test_that("two-level random residual covariance recovers shared shock log-variance (9.32b)", {
  set.seed(9322)
  G <- 45L; TT <- 40L
  rows <- vector("list", G)
  for (g in seq_len(G)) {
    mu1 <- rnorm(1, 0, 0.5); mu2 <- rnorm(1, 0, 0.5)
    lv1 <- rnorm(1, -0.3, 0.25); lv2 <- rnorm(1, -0.3, 0.25)
    lvf <- rnorm(1, -0.4, 0.35)
    y1 <- numeric(TT); y2 <- numeric(TT)
    y1[1] <- mu1 + rnorm(1); y2[1] <- mu2 + rnorm(1)
    for (t in 2:TT) {
      f_t <- rnorm(1, 0, exp(0.5 * lvf))
      y1[t] <- mu1 + 0.35 * (y1[t - 1] - mu1) + f_t + rnorm(1, 0, exp(0.5 * lv1))
      y2[t] <- mu2 + 0.30 * (y2[t - 1] - mu2) + f_t + rnorm(1, 0, exp(0.5 * lv2))
    }
    rows[[g]] <- data.frame(id = g, time = seq_len(TT), y1 = y1, y2 = y2)
  }
  dat <- do.call(rbind, rows)

  # Lavaan-style random residual covariance syntax
  m_lav <- dsem_model("
    y1 ~ lag(y1, 1) + lag(y2, 1)
    y2 ~ lag(y2, 1) + lag(y1, 1)
    logv1 | y1 ~~ y1
    logv2 | y2 ~~ y2
    logvf | y1 ~~ y2
  ", id = "id", time = "time")
  fit_lav <- dsem(m_lav, dat, chains = 2, iter = 300, warmup = 150, seed = 101)
  est <- fit_lav$estimates
  expect_true("mean_logvf" %in% est$parameter)
  expect_true("var_logvf" %in% est$parameter)
  expect_gt(est$mean[est$parameter == "mean_logvf"], -1.2)
  expect_lt(est$mean[est$parameter == "mean_logvf"], 0.3)
  expect_true(all(is.finite(est$mean)))

  # Section-based factor-loading covariance syntax (f BY y1@1 y2@1; logvf | f;)
  m_sec <- dsem_model("
    VARIABLE: CLUSTER = id; LAGGED = y1(1) y2(1);
    ANALYSIS: TYPE = TWOLEVEL RANDOM;
    MODEL:
    %WITHIN%
    y1 ON y1&1 y2&1;
    y2 ON y2&1 y1&1;
    f BY y1@1 y2@1;
    logv1 | y1;
    logv2 | y2;
    logvf | f;
    %BETWEEN%
    y1 y2 logv1 logv2 logvf WITH y1 y2 logv1 logv2 logvf;
  ", syntax = "mplus", time = "time")
  fit_sec <- dsem(m_sec, dat, chains = 2, iter = 250, warmup = 125, seed = 102)
  expect_true("mean_logvf" %in% fit_sec$estimates$parameter)
  expect_true(is.finite(dsem_dic(fit_sec)[["dic"]]))
})

test_that("two-level ARMA(1,1) exact collapsed kernel recovers negative MA(1) and positive marginal DIC", {
  set.seed(9333)
  G <- 40L; TT <- 50L
  rows <- vector("list", G)
  for (g in seq_len(G)) {
    mug <- rnorm(1, 0.2, 0.6)
    sg <- rnorm(1, 0.40, 0.10)
    e <- rnorm(TT, 0, sqrt(1.2))
    y <- numeric(TT)
    y[1] <- mug + e[1]
    for (t in 2:TT) {
      y[t] <- mug + sg * (y[t - 1] - mug) + e[t] - 0.30 * e[t - 1]
    }
    rows[[g]] <- data.frame(subject = g, time = seq_len(TT), y = y)
  }
  dat <- do.call(rbind, rows)
  txt <- "
    VARIABLE: CLUSTER = subject; LAGGED = y(1);
    ANALYSIS: TYPE = TWOLEVEL RANDOM;
    MODEL:
    %WITHIN%
    e BY y@1 (&1);
    y@0.01;
    s | y ON y&1;
    y ON e&1;
    %BETWEEN%
    y WITH s;
  "
  fit <- dsem(txt, dat, syntax = "mplus", time = "time",
              chains = 2, iter = 350, warmup = 175, seed = 202,
              control = list(prior_mode = "diffuse"))
  est <- fit$estimates
  ma_est <- est$mean[est$parameter == "y_on_lag1_e"]
  ar_est <- est$mean[est$parameter == "mean_s"]
  ve_est <- est$mean[est$parameter == "var_e"]
  expect_lt(ma_est, -0.10)
  expect_gt(ma_est, -0.55)
  expect_gt(ar_est, 0.15)
  expect_lt(ar_est, 0.65)
  expect_gt(ve_est, 0.8)
  expect_lt(ve_est, 1.7)
  dc <- dsem_dic(fit)
  expect_true(is.finite(dc[["dic"]]) && dc[["dic"]] > 0)
})

test_that("cross-classified DSEM supports DEFINE transforms and selective level-specific predictors (9.39)", {
  set.seed(939)
  N_id <- 25L; N_t <- 25L
  u_id <- rnorm(N_id, 0, 0.5)
  u_t <- rnorm(N_t, 0, 0.3)
  sx_t <- 0.4 + 0.03 * scale(seq_len(N_t), scale = FALSE)[, 1] + rnorm(N_t, 0, 0.08)
  rows <- vector("list", N_id)
  for (i in seq_len(N_id)) {
    x <- rnorm(N_t)
    y <- numeric(N_t)
    y[1] <- u_id[i] + u_t[1] + sx_t[1] * x[1] + rnorm(1, 0, 0.7)
    for (t in 2:N_t) {
      y[t] <- u_id[i] + u_t[t] + 0.25 * (y[t - 1] - u_id[i] - u_t[t - 1]) +
        sx_t[t] * x[t] + rnorm(1, 0, 0.7)
    }
    rows[[i]] <- data.frame(id = i, time = seq_len(N_t), y = y, x = x)
  }
  dat <- do.call(rbind, rows)

  txt <- "
    DEFINE:
      timet = (time - 13) / 10;
      xc = x - CLUSTER_MEAN(x);
    VARIABLE:
      CLUSTER = id time;
      LAGGED = y(1);
    ANALYSIS:
      TYPE = CROSSCLASSIFIED RANDOM;
    MODEL:
      %WITHIN%
      s | y ON y&1;
      sx | y ON xc;
      %BETWEEN id%
      y s sx WITH y s sx;
      %BETWEEN time%
      sx ON timet;
  "
  fit <- dsem(txt, dat, syntax = "mplus", chains = 2, iter = 300, warmup = 150, seed = 303)
  est <- fit$estimates
  expect_true("time_sx_on_timet" %in% est$parameter)
  expect_false("time_y_on_timet" %in% est$parameter)
  expect_gt(est$mean[est$parameter == "mean_sx"], 0.2)
  expect_gt(est$mean[est$parameter == "time_sx_on_timet"], 0.05)
})

test_that("cross-classified Dynamic CFA recovers invariance (9.40a) and random factor loadings (9.40b)", {
  set.seed(940)
  N_id <- 24L; N_t <- 24L; J <- 3L
  lam_true <- c(1.0, 0.9, 1.1)
  f_id <- rnorm(N_id, 0, 0.7)
  f_tm <- rnorm(N_t, 0, 0.5)
  rows <- vector("list", N_id)
  for (i in seq_len(N_id)) {
    l_i <- lam_true + rnorm(J, 0, 0.22)
    eta <- numeric(N_t)
    eta[1] <- rnorm(1, 0, 1)
    for (t in 2:N_t) eta[t] <- 0.35 * eta[t - 1] + rnorm(1, 0, 1)
    ymat <- matrix(0, N_t, J)
    for (j in seq_len(J)) {
      ymat[, j] <- l_i[j] * (eta + f_id[i] + f_tm) + rnorm(N_t, 0, 0.7)
    }
    rows[[i]] <- data.frame(id = i, time = seq_len(N_t),
                            y1 = ymat[, 1], y2 = ymat[, 2], y3 = ymat[, 3])
  }
  dat <- do.call(rbind, rows)

  # 9.40a: Cross-classified Dynamic CFA with measurement invariance
  txt_a <- "
    VARIABLE: CLUSTER = id time;
    ANALYSIS: TYPE = CROSSCLASSIFIED RANDOM;
    MODEL:
      %WITHIN%
      f BY y1-y3 (&1);
      s | f ON f&1;
      %BETWEEN id%
      f_id BY y1-y3;
      %BETWEEN time%
      f_tm BY y1-y3;
  "
  fit_a <- dsem(txt_a, dat, syntax = "mplus", chains = 2, iter = 250, warmup = 125, seed = 401)
  expect_true("id_var_f_id" %in% fit_a$estimates$parameter)
  expect_true("time_var_f_tm" %in% fit_a$estimates$parameter)
  expect_gt(fit_a$estimates$mean[fit_a$estimates$parameter == "mean_s"], 0.1)

  # 9.40b: Random factor loadings across subjects
  txt_b <- "
    VARIABLE: CLUSTER = id time;
    ANALYSIS: TYPE = CROSSCLASSIFIED RANDOM;
    MODEL:
      %WITHIN%
      s1-s3 | f BY y1-y3 (&1);
      f@1;
      ar | f ON f&1;
      %BETWEEN id%
      f_id BY y1-y3;
      %BETWEEN time%
      f_tm BY y1-y3;
  "
  fit_b <- dsem(txt_b, dat, syntax = "mplus", chains = 2, iter = 250, warmup = 125, seed = 402)
  est_b <- fit_b$estimates
  expect_true(all(c("mean_s1", "mean_s2", "mean_s3", "id_var_s1", "id_var_s2", "id_var_s3") %in% est_b$parameter))
  expect_gt(est_b$mean[est_b$parameter == "mean_s1"], 0.5)
  expect_gt(est_b$mean[est_b$parameter == "id_var_s1"], 0.0)
})

test_that("three-level DSEM3 recovers Level-2 AR(1), PSEM Lasso, and random residual covariance", {
  set.seed(3001)
  L3 <- 18L; D_per <- 12L; T_per <- 8L
  rows <- vector("list", L3 * D_per)
  idx <- 1L
  for (k in seq_len(L3)) {
    u3 <- rnorm(1, 0.5, 0.5)
    w3 <- rnorm(1)
    u2 <- numeric(D_per)
    u2[1] <- rnorm(1, 0, 0.5)
    for (d_i in 2:D_per) {
      u2[d_i] <- 0.45 * u2[d_i - 1] + rnorm(1, 0, 0.4)
    }
    for (d_i in seq_len(D_per)) {
      y <- numeric(T_per)
      mu_kd <- u3 + 0.4 * w3 + u2[d_i]
      y[1] <- mu_kd + rnorm(1, 0, 0.6)
      for (t in 2:T_per) {
        y[t] <- mu_kd + 0.30 * (y[t - 1] - mu_kd) + rnorm(1, 0, 0.6)
      }
      rows[[idx]] <- data.frame(person = k, day = d_i, obs = seq_len(T_per), y = y, w3 = w3)
      idx <- idx + 1L
    }
  }
  dat <- do.call(rbind, rows)

  txt_l2ar <- "
    VARIABLE: CLUSTER = person day; LAGGED = y(1);
    ANALYSIS: TYPE = THREELEVEL RANDOM;
    MODEL:
      %WITHIN%
      s | y ON y&1;
      %BETWEEN day%
      y ON y&1;
      %BETWEEN person%
      y ON w3;
      y WITH s;
  "
  fit_l2ar <- dsem(txt_l2ar, dat, syntax = "mplus", time = "obs",
                   chains = 2, iter = 300, warmup = 150, seed = 501)
  est <- fit_l2ar$estimates
  expect_true("l2_ar1_y" %in% est$parameter)
  expect_gt(est$mean[est$parameter == "l2_ar1_y"], 0.15)
  expect_lt(est$mean[est$parameter == "l2_ar1_y"], 0.75)
  expect_gt(est$mean[est$parameter == "l3_y_on_w3"], 0.1)
  expect_false("l3_s_on_w3" %in% est$parameter)

  # PSEM Bayesian Lasso mode on 3L model
  fit_pen <- dsem(txt_l2ar, dat, syntax = "mplus", time = "obs",
                  chains = 2, iter = 200, warmup = 100, seed = 502,
                  control = list(prior_mode = "penalized"))
  expect_true(all(is.finite(fit_pen$estimates$mean)))
})

test_that("continuous-time DSEM recovers irregular-interval OU drift/diffusion and CT IRF", {
  set.seed(777)
  # 1. N=1 Univariate CT-DSEM with irregular dt in [0.2, 2.5]
  TT <- 180L
  dt <- runif(TT - 1L, 0.25, 2.2)
  tm <- c(0, cumsum(dt))
  a_true <- -0.65; q_true <- 0.80; mu_true <- 1.2
  y <- numeric(TT)
  y[1] <- mu_true + rnorm(1, 0, sqrt(q_true / (-2 * a_true)))
  for (t in 2:TT) {
    ph <- exp(a_true * dt[t - 1L])
    v <- q_true * (1 - ph^2) / (-2 * a_true)
    y[t] <- mu_true + ph * (y[t - 1] - mu_true) + rnorm(1, 0, sqrt(v))
  }
  dat1 <- data.frame(time = tm, y = y)
  fit_ct1 <- dsem_ct("y ~ lag(y, 1)", dat1, time = "time",
                     chains = 2, iter = 400, warmup = 200, seed = 701)
  est1 <- fit_ct1$estimates
  expect_true(isTRUE(fit_ct1$model$continuous_time))
  expect_lt(est1$mean[est1$parameter == "drift_y"], -0.30)
  expect_gt(est1$mean[est1$parameter == "drift_y"], -1.20)
  expect_gt(est1$mean[est1$parameter == "diffusion_y"], 0.40)
  expect_lt(est1$mean[est1$parameter == "diffusion_y"], 1.40)
  expect_gt(est1$mean[est1$parameter == "(Intercept)"], 0.6)
  expect_lt(est1$mean[est1$parameter == "(Intercept)"], 1.8)

  # CT Impulse Response Function via exp(A * h)
  irf <- dsem_irf(fit_ct1, horizon = 5)
  expect_s3_class(irf, "DSEMirf")
  expect_lt(irf$data$q50[irf$data$horizon == 5], irf$data$q50[irf$data$horizon == 1])

  # 2. Two-Level CT-DSEM with irregular observation intervals
  G <- 25L; Tg <- 35L
  rows2 <- vector("list", G)
  for (g in seq_len(G)) {
    dt_g <- runif(Tg - 1L, 0.3, 1.8)
    tm_g <- c(0, cumsum(dt_g))
    mu_g <- rnorm(1, 0.5, 0.6)
    yg <- numeric(Tg)
    yg[1] <- mu_g + rnorm(1, 0, 1)
    for (t in 2:Tg) {
      ph <- exp(-0.55 * dt_g[t - 1L])
      v <- 0.70 * (1 - ph^2) / (2 * 0.55)
      yg[t] <- mu_g + ph * (yg[t - 1] - mu_g) + rnorm(1, 0, sqrt(v))
    }
    rows2[[g]] <- data.frame(id = g, time = tm_g, y = yg)
  }
  dat2 <- do.call(rbind, rows2)
  fit_ct2 <- dsem_ct("y ~ lag(y, 1)", dat2, id = "id", time = "time",
                     chains = 2, iter = 350, warmup = 175, seed = 702)
  est2 <- fit_ct2$estimates
  expect_lt(est2$mean[est2$parameter == "drift_y"], -0.25)
  expect_gt(est2$mean[est2$parameter == "drift_y"], -0.95)
  expect_gt(est2$mean[est2$parameter == "var_y"], 0.15)
})

test_that("two-level ordinal K=3 probit recovers ordered thresholds and AR(1) slope", {
  set.seed(888)
  G <- 30L; TT <- 35L
  rows <- vector("list", G)
  for (g in seq_len(G)) {
    mug <- rnorm(1, 0, 0.5)
    ys <- numeric(TT)
    ys[1] <- mug + rnorm(1)
    for (t in 2:TT) ys[t] <- mug + 0.40 * (ys[t - 1] - mug) + rnorm(1)
    u <- ifelse(ys < 0.0, 0L, ifelse(ys < 1.0, 1L, 2L))
    rows[[g]] <- data.frame(id = g, time = seq_len(TT), u = u)
  }
  dat <- do.call(rbind, rows)
  m_ord <- dsem_model("s | u ~ lag(u, 1)", id = "id", time = "time", categorical = "u")
  fit_ord <- dsem(m_ord, dat, chains = 2, iter = 300, warmup = 150, seed = 801)
  est <- fit_ord$estimates
  expect_true("threshold_u$2" %in% est$parameter)
  expect_gt(est$mean[est$parameter == "threshold_u$2"], 0.5)
  expect_lt(est$mean[est$parameter == "threshold_u$2"], 1.6)
  expect_gt(est$mean[est$parameter == "mean_s"], 0.15)
})

test_that("MODEL CONSTRAINT, MODEL TEST, strict Monte Carlo convergence, and dsem_spec extensions work", {
  set.seed(909)
  TT <- 90L
  y <- as.numeric(stats::arima.sim(list(ar = 0.45), n = TT)) + 0.5
  dat <- data.frame(time = seq_len(TT), y = y)
  txt <- "
    VARIABLE: LAGGED = y(1);
    ANALYSIS: ESTIMATOR = BAYES;
    MODEL: y ON y&1;
    MODEL CONSTRAINT:
      NEW(persist);
      persist = lag1_y * 2;
    MODEL TEST:
      lag1_y = 0;
  "
  fit <- dsem(txt, dat, syntax = "mplus", time = "time",
              chains = 2, iter = 250, warmup = 125, seed = 901,
              control = list(prior_mode = "diffuse"))
  expect_true(!is.null(fit$constraints))
  expect_true("persist" %in% fit$constraints$summary$name)
  expect_equal(fit$constraints$summary$mean[fit$constraints$summary$name == "persist"],
               2 * fit$estimates$mean[fit$estimates$parameter == "lag1_y"], tolerance = 1e-6)
  expect_true(!is.null(fit$wald_test))
  expect_true(fit$wald_test$wald$p_value < 0.05)

  # Strict Monte Carlo convergence gate
  mc <- dsem_monte_carlo(
    "y ~ lag(y, 1)", replications = 2L,
    generator = function(seed) simulate_dsem(n = 60L, ar = 0.4, intercept = 0, sigma = 1, seed = seed),
    truth = c(lag1_y = 0.4),
    convergence = "strict", rhat_max = 1.20,
    fit_args = list(chains = 2L, iter = 250L, warmup = 125L),
    seed = 902
  )
  expect_equal(mc$convergence, "strict")
  expect_true(all(is.numeric(mc$summary$convergence_rate)))

  # dsem_spec round-trip with continuous_time, random_residual_cov, l2_ar, prior_mode = "diffuse"
  sp_ct <- dsem_spec(outcomes = "y", time = "time", continuous_time = TRUE, prior_mode = "diffuse")
  m_ct <- as_dsem_model(sp_ct)
  expect_true(isTRUE(m_ct$continuous_time))
  sp_3l <- dsem_spec(outcomes = c("y1", "y2"), id = c("day", "person"), time = "obs",
                     level_type = "threelevel", random_residual_cov = TRUE, l2_ar = TRUE)
  m_3l <- as_dsem_model(sp_3l)
  expect_equal(m_3l$level_type, "threelevel")
  expect_true(any(m_3l$terms$op == "~~" & m_3l$terms$lhs == "y1" & m_3l$terms$rhs == "y2" & m_3l$terms$random))
  expect_true(any(m_3l$terms$level == "between_2" & m_3l$terms$lag == 1L))
})
