latent_spec <- "
  f =~ y1 + y2 + y3
  f ~ lag(f, 1)
"

test_that("latent Gaussian simulation and sampler recover a dynamic factor", {
  model <- dsem_model(latent_spec, time = "time")
  dat <- simulate_dsem(
    model, n = 160, intercept = 0, ar = 0.55, sigma = 0.7,
    loadings = c(1, 0.8, 1.2), indicator_intercepts = c(0.2, -0.1, 0.3),
    residual_sd = c(0.35, 0.4, 0.45), seed = 9301
  )
  fit <- dsem(model, dat, chains = 2, iter = 700, warmup = 350,
              seed = 9302, engine = "R")
  expect_s3_class(fit, "DSEMfit")
  expect_equal(fit$status, "validated-latent-gaussian-ar1")
  expect_equal(fit$identification$marker_indicator, "y1")
  expect_equal(fit$identification$state_intercept, 0)
  expect_equal(dim(fit$latent_draws), c(350L, 160L, 2L))
  expect_equal(nrow(fit$latent_states), 160L)
  expect_equal(dim(fitted(fit)), c(160L, 3L))
  joint_fitted <- Reduce(`+`, lapply(1:2, function(chain) {
    colMeans(sweep(sweep(fit$latent_draws[, , chain], 1L,
      fit$draws[, "loading_y2", chain], "*"), 1L,
      fit$draws[, "intercept_y2", chain], "+"))
  })) / 2
  expect_equal(unname(fitted(fit)[, "y2"]), unname(joint_fitted))
  expect_equal(unname(coef(fit)["ar1_f"]), 0.55, tolerance = 0.22)
  expect_equal(unname(coef(fit)["loading_y2"]), 0.8, tolerance = 0.3)
  expect_gt(stats::cor(fit$latent_states$mean,
                       attr(dat, "dsem_truth")$latent_state), 0.7)
})

test_that("latent sampler handles indicator missingness and rejects implicit gaps", {
  model <- dsem_model(latent_spec, time = "time")
  dat <- simulate_dsem(model, n = 60, seed = 81)
  dat$y2[seq(3, 30, by = 4)] <- NA_real_
  fit <- dsem(model, dat, chains = 1, iter = 160, warmup = 80,
              seed = 82, engine = "R")
  expect_true(is.finite(fit$fit$log_lik))
  expect_error(dsem(model, dat[-20, ], chains = 1, iter = 40, warmup = 20),
               "complete regular time grid")
  rust_fit <- dsem(model, dat, chains = 1, iter = 160, warmup = 80,
                   seed = 82, engine = "rust")
  expect_equal(fit$draws, rust_fit$draws, tolerance = 1e-10)
  expect_equal(rust_fit$engine, "rust")
})

test_that("lavaan-like and Mplus-like latent syntax compile equivalently", {
  lavaan_model <- dsem_model(latent_spec, time = "time")
  mplus_model <- dsem_model("
    ANALYSIS: TYPE = BAYES;
    MODEL:
      %WITHIN%
      f BY y1 y2 y3;
      f ON f&1;
  ", syntax = "mplus", time = "time")
  expect_equal(lavaan_model$terms, mplus_model$terms)
})

test_that("public Mplus 6.26 syntax features compile cleanly", {
  model <- dsem_model("
    TITLE: N=1 dynamic CFA;
    DATA: FILE = ex6.26.dat;
    VARIABLE: NAMES = y1-y4;
    ANALYSIS: ESTIMATOR = BAYES; BITERATIONS = (2000);
    MODEL: f BY y1-y4 (&1); f ON f&1;
    OUTPUT: TECH1 TECH8;
    PLOT: TYPE = PLOT3;
  ", syntax = "mplus", time = "time")
  expect_equal(model$terms$rhs[model$terms$op == "=~"],
               c("y1", "y2", "y3", "y4"))
  expect_equal(model$measurement_lags,
               data.frame(factor = "f", lag = 1L))
})
