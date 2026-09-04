ar1_spec <- "y ~ lag(y, 1)"

test_that("simulation is reproducible and stationary inputs are checked", {
  a <- simulate_dsem(n = 30, seed = 42)
  b <- simulate_dsem(n = 30, seed = 42)
  expect_identical(a, b)
  expect_error(simulate_dsem(ar = 1), "strictly between")
})

test_that("lag construction never bridges time gaps", {
  dat <- simulate_dsem(n = 12, seed = 44)
  dat <- dat[-6, ]
  model <- dsem_model(ar1_spec, time = "time")
  prepared <- DSEMr:::.dsem_prepare_ar1(model, dat)
  expect_equal(nrow(prepared$x), 9L)
  expect_false(7L %in% dat$time[prepared$row_index])

  half_step <- transform(dat, time = time / 2)
  half_model <- dsem_model(ar1_spec, time = "time",
                           metadata = list(time_interval = 0.5))
  expect_equal(nrow(DSEMr:::.dsem_prepare_ar1(half_model, half_step)$x), 9L)
})

test_that("Rust sufficient statistics agree with the R reference", {
  x <- cbind(1, c(-1, 0, 1, 2))
  y <- c(0, 1, 2, 4)
  r <- DSEMr:::.dsem_ar_sufficient_R(x, y)
  rust <- DSEMr:::.dsem_ar_sufficient_rust(x, y)
  expect_equal(r$xtx, rust$xtx, tolerance = 1e-12)
  expect_equal(r$xty, rust$xty, tolerance = 1e-12)
  expect_equal(r$yty, rust$yty, tolerance = 1e-12)
})

test_that("grouped Rust sufficient statistics agree with the R reference", {
  groups <- list(
    list(x = cbind(1, c(1, 2)), y = c(2, 4)),
    list(x = cbind(1, c(3, 4, 5)), y = c(6, 8, 9))
  )
  reference <- DSEMr:::.dsem_grouped_sufficient_R(groups)
  rust <- DSEMr:::.dsem_grouped_sufficient_rust(groups)
  expect_equal(reference$xtx, rust$xtx, tolerance = 1e-12)
  expect_equal(reference$xty, rust$xty, tolerance = 1e-12)
  expect_equal(reference$yty, rust$yty, tolerance = 1e-12)
  expect_equal(reference$group_sizes, rust$group_sizes)
})

test_that("reference and Rust paths produce the same posterior draws", {
  dat <- simulate_dsem(n = 100, intercept = 0.2, ar = 0.55, sigma = 0.7, seed = 77)
  model <- dsem_model(ar1_spec, time = "time")
  args <- list(model = model, data = dat, chains = 2, iter = 300,
               warmup = 150, seed = 991)
  fit_r <- do.call(dsem, c(args, list(engine = "R")))
  fit_rust <- do.call(dsem, c(args, list(engine = "rust")))
  expect_equal(fit_r$draws, fit_rust$draws, tolerance = 1e-12)
  expect_equal(unname(coef(fit_r)["lag1_y"]), 0.55, tolerance = 0.2)
  expect_s3_class(fit_rust, "DSEMfit")
  expect_equal(length(fitted(fit_rust)), nrow(dat) - 1L)
})

test_that("process worker count does not change chain draws", {
  skip_on_cran()
  probe <- try(parallel::makePSOCKcluster(1), silent = TRUE)
  if (inherits(probe, "try-error")) skip("Local socket creation is unavailable in this sandbox.")
  parallel::stopCluster(probe)
  dat <- simulate_dsem(n = 60, seed = 3)
  model <- dsem_model(ar1_spec, time = "time")
  sequential <- dsem(model, dat, chains = 2, iter = 120, warmup = 60,
                     seed = 808, engine = "R",
                     compute = dsem_compute(backend = "sequential"))
  parallel <- dsem(model, dat, chains = 2, iter = 120, warmup = 60,
                   seed = 808, engine = "R",
                   compute = dsem_compute(workers = 2, backend = "psock"))
  expect_identical(sequential$chain_seeds, parallel$chain_seeds)
  expect_equal(sequential$draws, parallel$draws, tolerance = 0)
})

test_that("standard extractors expose fitted state", {
  dat <- simulate_dsem(n = 50, seed = 9)
  fit <- dsem(ar1_spec, dat, chains = 1, iter = 80, warmup = 40, seed = 9)
  expect_named(coef(fit), c("(Intercept)", "lag1_y", "sigma2"))
  expect_equal(dim(vcov(fit)), c(3L, 3L))
  expect_equal(dsem_inspect(fit, "model")$hash, fit$model$hash)
  expect_equal(residuals(fit), fit$residuals)
})

test_that("two-level AR(1) estimates population and cluster dynamics", {
  dat <- simulate_dsem(n = 28, clusters = 10, intercept = 0, ar = 0.45,
                       random_sd = c(0.2, 0.05), seed = 17)
  model <- dsem_model(ar1_spec, id = "id", time = "time")
  fit <- dsem(model, dat, chains = 2, iter = 240, warmup = 120,
              seed = 71, engine = "R")
  expect_s3_class(fit, "DSEMfit")
  expect_equal(fit$status, "experimental-two-level-ar1")
  expect_equal(nrow(fit$random_effects), 10L)
  expect_equal(nrow(fit$random_effects_summary), 20L)
  expect_equal(dim(fit$random_effect_draws)[2:4], c(10L, 2L, 2L))
  expect_equal(length(fitted(fit)), 10L * 27L)
  expect_equal(unname(coef(fit)["lag1_y"]), 0.45, tolerance = 0.2)
  expect_true(all(c("tau_intercept", "tau_lag1_y", "sigma2") %in%
                    names(coef(fit))))
  expect_error(dsem(model, dat, chains = 1, iter = 20, warmup = 10,
                    engine = "rust"), "R reference engine")
})

test_that("two-level simulation is reproducible and records generating truth", {
  a <- simulate_dsem(n = 10, clusters = 4, ar = 0.4,
                     random_sd = c(0.1, 0.03), seed = 612)
  b <- simulate_dsem(n = 10, clusters = 4, ar = 0.4,
                     random_sd = c(0.1, 0.03), seed = 612)
  expect_identical(a, b)
  expect_named(a, c("id", "time", "y"))
  expect_equal(nrow(a), 40L)
  expect_equal(attr(a, "dsem_truth")$population[["ar"]], 0.4)
})
