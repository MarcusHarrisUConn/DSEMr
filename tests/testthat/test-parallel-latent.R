test_that("latent chains are identical across process counts", {
  skip_on_cran()
  probe <- try(parallel::makePSOCKcluster(1), silent = TRUE)
  if (inherits(probe, "try-error")) skip("Local sockets unavailable")
  parallel::stopCluster(probe)
  model <- dsem_model("f =~ y1 + y2 + y3\nf ~ lag(f, 1)", time = "time")
  dat <- simulate_dsem(model, n = 40, seed = 901)
  args <- list(model = model, data = dat, chains = 2, iter = 240,
               warmup = 120, seed = 902, engine = "rust")
  serial <- do.call(dsem, args)
  processes <- do.call(dsem, c(args, list(compute = dsem_compute(workers = 2, backend = "psock"))))
  expect_identical(serial$chain_seeds, processes$chain_seeds)
  expect_identical(serial$draws, processes$draws)
  expect_identical(serial$latent_draws, processes$latent_draws)
  expect_identical(fitted(serial), fitted(processes))
})

test_that("Monte Carlo summaries are identical across process counts", {
  skip_on_cran()
  probe <- try(parallel::makePSOCKcluster(1), silent = TRUE)
  if (inherits(probe, "try-error")) skip("Local sockets unavailable")
  parallel::stopCluster(probe)
  args <- list(model = "y ~ lag(y, 1)", replications = 2,
               generator = function(seed) DSEMr::simulate_dsem(n = 40, seed = seed),
               seed = 209, truth = c(lag1_y = 0.5),
               fit_args = list(chains = 2, iter = 300, warmup = 150, engine = "R"))
  serial <- do.call(dsem_monte_carlo, args)
  processes <- do.call(dsem_monte_carlo, c(args,
    list(compute = dsem_compute(workers = 2, backend = "psock"))))
  expect_identical(serial$replication_seeds, processes$replication_seeds)
  expect_identical(serial$estimates, processes$estimates)
  expect_identical(serial$summary, processes$summary)
})
