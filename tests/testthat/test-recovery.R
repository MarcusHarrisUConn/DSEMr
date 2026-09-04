test_that("two-level AR recovery holds across fixed reference scenarios", {
  skip_on_cran()
  scenarios <- data.frame(
    seed = c(1101L, 1102L, 1103L),
    ar = c(0.20, 0.45, 0.70),
    random_ar_sd = c(0.04, 0.08, 0.05)
  )
  recovered <- numeric(nrow(scenarios))
  for (i in seq_len(nrow(scenarios))) {
    dat <- simulate_dsem(
      n = 40, clusters = 20, intercept = 0.1, ar = scenarios$ar[i],
      sigma = 0.8, random_sd = c(0.2, scenarios$random_ar_sd[i]),
      seed = scenarios$seed[i]
    )
    model <- dsem_model("y ~ lag(y, 1)", id = "id", time = "time")
    fit <- dsem(model, dat, chains = 2, iter = 400, warmup = 200,
                seed = scenarios$seed[i] + 50L, engine = "R")
    recovered[i] <- unname(coef(fit)["lag1_y"])
  }
  expect_lt(mean(abs(recovered - scenarios$ar)), 0.12)
  expect_true(all(abs(recovered - scenarios$ar) < 0.20))
})
