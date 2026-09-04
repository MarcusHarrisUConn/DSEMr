# Development recovery study for the experimental two-level Gaussian AR(1).
# Run from the package root after devtools::load_all(). This intentionally uses
# more work than the unit-test recovery scenarios.

devtools::load_all(".")

conditions <- expand.grid(
  clusters = c(20L, 50L),
  n = c(30L, 60L),
  ar = c(0.2, 0.5, 0.75),
  random_ar_sd = c(0.03, 0.10),
  replication = seq_len(20L)
)

results <- vector("list", nrow(conditions))
model <- dsem_model("y ~ lag(y, 1)", id = "id", time = "time")

for (i in seq_len(nrow(conditions))) {
  condition <- conditions[i, ]
  seed <- DSEMr:::.dsem_stable_seed(20260904L, paste(condition, collapse = ":"))
  dat <- simulate_dsem(
    n = condition$n, clusters = condition$clusters, intercept = 0.1,
    ar = condition$ar, sigma = 0.8,
    random_sd = c(0.2, condition$random_ar_sd), seed = seed
  )
  fit <- dsem(model, dat, chains = 4, iter = 2000, warmup = 1000,
              seed = seed, engine = "R")
  estimate <- fit$estimates[fit$estimates$parameter == "lag1_y", ]
  results[[i]] <- cbind(condition, estimate,
                        covered = estimate$q2.5 <= condition$ar &&
                          estimate$q97.5 >= condition$ar)
}

recovery <- do.call(rbind, results)
summary <- aggregate(cbind(mean, covered) ~ clusters + n + ar + random_ar_sd,
                     recovery, mean)
summary$bias <- summary$mean - summary$ar
print(summary)
