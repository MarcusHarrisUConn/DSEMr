# Automated Quantitative Mplus Version 8 / Version 9 Parity Audit
# Compares DSEMr posterior medians (Mplus POINT = MEDIAN convention) directly
# against official Mplus User's Guide outputs stored in research/library/*.html.
#
# Run from the package root:
#   Rscript dev/verify-mplus-parity.R

library(DSEMr)

check_case <- function(label, inp, dat, targets, iter = 1500, warmup = 750, seed = 2026) {
  cat(sprintf("\n================ %s ================\n", label))
  t0 <- proc.time()[[3]]
  fit <- dsem(inp, dat, chains = 2, iter = iter, warmup = warmup, seed = seed)
  dt <- proc.time()[[3]] - t0
  med <- stats::coef(fit, point = "median")
  rhats <- stats::setNames(fit$diagnostics$rhat, fit$diagnostics$parameter)
  ess <- stats::setNames(fit$diagnostics$ess_bulk, fit$diagnostics$parameter)
  dic_v <- dsem_dic(fit)
  cat(sprintf("Elapsed: %.2fs | Status: %s | DIC: %s | pD: %s\n",
              dt, fit$status,
              if (is.finite(dic_v["dic"])) sprintf("%.1f", dic_v["dic"]) else "NA",
              if (is.finite(dic_v["pd"])) sprintf("%.1f", dic_v["pd"]) else "NA"))

  rows <- lapply(names(targets), function(nm) {
    mplus_val <- targets[[nm]]
    dsemr_val <- if (nm %in% names(med)) unname(med[nm]) else NA_real_
    data.frame(
      parameter = nm,
      mplus = mplus_val,
      dsemr = round(dsemr_val, 3),
      abs_diff = round(abs(dsemr_val - mplus_val), 3),
      rhat = round(unname(rhats[nm]), 3),
      ess = round(unname(ess[nm]), 1),
      stringsAsFactors = FALSE
    )
  })
  res <- do.call(rbind, rows)
  print(res, row.names = FALSE)
  invisible(list(fit = fit, table = res))
}

# 1. Mplus 6.26 (N=1 Dynamic CFA)
d626 <- read.table("research/library/ex6.26.dat", col.names = paste0("y", 1:4))
inp626 <- paste(readLines("research/library/ex6.26.inp", warn = FALSE), collapse = "\n")
check_case("Mplus 6.26 (Dynamic CFA)", inp626, d626, c(
  ar1_f = 0.310, factor_variance = 0.911,
  loading_y2 = 1.097, loading_y3 = 1.200, loading_y4 = 1.118,
  residual_variance_y1 = 1.004, residual_variance_y2 = 0.890,
  residual_variance_y3 = 1.038, residual_variance_y4 = 1.120
), seed = 626)

# 2. Mplus 6.26b (WNFS Dynamic CFA)
inp626b <- paste(readLines("research/library/ex6.26b.inp", warn = FALSE), collapse = "\n")
check_case("Mplus 6.26b (WNFS Dynamic CFA)", inp626b, d626, c(
  loading_y2 = 1.134, loading_y3 = 1.210, loading_y4 = 1.125,
  lagged_factor_effect_y1 = 0.299, lagged_factor_effect_y2 = 0.299,
  lagged_factor_effect_y3 = 0.401, lagged_factor_effect_y4 = 0.359
), seed = 6262)

# 3. Mplus 6.28 (2-Factor Cross-Lagged SEM)
d628 <- read.table("research/library/ex6.28.dat", col.names = c(paste0("y1", 1:4), paste0("y2", 1:4)))
inp628 <- paste(readLines("research/library/ex6.28.inp", warn = FALSE), collapse = "\n")
check_case("Mplus 6.28 (2-Factor Cross-Lagged SEM)", inp628, d628, c(
  ar1_f1 = 0.441, f1_on_lag1_f2 = 0.197, ar1_f2 = 0.414, f2_on_lag1_f1 = 0.141
), seed = 628)

# 4. Mplus 9.30 (2-Level Location-Scale + TINTERVAL + Distal Z)
d930 <- read.table("research/library/ex9.30.dat", col.names = c("z", "y", "w", "time", "subject"))
inp930 <- paste(readLines("research/library/ex9.30.inp", warn = FALSE), collapse = "\n")
r930 <- check_case("Mplus 9.30 (2L Location-Scale + Distal Z)", inp930, d930, c(
  mean_y = -0.017, mean_s = 0.297, mean_logv = -0.014,
  y_on_w = 0.290, s_on_w = 0.125, logv_on_w = 0.312,
  var_y = 0.107, var_s = 0.010, var_logv = 0.005,
  z_on_y = 0.393, z_on_s = 0.590, z_on_logv = 0.407, var_z = 0.050
), iter = 1600, warmup = 800, seed = 930)
cat("\nMplus 9.30 STDYX & R^2 Check:\n")
print(dsem_standardize(r930$fit, "stdyx"))
print(dsem_r2(r930$fit))
