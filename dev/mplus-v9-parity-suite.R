# Full Mplus Version 8 & Version 9 DSEM parity benchmark suite
# Run from the DSEMr package root after installing DSEMr:
#   Rscript dev/mplus-v9-parity-suite.R

library(DSEMr)

run_case <- function(label, inp_file, dat_file, col_names, iter = 1200, warmup = 600, seed = 2026) {
  cat(sprintf("\n--- %s ---\n", label))
  inp <- paste(readLines(inp_file, warn = FALSE), collapse = "\n")
  raw <- utils::read.table(dat_file)
  raw <- raw[, seq_along(col_names), drop = FALSE]
  colnames(raw) <- col_names
  t0 <- proc.time()[[3]]
  fit <- dsem(inp, raw, chains = 2, iter = iter, warmup = warmup, seed = seed)
  dt <- proc.time()[[3]] - t0
  cat(sprintf("Elapsed: %.2fs | Status: %s | DIC: %.1f | pD: %.1f\n",
              dt, fit$status, dsem_dic(fit)[["dic"]], dsem_dic(fit)[["pd"]]))
  print(round(utils::head(coef(fit), 12), 3))
  invisible(fit)
}

run_case("Mplus 6.26 (DAFS Dynamic CFA)", "research/library/ex6.26.inp",
         "research/library/ex6.26.dat", paste0("y", 1:4), seed = 626)
run_case("Mplus 6.26b (WNFS Dynamic CFA)", "research/library/ex6.26b.inp",
         "research/library/ex6.26.dat", paste0("y", 1:4), seed = 6262)
run_case("Mplus 6.27 (Binary Dynamic IRT)", "research/library/ex6.27.inp",
         "research/library/ex6.27.dat", paste0("u", 1:4), seed = 627)
run_case("Mplus 6.28 (2-Factor Cross-Lagged SEM)", "research/library/ex6.28.inp",
         "research/library/ex6.28.dat", c(paste0("y1", 1:4), paste0("y2", 1:4)), seed = 628)
run_case("Mplus 9.30 (2L Location-Scale + TINTERVAL + Distal Z)", "research/library/ex9.30.inp",
         "research/library/ex9.30.dat", c("z", "y", "w", "time", "subject"), iter = 800, warmup = 400, seed = 930)
run_case("Mplus 9.33 (2L Single-Indicator Measurement Error)", "research/library/ex9.33.inp",
         "research/library/ex9.33.dat", c("y", "subject"), iter = 800, warmup = 400, seed = 933)
run_case("Mplus 9.34 (2L Dynamic CFA + Random LogV)", "research/library/ex9.34.inp",
         "research/library/ex9.34.dat", c(paste0("y", 1:4), "subject"), iter = 600, warmup = 300, seed = 934)
run_case("Mplus 9.35 (2L Binary Dynamic IRT + Invariance)", "research/library/ex9.35.inp",
         "research/library/ex9.35part2.dat", c(paste0("u", 1:4), "subject"), iter = 600, warmup = 300, seed = 935)
run_case("Mplus 9.36 (2L 2-Factor Cross-Lagged SEM)", "research/library/ex9.36.inp",
         "research/library/ex9.36.dat", c(paste0("y1", 1:4), paste0("y2", 1:4), "subject"), iter = 500, warmup = 250, seed = 936)
run_case("Mplus 9.37 (2L AR1 + Groupmean X + Trend + 4 Random Slopes)", "research/library/ex9.37.inp",
         "research/library/ex9.37.dat", c("y", "x", "w", "xm", "time", "subject"), iter = 800, warmup = 400, seed = 937)
