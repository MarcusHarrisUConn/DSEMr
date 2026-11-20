# ==============================================================================
# DSEMr 0.1.0 — RStudio Interactive Walkthrough
# ==============================================================================
# Install from your local source tarball (offline, compiled Rust backend):
#   install.packages("DSEMr_0.1.0.tar.gz", repos = NULL, type = "source")
# Or directly from GitHub:
#   remotes::install_github("MarcusHarrisUConn/DSEMr")

library(DSEMr)

# ------------------------------------------------------------------------------
# Example 1: Three-Level DSEM (DSEM3: occasions within days within persons)
# ------------------------------------------------------------------------------
dat_3l <- simulate_dsem(
  n = 20, clusters = c(12, 4), id = c("day", "person"),
  intercept = 1.5, ar = 0.45, sigma = 0.7,
  random_sd = c(0.35, 0.08, 0.15), seed = 2026
)

m_3l <- mplus_to_dsem("
  VARIABLE:
    CLUSTER = person day;
  ANALYSIS:
    TYPE = THREELEVEL RANDOM;
    ESTIMATOR = BAYES;
  MODEL:
    %WITHIN%
      s | y ON y&1;
      logv | y;
    %BETWEEN day%
      y;
    %BETWEEN person%
      y s logv WITH y s logv;
", time = "time")

fit_3l <- dsem(m_3l, dat_3l, chains = 2, iter = 1000, warmup = 500, seed = 2026)

summary(fit_3l, standardized = TRUE)
dsem_dic(fit_3l)
head(dsem_inspect(fit_3l, "random_effects_summary"))

# Visual diagnostics, path diagram, APA table, LaTeX equations & reproducible code:
plot(fit_3l, type = "trace", parameter = "fixed_s")
plot(fit_3l, type = "intervals")
dsem_diagram(fit_3l, format = "plot")
dsem_table(fit_3l, format = "markdown", point = "median")
dsem_equations(fit_3l, substitute = TRUE)
dsem_code(fit_3l, data_name = "dat_3l")

# ------------------------------------------------------------------------------
# Example 2: Two-Level Location-Scale DSEM (TINTERVAL + Distal Z Benchmark)
# ------------------------------------------------------------------------------
d930 <- read.table("research/library/ex9.30.dat",
                   col.names = c("z", "y", "w", "time", "subject"))
inp930 <- paste(readLines("research/library/ex9.30.inp", warn = FALSE), collapse = "\n")

fit_930 <- dsem(inp930, d930, chains = 2, iter = 1600, warmup = 800, seed = 930)
summary(fit_930, standardized = TRUE, r2 = TRUE, point = "median")
dsem_dic(fit_930)
dsem_standardize(fit_930, type = "stdyx")
dsem_r2(fit_930)
dsem_table(fit_930, format = "latex", point = "median")
dsem_diagram(fit_930, format = "plot")
plot(fit_930, type = "intervals")

# ------------------------------------------------------------------------------
# Example 3: Model Comparison (DAFS 6.26 vs WNFS 6.26b) & FFBS Latent States
# ------------------------------------------------------------------------------
d626 <- read.table("research/library/ex6.26.dat", col.names = paste0("y", 1:4))
inp626 <- paste(readLines("research/library/ex6.26.inp", warn = FALSE), collapse = "\n")
inp626b <- paste(readLines("research/library/ex6.26b.inp", warn = FALSE), collapse = "\n")

fit_626 <- dsem(inp626, d626, chains = 2, iter = 1500, warmup = 750, seed = 626)
fit_626b <- dsem(inp626b, d626, chains = 2, iter = 1500, warmup = 750, seed = 6262)

cmp <- dsem_compare(DAFS_626 = fit_626, WNFS_626b = fit_626b)
cmp
dsem_table(cmp, format = "markdown")
plot(fit_626, type = "latent")
