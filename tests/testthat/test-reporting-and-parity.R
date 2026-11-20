test_that("dsem_compare and anova rank models by DIC with valid weights", {
  dat <- simulate_dsem(n = 80, intercept = 0.2, ar = 0.5, sigma = 0.7, seed = 101)
  dat$x <- stats::rnorm(80)
  f1 <- dsem("y ~ lag(y, 1)", dat, time = "time", chains = 2, iter = 160, warmup = 80, seed = 101)
  f2 <- dsem("y ~ lag(y, 1) + x", dat, time = "time", chains = 2, iter = 160, warmup = 80, seed = 101)

  cmp <- dsem_compare(AR1 = f1, AR1_X = f2)
  expect_s3_class(cmp, "DSEMcompare")
  expect_equal(nrow(cmp), 2L)
  expect_equal(min(cmp$delta_dic), 0)
  expect_equal(sum(cmp$weight), 1, tolerance = 1e-8)
  expect_true(all(c("model", "dic", "pd", "delta_dic", "weight", "max_rhat", "converged") %in% names(cmp)))

  an <- stats::anova(f1, f2)
  expect_s3_class(an, "DSEMcompare")
  expect_equal(nrow(an), 2L)
  expect_output(print(cmp), "Preferred by DIC")
})

test_that("dsem_table formats APA tables across data.frame, markdown, latex, and html", {
  dat <- simulate_dsem(n = 80, intercept = 0.2, ar = 0.5, sigma = 0.7, seed = 102)
  f1 <- dsem("y ~ lag(y, 1)", dat, time = "time", chains = 2, iter = 160, warmup = 80, seed = 102)
  f2 <- dsem("y ~ lag(y, 1)", dat, time = "time", chains = 2, iter = 160, warmup = 80, seed = 103)
  cmp <- dsem_compare(M1 = f1, M2 = f2)

  df_tab <- dsem_table(f1, format = "data.frame", include_diagnostics = TRUE)
  expect_s3_class(df_tab, "data.frame")
  expect_true(all(c("Section", "Parameter", "Estimate", "SD", "95% CrI", "p", "Rhat", "ESS") %in% names(df_tab)))

  md_tab <- dsem_table(f1, format = "markdown", point = "median")
  expect_s3_class(md_tab, "DSEMtable")
  expect_true(grepl("\\*Note\\.\\*", md_tab$text))

  tex_tab <- dsem_table(f1, format = "latex", standardized = TRUE)
  expect_true(grepl("\\\\toprule", tex_tab$text))
  expect_true(grepl("\\\\bottomrule", tex_tab$text))

  html_tab <- dsem_table(f1, format = "html", ci_level = 0.90)
  expect_true(grepl("90% CrI", html_tab$text, fixed = TRUE))
  expect_true(grepl("<table", html_tab$text, fixed = TRUE))

  cmp_tex <- dsem_table(cmp, format = "latex")
  expect_true(grepl("DIC", cmp_tex$text, fixed = TRUE))
})

test_that("dsem_equations generates symbolic and substituted LaTeX across all hierarchies", {
  # Single-level N=1
  d1 <- simulate_dsem(n = 60, ar = 0.45, sigma = 0.7, seed = 104)
  f1 <- dsem("y ~ lag(y, 1)", d1, time = "time", chains = 2, iter = 160, warmup = 80, seed = 104)
  eq_sym <- dsem_equations(f1)
  eq_sub <- dsem_equations(f1, substitute = TRUE, wrap = "dollar")
  expect_s3_class(eq_sym, "DSEMequations")
  expect_true(grepl("\\\\begin\\{align\\*\\}", eq_sym$latex))
  expect_true(grepl("\\$\\$", eq_sub$latex))
  expect_true(eq_sub$substituted)

  # Two-level location-scale + distal
  s2 <- dsem_spec(
    outcomes = "y", lags = 1L, predictors_between = "w", distal = "z",
    id = "subject", time = "time", random_slopes = TRUE, random_logv = TRUE
  )
  eq2 <- dsem_equations(s2)
  expect_true(grepl("Level 1", eq2$latex, fixed = TRUE))
  expect_true(grepl("Level 2", eq2$latex, fixed = TRUE))
  expect_true(grepl("Distal Outcome", eq2$latex, fixed = TRUE))

  # Three-level DSEM3
  s3 <- dsem_spec(
    outcomes = "y", lags = 1L, id = c("day", "person"), time = "time",
    random_slopes = TRUE, random_logv = TRUE
  )
  eq3 <- dsem_equations(s3)
  expect_true(grepl("Level 3", eq3$latex, fixed = TRUE))

  # Dynamic CFA
  m_cfa <- dsem_model("f =~ y1 + y2 + y3 + y4\nf ~ lag(f, 1)", time = "time")
  eq_cfa <- dsem_equations(m_cfa)
  expect_true(grepl("Measurement Model", eq_cfa$latex, fixed = TRUE))
})

test_that("dsem_diagram exports plot, svg, mermaid, and dot without graphics leaks", {
  s2 <- dsem_spec(
    outcomes = "y", lags = 1L, predictors_between = "w", distal = "z",
    id = "subject", time = "time", random_slopes = TRUE, random_logv = TRUE
  )
  dg_svg <- dsem_diagram(s2, format = "svg")
  expect_s3_class(dg_svg, "DSEMdiagram")
  expect_true(grepl("^<svg ", dg_svg$svg))
  expect_true(grepl("WITHIN \\(Level 1\\)", dg_svg$svg))
  expect_true(grepl("BETWEEN \\(Level 2\\)", dg_svg$svg))

  dg_mmd <- dsem_diagram(s2, format = "mermaid")
  expect_true(grepl("flowchart LR", dg_mmd$mermaid, fixed = TRUE))

  dg_dot <- dsem_diagram(s2, format = "dot")
  expect_true(grepl("digraph DSEM", dg_dot$dot, fixed = TRUE))

  # Base graphics plot renders cleanly into a temporary PDF device
  tmp_pdf <- tempfile(fileext = ".pdf")
  on.exit(unlink(tmp_pdf), add = TRUE)
  grDevices::pdf(tmp_pdf)
  on.exit(if (grDevices::dev.cur() > 1L) grDevices::dev.off(), add = TRUE)
  res_p <- dsem_diagram(s2, format = "plot")
  expect_s3_class(res_p, "DSEMdiagram")
})

test_that("dsem_spec, dsem_to_mplus, and dsem_code produce runnable reproducible workflows", {
  dat <- simulate_dsem(n = 60, ar = 0.5, sigma = 0.7, seed = 105)
  spec <- dsem_spec(
    outcomes = "y", lags = 1L, time = "time",
    chains = 2, iter = 160, warmup = 80, seed = 105
  )
  expect_s3_class(spec, "DSEMspec")

  fit <- dsem(spec, dat)
  expect_s3_class(fit, "DSEMfit")

  mplus_txt <- dsem_to_mplus(spec)
  expect_true(grepl("ESTIMATOR = BAYES;", mplus_txt, fixed = TRUE))
  expect_true(grepl("MODEL:", mplus_txt, fixed = TRUE))

  code_obj <- dsem_code(fit, data_name = "dat", include_outputs = FALSE)
  expect_s3_class(code_obj, "DSEMcode")
  expect_true(grepl("dsem\\(", code_obj$code))

  # Verify the generated script parses cleanly as valid R syntax
  parsed_expr <- parse(text = code_obj$code)
  expect_true(length(parsed_expr) >= 3L)
})

test_that("Mplus-parity standardization, R2, median point estimates, and categorical NA DIC work", {
  # Binary probit should report NA for DIC/pD (matching Mplus, never 0.0)
  dbin <- simulate_dsem(n = 70, intercept = 0.1, ar = 0.45, sigma = 1, family = "binary", seed = 106)
  fbin <- dsem("y ~ lag(y, 1)", dbin, family = "binary", time = "time",
               chains = 2, iter = 160, warmup = 80, seed = 106)
  expect_true(is.na(dsem_dic(fbin)[["dic"]]))
  expect_true(is.na(dsem_dic(fbin)[["pd"]]))

  # Two-level location-scale: check labeled within_std_s, between correlations in [-1,1], and dsem_r2
  set.seed(107)
  g_n <- 16; t_n <- 22
  w <- stats::rnorm(g_n)
  mu_i <- 0.35 * w + stats::rnorm(g_n, sd = 0.25)
  phi_i <- pmin(pmax(0.35 + 0.1 * w + stats::rnorm(g_n, sd = 0.05), -0.8), 0.8)
  lv_i <- 0.25 * w + stats::rnorm(g_n, sd = 0.1)
  z_i <- 0.5 * mu_i + 0.35 * lv_i + stats::rnorm(g_n, sd = 0.2)
  dat2 <- do.call(rbind, lapply(seq_len(g_n), function(g) {
    y <- numeric(t_n); sd_g <- exp(0.5 * lv_i[g]); y[1] <- stats::rnorm(1, 0, sd_g)
    for (t in 2:t_n) y[t] <- phi_i[g] * y[t - 1] + stats::rnorm(1, 0, sd_g)
    data.frame(subject = g, time = seq_len(t_n), y = mu_i[g] + y, w = w[g], z = z_i[g])
  }))
  spec2 <- dsem_spec(
    outcomes = "y", lags = 1L, predictors_between = "w", distal = "z",
    id = "subject", time = "time", random_slopes = TRUE, random_logv = TRUE,
    chains = 2, iter = 300, warmup = 150, seed = 107
  )
  fit2 <- dsem(spec2, dat2)

  # Point estimate median vs mean
  expect_equal(length(stats::coef(fit2, point = "median")), length(stats::coef(fit2, point = "mean")))
  s_med <- summary(fit2, standardized = TRUE, r2 = TRUE, point = "median")
  expect_equal(s_med$point, "median")
  expect_true(!is.null(s_med$r2))

  # Within-cluster standardized slope for labeled `s` and bounded residual correlations
  std2 <- dsem_standardize(fit2, type = "stdyx")
  expect_true("within_std_s" %in% std2$parameter)
  expect_true(all(std2$mean[grepl("^cov_", std2$parameter)] >= -1 &
                  std2$mean[grepl("^cov_", std2$parameter)] <= 1))
  expect_true(all(std2$mean[grepl("^var_", std2$parameter)] >= 0 &
                  std2$mean[grepl("^var_", std2$parameter)] <= 1))

  # Bayesian R^2 bounded in [0, 1] for both Within and Between levels
  r2_df <- dsem_r2(fit2)
  expect_true(any(grepl("Within", r2_df$level)))
  expect_true(any(r2_df$level == "Between"))
  expect_true(all(r2_df$mean >= 0 & r2_df$mean <= 1))
})
