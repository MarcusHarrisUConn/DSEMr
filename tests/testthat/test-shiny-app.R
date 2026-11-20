test_that("DSEMr Shiny app constructs cleanly and runs full guided workflow via testServer", {
  skip_if_not_installed("shiny")
  skip_if_not_installed("bslib")
  skip_if_not_installed("ggplot2")

  app <- DSEMr:::dsemr_shiny_app()
  expect_s3_class(app, "shiny.appobj")

  # Verify all curated example datasets generate cleanly
  for (k in names(DSEMr:::.dsem_example_datasets_meta())) {
    d <- DSEMr:::.dsem_generate_example_data(k)
    expect_true(is.data.frame(d) && nrow(d) > 20L)
  }

  # Full reactive testServer run: data -> guided 2L model -> fit -> analytics -> code export
  shiny::testServer(DSEMr:::.dsem_shiny_server, {
    session$setInputs(
      use_example = TRUE,
      example_key = "ema_2l",
      level_type = "twolevel",
      id_col = "subject",
      time_col = "time",
      use_tinterval = TRUE,
      tinterval_delta = 1,
      spec_mode = "guided",
      outcomes = "y",
      exog_vars = "x",
      between_preds = "w",
      distal_vars = character(),
      lags_str = "1",
      outcome_family = "gaussian",
      continuous_time = FALSE,
      residual_dynamics = FALSE,
      random_slopes = TRUE,
      random_logv = TRUE,
      random_res_cov = FALSE,
      l2_ar = FALSE,
      centering_mode = "latent",
      prior_mode = "default",
      model_label = "EMA_2L",
      mcmc_chains = 2L,
      mcmc_iter = 160L,
      mcmc_warmup = 80L,
      mcmc_thin = 1L,
      mcmc_seed = 42L,
      point_est = "mean",
      std_type = "unstd",
      table_digits = 3L,
      show_cluster_std = FALSE,
      irf_horizon = 5L,
      irf_orth = TRUE
    )

    df <- analysis_data()
    expect_true(is.data.frame(df) && nrow(df) > 100L)

    cm <- compiled_model()
    expect_true(cm$ok)
    expect_s3_class(cm$model, "DSEMmodel")
    expect_true(nzchar(builder_eq_string()))
    expect_true(grepl("<svg", as.character(output$builder_svg_diagram$html)))

    # Fit first model
    session$setInputs(fit_btn = 1L)
    fit1 <- current_fit()
    expect_s3_class(fit1, "DSEMfit")
    expect_true("mean_logv" %in% fit1$estimates$parameter)
    expect_true(nzchar(output$export_r_code))
    expect_true(nzchar(output$export_md_table))
    expect_true(nzchar(output$export_tex_table))
    expect_true(nzchar(output$export_mermaid))

    # Evaluate posterior hypothesis inside server
    session$setInputs(hyp_exprs = "persist = mean_s^2", eval_hyp_btn = 1L)
    expect_false(is.null(rv$hyp_result))
    expect_true("persist" %in% rv$hyp_result$summary$name)

    # Fit a second Continuous-Time model to exercise CT-DSEM + DIC comparison
    session$setInputs(
      example_key = "ct_ema",
      level_type = "twolevel",
      id_col = "subject",
      time_col = "time",
      use_tinterval = FALSE,
      outcomes = "y",
      exog_vars = character(),
      between_preds = character(),
      continuous_time = TRUE,
      model_label = "CT_EMA",
      fit_btn = 2L
    )
    expect_equal(length(rv$fits), 2L)
    fit2 <- current_fit()
    expect_true(isTRUE(fit2$model$continuous_time))
    irf <- irf_react()
    expect_s3_class(irf, "DSEMirf")
  })
})
