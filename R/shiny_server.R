# Reactive server logic for the DSEMr Shiny studio
utils::globalVariables(".data")

.dsem_df_to_html <- function(df, digits = 3L, max_rows = 120L, caption = NULL) {
  if (is.null(df) || !is.data.frame(df) || !nrow(df)) {
    return(shiny::tags$p(class = "text-muted fst-italic", "No rows to display."))
  }
  show_df <- utils::head(df, max_rows)
  fmt_col <- function(v) {
    if (is.integer(v)) {
      ifelse(is.na(v), "", as.character(v))
    } else if (is.numeric(v)) {
      fin <- v[is.finite(v)]
      if (length(fin) && all(abs(fin - round(fin)) < 1e-9)) {
        ifelse(is.na(v), "", formatC(round(v), digits = 0L, format = "f"))
      } else {
        ifelse(is.na(v), "", formatC(v, digits = digits, format = "f"))
      }
    } else {
      ifelse(is.na(v), "", as.character(v))
    }
  }
  cols <- lapply(show_df, fmt_col)
  hdr <- shiny::tags$thead(
    shiny::tags$tr(lapply(names(show_df), shiny::tags$th))
  )
  body_rows <- lapply(seq_len(nrow(show_df)), function(i) {
    shiny::tags$tr(lapply(cols, function(col) shiny::tags$td(col[i])))
  })
  tbl <- shiny::tags$table(
    class = "dsem-apa-table",
    if (!is.null(caption)) shiny::tags$caption(caption),
    hdr,
    shiny::tags$tbody(body_rows)
  )
  if (nrow(df) > max_rows) {
    shiny::tagList(
      tbl,
      shiny::tags$p(class = "text-muted small mt-1",
                    sprintf("Showing first %d of %d rows.", max_rows, nrow(df)))
    )
  } else {
    tbl
  }
}

.dsem_read_uploaded_file <- function(path, filename) {
  ext <- tolower(tools::file_ext(filename))
  if (ext == "rds") {
    obj <- readRDS(path)
    if (!is.data.frame(obj)) .dsem_abort("Uploaded .rds file must contain a data.frame.")
    return(as.data.frame(obj))
  }
  if (ext %in% c("csv")) {
    return(utils::read.csv(path, stringsAsFactors = FALSE, check.names = TRUE))
  }
  if (ext %in% c("tsv", "txt")) {
    return(utils::read.delim(path, stringsAsFactors = FALSE, check.names = TRUE))
  }
  if (ext %in% c("xlsx", "xls")) {
    if (!requireNamespace("readxl", quietly = TRUE)) {
      .dsem_abort("Reading Excel files requires the `readxl` package.")
    }
    return(as.data.frame(readxl::read_excel(path)))
  }
  if (ext %in% c("sav", "por", "dta", "sas7bdat")) {
    if (!requireNamespace("haven", quietly = TRUE)) {
      .dsem_abort("Reading SPSS/Stata/SAS files requires the `haven` package.")
    }
    out <- switch(
      ext,
      sav = haven::read_sav(path),
      por = haven::read_por(path),
      dta = haven::read_dta(path),
      sas7bdat = haven::read_sas(path)
    )
    out <- as.data.frame(out)
    for (nm in names(out)) {
      if (inherits(out[[nm]], "haven_labelled")) out[[nm]] <- as.numeric(out[[nm]])
    }
    return(out)
  }
  utils::read.table(path, header = TRUE, stringsAsFactors = FALSE)
}

.dsem_build_qmd_report <- function(fit, label, code_txt) {
  est_md <- dsem_table(fit, format = "markdown")$text
  eq_tex <- dsem_equations(fit)$latex
  dc <- dsem_dic(fit)
  paste(
    "---",
    sprintf("title: \"Bayesian DSEM Report: %s\"", label),
    "format: html",
    "execute:",
    "  echo: true",
    "  warning: false",
    "---",
    "",
    "## Model Specification",
    "",
    "$$",
    eq_tex,
    "$$",
    "",
    "## Model Fit & Convergence Summary",
    "",
    sprintf("- **Hierarchy:** `%s` | **Engine:** `%s`", fit$model$level_type, fit$engine),
    sprintf("- **MCMC:** %d chains x %d iterations (%d warmup, thin = %d)",
            fit$sampler$chains, fit$sampler$iter, fit$sampler$warmup, fit$sampler$thin),
    sprintf("- **DIC:** %s | **pD:** %s",
            if (is.finite(dc[["dic"]])) sprintf("%.2f", dc[["dic"]]) else "NA",
            if (is.finite(dc[["pd"]])) sprintf("%.2f", dc[["pd"]]) else "NA"),
    sprintf("- **Max Split-Rhat:** %.4f | **Min Bulk ESS:** %.1f",
            max(fit$estimates$rhat, na.rm = TRUE),
            min(fit$estimates$ess_bulk, na.rm = TRUE)),
    "",
    "## Posterior Parameter Estimates (APA 7th Ed.)",
    "",
    est_md,
    "",
    "## Reproducible R Code",
    "",
    "```r",
    code_txt,
    "```",
    sep = "\n"
  )
}

.dsem_shiny_server <- function(input, output, session) {
  ex_meta <- .dsem_example_datasets_meta()
  presets <- .dsem_syntax_presets()

  # Session registry of fitted models for side-by-side DIC comparison
  rv <- shiny::reactiveValues(
    fits = list(),
    active_fit = NULL,
    hyp_result = NULL,
    fit_counter = 0L
  )

  # ---- 1. Data reactive & dynamic UI selectors ----
  raw_data <- shiny::reactive({
    if (isTRUE(input$use_example)) {
      key <- if (!is.null(input$example_key) && nzchar(input$example_key)) input$example_key else "ema_2l"
      return(.dsem_generate_example_data(key))
    }
    shiny::req(input$upload_file)
    tryCatch(
      .dsem_read_uploaded_file(input$upload_file$datapath, input$upload_file$name),
      error = function(e) {
        shiny::showNotification(conditionMessage(e), type = "error", duration = 7)
        NULL
      }
    )
  })

  # Auto-configure hierarchy defaults when switching curated example dataset
  shiny::observeEvent(input$example_key, {
    if (!isTRUE(input$use_example)) return()
    meta <- ex_meta[[input$example_key]]
    if (is.null(meta)) return()
    shiny::updateSelectInput(session, "level_type", selected = meta$level_type)
    shiny::updateCheckboxInput(session, "use_tinterval", value = !is.na(meta$tinterval))
    if (!is.na(meta$tinterval)) {
      shiny::updateNumericInput(session, "tinterval_delta", value = meta$tinterval)
    }
    shiny::updateCheckboxInput(session, "continuous_time", value = isTRUE(meta$continuous_time))
    shiny::updateCheckboxInput(session, "l2_ar", value = identical(input$example_key, "dsem_3l"))
    shiny::updateCheckboxInput(session, "random_logv", value = identical(input$example_key, "ema_2l"))
    shiny::updateCheckboxInput(session, "random_res_cov", value = FALSE)
    if (identical(input$example_key, "dcfa_2l")) {
      shiny::updateRadioButtons(session, "spec_mode", selected = "syntax")
      shiny::updateTextAreaInput(session, "custom_syntax", value = presets$dcfa_2l)
    } else if (identical(input$spec_mode, "syntax") && identical(input$example_key, "ema_2l")) {
      shiny::updateTextAreaInput(session, "custom_syntax", value = presets$locscale_2l)
    }
  }, ignoreInit = FALSE)

  shiny::observeEvent(input$syntax_preset, {
    if (is.null(input$syntax_preset) || !nzchar(input$syntax_preset)) return()
    txt <- presets[[input$syntax_preset]]
    if (!is.null(txt)) {
      shiny::updateTextAreaInput(session, "custom_syntax", value = txt)
    }
  })

  output$data_source_badge <- shiny::renderUI({
    df <- raw_data()
    if (is.null(df)) return(shiny::tags$p(class = "text-muted small", "Upload a dataset or choose a built-in example."))
    src <- if (isTRUE(input$use_example)) "Built-in example" else input$upload_file$name
    shiny::tags$div(
      class = "small text-muted mt-1",
      sprintf("%s: %d rows x %d columns", src, nrow(df), ncol(df))
    )
  })

  col_names <- shiny::reactive({
    df <- raw_data()
    if (is.null(df)) character() else names(df)
  })

  num_cols <- shiny::reactive({
    df <- raw_data()
    if (is.null(df)) return(character())
    names(df)[vapply(df, is.numeric, logical(1))]
  })

  active_meta <- shiny::reactive({
    if (isTRUE(input$use_example) && !is.null(input$example_key)) ex_meta[[input$example_key]] else NULL
  })

  output$id_col_ui <- shiny::renderUI({
    cols <- col_names()
    m <- active_meta()
    sel <- if (!is.null(m) && nzchar(m$id) && m$id %in% cols) m$id else cols[1L]
    lbl <- if (identical(input$level_type, "threelevel")) "Level-2 cluster ID (e.g., day)" else "Cluster / subject ID column"
    shiny::selectInput("id_col", lbl, choices = cols, selected = sel)
  })

  output$id_l3_col_ui <- shiny::renderUI({
    cols <- col_names()
    m <- active_meta()
    sel <- if (!is.null(m) && nzchar(m$id_l3) && m$id_l3 %in% cols) m$id_l3 else cols[min(2L, length(cols))]
    shiny::selectInput("id_l3_col", "Level-3 cluster ID (e.g., person)", choices = cols, selected = sel)
  })

  output$time_col_ui <- shiny::renderUI({
    cols <- col_names()
    m <- active_meta()
    sel <- if (!is.null(m) && nzchar(m$time) && m$time %in% cols) {
      m$time
    } else if ("time" %in% cols) {
      "time"
    } else {
      cols[min(2L, length(cols))]
    }
    shiny::selectInput("time_col", "Time / occasion column", choices = cols, selected = sel)
  })

  selected_ids <- shiny::reactive({
    lt <- input$level_type
    if (identical(lt, "single")) return(NULL)
    cols <- col_names()
    m <- active_meta()
    id1 <- if (!is.null(input$id_col) && nzchar(input$id_col) && input$id_col %in% cols) {
      input$id_col
    } else if (!is.null(m) && nzchar(m$id) && m$id %in% cols) {
      m$id
    } else {
      NULL
    }
    if (identical(lt, "threelevel")) {
      id3 <- if (!is.null(input$id_l3_col) && nzchar(input$id_l3_col) && input$id_l3_col %in% cols) {
        input$id_l3_col
      } else if (!is.null(m) && nzchar(m$id_l3) && m$id_l3 %in% cols) {
        m$id_l3
      } else {
        NULL
      }
      if (is.null(id1) || is.null(id3)) return(NULL)
      return(c(id1, id3))
    }
    id1
  })

  selected_time <- shiny::reactive({
    cols <- col_names()
    m <- active_meta()
    if (!is.null(input$time_col) && nzchar(input$time_col) && input$time_col %in% cols) {
      input$time_col
    } else if (!is.null(m) && nzchar(m$time) && m$time %in% cols) {
      m$time
    } else if ("time" %in% cols) {
      "time"
    } else if (length(cols)) {
      cols[1L]
    } else {
      NULL
    }
  })

  analysis_data <- shiny::reactive({
    df <- raw_data()
    if (is.null(df)) return(NULL)
    tc <- selected_time()
    if (isTRUE(input$use_tinterval) && !is.null(tc) && tc %in% names(df)) {
      delta <- input$tinterval_delta
      if (is.numeric(delta) && length(delta) == 1L && is.finite(delta) && delta > 0) {
        id_arg <- selected_ids()
        id_ti <- if (length(id_arg) >= 1L) id_arg[1L] else NULL
        aligned <- tryCatch(
          dsem_tinterval(df, time = tc, delta = delta, id = id_ti),
          error = function(e) df
        )
        return(aligned)
      }
    }
    df
  })

  output$data_kpi_strip <- shiny::renderUI({
    df <- analysis_data()
    if (is.null(df)) return(NULL)
    n_rows <- nrow(df)
    ids <- selected_ids()
    n_cl <- if (!is.null(ids) && all(ids %in% names(df))) {
      length(unique(df[[ids[length(ids)]]]))
    } else {
      1L
    }
    t_med <- if (!is.null(ids) && ids[1L] %in% names(df)) {
      stats::median(as.numeric(table(df[[ids[1L]]])))
    } else {
      n_rows
    }
    vis_cols <- setdiff(names(df), ".dsem_bin")
    miss_pct <- 100 * mean(is.na(df[, vis_cols, drop = FALSE]))
    shiny::div(
      class = "kpi-grid",
      shiny::div(class = "kpi-card", shiny::div(class = "kpi-label", "Observations"), shiny::div(class = "kpi-value", format(n_rows, big.mark = ","))),
      shiny::div(class = "kpi-card", shiny::div(class = "kpi-label", "Top-Level Clusters"), shiny::div(class = "kpi-value", format(n_cl, big.mark = ","))),
      shiny::div(class = "kpi-card", shiny::div(class = "kpi-label", "Median T per Cluster"), shiny::div(class = "kpi-value", sprintf("%.1f", t_med))),
      shiny::div(class = "kpi-card", shiny::div(class = "kpi-label", "Missing Cells"), shiny::div(class = "kpi-value", sprintf("%.1f%%", miss_pct)))
    )
  })

  output$traj_var_ui <- shiny::renderUI({
    nc <- setdiff(num_cols(), c(selected_ids(), selected_time()))
    if (!length(nc)) nc <- num_cols()
    shiny::selectInput("traj_var", "Variable to visualize", choices = nc, selected = nc[1L], width = "240px")
  })

  output$trajectory_plot <- shiny::renderPlot({
    df <- raw_data()
    shiny::req(df)
    tc <- selected_time()
    nc <- setdiff(num_cols(), c(selected_ids(), tc))
    if (!length(nc)) nc <- num_cols()
    tv <- if (!is.null(input$traj_var) && input$traj_var %in% names(df)) input$traj_var else nc[1L]
    shiny::req(tc, tv, tc %in% names(df), tv %in% names(df))
    ids <- selected_ids()
    pdat <- data.frame(
      time_v = as.numeric(df[[tc]]),
      val_v = as.numeric(df[[tv]])
    )
    if (!is.null(ids) && ids[1L] %in% names(df)) {
      pdat$grp_v <- as.factor(df[[ids[1L]]])
      keep_cl <- utils::head(unique(pdat$grp_v), 12L)
      pdat <- pdat[pdat$grp_v %in% keep_cl & is.finite(pdat$val_v) & is.finite(pdat$time_v), , drop = FALSE]
      ggplot2::ggplot(pdat, ggplot2::aes(x = .data[["time_v"]], y = .data[["val_v"]],
                                         group = .data[["grp_v"]], color = .data[["grp_v"]])) +
        ggplot2::geom_line(alpha = 0.72, linewidth = 0.75) +
        ggplot2::geom_point(size = 1.3, alpha = 0.8) +
        ggplot2::labs(x = tc, y = tv,
                      title = sprintf("Sample Cluster Trajectories (up to 12 clusters) for %s", tv)) +
        ggplot2::theme_minimal(base_size = 12) +
        ggplot2::theme(legend.position = "none")
    } else {
      pdat <- pdat[is.finite(pdat$val_v) & is.finite(pdat$time_v), , drop = FALSE]
      ggplot2::ggplot(pdat, ggplot2::aes(x = .data[["time_v"]], y = .data[["val_v"]])) +
        ggplot2::geom_line(color = "#264653", linewidth = 0.85) +
        ggplot2::geom_point(color = "#3f6f78", size = 1.3) +
        ggplot2::labs(x = tc, y = tv,
                      title = sprintf("Time Series Trajectory: %s", tv)) +
        ggplot2::theme_minimal(base_size = 12)
    }
  })

  output$var_summary_table <- shiny::renderUI({
    df <- analysis_data()
    shiny::req(df)
    vis_nms <- setdiff(names(df), ".dsem_bin")
    rows <- lapply(vis_nms, function(nm) {
      x <- df[[nm]]
      is_n <- is.numeric(x)
      data.frame(
        Variable = nm,
        Type = class(x)[1L],
        Observed = as.integer(sum(!is.na(x))),
        `Miss %` = round(100 * mean(is.na(x)), 1),
        Mean = if (is_n) mean(x, na.rm = TRUE) else NA_real_,
        SD = if (is_n) stats::sd(x, na.rm = TRUE) else NA_real_,
        stringsAsFactors = FALSE, check.names = FALSE
      )
    })
    .dsem_df_to_html(do.call(rbind, rows), digits = 2L)
  })

  output$tinterval_diag_ui <- shiny::renderUI({
    df <- analysis_data()
    if (is.null(df) || !isTRUE(input$use_tinterval)) return(NULL)
    info <- attr(df, "tinterval_info")
    if (is.null(info)) return(NULL)
    shiny::tags$div(
      class = "alert alert-info py-2 small mb-2",
      sprintf("TINTERVAL aligned (delta = %g): %d observed rows expanded to %d grid rows (%d inserted missing grid cells).",
              info$delta, info$observed_rows, info$total_grid_rows, info$inserted_missing_rows)
    )
  })

  output$data_preview_table <- shiny::renderUI({
    df <- analysis_data()
    if (is.null(df)) return(NULL)
    vis_nms <- setdiff(names(df), ".dsem_bin")
    .dsem_df_to_html(df[, vis_nms, drop = FALSE], digits = 3L, max_rows = 80L)
  })

  # ---- 2. Model Builder UI selectors & live compilation ----
  model_vars <- shiny::reactive({
    setdiff(num_cols(), c(selected_ids(), selected_time()))
  })

  output$outcomes_ui <- shiny::renderUI({
    vars <- model_vars()
    m <- active_meta()
    sel <- if (!is.null(m) && length(intersect(m$outcomes, vars))) intersect(m$outcomes, vars) else vars[1L]
    shiny::selectizeInput("outcomes", "Dynamic outcome variable(s)", choices = vars, selected = sel, multiple = TRUE)
  })

  output$exog_ui <- shiny::renderUI({
    vars <- setdiff(model_vars(), input$outcomes)
    m <- active_meta()
    sel <- if (!is.null(m)) intersect(m$exog, vars) else character()
    shiny::selectizeInput("exog_vars", "Within-level exogenous covariate(s)", choices = vars, selected = sel, multiple = TRUE)
  })

  output$between_preds_ui <- shiny::renderUI({
    vars <- setdiff(model_vars(), c(input$outcomes, input$exog_vars))
    m <- active_meta()
    sel <- if (!is.null(m)) intersect(m$between_preds, vars) else character()
    shiny::selectizeInput("between_preds", "Between-level / L3 predictor(s)", choices = vars, selected = sel, multiple = TRUE)
  })

  output$distal_ui <- shiny::renderUI({
    vars <- setdiff(model_vars(), c(input$outcomes, input$exog_vars, input$between_preds))
    m <- active_meta()
    sel <- if (!is.null(m)) intersect(m$distal, vars) else character()
    shiny::selectizeInput("distal_vars", "Between-level distal outcome(s)", choices = vars, selected = sel, multiple = TRUE)
  })

  parsed_lags <- shiny::reactive({
    raw <- trimws(if (is.null(input$lags_str)) "1" else input$lags_str)
    toks <- suppressWarnings(as.integer(unlist(strsplit(raw, "[,\\s]+"))))
    toks <- sort(unique(toks[!is.na(toks) & toks >= 1L]))
    if (!length(toks)) 1L else toks
  })

  guided_spec <- shiny::reactive({
    vars <- model_vars()
    m <- active_meta()
    outs <- if (length(input$outcomes)) {
      input$outcomes
    } else if (!is.null(m) && length(intersect(m$outcomes, vars))) {
      intersect(m$outcomes, vars)
    } else {
      vars[1L]
    }
    shiny::req(length(outs) >= 1L)
    lt <- input$level_type
    ids <- selected_ids()
    tc <- selected_time()
    ti <- if (isTRUE(input$use_tinterval) && !isTRUE(input$continuous_time)) input$tinterval_delta else NULL
    cat_v <- if (!identical(input$outcome_family, "gaussian")) outs else character()
    ct_on <- isTRUE(input$continuous_time)
    dsem_spec(
      outcomes = outs,
      predictors_within = if (length(input$exog_vars)) input$exog_vars else character(),
      predictors_between = if (!identical(lt, "single") && length(input$between_preds)) input$between_preds else character(),
      distal = if (identical(lt, "twolevel") && length(input$distal_vars)) input$distal_vars else character(),
      lags = if (ct_on) 1L else parsed_lags(),
      level_type = lt,
      id = ids,
      time = tc,
      time_cluster = if (identical(lt, "crossclassified")) tc else NULL,
      random_slopes = !identical(lt, "single") && !ct_on && isTRUE(input$random_slopes),
      random_logv = !identical(lt, "single") && !ct_on && isTRUE(input$random_logv),
      random_residual_cov = !identical(lt, "single") && !ct_on && length(outs) >= 2L && isTRUE(input$random_res_cov),
      l2_ar = identical(lt, "threelevel") && !ct_on && isTRUE(input$l2_ar),
      residual_dynamics = !ct_on && isTRUE(input$residual_dynamics),
      continuous_time = ct_on,
      centering = if (!identical(lt, "single") && !is.null(input$centering_mode)) input$centering_mode else "latent",
      tinterval = ti,
      categorical = cat_v,
      prior_mode = if (!is.null(input$prior_mode)) input$prior_mode else "default"
    )
  })

  shiny::observeEvent(input$sync_to_syntax, {
    sp <- tryCatch(guided_spec(), error = function(e) NULL)
    if (is.null(sp)) return()
    shiny::updateTextAreaInput(session, "custom_syntax", value = dsem_to_lavaan(sp))
    shiny::updateSelectInput(session, "syntax_flavor", selected = "lavaan")
    shiny::updateCheckboxInput(session, "syntax_ct", value = isTRUE(sp$continuous_time))
    shiny::updateSelectInput(session, "syntax_prior_mode", selected = sp$prior_mode)
    shiny::updateRadioButtons(session, "spec_mode", selected = "syntax")
  })

  compiled_model <- shiny::reactive({
    tryCatch({
      if (identical(input$spec_mode, "guided")) {
        sp <- guided_spec()
        list(ok = TRUE, model = as_dsem_model(sp), spec = sp, error = NULL)
      } else {
        txt <- input$custom_syntax
        if (is.null(txt) || !nzchar(trimws(txt))) {
          return(list(ok = FALSE, model = NULL, spec = NULL, error = "Enter model syntax."))
        }
        flv <- if (!is.null(input$syntax_flavor)) input$syntax_flavor else "auto"
        ids <- selected_ids()
        tc <- selected_time()
        ti <- if (isTRUE(input$use_tinterval) && !isTRUE(input$syntax_ct)) input$tinterval_delta else NULL
        m <- dsem_model(
          txt,
          syntax = flv,
          id = ids,
          time = tc,
          level_type = if (identical(flv, "mplus")) "auto" else input$level_type,
          tinterval = ti,
          continuous_time = isTRUE(input$syntax_ct)
        )
        list(ok = TRUE, model = m, spec = NULL, error = NULL)
      }
    }, error = function(e) {
      list(ok = FALSE, model = NULL, spec = NULL, error = conditionMessage(e))
    })
  })

  output$readiness_ui <- shiny::renderUI({
    df <- analysis_data()
    cm <- compiled_model()
    items <- list()
    add_item <- function(status, msg) {
      cls <- switch(status, pass = "badge-pass", warn = "badge-warn", fail = "badge-fail")
      lbl <- toupper(status)
      items[[length(items) + 1L]] <<- shiny::div(
        class = "readiness-item",
        shiny::span(class = cls, lbl),
        shiny::span(msg)
      )
    }
    if (is.null(df) || !nrow(df)) {
      add_item("fail", "No dataset loaded.")
    } else {
      add_item("pass", sprintf("Dataset loaded (%d rows, %d columns).", nrow(df), ncol(df)))
    }
    if (!cm$ok) {
      add_item("fail", paste("Model compilation error:", cm$error))
      return(shiny::tagList(items))
    }
    m <- cm$model
    add_item("pass", sprintf("Compiled canonical DSEM (%s hierarchy%s, %d model terms).",
                             m$level_type,
                             if (isTRUE(m$continuous_time)) ", continuous-time" else "",
                             nrow(m$terms)))
    est_chk <- tryCatch({
      .dsem_assert_estimable(m)
      TRUE
    }, error = function(e) conditionMessage(e))
    if (isTRUE(est_chk)) {
      add_item("pass", "Estimator routing and identification checks passed.")
    } else {
      add_item("fail", est_chk)
    }
    if (!is.null(df) && !is.null(m$time) && !m$time %in% names(df)) {
      add_item("fail", sprintf("Time column `%s` not found in dataset.", m$time))
    }
    if (!is.null(df) && length(m$id) && !all(m$id %in% names(df))) {
      miss_id <- setdiff(m$id, names(df))
      add_item("fail", sprintf("Cluster ID column(s) missing from data: %s.", paste(miss_id, collapse = ", ")))
    }
    if (isTRUE(m$continuous_time) && isTRUE(input$use_tinterval)) {
      add_item("warn", "TINTERVAL is bypassed for Continuous-Time DSEM, which models exact irregular intervals directly.")
    }
    shiny::tagList(items)
  })

  output$builder_svg_diagram <- shiny::renderUI({
    cm <- compiled_model()
    if (!cm$ok) return(shiny::tags$p(class = "text-muted fst-italic", cm$error))
    svg_txt <- tryCatch(dsem_diagram(cm$model, format = "svg")$svg, error = function(e) NULL)
    if (is.null(svg_txt) || !nzchar(svg_txt)) return(shiny::tags$p(class = "text-muted", "Diagram unavailable for this model."))
    shiny::HTML(svg_txt)
  })

  builder_eq_string <- shiny::reactive({
    cm <- compiled_model()
    if (!cm$ok) return("")
    tryCatch(dsem_equations(cm$model)$latex, error = function(e) "")
  })

  output$builder_eq_tex <- shiny::renderText({
    builder_eq_string()
  })

  output$builder_equations_math <- shiny::renderUI({
    eq <- builder_eq_string()
    if (!nzchar(eq)) return(shiny::tags$p(class = "text-muted fst-italic", "Compile a valid model to preview equations."))
    # Normalize LaTeX helpers for MathJax 2 rendering while preserving pure LaTeX in builder_eq_tex
    eq_mj <- gsub("\\\\boldsymbol\\{([^}]+)\\}", "\\\\mathbf{\\1}", eq)
    eq_mj <- gsub("\\\\mathcal\\{N\\}", "\\\\mathrm{N}", eq_mj)
    eq_mj <- gsub("\\\\text\\{([^}]*)\\\\_([^}]*)\\}", "\\\\text{\\1_\\2}", eq_mj)
    eq_mj <- gsub("\\\\!", "", eq_mj, fixed = TRUE)
    wrapped <- if (grepl("\\\\begin\\{align", eq_mj)) eq_mj else sprintf("$$%s$$", eq_mj)
    shiny::withMathJax(shiny::HTML(wrapped))
  })

  output$preview_lavaan_txt <- shiny::renderText({
    cm <- compiled_model()
    if (!cm$ok) return(cm$error)
    if (!is.null(cm$spec)) dsem_to_lavaan(cm$spec) else cm$model$raw
  })

  output$preview_script_txt <- shiny::renderText({
    cm <- compiled_model()
    if (!cm$ok) return(cm$error)
    if (!is.null(cm$spec)) {
      dsem_to_mplus(cm$spec)
    } else {
      paste("# Custom syntax entered directly in Advanced Syntax Editor:", cm$model$raw, sep = "\n")
    }
  })

  # ---- 3. Estimate & Diagnostics ----
  active_prior_mode <- shiny::reactive({
    if (identical(input$spec_mode, "guided")) {
      if (!is.null(input$prior_mode)) input$prior_mode else "default"
    } else {
      if (!is.null(input$syntax_prior_mode)) input$syntax_prior_mode else "default"
    }
  })

  shiny::observeEvent(input$fit_btn, {
    cm <- compiled_model()
    df <- raw_data()
    if (is.null(df) || !cm$ok) {
      shiny::showNotification("Fix dataset or model specification errors before fitting.", type = "error")
      return()
    }
    m <- cm$model
    ch <- max(1L, as.integer(input$mcmc_chains))
    it <- max(100L, as.integer(input$mcmc_iter))
    wu <- min(it - 20L, max(20L, as.integer(input$mcmc_warmup)))
    th <- max(1L, as.integer(input$mcmc_thin))
    sd_val <- as.integer(input$mcmc_seed)
    pm <- active_prior_mode()
    lbl <- trimws(if (!is.null(input$model_label) && nzchar(trimws(input$model_label))) input$model_label else sprintf("Model %d", rv$fit_counter + 1L))

    fit_obj <- NULL
    shiny::withProgress(message = sprintf("Sampling Bayesian DSEM (%s)...", lbl), value = 0.25, {
      fit_obj <- tryCatch({
        dsem(
          m, df,
          chains = ch, iter = it, warmup = wu, thin = th, seed = sd_val,
          control = list(prior_mode = pm)
        )
      }, error = function(e) {
        shiny::showNotification(paste("Estimation failed:", conditionMessage(e)), type = "error", duration = 10)
        NULL
      })
      shiny::incProgress(0.75)
    })

    if (!is.null(fit_obj)) {
      rv$fit_counter <- rv$fit_counter + 1L
      rv$fits[[lbl]] <- fit_obj
      rv$active_fit <- lbl
      rv$hyp_result <- fit_obj$wald_test
      shiny::updateTextInput(session, "model_label", value = sprintf("Model %d", rv$fit_counter + 1L))
      shiny::showNotification(sprintf("Completed `%s` in %.2fs.", lbl, fit_obj$elapsed), type = "message")
    }
  })

  shiny::observeEvent(input$clear_fits_btn, {
    rv$fits <- list()
    rv$active_fit <- NULL
    rv$hyp_result <- NULL
    rv$fit_counter <- 0L
  })

  output$saved_model_selector_ui <- shiny::renderUI({
    nms <- names(rv$fits)
    if (!length(nms)) return(shiny::tags$p(class = "text-muted small", "No fitted models yet. Click 'Fit Bayesian DSEM'."))
    shiny::selectInput("selected_fit_name", "Inspect model", choices = nms, selected = rv$active_fit)
  })

  shiny::observeEvent(input$selected_fit_name, {
    if (!is.null(input$selected_fit_name) && input$selected_fit_name %in% names(rv$fits)) {
      rv$active_fit <- input$selected_fit_name
    }
  })

  current_fit <- shiny::reactive({
    if (is.null(rv$active_fit) || !rv$active_fit %in% names(rv$fits)) return(NULL)
    rv$fits[[rv$active_fit]]
  })

  output$fit_kpi_strip <- shiny::renderUI({
    fit <- current_fit()
    if (is.null(fit)) {
      return(shiny::div(class = "alert alert-secondary", "Configure your model and click 'Fit Bayesian DSEM' to estimate parameters and inspect MCMC diagnostics."))
    }
    max_rh <- suppressWarnings(max(fit$estimates$rhat, na.rm = TRUE))
    min_es <- suppressWarnings(min(fit$estimates$ess_bulk, na.rm = TRUE))
    dc <- dsem_dic(fit)
    rh_cls <- if (!is.finite(max_rh)) "" else if (max_rh <= 1.05) "kpi-ok" else if (max_rh <= 1.10) "kpi-warn" else "kpi-bad"
    es_cls <- if (!is.finite(min_es)) "" else if (min_es >= 100) "kpi-ok" else if (min_es >= 40) "kpi-warn" else "kpi-bad"
    shiny::div(
      class = "kpi-grid",
      shiny::div(class = paste("kpi-card", rh_cls), shiny::div(class = "kpi-label", "Max Split-Rhat"),
                 shiny::div(class = "kpi-value", if (is.finite(max_rh)) sprintf("%.3f", max_rh) else "NA")),
      shiny::div(class = paste("kpi-card", es_cls), shiny::div(class = "kpi-label", "Min Bulk ESS"),
                 shiny::div(class = "kpi-value", if (is.finite(min_es)) sprintf("%.0f", min_es) else "NA")),
      shiny::div(class = "kpi-card", shiny::div(class = "kpi-label", "DIC"),
                 shiny::div(class = "kpi-value", if (is.finite(dc[["dic"]])) sprintf("%.1f", dc[["dic"]]) else "NA")),
      shiny::div(class = "kpi-card", shiny::div(class = "kpi-label", "Effective Params (pD)"),
                 shiny::div(class = "kpi-value", if (is.finite(dc[["pd"]])) sprintf("%.1f", dc[["pd"]]) else "NA")),
      shiny::div(class = "kpi-card", shiny::div(class = "kpi-label", "Elapsed (Rust Engine)"),
                 shiny::div(class = "kpi-value", sprintf("%.2fs", fit$elapsed)))
    )
  })

  output$diag_params_ui <- shiny::renderUI({
    fit <- current_fit()
    if (is.null(fit)) return(NULL)
    p_all <- dimnames(fit$draws)[[2L]]
    sel <- utils::head(p_all, 6L)
    shiny::selectizeInput(
      "diag_params", "Parameters to plot (up to 9 recommended)",
      choices = p_all, selected = sel, multiple = TRUE, width = "100%"
    )
  })

  diag_selected_params <- shiny::reactive({
    fit <- current_fit()
    shiny::req(fit)
    p_all <- dimnames(fit$draws)[[2L]]
    sel <- intersect(input$diag_params, p_all)
    if (!length(sel)) utils::head(p_all, 6L) else sel
  })

  diag_draws_long <- shiny::reactive({
    fit <- current_fit()
    pars <- diag_selected_params()
    dr <- fit$draws[, pars, , drop = FALSE]
    n_it <- dim(dr)[1L]; n_p <- dim(dr)[2L]; n_ch <- dim(dr)[3L]
    data.frame(
      iteration = rep(seq_len(n_it), times = n_p * n_ch),
      parameter = factor(rep(rep(pars, each = n_it), times = n_ch), levels = pars),
      chain = factor(rep(paste("Chain", seq_len(n_ch)), each = n_it * n_p)),
      value = as.numeric(dr)
    )
  })

  output$diag_trace_plot <- shiny::renderPlot({
    df <- diag_draws_long(); shiny::req(nrow(df) > 0L)
    ggplot2::ggplot(df, ggplot2::aes(x = .data[["iteration"]], y = .data[["value"]], color = .data[["chain"]])) +
      ggplot2::geom_line(alpha = 0.72, linewidth = 0.45) +
      ggplot2::facet_wrap(~parameter, scales = "free_y") +
      ggplot2::labs(x = "Post-Warmup Draw", y = "Value", color = NULL, title = "MCMC Trace Plots") +
      ggplot2::theme_minimal(base_size = 11) +
      ggplot2::theme(legend.position = "bottom")
  })

  output$diag_density_plot <- shiny::renderPlot({
    df <- diag_draws_long(); shiny::req(nrow(df) > 0L)
    ggplot2::ggplot(df, ggplot2::aes(x = .data[["value"]], color = .data[["chain"]], fill = .data[["chain"]])) +
      ggplot2::geom_density(alpha = 0.22, linewidth = 0.65) +
      ggplot2::facet_wrap(~parameter, scales = "free") +
      ggplot2::labs(x = "Parameter Value", y = "Posterior Density", color = NULL, fill = NULL,
                    title = "Posterior Marginal Densities by Chain") +
      ggplot2::theme_minimal(base_size = 11) +
      ggplot2::theme(legend.position = "bottom")
  })

  output$diag_acf_plot <- shiny::renderPlot({
    fit <- current_fit()
    pars <- diag_selected_params()
    dr <- fit$draws[, pars, , drop = FALSE]
    n_it <- dim(dr)[1L]; n_ch <- dim(dr)[3L]
    max_lag <- min(20L, max(1L, n_it - 2L))
    rows <- lapply(pars, function(p) {
      ac_mat <- vapply(seq_len(n_ch), function(ch) {
        v <- dr[, p, ch]
        if (stats::sd(v) < 1e-12) return(rep(0, max_lag))
        as.numeric(stats::acf(v, lag.max = max_lag, plot = FALSE)$acf)[-1L]
      }, numeric(max_lag))
      ac_mean <- if (is.matrix(ac_mat)) rowMeans(ac_mat, na.rm = TRUE) else as.numeric(ac_mat)
      data.frame(parameter = factor(p, levels = pars), lag = seq_len(max_lag), acf = ac_mean)
    })
    df <- do.call(rbind, rows)
    ci_b <- 1.96 / sqrt(max(1, n_it * n_ch))
    ggplot2::ggplot(df, ggplot2::aes(x = .data[["lag"]], y = .data[["acf"]])) +
      ggplot2::geom_hline(yintercept = 0, color = "#5c6d7a") +
      ggplot2::geom_hline(yintercept = c(-ci_b, ci_b), linetype = "dashed", color = "#7a4e48") +
      ggplot2::geom_col(fill = "#3f6f78", width = 0.65) +
      ggplot2::facet_wrap(~parameter) +
      ggplot2::labs(x = "Lag", y = "Autocorrelation", title = "Chain-Averaged Autocorrelation (ACF)") +
      ggplot2::theme_minimal(base_size = 11)
  })

  output$diag_interval_plot <- shiny::renderPlot({
    fit <- current_fit(); shiny::req(fit)
    est <- fit$estimates
    est$parameter <- factor(est$parameter, levels = rev(est$parameter))
    ggplot2::ggplot(est, ggplot2::aes(x = .data[["mean"]], y = .data[["parameter"]])) +
      ggplot2::geom_vline(xintercept = 0, linetype = "dashed", color = "#7a4e48") +
      ggplot2::geom_errorbar(ggplot2::aes(xmin = .data[["q2.5"]], xmax = .data[["q97.5"]]),
                             width = 0.25, color = "#3f6f78", orientation = "y") +
      ggplot2::geom_point(size = 2.4, color = "#264653") +
      ggplot2::labs(x = "Posterior Mean & 95% Credible Interval", y = NULL,
                    title = "Posterior Parameter Estimates & 95% CrI") +
      ggplot2::theme_minimal(base_size = 11)
  })

  output$diag_full_table <- shiny::renderUI({
    fit <- current_fit()
    if (is.null(fit)) return(shiny::tags$p(class = "text-muted fst-italic", "Fit a model to view convergence diagnostics."))
    .dsem_df_to_html(fit$estimates, digits = 3L, max_rows = 150L)
  })

  output$diag_vif_table <- shiny::renderUI({
    fit <- current_fit()
    if (is.null(fit)) return(shiny::tags$p(class = "text-muted fst-italic", "Fit a model to view VIF diagnostics."))
    vt <- tryCatch(dsem_vif(fit), error = function(e) NULL)
    if (is.null(vt) || !nrow(vt)) {
      return(shiny::tags$p(class = "text-muted small", "VIF requires >= 2 predictors in an equation."))
    }
    .dsem_df_to_html(vt, digits = 3L)
  })

  # ---- 4. Results & Analytics ----
  output$apa_html_table <- shiny::renderUI({
    fit <- current_fit()
    if (is.null(fit)) return(shiny::tags$p(class = "text-muted fst-italic", "Fit a model first."))
    dig <- if (is.numeric(input$table_digits)) as.integer(input$table_digits) else 3L
    pt <- if (!is.null(input$point_est)) input$point_est else "mean"
    st <- if (!is.null(input$std_type)) input$std_type else "unstd"
    if (identical(st, "unstd")) {
      html_str <- tryCatch(
        dsem_table(fit, format = "html", digits = dig, point = pt,
                   caption = sprintf("Bayesian DSEM Posterior Estimates (%s)", rv$active_fit))$text,
        error = function(e) NULL
      )
      if (!is.null(html_str) && nzchar(html_str)) return(shiny::HTML(html_str))
      return(.dsem_df_to_html(fit$estimates, digits = dig))
    }
    std_df <- tryCatch(dsem_standardize(fit, type = st), error = function(e) NULL)
    if (is.null(std_df) || !nrow(std_df)) {
      return(shiny::tags$p(class = "text-muted", "Standardized estimates are unavailable for this model."))
    }
    .dsem_df_to_html(std_df, digits = dig, caption = sprintf("Standardized Estimates (%s)", toupper(st)))
  })

  output$r2_table_ui <- shiny::renderUI({
    fit <- current_fit()
    if (is.null(fit)) return(shiny::tags$p(class = "text-muted fst-italic", "Fit a model first."))
    r2_df <- tryCatch(dsem_r2(fit), error = function(e) NULL)
    if (is.null(r2_df) || !nrow(r2_df)) {
      return(shiny::tags$p(class = "text-muted small", "Bayesian R-squared not defined for this model type."))
    }
    disp <- data.frame(
      Level = r2_df$level,
      Variable = r2_df$variable,
      `R^2` = r2_df$mean,
      SD = r2_df$sd,
      `95% CrI` = sprintf("[%.3f, %.3f]", r2_df$q_lower, r2_df$q_upper),
      stringsAsFactors = FALSE, check.names = FALSE
    )
    .dsem_df_to_html(disp, digits = 3L)
  })

  output$cluster_std_table <- shiny::renderUI({
    fit <- current_fit()
    if (is.null(fit)) return(NULL)
    cs <- tryCatch(dsem_standardize(fit, type = "stdyx", cluster = TRUE), error = function(e) NULL)
    if (is.null(cs) || !nrow(cs)) {
      return(shiny::tags$p(class = "text-muted small", "Cluster-specific standardization applies to multilevel models."))
    }
    .dsem_df_to_html(cs, digits = 3L, max_rows = 40L)
  })

  pcor_df_react <- shiny::reactive({
    fit <- current_fit()
    if (is.null(fit)) return(NULL)
    tryCatch(dsem_partial_cor(fit), error = function(e) NULL)
  })

  output$pcor_table_ui <- shiny::renderUI({
    pc <- pcor_df_react()
    if (is.null(pc) || !nrow(pc)) {
      return(shiny::tags$p(class = "text-muted small", "Partial correlations require at least one covariance block with >= 2 correlated variables or random effects."))
    }
    .dsem_df_to_html(pc, digits = 3L)
  })

  output$pcor_plot <- shiny::renderPlot({
    pc <- pcor_df_react()
    shiny::req(pc, nrow(pc) >= 1L)
    pc$pair <- paste0(pc$var1, " - ", pc$var2, " (", pc$level, ")")
    ggplot2::ggplot(pc, ggplot2::aes(x = .data[["pcor_mean"]], y = .data[["pair"]])) +
      ggplot2::geom_vline(xintercept = 0, linetype = "dashed", color = "#7a4e48") +
      ggplot2::geom_errorbar(ggplot2::aes(xmin = .data[["pcor_q_lower"]], xmax = .data[["pcor_q_upper"]]),
                             width = 0.2, color = "#3f6f78", orientation = "y") +
      ggplot2::geom_point(size = 2.8, color = "#264653") +
      ggplot2::labs(x = "Posterior Partial Correlation (95% CrI)", y = NULL,
                    title = "Posterior Partial Correlation Network") +
      ggplot2::theme_minimal(base_size = 12)
  })

  output$avail_param_names <- shiny::renderText({
    fit <- current_fit()
    if (is.null(fit)) return("Fit a model to inspect parameter names.")
    paste(fit$estimates$parameter, collapse = ", ")
  })

  shiny::observeEvent(input$eval_hyp_btn, {
    fit <- current_fit()
    shiny::req(fit)
    lines <- trimws(unlist(strsplit(input$hyp_exprs, "\n", fixed = TRUE)))
    lines <- lines[nzchar(lines)]
    if (!length(lines)) {
      shiny::showNotification("Enter at least one parameter expression.", type = "warning")
      return()
    }
    # Support optional `name = expr` vs `lhs = rhs` contrast syntax
    expr_vec <- character()
    for (ln in lines) {
      if (grepl("^[A-Za-z_.][A-Za-z0-9_.]*\\s*=\\s*[^=]", ln) &&
          !(sub("\\s*=.*$", "", ln) %in% fit$estimates$parameter)) {
        eqp <- regexpr("=", ln, fixed = TRUE)
        nm <- trimws(substr(ln, 1L, eqp - 1L))
        rhs <- trimws(substr(ln, eqp + 1L, nchar(ln)))
        expr_vec <- c(expr_vec, stats::setNames(rhs, nm))
      } else {
        expr_vec <- c(expr_vec, ln)
      }
    }
    rv$hyp_result <- tryCatch(
      dsem_hypothesis(fit, hypotheses = expr_vec, joint_test = TRUE),
      error = function(e) {
        shiny::showNotification(conditionMessage(e), type = "error", duration = 7)
        NULL
      }
    )
  })

  output$hyp_wald_ui <- shiny::renderUI({
    hr <- rv$hyp_result
    if (is.null(hr) || is.null(hr$wald)) return(NULL)
    shiny::tags$div(
      class = "alert alert-info py-2 small mb-2",
      sprintf("Joint Multivariate Bayesian Wald Test: chi-sq(%d) = %.3f, p = %.4f",
              hr$wald$df, hr$wald$statistic, hr$wald$p_value)
    )
  })

  output$hyp_table_ui <- shiny::renderUI({
    hr <- rv$hyp_result
    if (is.null(hr)) {
      return(shiny::tags$p(class = "text-muted small", "Enter expressions on the left and click 'Evaluate Posterior Hypotheses'."))
    }
    .dsem_df_to_html(hr$summary, digits = 3L)
  })

  irf_react <- shiny::reactive({
    fit <- current_fit()
    if (is.null(fit)) return(NULL)
    h <- if (is.numeric(input$irf_horizon)) as.integer(input$irf_horizon) else 10L
    tryCatch(dsem_irf(fit, horizon = h, orthogonal = isTRUE(input$irf_orth)), error = function(e) NULL)
  })

  output$irf_note_ui <- shiny::renderUI({
    fit <- current_fit()
    if (is.null(fit)) return(NULL)
    if (isTRUE(fit$model$continuous_time)) {
      shiny::tags$div(class = "small text-muted mt-3", "Continuous-Time IRF computed via exact matrix exponential exp(A * h).")
    } else {
      shiny::tags$div(class = "small text-muted mt-3", "Discrete-Time IRF computed from posterior companion matrix draws.")
    }
  })

  output$irf_plot <- shiny::renderPlot({
    irf <- irf_react()
    shiny::req(irf, is.data.frame(irf$data), nrow(irf$data) >= 1L)
    d <- irf$data
    d$panel <- paste0(d$impulse, " -> ", d$response)
    ggplot2::ggplot(d, ggplot2::aes(x = .data[["horizon"]], y = .data[["q50"]])) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "#7a4e48") +
      ggplot2::geom_ribbon(ggplot2::aes(ymin = .data[["q_lower"]], ymax = .data[["q_upper"]]),
                           fill = "#6f9fa6", alpha = 0.28) +
      ggplot2::geom_line(color = "#264653", linewidth = 0.9) +
      ggplot2::facet_wrap(~panel, scales = "free_y") +
      ggplot2::labs(x = "Horizon", y = "Dynamic Response", title = "Bayesian Impulse Response Function (95% CrI)") +
      ggplot2::theme_minimal(base_size = 12)
  })

  output$irf_table_ui <- shiny::renderUI({
    irf <- irf_react()
    if (is.null(irf)) return(shiny::tags$p(class = "text-muted small", "IRF unavailable for this model."))
    .dsem_df_to_html(irf$data, digits = 3L, max_rows = 100L)
  })

  output$fscomp_controls_ui <- shiny::renderUI({
    fit <- current_fit()
    if (is.null(fit) || is.null(fit$random_effects_draws)) {
      return(shiny::tags$p(class = "text-muted small", "Cluster rankings require a multilevel or cross-classified model with random effects."))
    }
    p_nms <- dimnames(fit$random_effects_draws)[[3L]]
    shiny::selectInput("fscomp_param", "Random effect to rank", choices = p_nms, selected = p_nms[1L], width = "280px")
  })

  fscomp_react <- shiny::reactive({
    fit <- current_fit()
    if (is.null(fit) || is.null(fit$random_effects_draws) || is.null(input$fscomp_param)) return(NULL)
    tryCatch(dsem_fscomparison(fit, parameter = input$fscomp_param), error = function(e) NULL)
  })

  output$fscomp_plot <- shiny::renderPlot({
    fs <- fscomp_react()
    shiny::req(fs, is.data.frame(fs$ranking), nrow(fs$ranking) >= 1L)
    rk <- fs$ranking
    rk$rank_ord <- seq_len(nrow(rk))
    ggplot2::ggplot(rk, ggplot2::aes(x = .data[["rank_ord"]], y = .data[["mean"]])) +
      ggplot2::geom_errorbar(ggplot2::aes(ymin = .data[["q_lower"]], ymax = .data[["q_upper"]]),
                             width = 0.25, color = "#6f9fa6") +
      ggplot2::geom_point(size = 2.0, color = "#264653") +
      ggplot2::labs(x = "Cluster Rank (Lowest to Highest)", y = paste("Posterior", fs$parameter),
                    title = sprintf("Cluster Random-Effect Caterpillar Plot (%s)", fs$parameter)) +
      ggplot2::theme_minimal(base_size = 12)
  })

  output$fscomp_table_ui <- shiny::renderUI({
    fs <- fscomp_react()
    if (is.null(fs)) return(shiny::tags$p(class = "text-muted small", "No cluster random effects available."))
    .dsem_df_to_html(fs$ranking, digits = 3L, max_rows = 100L)
  })

  output$compare_table_ui <- shiny::renderUI({
    if (!length(rv$fits)) {
      return(shiny::tags$p(class = "text-muted fst-italic", "Fit one or more models in 'Estimate & Diagnostics' to compare DIC and pD."))
    }
    if (length(rv$fits) == 1L) {
      f1 <- rv$fits[[1L]]
      dc <- dsem_dic(f1)
      one_df <- data.frame(
        model = names(rv$fits)[1L],
        dic = dc[["dic"]], pd = dc[["pd"]], d_bar = dc[["d_bar"]],
        delta_dic = 0, weight = 1, stringsAsFactors = FALSE
      )
      return(.dsem_df_to_html(one_df, digits = 2L))
    }
    cmp_warn <- NULL
    cmp <- tryCatch(
      withCallingHandlers(
        do.call(dsem_compare, c(unname(rv$fits), list(names = names(rv$fits)))),
        warning = function(w) {
          cmp_warn <<- conditionMessage(w)
          invokeRestart("muffleWarning")
        }
      ),
      error = function(e) NULL
    )
    if (is.null(cmp) || !is.data.frame(cmp)) {
      rows <- lapply(names(rv$fits), function(nm) {
        dc <- dsem_dic(rv$fits[[nm]])
        data.frame(model = nm, dic = dc[["dic"]], pd = dc[["pd"]], d_bar = dc[["d_bar"]], stringsAsFactors = FALSE)
      })
      return(.dsem_df_to_html(do.call(rbind, rows), digits = 2L))
    }
    shiny::tagList(
      if (!is.null(cmp_warn)) shiny::tags$div(class = "alert alert-warning py-2 small mb-2", cmp_warn),
      .dsem_df_to_html(cmp, digits = 2L)
    )
  })

  # ---- 5. Report & Reproducible Code ----
  repro_code_text <- shiny::reactive({
    fit <- current_fit()
    if (!is.null(fit)) {
      return(tryCatch(dsem_code(fit, data_name = "my_data")$code, error = function(e) ""))
    }
    cm <- compiled_model()
    if (cm$ok && !is.null(cm$spec)) {
      return(tryCatch(dsem_code(cm$spec, data_name = "my_data")$code, error = function(e) ""))
    }
    if (cm$ok && !is.null(cm$model)) {
      return(tryCatch(dsem_code(cm$model, data_name = "my_data")$code, error = function(e) ""))
    }
    "# Compile or fit a DSEM model to generate reproducible R code."
  })

  output$export_r_code <- shiny::renderText(repro_code_text())

  output$export_md_table <- shiny::renderText({
    fit <- current_fit()
    if (is.null(fit)) return("Fit a model to generate an APA 7th Ed. Markdown table.")
    pt <- if (!is.null(input$point_est)) input$point_est else "mean"
    dsem_table(fit, format = "markdown", digits = 3L, point = pt)$text
  })

  output$export_tex_table <- shiny::renderText({
    fit <- current_fit()
    if (is.null(fit)) return("Fit a model to generate an APA 7th Ed. LaTeX table.")
    pt <- if (!is.null(input$point_est)) input$point_est else "mean"
    dsem_table(fit, format = "latex", digits = 3L, point = pt)$text
  })

  output$export_fitted_svg <- shiny::renderUI({
    fit <- current_fit()
    target <- if (!is.null(fit)) fit else if (compiled_model()$ok) compiled_model()$model else NULL
    if (is.null(target)) return(shiny::tags$p(class = "text-muted", "No compiled model."))
    svg <- tryCatch(dsem_diagram(target, format = "svg")$svg, error = function(e) NULL)
    if (is.null(svg) || !nzchar(svg)) return(NULL)
    shiny::HTML(svg)
  })

  output$export_mermaid <- shiny::renderText({
    fit <- current_fit()
    target <- if (!is.null(fit)) fit else if (compiled_model()$ok) compiled_model()$model else NULL
    if (is.null(target)) return("")
    tryCatch(dsem_diagram(target, format = "mermaid")$mermaid, error = function(e) "")
  })

  output$dl_r_code <- shiny::downloadHandler(
    filename = function() "dsemr_analysis.R",
    content = function(file) writeLines(repro_code_text(), file)
  )

  output$dl_qmd_report <- shiny::downloadHandler(
    filename = function() "dsemr_report.qmd",
    content = function(file) {
      fit <- current_fit()
      if (is.null(fit)) {
        writeLines("# Fit a model first before exporting a Quarto report.", file)
      } else {
        writeLines(.dsem_build_qmd_report(fit, rv$active_fit, repro_code_text()), file)
      }
    }
  )

  output$dl_svg_diagram <- shiny::downloadHandler(
    filename = function() "dsemr_path_diagram.svg",
    content = function(file) {
      target <- if (!is.null(current_fit())) current_fit() else compiled_model()$model
      svg <- if (!is.null(target)) dsem_diagram(target, format = "svg")$svg else "<svg/>"
      writeLines(svg, file)
    }
  )

  output$dl_csv_est <- shiny::downloadHandler(
    filename = function() "dsemr_estimates.csv",
    content = function(file) {
      fit <- current_fit()
      if (!is.null(fit)) utils::write.csv(fit$estimates, file, row.names = FALSE)
      else utils::write.csv(data.frame(), file, row.names = FALSE)
    }
  )

  output$dl_zip_bundle <- shiny::downloadHandler(
    filename = function() "dsemr_reproducibility_bundle.zip",
    content = function(file) {
      tmp <- tempfile("dsemr_bundle_")
      dir.create(tmp, recursive = TRUE)
      on.exit(unlink(tmp, recursive = TRUE), add = TRUE)
      writeLines(repro_code_text(), file.path(tmp, "analysis.R"))
      fit <- current_fit()
      if (!is.null(fit)) {
        writeLines(.dsem_build_qmd_report(fit, rv$active_fit, repro_code_text()), file.path(tmp, "report.qmd"))
        writeLines(dsem_table(fit, format = "markdown")$text, file.path(tmp, "table_apa.md"))
        writeLines(dsem_table(fit, format = "latex")$text, file.path(tmp, "table_apa.tex"))
        writeLines(dsem_equations(fit)$latex, file.path(tmp, "equations.tex"))
        writeLines(dsem_diagram(fit, format = "svg")$svg, file.path(tmp, "path_diagram.svg"))
        utils::write.csv(fit$estimates, file.path(tmp, "estimates.csv"), row.names = FALSE)
      }
      old_wd <- setwd(tmp); on.exit(setwd(old_wd), add = TRUE)
      utils::zip(file, files = list.files(tmp))
    }
  )
}
