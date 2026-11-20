# Built-in curated DSEM example datasets & model templates for the Shiny studio

.dsem_app_www_dir <- function() {
  local_candidates <- c(
    file.path(getwd(), "inst", "app", "www"),
    file.path(getwd(), "www")
  )
  existing_local <- local_candidates[dir.exists(local_candidates)]
  if (length(existing_local)) {
    return(normalizePath(existing_local[[1L]], winslash = "/", mustWork = TRUE))
  }
  pkg_www <- system.file("app", "www", package = "DSEMr")
  if (nzchar(pkg_www) && dir.exists(pkg_www)) return(pkg_www)
  ""
}

.dsem_copy_btn <- function(target_id, label = "Copy") {
  shiny::tags$button(
    type = "button",
    class = "btn btn-outline-secondary copy-button",
    onclick = sprintf("dsemCopyText('%s', this)", target_id),
    label
  )
}

.dsem_step_hdr <- function(num, title) {
  shiny::div(
    class = "builder-step-title",
    shiny::span(class = "step-num", as.character(num)),
    shiny::span(title)
  )
}

.dsem_example_datasets_meta <- function() {
  list(
    ema_2l = list(
      label = "Two-Level Location-Scale EMA (y, x, w, z, irregular time)",
      level_type = "twolevel",
      id = "subject",
      id_l3 = "",
      time = "time",
      outcomes = "y",
      exog = "x",
      between_preds = "w",
      distal = "z",
      tinterval = 1
    ),
    var_2l = list(
      label = "Two-Level Bivariate Cross-Lagged VAR (y1, y2, w)",
      level_type = "twolevel",
      id = "subject",
      id_l3 = "",
      time = "time",
      outcomes = c("y1", "y2"),
      exog = character(),
      between_preds = "w",
      distal = character(),
      tinterval = NA_real_
    ),
    dsem_3l = list(
      label = "Three-Level Intensive Longitudinal (obs in day in person)",
      level_type = "threelevel",
      id = "day",
      id_l3 = "person",
      time = "obs",
      outcomes = "y",
      exog = character(),
      between_preds = "w3",
      distal = character(),
      tinterval = NA_real_
    ),
    cc_2way = list(
      label = "Cross-Classified Subject x Occasion (id, time, y, x)",
      level_type = "crossclassified",
      id = "id",
      id_l3 = "",
      time = "time",
      outcomes = "y",
      exog = "x",
      between_preds = character(),
      distal = character(),
      tinterval = NA_real_
    ),
    ct_ema = list(
      label = "Continuous-Time Irregular EMA (subject, continuous time, y)",
      level_type = "twolevel",
      id = "subject",
      id_l3 = "",
      time = "time",
      outcomes = "y",
      exog = character(),
      between_preds = character(),
      distal = character(),
      tinterval = NA_real_,
      continuous_time = TRUE
    ),
    dcfa_2l = list(
      label = "Two-Level Dynamic CFA (4 continuous indicators y1-y4)",
      level_type = "twolevel",
      id = "subject",
      id_l3 = "",
      time = "time",
      outcomes = c("y1", "y2", "y3", "y4"),
      exog = character(),
      between_preds = character(),
      distal = character(),
      tinterval = NA_real_
    ),
    n1_bivar = list(
      label = "Single-Level (N=1) Bivariate VAR(1) (y1, y2, x, time)",
      level_type = "single",
      id = "",
      id_l3 = "",
      time = "time",
      outcomes = c("y1", "y2"),
      exog = "x",
      between_preds = character(),
      distal = character(),
      tinterval = NA_real_
    )
  )
}

.dsem_generate_example_data <- function(key) {
  restore_rng <- .dsem_rng_scope(20260901L)
  on.exit(restore_rng(), add = TRUE)

  if (identical(key, "ema_2l")) {
    G <- 35L; TT <- 30L
    rows <- vector("list", G)
    for (g in seq_len(G)) {
      w <- stats::rnorm(1)
      mu_y <- 0.35 * w + stats::rnorm(1, 0, 0.45)
      s_y <- min(max(0.32 + 0.10 * w + stats::rnorm(1, 0, 0.08), -0.8), 0.8)
      lv_y <- -0.15 + 0.25 * w + stats::rnorm(1, 0, 0.20)
      z <- 0.2 + 0.40 * mu_y + 0.35 * lv_y + stats::rnorm(1, 0, 0.35)
      x <- stats::rnorm(TT)
      y <- numeric(TT)
      y[1L] <- mu_y + stats::rnorm(1, 0, exp(0.5 * lv_y))
      for (t in 2:TT) {
        y[t] <- mu_y + s_y * (y[t - 1L] - mu_y) + 0.25 * x[t] + stats::rnorm(1, 0, exp(0.5 * lv_y))
      }
      # Drop ~12% observations to showcase TINTERVAL alignment
      keep <- sort(unique(c(1L, sample.int(TT, size = round(0.88 * TT)))))
      rows[[g]] <- data.frame(
        subject = g, time = as.numeric(keep),
        y = round(y[keep], 4), x = round(x[keep], 4),
        w = round(w, 4), z = round(z, 4)
      )
    }
    return(do.call(rbind, rows))
  }

  if (identical(key, "var_2l")) {
    G <- 30L; TT <- 35L
    rows <- vector("list", G)
    for (g in seq_len(G)) {
      w <- stats::rnorm(1)
      mu1 <- 0.3 * w + stats::rnorm(1, 0, 0.5)
      mu2 <- -0.2 * w + stats::rnorm(1, 0, 0.5)
      s11 <- min(max(0.35 + stats::rnorm(1, 0, 0.08), -0.7), 0.7)
      s22 <- min(max(0.30 + stats::rnorm(1, 0, 0.08), -0.7), 0.7)
      y1 <- numeric(TT); y2 <- numeric(TT)
      y1[1L] <- mu1 + stats::rnorm(1); y2[1L] <- mu2 + stats::rnorm(1)
      for (t in 2:TT) {
        sh <- stats::rnorm(1, 0, 0.45)
        y1[t] <- mu1 + s11 * (y1[t - 1L] - mu1) + 0.15 * (y2[t - 1L] - mu2) + sh + stats::rnorm(1, 0, 0.65)
        y2[t] <- mu2 + 0.20 * (y1[t - 1L] - mu1) + s22 * (y2[t - 1L] - mu2) + sh + stats::rnorm(1, 0, 0.65)
      }
      rows[[g]] <- data.frame(subject = g, time = seq_len(TT),
                              y1 = round(y1, 4), y2 = round(y2, 4), w = round(w, 4))
    }
    return(do.call(rbind, rows))
  }

  if (identical(key, "dsem_3l")) {
    L3 <- 14L; D_per <- 8L; T_per <- 6L
    rows <- vector("list", L3 * D_per); idx <- 1L
    for (k in seq_len(L3)) {
      w3 <- stats::rnorm(1)
      u3 <- 0.4 * w3 + stats::rnorm(1, 0, 0.45)
      u2 <- numeric(D_per); u2[1L] <- stats::rnorm(1, 0, 0.35)
      for (d_i in 2:D_per) u2[d_i] <- 0.40 * u2[d_i - 1L] + stats::rnorm(1, 0, 0.35)
      for (d_i in seq_len(D_per)) {
        mu_kd <- u3 + u2[d_i]
        y <- numeric(T_per); y[1L] <- mu_kd + stats::rnorm(1, 0, 0.6)
        for (t in 2:T_per) y[t] <- mu_kd + 0.30 * (y[t - 1L] - mu_kd) + stats::rnorm(1, 0, 0.6)
        rows[[idx]] <- data.frame(person = k, day = d_i, obs = seq_len(T_per),
                                  y = round(y, 4), w3 = round(w3, 4))
        idx <- idx + 1L
      }
    }
    return(do.call(rbind, rows))
  }

  if (identical(key, "cc_2way")) {
    N_id <- 22L; N_t <- 22L
    u_id <- stats::rnorm(N_id, 0, 0.5); u_t <- stats::rnorm(N_t, 0, 0.35)
    rows <- vector("list", N_id)
    for (i in seq_len(N_id)) {
      x <- stats::rnorm(N_t); y <- numeric(N_t)
      y[1L] <- u_id[i] + u_t[1L] + 0.35 * x[1L] + stats::rnorm(1, 0, 0.7)
      for (t in 2:N_t) {
        y[t] <- u_id[i] + u_t[t] + 0.30 * (y[t - 1L] - u_id[i] - u_t[t - 1L]) +
          0.35 * x[t] + stats::rnorm(1, 0, 0.7)
      }
      rows[[i]] <- data.frame(id = i, time = seq_len(N_t),
                              y = round(y, 4), x = round(x, 4))
    }
    return(do.call(rbind, rows))
  }

  if (identical(key, "ct_ema")) {
    G <- 24L; Tg <- 30L
    rows <- vector("list", G)
    for (g in seq_len(G)) {
      dt <- stats::runif(Tg - 1L, 0.25, 2.2)
      tm <- c(0, cumsum(dt))
      mu_g <- stats::rnorm(1, 0.5, 0.55)
      y <- numeric(Tg); y[1L] <- mu_g + stats::rnorm(1, 0, 0.8)
      for (t in 2:Tg) {
        ph <- exp(-0.60 * dt[t - 1L])
        v <- 0.75 * (1 - ph^2) / (2 * 0.60)
        y[t] <- mu_g + ph * (y[t - 1L] - mu_g) + stats::rnorm(1, 0, sqrt(v))
      }
      rows[[g]] <- data.frame(subject = g, time = round(tm, 4), y = round(y, 4))
    }
    return(do.call(rbind, rows))
  }

  if (identical(key, "dcfa_2l")) {
    G <- 25L; TT <- 25L
    lam <- c(1.0, 0.95, 1.05, 0.90)
    rows <- vector("list", G)
    for (g in seq_len(G)) {
      fb <- stats::rnorm(1, 0, 0.7)
      sg <- min(max(0.35 + stats::rnorm(1, 0, 0.08), -0.7), 0.7)
      eta <- numeric(TT); eta[1L] <- stats::rnorm(1)
      for (t in 2:TT) eta[t] <- sg * eta[t - 1L] + stats::rnorm(1)
      ym <- matrix(0, TT, 4L)
      for (j in 1:4) ym[, j] <- 0.6 * fb + lam[j] * eta + stats::rnorm(TT, 0, 0.7)
      rows[[g]] <- data.frame(subject = g, time = seq_len(TT),
                              y1 = round(ym[, 1L], 4), y2 = round(ym[, 2L], 4),
                              y3 = round(ym[, 3L], 4), y4 = round(ym[, 4L], 4))
    }
    return(do.call(rbind, rows))
  }

  # Default: n1_bivar
  TT <- 120L
  x <- stats::rnorm(TT); y1 <- numeric(TT); y2 <- numeric(TT)
  for (t in 2:TT) {
    e1 <- stats::rnorm(1, 0, 0.75)
    y1[t] <- 0.2 + 0.45 * y1[t - 1L] + 0.18 * y2[t - 1L] + 0.30 * x[t] + e1
    y2[t] <- -0.1 + 0.22 * y1[t - 1L] + 0.38 * y2[t - 1L] + 0.40 * e1 + stats::rnorm(1, 0, 0.6)
  }
  data.frame(time = seq_len(TT), y1 = round(y1, 4), y2 = round(y2, 4), x = round(x, 4))
}

.dsem_syntax_presets <- function() {
  list(
    ar1_n1 = "y ~ lag(y, 1)",
    var1_n1 = "y1 ~ lag(y1, 1) + lag(y2, 1) + x\ny2 ~ lag(y1, 1) + lag(y2, 1)",
    locscale_2l = paste(
      "level: 1",
      "  s | y ~ lag(y, 1) + x",
      "  logv | y ~~ y",
      "level: 2",
      "  y ~ w",
      "  s ~ w",
      "  logv ~ w",
      "  y ~~ s + logv",
      "  s ~~ logv",
      "  z ~ y + s + logv",
      sep = "\n"
    ),
    rand_res_cov = paste(
      "level: 1",
      "  s11 | y1 ~ lag(y1, 1) + lag(y2, 1)",
      "  s22 | y2 ~ lag(y2, 1) + lag(y1, 1)",
      "  logv1 | y1 ~~ y1",
      "  logv2 | y2 ~~ y2",
      "  logvf | y1 ~~ y2",
      "level: 2",
      "  y1 ~~ y2 + s11 + s22 + logv1 + logv2 + logvf",
      sep = "\n"
    ),
    arma11_2l = paste(
      "VARIABLE: CLUSTER = subject; LAGGED = y(1);",
      "ANALYSIS: TYPE = TWOLEVEL RANDOM;",
      "MODEL:",
      "  %WITHIN%",
      "  e BY y@1 (&1);",
      "  y@0.01;",
      "  s | y ON y&1;",
      "  y ON e&1;",
      "  %BETWEEN%",
      "  y WITH s;",
      sep = "\n"
    ),
    dcfa_2l = paste(
      "VARIABLE: CLUSTER = subject;",
      "ANALYSIS: TYPE = TWOLEVEL RANDOM;",
      "MODEL:",
      "  %WITHIN%",
      "  f BY y1-y4 (&1);",
      "  s | f ON f&1;",
      "  %BETWEEN%",
      "  fb BY y1-y4*;",
      "  fb@1;",
      "  fb WITH s;",
      sep = "\n"
    ),
    rdsem_2l = paste(
      "level: 1",
      "  y ~ x",
      "  s | y ~ residual(lag(y, 1))",
      "level: 2",
      "  y ~~ s",
      sep = "\n"
    ),
    dsem_3l = paste(
      "VARIABLE: CLUSTER = person day; LAGGED = y(1);",
      "ANALYSIS: TYPE = THREELEVEL RANDOM;",
      "MODEL:",
      "  %WITHIN%",
      "  s | y ON y&1;",
      "  %BETWEEN day%",
      "  y ON y&1;",
      "  %BETWEEN person%",
      "  y ON w3;",
      "  y WITH s;",
      sep = "\n"
    ),
    cc_2way = paste(
      "VARIABLE: CLUSTER = id time; LAGGED = y(1);",
      "ANALYSIS: TYPE = CROSSCLASSIFIED RANDOM;",
      "MODEL:",
      "  %WITHIN%",
      "  s | y ON y&1;",
      "  sx | y ON x;",
      "  %BETWEEN id%",
      "  y s sx WITH y s sx;",
      "  %BETWEEN time%",
      "  y WITH sx;",
      sep = "\n"
    )
  )
}

.dsem_shiny_ui <- function() {
  www_dir <- .dsem_app_www_dir()
  if (nzchar(www_dir)) {
    shiny::addResourcePath("dsemr-assets", www_dir)
  }

  theme <- bslib::bs_theme(
    version = 5,
    bootswatch = "flatly",
    primary = "#264653",
    secondary = "#6e7f8d",
    base_font = bslib::font_collection("Source Sans Pro", "Segoe UI", "Arial", "sans-serif"),
    heading_font = bslib::font_collection("Source Serif Pro", "Georgia", "Times New Roman", "serif")
  )

  ex_meta <- .dsem_example_datasets_meta()
  ex_choices <- stats::setNames(names(ex_meta), vapply(ex_meta, `[[`, character(1), "label"))

  bslib::page_navbar(
    title = "DSEMr Studio",
    id = "main_nav",
    theme = theme,
    header = shiny::tags$head(
      if (nzchar(www_dir)) shiny::tags$link(rel = "stylesheet", type = "text/css", href = "dsemr-assets/style.css"),
      shiny::withMathJax(),
      shiny::tags$script(shiny::HTML("
        function dsemCopyText(id, btn) {
          const el = document.getElementById(id);
          if (!el) return;
          const text = el.innerText || el.textContent || '';
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(function() {
              if (btn) {
                const prev = btn.innerText;
                btn.innerText = 'Copied!';
                setTimeout(function() { btn.innerText = prev; }, 1400);
              }
            });
          }
        }
      "))
    ),
    bslib::nav_spacer(),
    bslib::nav_item(shiny::div(class = "theme-toggle", bslib::input_dark_mode(id = "theme_mode", mode = "light"))),

    # =========================================================================
    # TAB 1: DATA & DESIGN
    # =========================================================================
    bslib::nav_panel(
      "Data & Design",
      bslib::layout_sidebar(
        sidebar = bslib::sidebar(
          width = 340,
          shiny::div(
            class = "builder-step",
            .dsem_step_hdr(1, "Data Source"),
            shiny::checkboxInput("use_example", "Use curated example dataset", TRUE),
            shiny::conditionalPanel(
              "input.use_example == true",
              shiny::selectInput("example_key", "Example DSEM dataset",
                                 choices = ex_choices, selected = "ema_2l")
            ),
            shiny::conditionalPanel(
              "input.use_example == false",
              shiny::fileInput(
                "upload_file",
                "Upload CSV, TSV, RDS, Excel, or SPSS/Stata",
                accept = c(".csv", ".tsv", ".txt", ".rds", ".xlsx", ".xls", ".sav", ".dta", ".sas7bdat")
              )
            ),
            shiny::uiOutput("data_source_badge")
          ),
          shiny::div(
            class = "builder-step",
            .dsem_step_hdr(2, "Hierarchy & Time Structure"),
            shiny::selectInput(
              "level_type", "Hierarchical design",
              choices = c(
                "Two-Level (occasions in persons)" = "twolevel",
                "Single-Level (N = 1 time series)" = "single",
                "Three-Level (occasions in L2 in L3)" = "threelevel",
                "Cross-Classified (subject x occasion)" = "crossclassified"
              ),
              selected = "twolevel"
            ),
            shiny::conditionalPanel(
              "input.level_type != 'single'",
              shiny::uiOutput("id_col_ui")
            ),
            shiny::conditionalPanel(
              "input.level_type == 'threelevel'",
              shiny::uiOutput("id_l3_col_ui")
            ),
            shiny::uiOutput("time_col_ui"),
            shiny::checkboxInput("use_tinterval", "Apply TINTERVAL grid alignment", FALSE),
            shiny::conditionalPanel(
              "input.use_tinterval == true",
              shiny::numericInput("tinterval_delta", "Grid interval width (delta)", value = 1, min = 0.01, step = 0.25)
            )
          )
        ),
        shiny::uiOutput("data_kpi_strip"),
        bslib::layout_columns(
          col_widths = c(7, 5),
          bslib::card(
            bslib::card_header("Intensive Longitudinal Trajectories"),
            shiny::uiOutput("traj_var_ui"),
            shiny::plotOutput("trajectory_plot", height = "310px")
          ),
          bslib::card(
            bslib::card_header("Variable Summary & Missingness"),
            shiny::div(class = "dsem-table-wrap", shiny::uiOutput("var_summary_table"))
          )
        ),
        bslib::card(
          bslib::card_header("Dataset Preview (First 80 Rows) & TINTERVAL Grid Diagnostics"),
          shiny::uiOutput("tinterval_diag_ui"),
          shiny::div(class = "dsem-table-wrap", shiny::uiOutput("data_preview_table"))
        )
      )
    ),

    # =========================================================================
    # TAB 2: MODEL BUILDER
    # =========================================================================
    bslib::nav_panel(
      "Model Builder",
      bslib::layout_sidebar(
        sidebar = bslib::sidebar(
          width = 365,
          shiny::div(
            class = "builder-step",
            .dsem_step_hdr(1, "Specification Mode"),
            shiny::radioButtons(
              "spec_mode", NULL,
              choices = c("Guided Builder (dsem_spec)" = "guided",
                          "Advanced Syntax Editor" = "syntax"),
              selected = "guided", inline = TRUE
            )
          ),
          shiny::conditionalPanel(
            "input.spec_mode == 'guided'",
            shiny::div(
              class = "builder-step",
              .dsem_step_hdr(2, "Within-Level Dynamics"),
              shiny::uiOutput("outcomes_ui"),
              shiny::uiOutput("exog_ui"),
              shiny::textInput("lags_str", "Active lags (comma-separated)", value = "1"),
              shiny::selectInput(
                "outcome_family", "Outcome distribution",
                choices = c("Gaussian (Continuous)" = "gaussian",
                            "Binary Probit (0/1)" = "binary",
                            "Ordinal Probit (K >= 3)" = "ordinal"),
                selected = "gaussian"
              ),
              shiny::checkboxInput("continuous_time", "Continuous-Time DSEM (Ornstein-Uhlenbeck)", FALSE),
              shiny::conditionalPanel(
                "input.continuous_time == false",
                shiny::checkboxInput("residual_dynamics", "Residual DSEM (RDSEM)", FALSE)
              ),
              shiny::conditionalPanel(
                "input.level_type != 'single' && input.continuous_time == false",
                shiny::checkboxInput("random_slopes", "Random AR / cross-lag slopes", TRUE),
                shiny::checkboxInput("random_logv", "Random log innovation variance (logv)", FALSE),
                shiny::checkboxInput("random_res_cov", "Random residual covariance (logvf, >=2 outcomes)", FALSE)
              ),
              shiny::conditionalPanel(
                "input.level_type == 'threelevel' && input.continuous_time == false",
                shiny::checkboxInput("l2_ar", "Level-2 autoregressive dynamics (L2 AR(1))", FALSE)
              )
            ),
            shiny::div(
              class = "builder-step",
              .dsem_step_hdr(3, "Between-Level Structure & Priors"),
              shiny::conditionalPanel(
                "input.level_type != 'single'",
                shiny::uiOutput("between_preds_ui"),
                shiny::uiOutput("distal_ui"),
                shiny::selectInput("centering_mode", "Within centering",
                                   choices = c("Latent mean centering" = "latent",
                                               "Group-mean centering" = "groupmean"),
                                   selected = "latent")
              ),
              shiny::selectInput(
                "prior_mode", "Prior family",
                choices = c(
                  "Default (Weakly Informative)" = "default",
                  "Diffuse (Flat / IG(-1, 0))" = "diffuse",
                  "Penalized Bayesian Lasso (PSEM)" = "penalized"
                ),
                selected = "default"
              ),
              shiny::actionButton("sync_to_syntax", "Copy Guided Model to Syntax Editor",
                                  class = "btn-outline-secondary btn-sm w-100 mt-1")
            )
          ),
          shiny::conditionalPanel(
            "input.spec_mode == 'syntax'",
            shiny::div(
              class = "builder-step",
              .dsem_step_hdr(2, "Syntax Editor & Presets"),
              shiny::selectInput(
                "syntax_preset", "Load preset template",
                choices = c(
                  "-- Choose template --" = "",
                  "Two-Level Location-Scale + logv + distal z" = "locscale_2l",
                  "Two-Level Bivariate + Random Residual Covariance (logvf)" = "rand_res_cov",
                  "Two-Level ARMA(1,1)" = "arma11_2l",
                  "Two-Level Dynamic CFA (4 indicators)" = "dcfa_2l",
                  "Two-Level RDSEM" = "rdsem_2l",
                  "Three-Level DSEM3 + Level-2 AR(1)" = "dsem_3l",
                  "Cross-Classified Subject x Occasion DSEM" = "cc_2way",
                  "Single-Level Bivariate VAR(1)" = "var1_n1",
                  "Single-Level AR(1)" = "ar1_n1"
                ),
                selected = ""
              ),
              shiny::selectInput("syntax_flavor", "Syntax parser",
                                 choices = c("Auto-detect" = "auto",
                                             "Extended lavaan" = "lavaan",
                                             "Section-based SEM script" = "mplus"),
                                 selected = "auto"),
              shiny::textAreaInput(
                "custom_syntax", "Model specification",
                value = .dsem_syntax_presets()$locscale_2l,
                rows = 12, width = "100%"
              ),
              shiny::checkboxInput("syntax_ct", "Estimate in Continuous Time (CT-DSEM)", FALSE),
              shiny::selectInput(
                "syntax_prior_mode", "Prior family",
                choices = c(
                  "Default (Weakly Informative)" = "default",
                  "Diffuse (Flat / IG(-1, 0))" = "diffuse",
                  "Penalized Bayesian Lasso (PSEM)" = "penalized"
                ),
                selected = "default"
              )
            )
          )
        ),
        bslib::layout_columns(
          col_widths = c(5, 7),
          bslib::card(
            bslib::card_header("Pre-Fit Readiness Checklist"),
            shiny::uiOutput("readiness_ui")
          ),
          bslib::card(
            bslib::card_header("Live Vector Path Diagram (SVG)"),
            shiny::div(class = "svg-diagram-box", shiny::uiOutput("builder_svg_diagram"))
          )
        ),
        bslib::layout_columns(
          col_widths = c(6, 6),
          bslib::card(
            bslib::card_header(
              shiny::div(class = "d-flex justify-content-between align-items-center",
                         shiny::span("LaTeX Mathematical Equations"),
                         .dsem_copy_btn("builder_eq_tex", "Copy LaTeX"))
            ),
            shiny::div(class = "dsem-equation-box", shiny::uiOutput("builder_equations_math")),
            shiny::tags$details(
              class = "mt-2",
              shiny::tags$summary("Raw LaTeX source"),
              shiny::verbatimTextOutput("builder_eq_tex")
            )
          ),
          bslib::card(
            bslib::card_header("Compiled Canonical Syntax"),
            bslib::navset_card_tab(
              bslib::nav_panel("Extended lavaan", shiny::verbatimTextOutput("preview_lavaan_txt")),
              bslib::nav_panel("Section-Based SEM", shiny::verbatimTextOutput("preview_script_txt"))
            )
          )
        )
      )
    ),

    # =========================================================================
    # TAB 3: ESTIMATE & DIAGNOSTICS
    # =========================================================================
    bslib::nav_panel(
      "Estimate & Diagnostics",
      bslib::layout_sidebar(
        sidebar = bslib::sidebar(
          width = 330,
          shiny::div(
            class = "builder-step",
            .dsem_step_hdr(1, "MCMC Sampler Settings"),
            shiny::textInput("model_label", "Model label (for comparison)", value = "Model 1"),
            shiny::numericInput("mcmc_chains", "MCMC chains", value = 2L, min = 1L, max = 8L, step = 1L),
            shiny::numericInput("mcmc_iter", "Total iterations per chain", value = 600L, min = 100L, max = 20000L, step = 100L),
            shiny::numericInput("mcmc_warmup", "Warmup (burn-in) iterations", value = 300L, min = 50L, max = 10000L, step = 50L),
            shiny::numericInput("mcmc_thin", "Thinning interval", value = 1L, min = 1L, max = 20L, step = 1L),
            shiny::numericInput("mcmc_seed", "Deterministic master seed", value = 20260813L, min = 1L, step = 1L),
            shiny::selectInput("point_est", "Posterior point summary",
                               choices = c("Posterior Mean" = "mean", "Posterior Median" = "median"),
                               selected = "mean"),
            shiny::actionButton("fit_btn", "Fit Bayesian DSEM", class = "btn-primary w-100 mt-2"),
            shiny::actionButton("clear_fits_btn", "Clear Saved Models", class = "btn-outline-secondary btn-sm w-100 mt-2")
          ),
          shiny::div(
            class = "builder-step",
            .dsem_step_hdr(2, "Active Saved Model"),
            shiny::uiOutput("saved_model_selector_ui")
          )
        ),
        shiny::uiOutput("fit_kpi_strip"),
        bslib::card(
          bslib::card_header("MCMC Diagnostic Visualizations"),
          bslib::navset_card_tab(
            bslib::nav_panel("Trace Plots", shiny::plotOutput("diag_trace_plot", height = "340px")),
            bslib::nav_panel("Posterior Densities", shiny::plotOutput("diag_density_plot", height = "340px")),
            bslib::nav_panel("Autocorrelation (ACF)", shiny::plotOutput("diag_acf_plot", height = "340px")),
            bslib::nav_panel("Credible Intervals", shiny::plotOutput("diag_interval_plot", height = "340px"))
          )
        ),
        bslib::layout_columns(
          col_widths = c(8, 4),
          bslib::card(
            bslib::card_header("Full Posterior & Convergence Diagnostics"),
            shiny::div(class = "dsem-table-wrap", shiny::uiOutput("diag_full_table"))
          ),
          bslib::card(
            bslib::card_header("Covariate Collinearity (VIF)"),
            shiny::div(class = "dsem-table-wrap", shiny::uiOutput("diag_vif_table"))
          )
        )
      )
    ),

    # =========================================================================
    # TAB 4: RESULTS & ANALYTICS
    # =========================================================================
    bslib::nav_panel(
      "Results & Analytics",
      bslib::navset_card_underline(
        bslib::nav_panel(
          "APA Tables & Standardization",
          bslib::layout_columns(
            col_widths = c(4, 4, 4),
            shiny::selectInput(
              "std_type", "Standardization type",
              choices = c("Unstandardized" = "unstd",
                          "STDYX (Full Standardized)" = "stdyx",
                          "STDY (Outcome Standardized)" = "stdy",
                          "STD (Latent Standardized)" = "std"),
              selected = "unstd"
            ),
            shiny::numericInput("table_digits", "Decimal digits", value = 3L, min = 2L, max = 5L),
            shiny::checkboxInput("show_cluster_std", "Include cluster-specific STDYX preview", FALSE)
          ),
          bslib::layout_columns(
            col_widths = c(8, 4),
            bslib::card(
              bslib::card_header("Publication-Ready Parameter Table (APA 7th Ed.)"),
              shiny::div(class = "dsem-table-wrap", shiny::uiOutput("apa_html_table"))
            ),
            bslib::card(
              bslib::card_header("Within & Between Bayesian R-Squared"),
              shiny::div(class = "dsem-table-wrap", shiny::uiOutput("r2_table_ui"))
            )
          ),
          shiny::conditionalPanel(
            "input.show_cluster_std == true",
            bslib::card(
              bslib::card_header("Cluster-Specific Within Standardization (First 40 Rows)"),
              shiny::div(class = "dsem-table-wrap", shiny::uiOutput("cluster_std_table"))
            )
          )
        ),
        bslib::nav_panel(
          "Partial Correlations",
          bslib::layout_columns(
            col_widths = c(6, 6),
            bslib::card(
              bslib::card_header("Posterior Zero-Order vs. Partial Correlations"),
              shiny::div(class = "dsem-table-wrap", shiny::uiOutput("pcor_table_ui"))
            ),
            bslib::card(
              bslib::card_header("Partial Correlation Forest Plot"),
              shiny::plotOutput("pcor_plot", height = "340px")
            )
          )
        ),
        bslib::nav_panel(
          "Hypothesis & Wald Tests",
          bslib::layout_columns(
            col_widths = c(5, 7),
            bslib::card(
              bslib::card_header("Derived Parameter & Contrast Builder"),
              shiny::helpText("Enter one expression per line (e.g., `persist = lag1_y^2` or `diff = mean_s12 - mean_s21`). Available parameters are listed below."),
              shiny::textAreaInput("hyp_exprs", "Expressions", value = "", rows = 5, width = "100%"),
              shiny::actionButton("eval_hyp_btn", "Evaluate Posterior Hypotheses", class = "btn-primary btn-sm"),
              shiny::tags$hr(),
              shiny::tags$strong("Available parameter names:"),
              shiny::verbatimTextOutput("avail_param_names")
            ),
            bslib::card(
              bslib::card_header("Posterior Contrast Summaries & Joint Bayesian Wald Test"),
              shiny::uiOutput("hyp_wald_ui"),
              shiny::div(class = "dsem-table-wrap", shiny::uiOutput("hyp_table_ui"))
            )
          )
        ),
        bslib::nav_panel(
          "Impulse Response (IRF)",
          bslib::layout_columns(
            col_widths = c(4, 4, 4),
            shiny::sliderInput("irf_horizon", "Horizon (steps / continuous time units)", min = 3L, max = 25L, value = 10L),
            shiny::checkboxInput("irf_orth", "Orthogonalized Cholesky shock", TRUE),
            shiny::uiOutput("irf_note_ui")
          ),
          bslib::layout_columns(
            col_widths = c(7, 5),
            bslib::card(
              bslib::card_header("Bayesian Impulse Response Curves (95% CrI)"),
              shiny::plotOutput("irf_plot", height = "360px")
            ),
            bslib::card(
              bslib::card_header("IRF Numerical Summary"),
              shiny::div(class = "dsem-table-wrap", shiny::uiOutput("irf_table_ui"))
            )
          )
        ),
        bslib::nav_panel(
          "Cluster Rankings",
          shiny::uiOutput("fscomp_controls_ui"),
          bslib::layout_columns(
            col_widths = c(7, 5),
            bslib::card(
              bslib::card_header("Posterior Cluster Random-Effect Caterpillar Plot"),
              shiny::plotOutput("fscomp_plot", height = "380px")
            ),
            bslib::card(
              bslib::card_header("Ranked Cluster Factor Scores & Exceedance Probabilities"),
              shiny::div(class = "dsem-table-wrap", shiny::uiOutput("fscomp_table_ui"))
            )
          )
        ),
        bslib::nav_panel(
          "Model Comparison (DIC)",
          bslib::card(
            bslib::card_header("Bayesian DIC & pD Model Comparison Across Saved Fits"),
            shiny::div(class = "dsem-table-wrap", shiny::uiOutput("compare_table_ui"))
          )
        )
      )
    ),

    # =========================================================================
    # TAB 5: REPORT & REPRODUCIBLE CODE
    # =========================================================================
    bslib::nav_panel(
      "Report & Code",
      bslib::layout_columns(
        col_widths = c(4, 8),
        bslib::card(
          bslib::card_header("Export & Reproducibility Downloads"),
          shiny::helpText("Download standalone reproducible scripts, publication-ready tables, vector path diagrams, Quarto manuscripts, or a complete reproducibility archive."),
          shiny::downloadButton("dl_r_code", "Download Reproducible R Script (.R)", class = "btn-primary w-100 mb-2"),
          shiny::downloadButton("dl_qmd_report", "Download Quarto Report (.qmd)", class = "btn-outline-secondary w-100 mb-2"),
          shiny::downloadButton("dl_svg_diagram", "Download Path Diagram (.svg)", class = "btn-outline-secondary w-100 mb-2"),
          shiny::downloadButton("dl_csv_est", "Download Estimates (.csv)", class = "btn-outline-secondary w-100 mb-2"),
          shiny::downloadButton("dl_zip_bundle", "Download Full Reproducibility Bundle (.zip)", class = "btn-outline-primary w-100")
        ),
        bslib::card(
          bslib::card_header("Generated Code, Tables & Diagrams"),
          bslib::navset_card_tab(
            bslib::nav_panel(
              "Reproducible R Script",
              shiny::div(class = "d-flex justify-content-end mb-2", .dsem_copy_btn("export_r_code", "Copy R Code")),
              shiny::verbatimTextOutput("export_r_code")
            ),
            bslib::nav_panel(
              "APA Markdown Table",
              shiny::div(class = "d-flex justify-content-end mb-2", .dsem_copy_btn("export_md_table", "Copy Markdown")),
              shiny::verbatimTextOutput("export_md_table")
            ),
            bslib::nav_panel(
              "APA LaTeX Table",
              shiny::div(class = "d-flex justify-content-end mb-2", .dsem_copy_btn("export_tex_table", "Copy LaTeX")),
              shiny::verbatimTextOutput("export_tex_table")
            ),
            bslib::nav_panel(
              "Fitted SVG Path Diagram",
              shiny::div(class = "svg-diagram-box", shiny::uiOutput("export_fitted_svg"))
            ),
            bslib::nav_panel(
              "Mermaid / DOT Graph",
              shiny::div(class = "d-flex justify-content-end mb-2", .dsem_copy_btn("export_mermaid", "Copy Mermaid")),
              shiny::verbatimTextOutput("export_mermaid")
            )
          )
        )
      )
    )
  )
}
