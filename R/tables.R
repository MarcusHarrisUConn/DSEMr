#' Format publication-ready APA 7th Edition tables for DSEMr results
#'
#' Produces APA 7th Edition formatted parameter or model-comparison tables from
#' a `DSEMfit` or `DSEMcompare` object in four formats: tidy `"data.frame"`,
#' `"markdown"` (for Quarto/R Markdown), `"latex"` (using `booktabs` and
#' `threeparttable`), and `"html"` (strict APA horizontal-rule styling suitable
#' for RStudio Viewer and Shiny applications). Parameters are grouped into
#' conceptual multilevel/SEM sections with posterior medians (recommended for
#' skewed variance/covariance posteriors) or means, posterior SDs, credible
#' intervals, one-tailed Bayesian \eqn{p}-values, and model-fit footers (`N`,
#' `DIC`, `pD`, `R^2`).
#'
#' @param x A fitted `DSEMfit` object or a `DSEMcompare` object from
#'   [dsem_compare()].
#' @param format Output format: `"markdown"`, `"latex"`, `"html"`, or
#'   `"data.frame"`.
#' @param standardized Logical; if `TRUE`, reports `STDYX` standardized
#'   estimates from [dsem_standardize()] (only applies to `DSEMfit`).
#' @param point Point estimate convention: `"median"` (default) or `"mean"`.
#' @param digits Number of decimal places (default `3`).
#' @param ci_level Credible interval probability (default `0.95`).
#' @param include_diagnostics Logical; if `TRUE`, appends split `R-hat` and
#'   bulk `ESS` columns.
#' @param include_fit Logical; if `TRUE`, appends model-fit summary information
#'   (`N`, `DIC`, `pD`, max `R-hat`, and Bayesian \eqn{R^2}) in the table note.
#' @param caption Optional table title/caption.
#' @param note Optional custom APA table note appended after the automatic
#'   statistical legend.
#' @return A `DSEMtable` object (when `format` is `"markdown"`, `"latex"`, or
#'   `"html"`, containing the formatted string in `$text` and underlying table
#'   in `$data`), or a `data.frame` when `format = "data.frame"`.
#' @examples
#' dat <- simulate_dsem(n = 80, ar = 0.5, sigma = 0.7, seed = 21)
#' fit <- dsem("y ~ lag(y, 1)", dat, chains = 2, iter = 200, warmup = 100, seed = 21)
#' dsem_table(fit, format = "markdown")
#' dsem_table(fit, format = "latex")
#' @export
dsem_table <- function(x,
                       format = c("markdown", "latex", "html", "data.frame"),
                       standardized = FALSE,
                       point = c("median", "mean"),
                       digits = 3L,
                       ci_level = 0.95,
                       include_diagnostics = FALSE,
                       include_fit = TRUE,
                       caption = NULL,
                       note = NULL) {
  format <- match.arg(format)
  point <- match.arg(point)
  digits <- .dsem_integer(digits, "digits", 0)
  if (!is.numeric(ci_level) || length(ci_level) != 1L || !is.finite(ci_level) || ci_level <= 0 || ci_level >= 1) {
    .dsem_abort("`ci_level` must be strictly between 0 and 1.")
  }

  if (inherits(x, "DSEMcompare")) {
    return(.dsem_table_compare(x, format = format, digits = digits, caption = caption, note = note))
  }
  if (!inherits(x, "DSEMfit")) {
    .dsem_abort("`x` must be a `DSEMfit` or `DSEMcompare` object.")
  }

  ci_pct <- round(100 * ci_level)
  if (isTRUE(standardized)) {
    raw_df <- dsem_standardize(x, type = "stdyx", prob = ci_level)
    est_val <- if (point == "median") raw_df$q50 else raw_df$mean
    sd_val <- raw_df$sd
    lo_val <- raw_df$q_lower
    hi_val <- raw_df$q_upper
    p_val <- raw_df$p_one_tailed
    param_nm <- raw_df$parameter
    sec_lbl <- paste0(raw_df$level, "-Level Standardized (STDYX)")
    rhat_val <- rep(NA_real_, nrow(raw_df))
    ess_val <- rep(NA_real_, nrow(raw_df))
  } else {
    est_df <- x$estimates
    param_nm <- est_df$parameter
    est_val <- if (point == "median" && "q50" %in% names(est_df)) est_df$q50 else est_df$mean
    sd_val <- est_df$sd
    if (abs(ci_level - 0.95) < 1e-8 && all(c("q2.5", "q97.5") %in% names(est_df))) {
      lo_val <- est_df$q2.5
      hi_val <- est_df$q97.5
    } else {
      ci_mat <- stats::confint(x, level = ci_level)
      lo_val <- ci_mat[param_nm, 1L]
      hi_val <- ci_mat[param_nm, 2L]
    }
    p_val <- if ("p_one_tailed" %in% names(est_df)) est_df$p_one_tailed else rep(NA_real_, nrow(est_df))
    rhat_val <- if ("rhat" %in% names(est_df)) est_df$rhat else rep(NA_real_, nrow(est_df))
    ess_val <- if ("ess_bulk" %in% names(est_df)) est_df$ess_bulk else est_df$ess
    sec_lbl <- vapply(param_nm, .dsem_classify_param, character(1), level_type = x$model$level_type)
  }

  sig_flag <- ifelse(is.finite(lo_val) & is.finite(hi_val) & (lo_val > 0 | hi_val < 0), "*", "")

  fmt_num <- function(v, d = digits) {
    ifelse(is.finite(v), formatC(v, format = "f", digits = d), "\u2014")
  }
  fmt_p <- function(v) {
    vapply(v, function(val) {
      if (!is.finite(val)) return("\u2014")
      if (val < 0.001) return("<.001")
      sub("^0\\.", ".", formatC(val, format = "f", digits = 3L))
    }, character(1))
  }

  ci_col_nm <- sprintf("%d%% CrI", ci_pct)
  tab_df <- data.frame(
    Section = sec_lbl,
    Parameter = vapply(param_nm, .dsem_pretty_param_label, character(1)),
    Estimate = paste0(fmt_num(est_val), sig_flag),
    SD = fmt_num(sd_val),
    CrI = paste0("[", fmt_num(lo_val), ", ", fmt_num(hi_val), "]"),
    p = fmt_p(p_val),
    stringsAsFactors = FALSE
  )
  names(tab_df)[names(tab_df) == "CrI"] <- ci_col_nm

  if (isTRUE(include_diagnostics) && !isTRUE(standardized)) {
    tab_df$Rhat <- fmt_num(rhat_val, 3L)
    tab_df$ESS <- ifelse(is.finite(ess_val), formatC(round(ess_val), format = "d"), "\u2014")
  }

  # Sort rows by canonical section order while preserving within-section order
  sec_order <- c(
    "Within-Level Dynamic & Structural Effects",
    "Measurement Model (Factor Loadings)",
    "Between-Level Regressions",
    "Means, Intercepts & Thresholds",
    "Random-Effect Variances & Covariances",
    "Residual Variances"
  )
  ord_key <- match(tab_df$Section, sec_order)
  ord_key[is.na(ord_key)] <- 99L
  tab_df <- tab_df[order(ord_key, seq_len(nrow(tab_df))), , drop = FALSE]
  rownames(tab_df) <- NULL

  if (format == "data.frame") {
    return(tab_df)
  }

  # Build APA caption & note
  if (is.null(caption)) {
    hier_label <- switch(
      x$model$level_type,
      single = "Single-Level",
      twolevel = "Two-Level",
      threelevel = "Three-Level (DSEM3)",
      crossclassified = "Cross-Classified",
      x$model$level_type
    )
    scale_label <- if (isTRUE(standardized)) "Standardized (STDYX) " else ""
    caption <- sprintf("Bayesian %s DSEM %sParameter Estimates", hier_label, scale_label)
  }

  note_parts <- c(
    sprintf("Point estimates represent posterior %ss; SD = posterior standard deviation; %d%% CrI = equal-tailed %d%% credible interval; p = one-tailed Bayesian p-value.",
            point, ci_pct, ci_pct),
    sprintf("* %d%% credible interval excludes zero.", ci_pct)
  )
  if (isTRUE(include_fit)) {
    n_obs <- if (is.matrix(x$observed)) nrow(x$observed) else length(x$observed)
    dic_v <- dsem_dic(x)
    max_rh <- max(x$diagnostics$rhat, na.rm = TRUE)
    fit_bits <- c(
      sprintf("N = %d", n_obs),
      sprintf("Chains = %d (%d iter, %d warmup)", x$sampler$chains, x$sampler$iter, x$sampler$warmup)
    )
    if (is.finite(dic_v["dic"])) {
      fit_bits <- c(fit_bits, sprintf("DIC = %.2f", dic_v["dic"]), sprintf("pD = %.2f", dic_v["pd"]))
    }
    if (is.finite(max_rh)) fit_bits <- c(fit_bits, sprintf("Max R-hat = %.3f", max_rh))
    r2_df <- tryCatch(dsem_r2(x), error = function(e) NULL)
    if (!is.null(r2_df) && nrow(r2_df) > 0L) {
      r2_str <- paste(sprintf("%s %s R^2 = %.3f", r2_df$level, r2_df$variable, r2_df$mean), collapse = "; ")
      fit_bits <- c(fit_bits, r2_str)
    }
    note_parts <- c(note_parts, paste0("Model fit: ", paste(fit_bits, collapse = ", "), "."))
  }
  if (!is.null(note) && nzchar(note)) note_parts <- c(note_parts, note)
  full_note <- paste(note_parts, collapse = " ")

  txt <- .dsem_render_apa_table(
    tab_df, format = format, caption = caption, note = full_note, grouped = TRUE
  )
  out <- list(format = format, text = txt, data = tab_df, caption = caption, note = full_note)
  class(out) <- c("DSEMtable", "list")
  out
}

.dsem_table_compare <- function(x, format, digits, caption, note) {
  fmt_num <- function(v, d = digits) ifelse(is.finite(v), formatC(v, format = "f", digits = d), "\u2014")
  tab_df <- data.frame(
    Model = x$model,
    Hierarchy = x$hierarchy,
    N = formatC(x$nobs, format = "d"),
    pD = fmt_num(x$pd, 2L),
    DIC = fmt_num(x$dic, 2L),
    Delta_DIC = fmt_num(x$delta_dic, 2L),
    Weight = fmt_num(x$weight, 3L),
    Max_Rhat = fmt_num(x$max_rhat, 3L),
    stringsAsFactors = FALSE
  )
  names(tab_df)[names(tab_df) == "Delta_DIC"] <- if (format == "latex") "$\\Delta$DIC" else "\u0394DIC"
  names(tab_df)[names(tab_df) == "Max_Rhat"] <- if (format == "latex") "Max $\\hat{R}$" else "Max R-hat"

  if (format == "data.frame") return(tab_df)
  if (is.null(caption)) caption <- "Bayesian DSEM Model Comparison"
  note_parts <- c(
    "DIC = Deviance Information Criterion (lower indicates better fit); pD = effective number of parameters; Weight = DIC model weight.",
    attr(x, "notes"),
    if (!is.null(note) && nzchar(note)) note else NULL
  )
  full_note <- paste(note_parts, collapse = " ")
  txt <- .dsem_render_apa_table(tab_df, format = format, caption = caption, note = full_note, grouped = FALSE)
  out <- list(format = format, text = txt, data = tab_df, caption = caption, note = full_note)
  class(out) <- c("DSEMtable", "list")
  out
}

.dsem_render_apa_table <- function(df, format, caption, note, grouped = TRUE) {
  cols <- if (grouped) setdiff(names(df), "Section") else names(df)
  n_col <- length(cols)

  if (format == "markdown") {
    lines <- c(
      paste0("**", caption, "**"),
      "",
      paste0("| ", paste(cols, collapse = " | "), " |"),
      paste0("| ", paste(c(":---", rep("---:", n_col - 1L)), collapse = " | "), " |")
    )
    if (grouped) {
      for (sec in unique(df$Section)) {
        lines <- c(lines, paste0("| **", sec, "** |", paste(rep(" |", n_col - 1L), collapse = "")))
        sub <- df[df$Section == sec, cols, drop = FALSE]
        for (i in seq_len(nrow(sub))) {
          lines <- c(lines, paste0("| ", paste(as.character(sub[i, ]), collapse = " | "), " |"))
        }
      }
    } else {
      for (i in seq_len(nrow(df))) {
        lines <- c(lines, paste0("| ", paste(as.character(df[i, cols, drop = FALSE]), collapse = " | "), " |"))
      }
    }
    lines <- c(lines, "", paste0("*Note.* ", note))
    return(paste(lines, collapse = "\n"))
  }

  if (format == "latex") {
    esc <- function(s) {
      s <- gsub("\\", "\\textbackslash{}", s, fixed = TRUE)
      s <- gsub("_", "\\_", s, fixed = TRUE)
      s <- gsub("%", "\\%", s, fixed = TRUE)
      s <- gsub("&", "\\&", s, fixed = TRUE)
      s <- gsub("#", "\\#", s, fixed = TRUE)
      s <- gsub("$", "\\$", s, fixed = TRUE)
      s <- gsub("\u2014", "---", s, fixed = TRUE)
      s
    }
    esc_hdr <- function(s) {
      # Preserve explicit math delimiters in headers
      if (grepl("\\$", s)) return(s)
      s <- gsub("%", "\\%", s, fixed = TRUE)
      s <- gsub("_", "\\_", s, fixed = TRUE)
      if (s %in% c("Estimate", "Est.", "SD", "p", "N")) paste0("\\textit{", s, "}") else s
    }
    align_str <- paste0("l", paste(rep("r", n_col - 1L), collapse = ""))
    lines <- c(
      "\\begin{table}[htbp]",
      "\\centering",
      "\\begin{threeparttable}",
      paste0("\\caption{", esc(caption), "}"),
      paste0("\\begin{tabular}{", align_str, "}"),
      "\\toprule",
      paste0(paste(vapply(cols, esc_hdr, character(1)), collapse = " & "), " \\\\"),
      "\\midrule"
    )
    if (grouped) {
      secs <- unique(df$Section)
      for (si in seq_along(secs)) {
        sec <- secs[si]
        if (si > 1L) lines <- c(lines, "\\addlinespace")
        lines <- c(lines, sprintf("\\multicolumn{%d}{l}{\\textit{%s}} \\\\", n_col, esc(sec)))
        sub <- df[df$Section == sec, cols, drop = FALSE]
        for (i in seq_len(nrow(sub))) {
          row_vals <- vapply(as.character(sub[i, ]), esc, character(1))
          row_vals[1L] <- paste0("\\quad ", row_vals[1L])
          lines <- c(lines, paste0(paste(row_vals, collapse = " & "), " \\\\"))
        }
      }
    } else {
      for (i in seq_len(nrow(df))) {
        row_vals <- vapply(as.character(df[i, cols, drop = FALSE]), esc, character(1))
        lines <- c(lines, paste0(paste(row_vals, collapse = " & "), " \\\\"))
      }
    }
    lines <- c(
      lines,
      "\\bottomrule",
      "\\end{tabular}",
      "\\begin{tablenotes}[flushleft]",
      "\\small",
      paste0("\\item \\textit{Note.} ", esc(note)),
      "\\end{tablenotes}",
      "\\end{threeparttable}",
      "\\end{table}"
    )
    return(paste(lines, collapse = "\n"))
  }

  # APA 7th Edition HTML (horizontal rules only, italicized caption & statistical headers)
  esc_h <- function(s) {
    s <- gsub("&", "&amp;", s, fixed = TRUE)
    s <- gsub("<", "&lt;", s, fixed = TRUE)
    s <- gsub(">", "&gt;", s, fixed = TRUE)
    s
  }
  th_cells <- vapply(seq_along(cols), function(j) {
    al <- if (j == 1L) "left" else "right"
    lbl <- esc_h(cols[j])
    if (cols[j] %in% c("Estimate", "SD", "p", "N")) lbl <- paste0("<em>", lbl, "</em>")
    sprintf('<th style="text-align:%s;padding:6px 10px;border-top:2px solid currentColor;border-bottom:1px solid currentColor;font-weight:600;">%s</th>', al, lbl)
  }, character(1))

  body_rows <- character()
  if (grouped) {
    for (sec in unique(df$Section)) {
      body_rows <- c(
        body_rows,
        sprintf('<tr><td colspan="%d" style="padding:8px 10px 4px 10px;font-style:italic;font-weight:600;">%s</td></tr>', n_col, esc_h(sec))
      )
      sub <- df[df$Section == sec, cols, drop = FALSE]
      for (i in seq_len(nrow(sub))) {
        tds <- vapply(seq_along(cols), function(j) {
          al <- if (j == 1L) "left" else "right"
          pad_l <- if (j == 1L) "22px" else "10px"
          sprintf('<td style="text-align:%s;padding:4px 10px 4px %s;font-variant-numeric:tabular-nums;">%s</td>', al, pad_l, esc_h(as.character(sub[i, j])))
        }, character(1))
        body_rows <- c(body_rows, paste0("<tr>", paste(tds, collapse = ""), "</tr>"))
      }
    }
  } else {
    for (i in seq_len(nrow(df))) {
      tds <- vapply(seq_along(cols), function(j) {
        al <- if (j == 1L) "left" else "right"
        sprintf('<td style="text-align:%s;padding:5px 10px;font-variant-numeric:tabular-nums;">%s</td>', al, esc_h(as.character(df[i, j])))
      }, character(1))
      body_rows <- c(body_rows, paste0("<tr>", paste(tds, collapse = ""), "</tr>"))
    }
  }

  paste(c(
    '<figure class="dsemr-apa-table" style="margin:1em 0;font-family:\'Times New Roman\',Georgia,serif;font-size:14px;line-height:1.45;">',
    sprintf('  <figcaption style="text-align:left;font-style:italic;margin-bottom:6px;">%s</figcaption>', esc_h(caption)),
    '  <table style="border-collapse:collapse;width:100%;border-bottom:2px solid currentColor;">',
    paste0("    <thead><tr>", paste(th_cells, collapse = ""), "</tr></thead>"),
    paste0("    <tbody>\n      ", paste(body_rows, collapse = "\n      "), "\n    </tbody>"),
    "  </table>",
    sprintf('  <p style="margin-top:6px;font-size:12px;"><em>Note.</em> %s</p>', esc_h(note)),
    "</figure>"
  ), collapse = "\n")
}

.dsem_classify_param <- function(p_name, level_type = "single") {
  if (grepl("^(loading_|between_loading_)", p_name)) return("Measurement Model (Factor Loadings)")
  if (grepl("^(sigma2|within_cov_|residual_variance_|between_residual_variance_|factor_variance|factor_cov_)", p_name)) return("Within-Level Variances & Covariances")
  if (grepl("^(var_|cov_|l3_var_|l3_cov_|l2_var_|l2_cov_|id_var_|id_cov_|time_var_|time_cov_|between_var_|between_cov_|tau_)", p_name)) {
    return("Random-Effect Variances & Covariances")
  }
  if (grepl("^(\\(Intercept\\)|intercept_|mean_|fixed_|threshold_)", p_name)) {
    return("Means, Intercepts & Thresholds")
  }
  if (level_type != "single" && grepl("_on_", p_name) && !grepl("_on_lag[0-9]+_", p_name)) {
    return("Between-Level Regressions")
  }
  "Within-Level Dynamic & Structural Effects"
}

.dsem_pretty_param_label <- function(p_name) {
  if (identical(p_name, "(Intercept)")) return("Intercept")
  s <- p_name
  s <- sub("^l3_var_", "L3 Var(", s)
  s <- sub("^l3_cov_", "L3 Cov(", s)
  s <- sub("^l2_var_", "L2 Var(", s)
  s <- sub("^l2_cov_", "L2 Cov(", s)
  s <- sub("^between_var_", "Between Var(", s)
  s <- sub("^between_cov_", "Between Cov(", s)
  s <- sub("^within_cov_", "Within Cov(", s)
  s <- sub("^id_var_", "Person Var(", s)
  s <- sub("^id_cov_", "Person Cov(", s)
  s <- sub("^time_var_", "Occasion Var(", s)
  s <- sub("^time_cov_", "Occasion Cov(", s)
  s <- sub("^var_", "Var(", s)
  s <- sub("^cov_", "Cov(", s)
  if (grepl("\\($", sub("[^()]+$", "", s)) && !grepl("\\)$", s)) s <- paste0(s, ")")
  s <- sub("^fixed_", "Fixed ", s)
  s <- sub("^mean_", "Mean ", s)
  s <- sub("^intercept_", "Intercept ", s)
  s <- sub("^threshold_", "Threshold ", s)
  s <- sub("^loading_", "Loading ", s)
  s <- sub("^between_loading_", "Between Loading ", s)
  s <- sub("^residual_variance_", "Resid. Var. ", s)
  s <- sub("^between_residual_variance_", "Between Resid. Var. ", s)
  s <- sub("^factor_variance", "Factor Var.", s)
  s <- sub("^lagged_factor_effect_", "Lagged Factor -> ", s)
  s <- gsub("_on_lag([0-9]+)_", " ON Lag\\1 ", s)
  s <- gsub("_on_", " ON ", s)
  s <- sub("^ar1_", "AR(1) ", s)
  s <- sub("^lag([0-9]+)_", "Lag\\1 ", s)
  s
}

#' @exportS3Method base::print
print.DSEMtable <- function(x, ...) {
  cat(x$text, "\n")
  invisible(x)
}

#' @exportS3Method base::as.character
as.character.DSEMtable <- function(x, ...) x$text
