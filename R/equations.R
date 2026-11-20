#' Generate publication-ready LaTeX equations for a DSEMr model or fit
#'
#' Automatically constructs multilevel or state-space LaTeX equations matching
#' the exact statistical parameterization of a `DSEMmodel`, `DSEMspec`, or
#' `DSEMfit` object. Supports single-level AR/VAR/RDSEM, dynamic confirmatory
#' factor / IRT / WNFS models, two-level location-scale DSEM (with latent
#' centering, random slopes, random log residual variances, between-level
#' covariates, and distal outcomes), three-level DSEM (DSEM3),
#' and cross-classified subject-by-time models. When passed a `DSEMfit` with
#' `substitute = TRUE`, posterior point estimates are substituted directly into
#' the equations.
#'
#' @param object A `DSEMfit`, `DSEMmodel`, or `DSEMspec` object.
#' @param style Equation presentation style: `"multilevel"` (default; Level 1 /
#'   Level 2 / Level 3 decomposition) or `"compact"`.
#' @param substitute Logical; if `TRUE` and `object` is a `DSEMfit`, replaces
#'   symbolic parameters with fitted posterior estimates.
#' @param point Point estimate used when `substitute = TRUE`: `"median"`
#'   (default) or `"mean"`.
#' @param digits Decimal places when substituting estimates (default `2`).
#' @param wrap Outer LaTeX environment: `"align"`, `"aligned"`, `"dollar"`
#'   (`$$...$$`), or `"none"`.
#' @return A `DSEMequations` object with `$latex`, `$lines`, `$hierarchy`, and
#'   S3 methods [print()] and [as.character()].
#' @examples
#' dat <- simulate_dsem(n = 60, ar = 0.45, sigma = 0.7, seed = 31)
#' fit <- dsem("y ~ lag(y, 1)", dat, chains = 2, iter = 160, warmup = 80, seed = 31)
#' dsem_equations(fit)
#' dsem_equations(fit, substitute = TRUE)
#' @export
dsem_equations <- function(object,
                           style = c("multilevel", "compact"),
                           substitute = FALSE,
                           point = c("median", "mean"),
                           digits = 2L,
                           wrap = c("align", "aligned", "dollar", "none")) {
  style <- match.arg(style)
  point <- match.arg(point)
  wrap <- match.arg(wrap)
  digits <- .dsem_integer(digits, "digits", 0)

  fit_obj <- NULL
  if (inherits(object, "DSEMfit")) {
    fit_obj <- object
    model <- object$model
  } else if (inherits(object, "DSEMspec")) {
    model <- as_dsem_model(object)
  } else if (inherits(object, "DSEMmodel")) {
    model <- object
  } else {
    .dsem_abort("`object` must be a `DSEMfit`, `DSEMmodel`, or `DSEMspec`.")
  }

  cf <- if (isTRUE(substitute) && !is.null(fit_obj)) {
    stats::coef(fit_obj, point = point)
  } else {
    NULL
  }

  tex_var <- function(v) {
    v <- gsub("_", "\\_", v, fixed = TRUE)
    if (nchar(v) == 1L) v else paste0("\\text{", v, "}")
  }
  fmt_est <- function(val) {
    formatC(val, format = "f", digits = digits)
  }
  lookup_coef <- function(cands, fallback_sym) {
    if (!is.null(cf)) {
      hit <- intersect(cands, names(cf))
      if (length(hit)) return(fmt_est(cf[[hit[1L]]]))
    }
    fallback_sym
  }
  signed_term <- function(coef_str, expr_str, first = FALSE) {
    num_v <- suppressWarnings(as.numeric(coef_str))
    if (!is.na(num_v)) {
      if (first) {
        if (num_v < 0) paste0("- ", fmt_est(abs(num_v)), " ", expr_str)
        else paste0(fmt_est(num_v), " ", expr_str)
      } else {
        if (num_v < 0) paste0(" - ", fmt_est(abs(num_v)), " ", expr_str)
        else paste0(" + ", fmt_est(num_v), " ", expr_str)
      }
    } else {
      if (first) paste0(coef_str, " ", expr_str) else paste0(" + ", coef_str, " ", expr_str)
    }
  }

  terms <- model$terms
  lvl_type <- model$level_type
  has_latent <- any(terms$op == "=~")
  is_cat <- !identical(model$family, "gaussian") || length(model$categorical) > 0L
  eq_lines <- character()

  if (has_latent) {
    # Dynamic CFA / SEM / WNFS / Dynamic IRT
    meas_w <- terms[terms$op == "=~" & terms$level == "within", , drop = FALSE]
    regs_w <- terms[terms$op == "~" & terms$rhs != "1" & terms$level == "within", , drop = FALSE]
    w_factors <- unique(meas_w$lhs)
    items <- unique(meas_w$rhs)
    idx_sub <- if (lvl_type == "single") "t" else "it"
    idx_lag <- if (lvl_type == "single") "t-1" else "i,t-1"

    eq_lines <- c(eq_lines, "\\text{Measurement Model:}")
    for (j_i in seq_along(items)) {
      it_nm <- items[j_i]
      f_rows <- meas_w[meas_w$rhs == it_nm, , drop = FALSE]
      lhs_sym <- if (it_nm %in% model$categorical || is_cat) {
        paste0(tex_var(it_nm), "^{*}_{", idx_sub, "}")
      } else {
        paste0(tex_var(it_nm), "_{", idx_sub, "}")
      }
      nu_str <- if (lvl_type == "single") {
        lookup_coef(c(paste0("intercept_", it_nm)), paste0("\\nu_{", j_i, "}"))
      } else {
        paste0("\\nu_{", j_i, "i}")
      }
      rhs_parts <- if (it_nm %in% model$categorical || is_cat) "" else nu_str
      for (r_k in seq_len(nrow(f_rows))) {
        fnm <- f_rows$lhs[r_k]
        lam_str <- lookup_coef(
          c(paste0("loading_", fnm, "_", it_nm), paste0("loading_", it_nm)),
          if (j_i == 1L && length(w_factors) == 1L) "1" else paste0("\\lambda_{", j_i, "}")
        )
        rhs_parts <- paste0(rhs_parts, signed_term(lam_str, paste0("\\eta_{", tex_var(fnm), ",", idx_sub, "}"), first = !nzchar(rhs_parts)))
      }
      # WNFS lagged factor effect directly on item
      wnfs_r <- regs_w[regs_w$lhs == it_nm & regs_w$rhs %in% w_factors & regs_w$lag > 0L, , drop = FALSE]
      for (w_k in seq_len(nrow(wnfs_r))) {
        fnm <- wnfs_r$rhs[w_k]
        g_str <- lookup_coef(paste0("lagged_factor_effect_", it_nm), paste0("\\gamma_{", j_i, "}"))
        rhs_parts <- paste0(rhs_parts, signed_term(g_str, paste0("\\eta_{", tex_var(fnm), ",", idx_lag, "}")))
      }
      rhs_parts <- paste0(rhs_parts, " + \\varepsilon_{", j_i, ",", idx_sub, "}")
      eq_lines <- c(eq_lines, paste0(lhs_sym, " &= ", rhs_parts))
    }

    # Latent structural transition equation(s)
    lat_regs <- regs_w[regs_w$lhs %in% w_factors, , drop = FALSE]
    if (nrow(lat_regs)) {
      eq_lines <- c(eq_lines, "\\text{Latent Dynamic Transition:}")
      for (fnm in unique(lat_regs$lhs)) {
        sub_r <- lat_regs[lat_regs$lhs == fnm, , drop = FALSE]
        rhs_s <- ""
        for (r_k in seq_len(nrow(sub_r))) {
          rn <- sub_r$rhs[r_k]; lg <- sub_r$lag[r_k]
          lbl <- sub_r$label[r_k]
          phi_cands <- c(
            if (nzchar(lbl)) c(paste0("mean_", lbl), lbl) else character(),
            paste0("ar1_", fnm),
            paste0(fnm, "_on_lag", lg, "_", rn)
          )
          phi_sym <- if (lvl_type != "single" && sub_r$random[r_k]) {
            paste0("\\phi_{", if (nzchar(lbl)) tex_var(lbl) else fnm, ",i}")
          } else {
            paste0("\\phi_{", tex_var(fnm), ",", tex_var(rn), "}")
          }
          phi_str <- lookup_coef(phi_cands, phi_sym)
          lag_sub <- if (lvl_type == "single") paste0("t-", lg) else paste0("i,t-", lg)
          rhs_s <- paste0(rhs_s, signed_term(phi_str, paste0("\\eta_{", tex_var(rn), ",", lag_sub, "}"), first = !nzchar(rhs_s)))
        }
        eq_lines <- c(eq_lines, paste0("\\eta_{", tex_var(fnm), ",", idx_sub, "} &= ", rhs_s, " + \\zeta_{", tex_var(fnm), ",", idx_sub, "}"))
      }
    }
    if (is_cat) {
      eq_lines <- c(eq_lines, paste0("u_{j,", idx_sub, "} &= \\mathbb{I}\\!\\left(u^{*}_{j,", idx_sub, "} > \\tau_j\\right), \\quad \\varepsilon_{j,", idx_sub, "} \\sim \\mathcal{N}(0, 1)"))
    }
  } else if (lvl_type == "single") {
    # Single-level N=1 AR(p) / VAR(p) / RDSEM / Probit
    regs <- terms[terms$op == "~" & terms$rhs != "1", , drop = FALSE]
    outcomes <- unique(regs$lhs)
    is_rdsem <- any(terms$residual)
    d <- length(outcomes)
    for (out_nm in outcomes) {
      sub_r <- regs[regs$lhs == out_nm, , drop = FALSE]
      int_str <- lookup_coef(
        c("(Intercept)", paste0("intercept_", out_nm), paste0("threshold_", out_nm, "$1")),
        paste0("\\alpha_{", tex_var(out_nm), "}")
      )
      lhs_sym <- if (out_nm %in% model$categorical || is_cat) {
        paste0(tex_var(out_nm), "^{*}_t")
      } else {
        paste0(tex_var(out_nm), "_t")
      }
      if (is_rdsem) {
        ex_r <- sub_r[sub_r$lag == 0L, , drop = FALSE]
        lag_r <- sub_r[sub_r$lag > 0L, , drop = FALSE]
        rhs_s <- int_str
        for (k in seq_len(nrow(ex_r))) {
          rn <- ex_r$rhs[k]
          b_str <- lookup_coef(c(rn, paste0(out_nm, "_on_", rn)), paste0("\\beta_{", tex_var(rn), "}"))
          rhs_s <- paste0(rhs_s, signed_term(b_str, paste0(tex_var(rn), "_t")))
        }
        eq_lines <- c(eq_lines, paste0(lhs_sym, " &= ", rhs_s, " + u_{", tex_var(out_nm), ",t}"))
        u_rhs <- ""
        for (k in seq_len(nrow(lag_r))) {
          rn <- lag_r$rhs[k]; lg <- lag_r$lag[k]
          p_str <- lookup_coef(c(paste0("lag", lg, "_", rn), paste0(out_nm, "_on_lag", lg, "_", rn)),
                               paste0("\\phi_{", lg, "}"))
          u_rhs <- paste0(u_rhs, signed_term(p_str, paste0("u_{", tex_var(rn), ",t-", lg, "}"), first = !nzchar(u_rhs)))
        }
        eq_lines <- c(eq_lines, paste0("u_{", tex_var(out_nm), ",t} &= ", u_rhs, " + \\varepsilon_{", tex_var(out_nm), ",t}"))
      } else {
        rhs_s <- int_str
        for (k in seq_len(nrow(sub_r))) {
          rn <- sub_r$rhs[k]; lg <- sub_r$lag[k]
          if (lg > 0L) {
            p_str <- lookup_coef(
              c(paste0("lag", lg, "_", rn), paste0(out_nm, "_on_lag", lg, "_", rn)),
              if (d == 1L) paste0("\\phi_{", lg, "}") else paste0("\\phi_{", tex_var(out_nm), ",", tex_var(rn), "}^{(", lg, ")}")
            )
            rhs_s <- paste0(rhs_s, signed_term(p_str, paste0(tex_var(rn), "_{t-", lg, "}")))
          } else {
            b_str <- lookup_coef(c(rn, paste0(out_nm, "_on_", rn)), paste0("\\beta_{", tex_var(rn), "}"))
            rhs_s <- paste0(rhs_s, signed_term(b_str, paste0(tex_var(rn), "_t")))
          }
        }
        eq_lines <- c(eq_lines, paste0(lhs_sym, " &= ", rhs_s, " + \\varepsilon_{", tex_var(out_nm), ",t}"))
      }
    }
    sig_str <- lookup_coef(c("sigma2", paste0("sigma2_", outcomes[1L])), "\\sigma^2")
    if (is_cat) {
      eq_lines <- c(eq_lines, "\\varepsilon_t &\\sim \\mathcal{N}(0, 1)")
    } else if (d == 1L) {
      eq_lines <- c(eq_lines, paste0("\\varepsilon_t &\\sim \\mathcal{N}(0, ", sig_str, ")"))
    } else {
      eq_lines <- c(eq_lines, "\\boldsymbol{\\varepsilon}_t &\\sim \\mathcal{N}(\\mathbf{0}, \\boldsymbol{\\Sigma})")
    }
  } else if (lvl_type == "threelevel") {
    # Mplus Version 9 Three-Level DSEM (DSEM3)
    w_regs <- terms[terms$op == "~" & terms$rhs != "1" & terms$level == "within", , drop = FALSE]
    b2_regs <- terms[terms$op == "~" & terms$rhs != "1" & terms$level == "between_2", , drop = FALSE]
    b3_regs <- terms[terms$op == "~" & terms$rhs != "1" & terms$level == "between_3", , drop = FALSE]
    w_vars <- terms[terms$op == "~~" & terms$level == "within" & terms$lhs == terms$rhs, , drop = FALSE]
    outcomes <- unique(w_regs$lhs)
    has_lv <- any(w_vars$random)

    eq_lines <- c(eq_lines, "\\textbf{Level 1 (Momentary / Within):}")
    for (out_nm in outcomes) {
      sub_r <- w_regs[w_regs$lhs == out_nm, , drop = FALSE]
      eq_lines <- c(eq_lines, paste0(tex_var(out_nm), "_{tdi} &= \\mu_{", tex_var(out_nm), ",di} + ", tex_var(out_nm), "^{(w)}_{tdi}"))
      rhs_w <- ""
      for (k in seq_len(nrow(sub_r))) {
        rn <- sub_r$rhs[k]; lg <- sub_r$lag[k]; lbl <- sub_r$label[k]
        sym <- if (sub_r$random[k]) {
          paste0(if (nzchar(lbl)) tex_var(lbl) else "\\phi", "_{di}")
        } else {
          lookup_coef(c(paste0("fixed_", lbl), paste0("fixed_lag", lg, "_", rn)), paste0("\\phi_{", lg, "}"))
        }
        pred_tex <- if (lg > 0L) paste0(tex_var(rn), "^{(w)}_{t-", lg, ",di}") else paste0(tex_var(rn), "^{(w)}_{tdi}")
        rhs_w <- paste0(rhs_w, signed_term(sym, pred_tex, first = !nzchar(rhs_w)))
      }
      eq_lines <- c(eq_lines, paste0(tex_var(out_nm), "^{(w)}_{tdi} &= ", rhs_w, " + \\varepsilon_{tdi}, \\quad \\varepsilon_{tdi} \\sim \\mathcal{N}\\!\\left(0, \\exp(\\omega_{di})\\right)"))
    }

    eq_lines <- c(eq_lines, "\\textbf{Level 2 (Nested Context / Day } d \\textbf{ within } i\\textbf{):}")
    mu_eq <- paste0("\\mu_{", tex_var(outcomes[1L]), ",di} &= \\mu_{", tex_var(outcomes[1L]), ",i}^{(3)}")
    if (nrow(b2_regs)) {
      for (k in seq_len(nrow(b2_regs))) {
        w2 <- b2_regs$rhs[k]
        b_s <- lookup_coef(paste0("l2_", b2_regs$lhs[k], "_on_", w2), paste0("\\gamma^{(2)}_{", tex_var(w2), "}"))
        mu_eq <- paste0(mu_eq, signed_term(b_s, paste0(tex_var(w2), "_{di}")))
      }
    }
    eq_lines <- c(eq_lines, paste0(mu_eq, " + u^{(2)}_{di}, \\quad u^{(2)}_{di} \\sim \\mathcal{N}(0, \\boldsymbol{\\Omega}^{(2)})"))

    eq_lines <- c(eq_lines, "\\textbf{Level 3 (Cluster / Person } i\\textbf{):}")
    re_syms <- c(paste0("\\mu_{", tex_var(outcomes[1L]), ",i}^{(3)}"))
    fixed_syms <- c(lookup_coef(paste0("fixed_", outcomes[1L]), paste0("\\gamma_{0,", tex_var(outcomes[1L]), "}")))
    for (k in seq_len(nrow(w_regs))) {
      if (w_regs$random[k]) {
        lbl <- if (nzchar(w_regs$label[k])) w_regs$label[k] else paste0("lag", w_regs$lag[k], "_", w_regs$rhs[k])
        re_syms <- c(re_syms, paste0(tex_var(lbl), "_{i}^{(3)}"))
        fixed_syms <- c(fixed_syms, lookup_coef(paste0("fixed_", lbl), paste0("\\gamma_{0,", tex_var(lbl), "}")))
      }
    }
    if (has_lv) {
      lv_lbl <- w_vars$label[w_vars$random][1L]
      if (!nzchar(lv_lbl)) lv_lbl <- paste0("logv_", outcomes[1L])
      re_syms <- c(re_syms, "\\omega_{i}^{(3)}")
      fixed_syms <- c(fixed_syms, lookup_coef(paste0("fixed_", lv_lbl), "\\gamma_{0,\\omega}"))
    }
    for (idx_q in seq_along(re_syms)) {
      line_q <- paste0(re_syms[idx_q], " &= ", fixed_syms[idx_q])
      if (nrow(b3_regs)) {
        for (k in seq_len(nrow(b3_regs))) {
          w3 <- b3_regs$rhs[k]
          if (idx_q == 1L && b3_regs$lhs[k] == outcomes[1L]) {
            b3_s <- lookup_coef(paste0("l3_", outcomes[1L], "_on_", w3), paste0("\\gamma^{(3)}_{", tex_var(w3), "}"))
            line_q <- paste0(line_q, signed_term(b3_s, paste0(tex_var(w3), "_i")))
          }
        }
      }
      eq_lines <- c(eq_lines, paste0(line_q, " + u^{(3)}_{", idx_q, ",i}"))
    }
    eq_lines <- c(eq_lines, "\\mathbf{u}^{(3)}_i &\\sim \\mathcal{N}(\\mathbf{0}, \\boldsymbol{\\Omega}^{(3)})")
  } else if (lvl_type == "crossclassified") {
    w_regs <- terms[terms$op == "~" & terms$rhs != "1" & terms$level == "within", , drop = FALSE]
    outcomes <- unique(w_regs$lhs)
    out_nm <- outcomes[1L]
    eq_lines <- c(
      "\\textbf{Cross-Classified Decomposition (Subject } i \\times \\textbf{ Occasion } t\\textbf{):}",
      paste0(tex_var(out_nm), "_{it} &= \\mu + \\alpha_i + \\tau_t + ", tex_var(out_nm), "^{(w)}_{it}"),
      paste0(tex_var(out_nm), "^{(w)}_{it} &= ",
             lookup_coef(c("mean_lag1_y", paste0("mean_lag1_", out_nm)), "\\phi_i"),
             " ", tex_var(out_nm), "^{(w)}_{i,t-1} + \\varepsilon_{it}, \\quad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2)"),
      paste0("\\alpha_i &\\sim \\mathcal{N}(0, ", lookup_coef(paste0("id_var_", out_nm), "\\sigma^2_{\\text{id}}"), "), \\quad \\tau_t \\sim \\mathcal{N}(0, ", lookup_coef(paste0("time_var_", out_nm), "\\sigma^2_{\\text{time}}"), ")")
    )
  } else {
    # Two-Level DSEM / Location-Scale / RDSEM (Mplus 9.30, 9.37, etc.)
    w_regs <- terms[terms$op == "~" & terms$rhs != "1" & terms$level == "within", , drop = FALSE]
    b_regs <- terms[terms$op == "~" & terms$rhs != "1" & terms$level == "between", , drop = FALSE]
    w_vars <- terms[terms$op == "~~" & terms$level == "within" & terms$lhs == terms$rhs, , drop = FALSE]
    outcomes <- unique(w_regs$lhs)
    has_lv <- any(w_vars$random)
    is_rdsem <- any(terms$residual)

    eq_lines <- c(eq_lines, "\\textbf{Level 1 (Within-Cluster):}")
    re_list <- outcomes
    for (out_nm in outcomes) {
      sub_r <- w_regs[w_regs$lhs == out_nm, , drop = FALSE]
      if (!identical(model$centering, "none")) {
        eq_lines <- c(eq_lines, paste0(tex_var(out_nm), "_{it} &= \\mu_{", tex_var(out_nm), ",i} + ", tex_var(out_nm), "^{(w)}_{it}"))
      }
      rhs_w <- ""
      for (k in seq_len(nrow(sub_r))) {
        rn <- sub_r$rhs[k]; lg <- sub_r$lag[k]; lbl <- sub_r$label[k]
        sl_nm <- if (nzchar(lbl)) lbl else if (lg > 0L) paste0("lag", lg, "_", rn) else rn
        if (sub_r$random[k]) {
          re_list <- unique(c(re_list, sl_nm))
          sym <- paste0(if (nzchar(lbl)) tex_var(lbl) else if (lg > 0L) "\\phi" else "\\beta", "_{i}")
        } else {
          sym <- lookup_coef(c(sl_nm, paste0(out_nm, "_on_", sl_nm)), if (lg > 0L) paste0("\\phi_{", lg, "}") else paste0("\\beta_{", tex_var(rn), "}"))
        }
        pred_tex <- if (lg > 0L) {
          if (is_rdsem) paste0("u_{", tex_var(rn), ",i,t-", lg, "}")
          else if (!identical(model$centering, "none")) paste0(tex_var(rn), "^{(w)}_{i,t-", lg, "}")
          else paste0(tex_var(rn), "_{i,t-", lg, "}")
        } else {
          if (!identical(model$centering, "none")) paste0(tex_var(rn), "^{(w)}_{it}") else paste0(tex_var(rn), "_{it}")
        }
        rhs_w <- paste0(rhs_w, signed_term(sym, pred_tex, first = !nzchar(rhs_w)))
      }
      var_tex <- if (has_lv) {
        lv_lbl <- w_vars$label[w_vars$lhs == out_nm & w_vars$random]
        lv_nm <- if (length(lv_lbl) && nzchar(lv_lbl[1L])) lv_lbl[1L] else paste0("logv_", out_nm)
        re_list <- unique(c(re_list, lv_nm))
        paste0("\\exp(", tex_var(lv_nm), "_i)")
      } else {
        lookup_coef(c("sigma2", paste0("sigma2_", out_nm)), "\\sigma^2")
      }
      lhs_w <- if (!identical(model$centering, "none")) paste0(tex_var(out_nm), "^{(w)}_{it}") else paste0(tex_var(out_nm), "_{it}")
      eq_lines <- c(eq_lines, paste0(lhs_w, " &= ", rhs_w, " + \\varepsilon_{it}, \\quad \\varepsilon_{it} \\sim \\mathcal{N}\\!\\left(0, ", var_tex, "\\right)"))
    }

    eq_lines <- c(eq_lines, "\\textbf{Level 2 (Between-Cluster):}")
    distal_z <- setdiff(unique(b_regs$lhs), re_list)
    for (rn in re_list) {
      lhs_b <- if (rn %in% outcomes) paste0("\\mu_{", tex_var(rn), ",i}") else paste0(tex_var(rn), "_i")
      int_b <- lookup_coef(
        c(paste0("mean_", rn), if (rn %in% outcomes) "(Intercept)" else rn),
        paste0("\\gamma_{0,", tex_var(rn), "}")
      )
      rhs_b <- int_b
      b_sub <- b_regs[b_regs$lhs == rn, , drop = FALSE]
      for (k in seq_len(nrow(b_sub))) {
        wn <- b_sub$rhs[k]
        g_str <- lookup_coef(paste0(rn, "_on_", wn), paste0("\\gamma_{", tex_var(rn), ",", tex_var(wn), "}"))
        rhs_b <- paste0(rhs_b, signed_term(g_str, paste0(tex_var(wn), "_i")))
      }
      eq_lines <- c(eq_lines, paste0(lhs_b, " &= ", rhs_b, " + u_{", tex_var(rn), ",i}"))
    }
    eq_lines <- c(eq_lines, "\\mathbf{u}_i &\\sim \\mathcal{N}(\\mathbf{0}, \\boldsymbol{\\Omega}_B)")

    if (length(distal_z)) {
      eq_lines <- c(eq_lines, "\\textbf{Distal Outcome Model:}")
      for (zn in distal_z) {
        z_sub <- b_regs[b_regs$lhs == zn, , drop = FALSE]
        rhs_z <- lookup_coef(paste0("intercept_", zn), paste0("\\beta_{0,", tex_var(zn), "}"))
        for (k in seq_len(nrow(z_sub))) {
          pr <- z_sub$rhs[k]
          bz_s <- lookup_coef(paste0(zn, "_on_", pr), paste0("\\beta_{", tex_var(zn), ",", tex_var(pr), "}"))
          pr_tex <- if (pr %in% outcomes) paste0("\\mu_{", tex_var(pr), ",i}") else paste0(tex_var(pr), "_i")
          rhs_z <- paste0(rhs_z, signed_term(bz_s, pr_tex))
        }
        vz_s <- lookup_coef(paste0("var_", zn), paste0("\\sigma^2_{", tex_var(zn), "}"))
        eq_lines <- c(eq_lines, paste0(tex_var(zn), "_i &= ", rhs_z, " + e_{", tex_var(zn), ",i}, \\quad e_{", tex_var(zn), ",i} \\sim \\mathcal{N}(0, ", vz_s, ")"))
      }
    }
  }

  body <- paste(eq_lines, collapse = " \\\\\n")
  latex <- switch(
    wrap,
    align = paste0("\\begin{align*}\n", body, "\n\\end{align*}"),
    aligned = paste0("\\begin{aligned}\n", body, "\n\\end{aligned}"),
    dollar = paste0("$$\n\\begin{aligned}\n", body, "\n\\end{aligned}\n$$"),
    none = body
  )

  out <- list(
    latex = latex,
    lines = eq_lines,
    hierarchy = lvl_type,
    substituted = isTRUE(substitute) && !is.null(fit_obj)
  )
  class(out) <- c("DSEMequations", "list")
  out
}

#' @exportS3Method base::print
print.DSEMequations <- function(x, ...) {
  cat(x$latex, "\n")
  invisible(x)
}

#' @exportS3Method base::as.character
as.character.DSEMequations <- function(x, ...) x$latex
