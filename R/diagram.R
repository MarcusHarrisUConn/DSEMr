#' Render publication-style visual path diagrams for DSEMr models and fits
#'
#' Generates multilevel dynamic structural equation path diagrams from a
#' `DSEMmodel`, `DSEMspec`, or `DSEMfit` object. Follows standard DSEM visual
#' conventions: rectangles denote observed variables, circles/ellipses denote
#' latent variables and random effects, dashed arrows denote lagged transitions
#' (\eqn{t - k \to t}), filled dots on within-level paths indicate random slopes
#' or random log residual variances, and multilevel models are separated into
#' clear Within and Between (Level 2 / Level 3) bands. Supports direct base-R
#' rendering in RStudio (`"plot"`), standalone vector `"svg"` (for HTML,
#' Quarto, and Shiny apps), `"mermaid"`, and Graphviz `"dot"`.
#'
#' @param object A `DSEMfit`, `DSEMmodel`, or `DSEMspec` object.
#' @param format Output format: `"plot"` (draws in current graphics device),
#'   `"svg"`, `"mermaid"`, or `"dot"`.
#' @param standardized Logical; if `TRUE` and `object` is a `DSEMfit`, labels
#'   edges with `STDYX` standardized estimates instead of unstandardized
#'   estimates.
#' @param show_estimates Logical; if `TRUE` and `object` is a `DSEMfit`,
#'   annotates paths with posterior point estimates (and `*` when the 95% CrI
#'   excludes zero).
#' @param point Point estimate convention when `show_estimates = TRUE`:
#'   `"median"` (default) or `"mean"`.
#' @param digits Decimal places for edge annotations (default `2`).
#' @param main Optional diagram title.
#' @return For `format = "plot"`, invisibly returns a `DSEMdiagram` list
#'   containing `$nodes`, `$edges`, `$svg`, `$mermaid`, and `$dot`. For
#'   `"svg"`, `"mermaid"`, or `"dot"`, returns a `DSEMdiagram` object whose
#'   [print()] and [as.character()] methods yield the requested diagram source.
#' @examples
#' dat <- simulate_dsem(n = 60, ar = 0.5, sigma = 0.7, seed = 41)
#' fit <- dsem("y ~ lag(y, 1)", dat, chains = 2, iter = 160, warmup = 80, seed = 41)
#' dsem_diagram(fit, format = "mermaid")
#' dsem_diagram(fit, format = "svg")
#' @export
dsem_diagram <- function(object,
                         format = c("plot", "svg", "mermaid", "dot"),
                         standardized = FALSE,
                         show_estimates = TRUE,
                         point = c("median", "mean"),
                         digits = 2L,
                         main = NULL) {
  format <- match.arg(format)
  point <- match.arg(point)
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

  graph <- .dsem_build_diagram_graph(
    model = model, fit_obj = fit_obj,
    standardized = standardized, show_estimates = show_estimates,
    point = point, digits = digits, main = main
  )

  svg_txt <- .dsem_diagram_to_svg(graph)
  mermaid_txt <- .dsem_diagram_to_mermaid(graph)
  dot_txt <- .dsem_diagram_to_dot(graph)

  out <- list(
    format = format,
    title = graph$title,
    nodes = graph$nodes,
    edges = graph$edges,
    svg = svg_txt,
    mermaid = mermaid_txt,
    dot = dot_txt,
    text = switch(format, svg = svg_txt, mermaid = mermaid_txt, dot = dot_txt, plot = mermaid_txt)
  )
  class(out) <- c("DSEMdiagram", "list")

  if (format == "plot") {
    .dsem_diagram_plot_base(graph)
    return(invisible(out))
  }
  out
}

.dsem_build_diagram_graph <- function(model, fit_obj, standardized, show_estimates, point, digits, main) {
  terms <- model$terms
  lvl_type <- model$level_type
  is_ml <- lvl_type %in% c("twolevel", "threelevel", "crossclassified")

  est_lookup <- list()
  sig_lookup <- list()
  if (isTRUE(show_estimates) && !is.null(fit_obj)) {
    if (isTRUE(standardized)) {
      sdf <- dsem_standardize(fit_obj, type = "stdyx")
      vals <- if (point == "median") sdf$q50 else sdf$mean
      for (i in seq_len(nrow(sdf))) {
        est_lookup[[sdf$parameter[i]]] <- vals[i]
        sig_lookup[[sdf$parameter[i]]] <- is.finite(sdf$q_lower[i]) && is.finite(sdf$q_upper[i]) &&
          (sdf$q_lower[i] > 0 || sdf$q_upper[i] < 0)
      }
    } else {
      edf <- fit_obj$estimates
      vals <- if (point == "median" && "q50" %in% names(edf)) edf$q50 else edf$mean
      for (i in seq_len(nrow(edf))) {
        est_lookup[[edf$parameter[i]]] <- vals[i]
        sig_lookup[[edf$parameter[i]]] <- is.finite(edf$q2.5[i]) && is.finite(edf$q97.5[i]) &&
          (edf$q2.5[i] > 0 || edf$q97.5[i] < 0)
      }
    }
  }

  fmt_edge <- function(cands, fallback = "") {
    if (!length(est_lookup)) return(list(label = fallback, sig = NA))
    hit <- intersect(cands, names(est_lookup))
    if (!length(hit)) return(list(label = fallback, sig = NA))
    v <- est_lookup[[hit[1L]]]
    sg <- isTRUE(sig_lookup[[hit[1L]]])
    lbl <- paste0(formatC(v, format = "f", digits = digits), if (sg) "*" else "")
    list(label = lbl, sig = sg)
  }

  nodes <- data.frame(
    id = character(), label = character(), shape = character(),
    band = character(), x = numeric(), y = numeric(),
    stringsAsFactors = FALSE
  )
  edges <- data.frame(
    from = character(), to = character(), label = character(),
    style = character(), random_dot = logical(), dot_label = character(),
    sig = logical(), stringsAsFactors = FALSE
  )

  add_node <- function(id, label, shape, band, x, y) {
    if (!id %in% nodes$id) {
      nodes <<- rbind(nodes, data.frame(
        id = id, label = label, shape = shape, band = band, x = x, y = y,
        stringsAsFactors = FALSE
      ))
    }
  }
  add_edge <- function(from, to, label = "", style = "solid", random_dot = FALSE, dot_label = "", sig = NA) {
    edges <<- rbind(edges, data.frame(
      from = from, to = to, label = label, style = style,
      random_dot = isTRUE(random_dot), dot_label = dot_label,
      sig = sig, stringsAsFactors = FALSE
    ))
  }

  w_meas <- terms[terms$op == "=~" & terms$level == "within", , drop = FALSE]
  w_regs <- terms[terms$op == "~" & terms$rhs != "1" & terms$level == "within", , drop = FALSE]
  w_vars <- terms[terms$op == "~~" & terms$level == "within" & terms$lhs == terms$rhs, , drop = FALSE]
  b_regs <- terms[terms$op == "~" & terms$rhs != "1" & terms$level %in% c("between", "between_2", "between_3"), , drop = FALSE]
  b_covs <- terms[terms$op == "~~" & terms$level %in% c("between", "between_2", "between_3") & terms$lhs != terms$rhs, , drop = FALSE]

  # Vertical bands: if multilevel, Within occupies y in [0.56, 0.92], Between occupies y in [0.10, 0.44]
  y_w_top <- if (is_ml) 0.84 else 0.72
  y_w_mid <- if (is_ml) 0.70 else 0.50
  y_w_bot <- if (is_ml) 0.58 else 0.24

  if (nrow(w_meas) > 0L) {
    w_factors <- unique(w_meas$lhs)
    items <- unique(w_meas$rhs)
    nf <- length(w_factors)
    ni <- length(items)

    # Lagged factor(s) on left, current factor(s) in center, indicators on right/bottom
    f_ys <- if (nf == 1L) y_w_mid else seq(y_w_top, y_w_bot + 0.04, length.out = nf)
    for (k in seq_len(nf)) {
      fn <- w_factors[k]
      add_node(paste0("w_", fn, "_lag"), paste0(fn, "(t-1)"), "circle", "within", 0.20, f_ys[k])
      add_node(paste0("w_", fn), paste0(fn, "(t)"), "circle", "within", 0.52, f_ys[k])
    }
    it_ys <- if (ni == 1L) y_w_mid else seq(y_w_top + 0.03, y_w_bot, length.out = ni)
    for (j in seq_len(ni)) {
      add_node(paste0("w_", items[j]), items[j], "rect", "within", 0.84, it_ys[j])
    }
    for (r in seq_len(nrow(w_meas))) {
      fn <- w_meas$lhs[r]; it <- w_meas$rhs[r]
      info <- fmt_edge(c(paste0("loading_", fn, "_", it), paste0("loading_", it)), "")
      add_edge(paste0("w_", fn), paste0("w_", it), info$label, "solid", FALSE, "", info$sig)
    }
    for (r in seq_len(nrow(w_regs))) {
      lhs <- w_regs$lhs[r]; rhs <- w_regs$rhs[r]; lg <- w_regs$lag[r]; lbl <- w_regs$label[r]
      from_id <- if (lg > 0L && rhs %in% w_factors) paste0("w_", rhs, "_lag") else paste0("w_", rhs)
      to_id <- paste0("w_", lhs)
      cands <- c(
        if (nzchar(lbl)) c(paste0("within_std_", lbl), paste0("mean_", lbl), lbl) else character(),
        paste0("ar1_", lhs), paste0(lhs, "_on_lag", lg, "_", rhs), paste0("lagged_factor_effect_", lhs)
      )
      info <- fmt_edge(cands, if (nzchar(lbl)) lbl else "")
      add_edge(from_id, to_id, info$label, if (lg > 0L) "dashed" else "solid", w_regs$random[r], lbl, info$sig)
    }
  } else {
    outcomes <- unique(w_regs$lhs)
    exog_w <- setdiff(unique(w_regs$rhs[w_regs$lag == 0L]), outcomes)
    d <- length(outcomes)
    y_pos <- if (d == 1L) y_w_mid else seq(y_w_top, y_w_bot, length.out = d)

    for (j in seq_len(d)) {
      onm <- outcomes[j]
      add_node(paste0("w_", onm, "_lag"), paste0(onm, "(t-1)"), "rect", "within", 0.22, y_pos[j])
      add_node(paste0("w_", onm), paste0(onm, "(t)"), "rect", "within", 0.72, y_pos[j])
    }
    if (length(exog_w)) {
      ex_xs <- seq(0.36, 0.56, length.out = length(exog_w))
      for (k in seq_along(exog_w)) {
        add_node(paste0("w_", exog_w[k]), paste0(exog_w[k], "(t)"), "rect", "within", ex_xs[k], y_w_top + 0.06)
      }
    }
    for (r in seq_len(nrow(w_regs))) {
      lhs <- w_regs$lhs[r]; rhs <- w_regs$rhs[r]; lg <- w_regs$lag[r]; lbl <- w_regs$label[r]
      from_id <- if (lg > 0L) paste0("w_", rhs, "_lag") else paste0("w_", rhs)
      to_id <- paste0("w_", lhs)
      cands <- c(
        if (nzchar(lbl)) c(paste0("within_std_", lbl), paste0("mean_", lbl), paste0("fixed_", lbl), lbl) else character(),
        if (lg > 0L) c(paste0("within_std_lag", lg, "_", rhs), paste0("lag", lg, "_", rhs), paste0("mean_lag", lg, "_", rhs), paste0(lhs, "_on_lag", lg, "_", rhs))
        else c(rhs, paste0(lhs, "_on_", rhs))
      )
      info <- fmt_edge(cands, if (nzchar(lbl)) lbl else "")
      add_edge(from_id, to_id, info$label, if (lg > 0L) "dashed" else "solid", w_regs$random[r], lbl, info$sig)
    }
    # Random log-variance indicator node (residual arrow with random-effect dot)
    lv_rows <- w_vars[w_vars$random, , drop = FALSE]
    for (k in seq_len(nrow(lv_rows))) {
      onm <- lv_rows$lhs[k]
      lbl <- if (nzchar(lv_rows$label[k])) lv_rows$label[k] else "logv"
      add_node(paste0("w_err_", onm), "e(t)", "circle", "within", 0.90, y_pos[match(onm, outcomes)])
      info <- fmt_edge(c(paste0("mean_", lbl), paste0("fixed_", lbl)), lbl)
      add_edge(paste0("w_err_", onm), paste0("w_", onm), info$label, "solid", TRUE, lbl, info$sig)
    }
  }

  # Between-level nodes & edges (Level 2 / Level 3)
  if (is_ml) {
    re_cands <- unique(c(
      unique(w_regs$lhs),
      w_regs$label[w_regs$random & nzchar(w_regs$label)],
      w_vars$label[w_vars$random & nzchar(w_vars$label)]
    ))
    if (!length(re_cands)) re_cands <- unique(terms$lhs[terms$level != "within"])
    distal_z <- setdiff(unique(b_regs$lhs), re_cands)
    exog_w_b <- setdiff(unique(b_regs$rhs), c(re_cands, distal_z))

    # Position between exogenous predictors on left, random effects in center, distal outcomes on right
    if (length(exog_w_b)) {
      w_ys <- if (length(exog_w_b) == 1L) 0.26 else seq(0.38, 0.14, length.out = length(exog_w_b))
      for (k in seq_along(exog_w_b)) {
        add_node(paste0("b_", exog_w_b[k]), exog_w_b[k], "rect", "between", 0.14, w_ys[k])
      }
    }
    re_xs <- if (length(re_cands) == 1L) 0.50 else seq(0.34, 0.66, length.out = length(re_cands))
    re_ys <- if (length(re_cands) <= 2L) rep(0.26, length(re_cands)) else seq(0.38, 0.14, length.out = length(re_cands))
    for (k in seq_along(re_cands)) {
      rn <- re_cands[k]
      # Spread vertically when predictors/distal exist so edges don't overlap
      xk <- if (length(exog_w_b) || length(distal_z)) 0.50 else re_xs[k]
      yk <- if (length(exog_w_b) || length(distal_z)) re_ys[k] else 0.26
      add_node(paste0("b_", rn), rn, "circle", "between", xk, yk)
    }
    if (length(distal_z)) {
      z_ys <- if (length(distal_z) == 1L) 0.26 else seq(0.38, 0.14, length.out = length(distal_z))
      for (k in seq_along(distal_z)) {
        add_node(paste0("b_", distal_z[k]), distal_z[k], "rect", "between", 0.86, z_ys[k])
      }
    }
    for (r in seq_len(nrow(b_regs))) {
      lhs <- b_regs$lhs[r]; rhs <- b_regs$rhs[r]
      add_node(paste0("b_", lhs), lhs, if (lhs %in% re_cands) "circle" else "rect", "between", 0.65, 0.26)
      add_node(paste0("b_", rhs), rhs, if (rhs %in% re_cands) "circle" else "rect", "between", 0.25, 0.26)
      cands <- c(paste0(lhs, "_on_", rhs), paste0("l3_", lhs, "_on_", rhs), paste0("l2_", lhs, "_on_", rhs))
      info <- fmt_edge(cands, "")
      add_edge(paste0("b_", rhs), paste0("b_", lhs), info$label, "solid", FALSE, "", info$sig)
    }
    # Add up to 3 between covariances for clean visual clarity
    if (nrow(b_covs) > 0L) {
      for (r in seq_len(min(nrow(b_covs), 3L))) {
        a <- b_covs$lhs[r]; b <- b_covs$rhs[r]
        if (paste0("b_", a) %in% nodes$id && paste0("b_", b) %in% nodes$id) {
          cands <- c(paste0("cov_", a, "_", b), paste0("cov_", b, "_", a),
                     paste0("l3_cov_", a, "_", b), paste0("l3_cov_", b, "_", a))
          info <- fmt_edge(cands, "")
          add_edge(paste0("b_", a), paste0("b_", b), info$label, "cov", FALSE, "", info$sig)
        }
      }
    }
  }

  title_str <- if (!is.null(main)) {
    main
  } else {
    paste0(
      switch(lvl_type, single = "Single-Level", twolevel = "Two-Level",
             threelevel = "Three-Level (DSEM3)", crossclassified = "Cross-Classified", lvl_type),
      " DSEM Path Diagram",
      if (isTRUE(standardized) && !is.null(fit_obj)) " (STDYX)" else ""
    )
  }

  list(title = title_str, level_type = lvl_type, is_ml = is_ml, nodes = nodes, edges = edges)
}

.dsem_diagram_plot_base <- function(graph) {
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(old_par), add = TRUE)
  graphics::par(mar = c(1.2, 1.2, 2.6, 1.2), xaxs = "i", yaxs = "i")
  graphics::plot.new()
  graphics::plot.window(xlim = c(0, 1), ylim = c(0, 1))
  graphics::title(main = graph$title, cex.main = 1.05)

  # Draw multilevel swimlanes
  if (graph$is_ml) {
    graphics::rect(0.02, 0.52, 0.98, 0.96, border = "#94a3b8", lty = 2, col = grDevices::adjustcolor("#f8fafc", 0.6))
    graphics::text(0.04, 0.935, "WITHIN (Level 1)", adj = c(0, 1), cex = 0.78, font = 2, col = "#475569")
    graphics::rect(0.02, 0.04, 0.98, 0.48, border = "#94a3b8", lty = 2, col = grDevices::adjustcolor("#eff6ff", 0.45))
    b_lbl <- if (graph$level_type == "threelevel") "BETWEEN (Level 2 & Level 3)" else "BETWEEN (Level 2)"
    graphics::text(0.04, 0.455, b_lbl, adj = c(0, 1), cex = 0.78, font = 2, col = "#475569")
  }

  nodes <- graph$nodes
  edges <- graph$edges

  # Draw edges first
  for (i in seq_len(nrow(edges))) {
    e <- edges[i, ]
    n1 <- nodes[nodes$id == e$from, , drop = FALSE]
    n2 <- nodes[nodes$id == e$to, , drop = FALSE]
    if (!nrow(n1) || !nrow(n2)) next
    x1 <- n1$x[1L]; y1 <- n1$y[1L]; x2 <- n2$x[1L]; y2 <- n2$y[1L]
    dx <- x2 - x1; dy <- y2 - y1
    dist <- sqrt(dx^2 + dy^2)
    if (dist < 1e-6) next
    ux <- dx / dist; uy <- dy / dist
    r_pad <- 0.068
    sx <- x1 + ux * r_pad; sy <- y1 + uy * r_pad
    ex <- x2 - ux * r_pad; ey <- y2 - uy * r_pad

    col_e <- if (isTRUE(e$sig)) "#1d4ed8" else if (identical(e$sig, FALSE)) "#64748b" else "#334155"
    lwd_e <- if (isTRUE(e$sig)) 2.1 else 1.3
    lty_e <- if (e$style == "dashed") 2 else if (e$style == "cov") 3 else 1

    if (e$style == "cov") {
      # Gentle arc between between-level random effects
      mx <- 0.5 * (x1 + x2) - 0.075 * uy
      my <- 0.5 * (y1 + y2) + 0.075 * ux
      tt <- seq(0, 1, length.out = 25)
      bx <- (1 - tt)^2 * sx + 2 * (1 - tt) * tt * mx + tt^2 * ex
      by <- (1 - tt)^2 * sy + 2 * (1 - tt) * tt * my + tt^2 * ey
      graphics::lines(bx, by, col = col_e, lwd = lwd_e, lty = lty_e)
      if (nzchar(e$label)) {
        graphics::text(mx, my, e$label, cex = 0.72, col = col_e)
      }
    } else {
      graphics::arrows(sx, sy, ex, ey, length = 0.09, col = col_e, lwd = lwd_e, lty = lty_e)
      mx <- 0.5 * (sx + ex); my <- 0.5 * (sy + ey)
      if (isTRUE(e$random_dot)) {
        graphics::points(mx, my, pch = 19, cex = 1.25, col = "#0f172a")
      }
      if (nzchar(e$label)) {
        off_y <- if (abs(dy) < 0.05) 0.035 else 0.025
        graphics::text(mx, my + off_y, e$label, cex = 0.76, font = if (isTRUE(e$sig)) 2 else 1, col = col_e)
      }
    }
  }

  # Draw nodes on top
  theta_c <- seq(0, 2 * pi, length.out = 60)
  for (i in seq_len(nrow(nodes))) {
    nd <- nodes[i, ]
    if (nd$shape == "circle") {
      rx <- 0.058; ry <- 0.048
      graphics::polygon(nd$x + rx * cos(theta_c), nd$y + ry * sin(theta_c),
                        col = "#ffffff", border = "#1e3a8a", lwd = 1.6)
    } else {
      rx <- 0.068; ry <- 0.042
      graphics::rect(nd$x - rx, nd$y - ry, nd$x + rx, nd$y + ry,
                     col = "#ffffff", border = "#0f172a", lwd = 1.6)
    }
    graphics::text(nd$x, nd$y, nd$label, cex = 0.80, font = 2, col = "#0f172a")
  }
  invisible(TRUE)
}

.dsem_diagram_to_svg <- function(graph) {
  W <- 760; H <- if (graph$is_ml) 480 else 280
  sx <- function(x) round(x * W, 1)
  sy <- function(y) round((1 - y) * H, 1)
  esc <- function(s) gsub(">", "&gt;", gsub("<", "&lt;", gsub("&", "&amp;", s, fixed = TRUE), fixed = TRUE), fixed = TRUE)

  parts <- c(
    sprintf('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="100%%" style="max-width:%dpx;font-family:system-ui,-apple-system,sans-serif;">', W, H, W),
    '  <defs>',
    '    <marker id="dsem-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#1d4ed8"/></marker>',
    '    <marker id="dsem-arrow-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b"/></marker>',
    '  </defs>',
    sprintf('  <rect x="0" y="0" width="%d" height="%d" rx="10" fill="#ffffff" stroke="#e2e8f0"/>', W, H),
    sprintf('  <text x="%d" y="24" text-anchor="middle" font-size="14" font-weight="600" fill="#0f172a">%s</text>', W / 2, esc(graph$title))
  )

  if (graph$is_ml) {
    parts <- c(
      parts,
      sprintf('  <rect x="16" y="38" width="%d" height="%d" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-dasharray="5,4"/>', W - 32, round(H * 0.42)),
      '  <text x="28" y="56" font-size="11" font-weight="700" fill="#475569">WITHIN (Level 1)</text>',
      sprintf('  <rect x="16" y="%d" width="%d" height="%d" rx="8" fill="#eff6ff" fill-opacity="0.5" stroke="#cbd5e1" stroke-dasharray="5,4"/>', round(H * 0.52), W - 32, round(H * 0.43)),
      sprintf('  <text x="28" y="%d" font-size="11" font-weight="700" fill="#475569">%s</text>', round(H * 0.52) + 18, if (graph$level_type == "threelevel") "BETWEEN (Level 2 &amp; Level 3)" else "BETWEEN (Level 2)")
    )
  }

  nodes <- graph$nodes; edges <- graph$edges
  for (i in seq_len(nrow(edges))) {
    e <- edges[i, ]
    n1 <- nodes[nodes$id == e$from, , drop = FALSE]
    n2 <- nodes[nodes$id == e$to, , drop = FALSE]
    if (!nrow(n1) || !nrow(n2)) next
    x1 <- sx(n1$x[1L]); y1 <- sy(n1$y[1L]); x2 <- sx(n2$x[1L]); y2 <- sy(n2$y[1L])
    dx <- x2 - x1; dy <- y2 - y1; dist <- sqrt(dx^2 + dy^2)
    if (dist < 1) next
    ux <- dx / dist; uy <- dy / dist
    p1x <- round(x1 + ux * 46, 1); p1y <- round(y1 + uy * 26, 1)
    p2x <- round(x2 - ux * 48, 1); p2y <- round(y2 - uy * 28, 1)
    col_e <- if (isTRUE(e$sig)) "#1d4ed8" else "#64748b"
    mk <- if (isTRUE(e$sig)) "url(#dsem-arrow)" else "url(#dsem-arrow-muted)"
    dash <- if (e$style == "dashed") ' stroke-dasharray="6,4"' else if (e$style == "cov") ' stroke-dasharray="3,3"' else ""
    sw <- if (isTRUE(e$sig)) "2.2" else "1.5"

    if (e$style == "cov") {
      cx <- round(0.5 * (p1x + p2x) - 36 * uy, 1)
      cy <- round(0.5 * (p1y + p2y) + 36 * ux, 1)
      parts <- c(parts, sprintf('  <path d="M %.1f %.1f Q %.1f %.1f %.1f %.1f" fill="none" stroke="%s" stroke-width="%s"%s/>', p1x, p1y, cx, cy, p2x, p2y, col_e, sw, dash))
      if (nzchar(e$label)) {
        parts <- c(parts, sprintf('  <text x="%.1f" y="%.1f" text-anchor="middle" font-size="11" fill="%s">%s</text>', cx, cy - 4, col_e, esc(e$label)))
      }
    } else {
      parts <- c(parts, sprintf('  <line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="%s"%s marker-end="%s"/>', p1x, p1y, p2x, p2y, col_e, sw, dash, mk))
      mx <- round(0.5 * (p1x + p2x), 1); my <- round(0.5 * (p1y + p2y), 1)
      if (isTRUE(e$random_dot)) {
        parts <- c(parts, sprintf('  <circle cx="%.1f" cy="%.1f" r="5" fill="#0f172a"/>', mx, my))
      }
      if (nzchar(e$label)) {
        parts <- c(parts, sprintf('  <text x="%.1f" y="%.1f" text-anchor="middle" font-size="11.5" font-weight="%s" fill="%s">%s</text>', mx, my - 9, if (isTRUE(e$sig)) "700" else "500", col_e, esc(e$label)))
      }
    }
  }

  for (i in seq_len(nrow(nodes))) {
    nd <- nodes[i, ]
    cx <- sx(nd$x); cy <- sy(nd$y)
    if (nd$shape == "circle") {
      parts <- c(parts, sprintf('  <ellipse cx="%.1f" cy="%.1f" rx="42" ry="22" fill="#ffffff" stroke="#1e3a8a" stroke-width="1.8"/>', cx, cy))
    } else {
      parts <- c(parts, sprintf('  <rect x="%.1f" y="%.1f" width="90" height="36" rx="5" fill="#ffffff" stroke="#0f172a" stroke-width="1.8"/>', cx - 45, cy - 18))
    }
    parts <- c(parts, sprintf('  <text x="%.1f" y="%.1f" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="600" fill="#0f172a">%s</text>', cx, cy, esc(nd$label)))
  }

  paste(c(parts, "</svg>"), collapse = "\n")
}

.dsem_diagram_to_mermaid <- function(graph) {
  lines <- c("flowchart LR")
  nodes <- graph$nodes; edges <- graph$edges
  fmt_node <- function(nd) {
    if (nd$shape == "circle") sprintf('  %s(("%s"))', nd$id, nd$label)
    else sprintf('  %s["%s"]', nd$id, nd$label)
  }
  w_nodes <- nodes[nodes$band == "within", , drop = FALSE]
  b_nodes <- nodes[nodes$band == "between", , drop = FALSE]
  if (nrow(w_nodes)) {
    lines <- c(lines, '  subgraph Within["Within (Level 1)"]')
    for (i in seq_len(nrow(w_nodes))) lines <- c(lines, paste0("  ", fmt_node(w_nodes[i, ])))
    lines <- c(lines, "  end")
  }
  if (nrow(b_nodes)) {
    b_lbl <- if (graph$level_type == "threelevel") "Between (Level 2 & Level 3)" else "Between (Level 2)"
    lines <- c(lines, sprintf('  subgraph Between["%s"]', b_lbl))
    for (i in seq_len(nrow(b_nodes))) lines <- c(lines, paste0("  ", fmt_node(b_nodes[i, ])))
    lines <- c(lines, "  end")
  }
  for (i in seq_len(nrow(edges))) {
    e <- edges[i, ]
    lbl <- if (nzchar(e$label)) {
      if (isTRUE(e$random_dot)) paste0(e$label, " [random]") else e$label
    } else if (isTRUE(e$random_dot)) {
      "random"
    } else {
      ""
    }
    arrow <- if (e$style == "dashed") "-.->" else if (e$style == "cov") "<-->" else "-->"
    if (nzchar(lbl)) {
      lines <- c(lines, sprintf('  %s %s|"%s"| %s', e$from, arrow, lbl, e$to))
    } else {
      lines <- c(lines, sprintf("  %s %s %s", e$from, arrow, e$to))
    }
  }
  paste(lines, collapse = "\n")
}

.dsem_diagram_to_dot <- function(graph) {
  lines <- c(
    "digraph DSEM {",
    "  rankdir=LR;",
    sprintf('  label="%s"; labelloc="t"; fontsize=14;', gsub('"', '\\"', graph$title, fixed = TRUE)),
    '  node [fontname="Helvetica", fontsize=11];',
    '  edge [fontname="Helvetica", fontsize=10];'
  )
  nodes <- graph$nodes; edges <- graph$edges
  for (i in seq_len(nrow(nodes))) {
    nd <- nodes[i, ]
    sh <- if (nd$shape == "circle") "ellipse" else "box"
    lines <- c(lines, sprintf('  "%s" [label="%s", shape=%s];', nd$id, nd$label, sh))
  }
  for (i in seq_len(nrow(edges))) {
    e <- edges[i, ]
    attrs <- character()
    if (nzchar(e$label)) attrs <- c(attrs, sprintf('label="%s"', e$label))
    if (e$style == "dashed") attrs <- c(attrs, "style=dashed")
    if (e$style == "cov") attrs <- c(attrs, 'dir=both, style=dotted')
    if (isTRUE(e$sig)) attrs <- c(attrs, 'color="#1d4ed8", penwidth=1.8')
    attr_s <- if (length(attrs)) paste0(" [", paste(attrs, collapse = ", "), "]") else ""
    lines <- c(lines, sprintf('  "%s" -> "%s"%s;', e$from, e$to, attr_s))
  }
  paste(c(lines, "}"), collapse = "\n")
}

#' @exportS3Method base::print
print.DSEMdiagram <- function(x, ...) {
  cat(x$text, "\n")
  invisible(x)
}

#' @exportS3Method base::as.character
as.character.DSEMdiagram <- function(x, ...) x$text
