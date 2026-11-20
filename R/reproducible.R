#' Create a declarative, GUI-ready DSEM specification (`DSEMspec`)
#'
#' Constructs a validated, structured model specification designed for
#' programmatic pipelines and interactive drag-and-drop Shiny builders (such as
#' `mlmr`-style interfaces). A `DSEMspec` compiles cleanly to a [dsem_model()],
#' exports both `lavaan`-style and standard section-based SEM `.inp` syntax via
#' [dsem_to_lavaan()] and [dsem_to_mplus()], previews live path diagrams
#' ([dsem_diagram()]) and LaTeX equations ([dsem_equations()]), and generates
#' standalone reproducible R scripts ([dsem_code()]).
#'
#' @param outcomes Character vector of endogenous outcome variable names
#'   (observed outcomes or factor names when `factors` is supplied).
#' @param lags Positive integer vector of autoregressive/cross-lag orders
#'   (default `1L`).
#' @param predictors_within Optional character vector of time-varying Level-1
#'   covariates.
#' @param predictors_between Optional character vector of Level-2 (or Level-3
#'   for three-level) cluster-level predictors of random effects.
#' @param predictors_between_l2 Optional character vector of Level-2 predictors
#'   in a three-level DSEM3 model.
#' @param distal Optional character vector of cluster-level distal outcomes
#'   regressed on random effects (e.g., `z ON y s logv`).
#' @param factors Optional named list mapping latent factor names to character
#'   vectors of observed indicators (e.g., `list(f = c("y1", "y2", "y3", "y4"))`).
#' @param id Optional cluster identifier column name(s): `NULL` (single-level),
#'   length 1 (two-level or cross-classified), or length 2
#'   `c("level2", "level3")` (three-level DSEM3).
#' @param time Optional time index column name.
#' @param time_cluster Optional occasion cluster column for cross-classified
#'   models.
#' @param level_type Hierarchy: `"auto"`, `"single"`, `"twolevel"`,
#'   `"threelevel"`, or `"crossclassified"`.
#' @param random_slopes Logical or character vector of slope names that vary
#'   randomly across clusters (default `TRUE` when multilevel).
#' @param random_logv Logical; if `TRUE`, models a cluster-specific random log
#'   residual variance (`logv | y`).
#' @param residual_dynamics Logical; if `TRUE`, uses Residual DSEM (`RDSEM`)
#'   autoregressive residuals.
#' @param categorical Optional character vector of binary/ordinal variable names.
#' @param censored Optional Tobit censoring specification (character vector or
#'   named list, passed to [dsem_model()]).
#' @param family Outcome family (`"gaussian"`, `"binary"`, or `"ordinal"`).
#' @param tinterval Optional positive numeric `TINTERVAL` bin width.
#' @param centering Centering policy (`"latent"`, `"groupmean"`, `"grandmean"`,
#'   or `"none"`).
#' @param prior_mode Prior specification (`"default"`, `"mplus"` for diffuse
#'   priors, or `"penalized"` for Bayesian Lasso PSEM shrinkage).
#' @param chains,iter,warmup,thin,seed Default MCMC sampler settings stored
#'   with the specification for reproducible script generation.
#' @return A `DSEMspec` object.
#' @examples
#' spec <- dsem_spec(
#'   outcomes = "y",
#'   lags = 1L,
#'   predictors_between = "w",
#'   distal = "z",
#'   id = "subject",
#'   time = "time",
#'   random_slopes = TRUE,
#'   random_logv = TRUE,
#'   tinterval = 1
#' )
#' print(spec)
#' dsem_to_mplus(spec)
#' dsem_code(spec, data_name = "ema_data")
#' @export
dsem_spec <- function(outcomes,
                      lags = 1L,
                      predictors_within = character(),
                      predictors_between = character(),
                      predictors_between_l2 = character(),
                      distal = character(),
                      factors = list(),
                      id = NULL,
                      time = NULL,
                      time_cluster = NULL,
                      level_type = c("auto", "single", "twolevel", "threelevel", "crossclassified"),
                      random_slopes = NULL,
                      random_logv = FALSE,
                      residual_dynamics = FALSE,
                      categorical = character(),
                      censored = NULL,
                      family = c("gaussian", "binary", "ordinal"),
                      tinterval = NULL,
                      centering = c("latent", "groupmean", "grandmean", "none"),
                      prior_mode = c("default", "mplus", "penalized"),
                      chains = 4L,
                      iter = 2000L,
                      warmup = floor(iter / 2),
                      thin = 1L,
                      seed = 20260813L) {
  if (!is.character(outcomes) || !length(outcomes) || anyNA(outcomes) || any(!nzchar(outcomes))) {
    .dsem_abort("`outcomes` must be a non-empty character vector of variable names.")
  }
  lags <- as.integer(lags)
  if (!length(lags) || anyNA(lags) || any(lags < 1L)) {
    .dsem_abort("`lags` must contain positive integer lag orders.")
  }
  level_type <- match.arg(level_type)
  family <- match.arg(family)
  centering <- match.arg(centering)
  prior_mode <- match.arg(prior_mode)
  if (!is.list(factors)) .dsem_abort("`factors` must be a named list of indicator vectors.")
  if (length(factors) && (is.null(names(factors)) || any(!nzchar(names(factors))))) {
    .dsem_abort("Every element of `factors` must have a non-empty factor name.")
  }
  if (!is.null(id) && (!is.character(id) || length(id) > 2L || anyNA(id) || any(!nzchar(id)))) {
    .dsem_abort("`id` must be NULL or one or two column names.")
  }
  if (level_type == "auto") {
    level_type <- if (!is.null(time_cluster)) {
      "crossclassified"
    } else if (length(id) == 2L) {
      "threelevel"
    } else if (length(id) == 1L) {
      "twolevel"
    } else {
      "single"
    }
  }
  if (is.null(random_slopes)) {
    random_slopes <- level_type %in% c("twolevel", "threelevel", "crossclassified")
  }
  if (length(categorical) && family == "gaussian") family <- "binary"

  spec <- list(
    outcomes = unique(outcomes),
    lags = sort(unique(lags)),
    predictors_within = unique(as.character(predictors_within)),
    predictors_between = unique(as.character(predictors_between)),
    predictors_between_l2 = unique(as.character(predictors_between_l2)),
    distal = unique(as.character(distal)),
    factors = factors,
    id = id,
    time = time,
    time_cluster = time_cluster,
    level_type = level_type,
    random_slopes = random_slopes,
    random_logv = isTRUE(random_logv),
    residual_dynamics = isTRUE(residual_dynamics),
    categorical = unique(as.character(categorical)),
    censored = .dsem_normalize_censored(censored),
    family = family,
    tinterval = tinterval,
    centering = centering,
    prior_mode = prior_mode,
    chains = .dsem_integer(chains, "chains", 1),
    iter = .dsem_integer(iter, "iter", 4),
    warmup = .dsem_integer(warmup, "warmup", 0),
    thin = .dsem_integer(thin, "thin", 1),
    seed = .dsem_integer(seed, "seed", 0)
  )
  class(spec) <- c("DSEMspec", "list")
  spec
}

#' Compile a `DSEMspec` object into a canonical `DSEMmodel`
#'
#' @param spec A `DSEMspec` object created by [dsem_spec()].
#' @param syntax Target intermediate syntax (`"lavaan"` or `"mplus"`).
#' @return A compiled `DSEMmodel` object ready for [dsem()].
#' @examples
#' spec <- dsem_spec(outcomes = "y", lags = 1L, time = "time")
#' as_dsem_model(spec)
#' @export
as_dsem_model <- function(spec, syntax = c("lavaan", "mplus")) {
  if (!inherits(spec, "DSEMspec")) .dsem_abort("`spec` must be a `DSEMspec` object.")
  syntax <- match.arg(syntax)
  syn_str <- if (syntax == "mplus") dsem_to_mplus(spec) else dsem_to_lavaan(spec)
  dsem_model(
    syn_str,
    syntax = syntax,
    id = spec$id,
    time = spec$time,
    time_cluster = spec$time_cluster,
    family = spec$family,
    level_type = spec$level_type,
    categorical = if (length(spec$categorical)) spec$categorical else NULL,
    censored = if (length(spec$censored)) spec$censored else NULL,
    tinterval = spec$tinterval,
    centering = spec$centering
  )
}

#' Export a DSEMr specification, model, or fit to `lavaan`-style DSEM syntax
#'
#' @param object A `DSEMspec`, `DSEMmodel`, or `DSEMfit` object.
#' @return Character string containing `DSEMr` `lavaan`-style model syntax.
#' @examples
#' spec <- dsem_spec(outcomes = "y", lags = 1L, id = "id", time = "time", random_logv = TRUE)
#' cat(dsem_to_lavaan(spec))
#' @export
dsem_to_lavaan <- function(object) {
  if (inherits(object, "DSEMfit")) object <- object$model
  if (inherits(object, "DSEMmodel")) {
    if (identical(object$syntax, "lavaan")) return(object$source)
    return(.dsem_terms_to_lavaan(object))
  }
  if (!inherits(object, "DSEMspec")) .dsem_abort("`object` must be a `DSEMspec`, `DSEMmodel`, or `DSEMfit`.")
  .dsem_spec_to_lavaan(object)
}

#' Export a DSEMr specification, model, or fit to Mplus `.inp` script syntax
#'
#' Translates a `DSEMspec`, `DSEMmodel`, or `DSEMfit` into a section-based
#' `.inp` model script for workflow interoperability and documentation
#' alongside reproducible R code.
#'
#' @param object A `DSEMspec`, `DSEMmodel`, or `DSEMfit` object.
#' @param data_file File name placed in the `DATA: FILE = ...;` block.
#' @param title Optional `TITLE:` string.
#' @return Character string containing section-based `.inp` syntax.
#' @examples
#' m <- dsem_model("y ~ lag(y, 1)", time = "time")
#' cat(dsem_to_mplus(m))
#' @export
dsem_to_mplus <- function(object, data_file = "data.dat", title = "DSEMr Generated SEM Input") {
  fit_obj <- NULL
  if (inherits(object, "DSEMfit")) {
    fit_obj <- object
    object <- object$model
  }
  if (inherits(object, "DSEMspec")) {
    return(.dsem_spec_to_mplus(object, data_file = data_file, title = title))
  }
  if (inherits(object, "DSEMmodel")) {
    if (identical(object$syntax, "mplus")) return(object$source)
    return(.dsem_model_to_mplus(object, fit_obj = fit_obj, data_file = data_file, title = title))
  }
  .dsem_abort("`object` must be a `DSEMspec`, `DSEMmodel`, or `DSEMfit`.")
}

#' Generate a complete, standalone reproducible R script for a DSEMr workflow
#'
#' Emits a clean, self-contained R script that reproduces model compilation,
#' estimation (with exact `chains`, `iter`, `warmup`, `thin`, `seed`, and
#' `control`), diagnostics, standardized estimates, Bayesian \eqn{R^2}, APA
#' tables, LaTeX equations, and visual path diagrams. Designed both for
#' manuscript replication packages and for code-export buttons in interactive
#' Shiny applications.
#'
#' @param object A `DSEMfit`, `DSEMmodel`, or `DSEMspec` object.
#' @param syntax Model specification syntax embedded in the script: `"lavaan"`
#'   or `"mplus"`.
#' @param data_name Name of the data frame object referenced in the generated
#'   script (default `"dat"`).
#' @param include_outputs Logical; if `TRUE`, includes calls to `summary()`,
#'   `dsem_dic()`, `dsem_r2()`, `dsem_table()`, `dsem_equations()`, and
#'   `dsem_diagram()`.
#' @param include_session_info Logical; if `TRUE`, appends `sessionInfo()`.
#' @return A `DSEMcode` object containing `$code` (character string) with
#'   [print()] and [as.character()] methods.
#' @examples
#' dat <- simulate_dsem(n = 60, ar = 0.45, sigma = 0.7, seed = 51)
#' fit <- dsem("y ~ lag(y, 1)", dat, chains = 2, iter = 160, warmup = 80, seed = 51)
#' dsem_code(fit, data_name = "dat")
#' @export
dsem_code <- function(object,
                      syntax = c("lavaan", "mplus"),
                      data_name = "dat",
                      include_outputs = TRUE,
                      include_session_info = FALSE) {
  syntax <- match.arg(syntax)
  if (!is.character(data_name) || length(data_name) != 1L || !nzchar(data_name)) {
    .dsem_abort("`data_name` must be a non-empty character string.")
  }

  if (inherits(object, "DSEMfit")) {
    model <- object$model
    chains <- object$sampler$chains
    iter <- object$sampler$iter
    warmup <- object$sampler$warmup
    thin <- object$sampler$thin
    seed <- object$master_seed
    prior_mode <- if (!is.null(object$prior$prior_mode)) object$prior$prior_mode else "default"
    model_str <- if (syntax == "mplus") dsem_to_mplus(object) else dsem_to_lavaan(object)
  } else if (inherits(object, "DSEMspec")) {
    model <- as_dsem_model(object, syntax = syntax)
    chains <- object$chains
    iter <- object$iter
    warmup <- object$warmup
    thin <- object$thin
    seed <- object$seed
    prior_mode <- object$prior_mode
    model_str <- if (syntax == "mplus") dsem_to_mplus(object) else dsem_to_lavaan(object)
  } else if (inherits(object, "DSEMmodel")) {
    model <- object
    chains <- 4L; iter <- 2000L; warmup <- 1000L; thin <- 1L; seed <- 20260813L
    prior_mode <- "default"
    model_str <- if (syntax == "mplus") dsem_to_mplus(object) else dsem_to_lavaan(object)
  } else {
    .dsem_abort("`object` must be a `DSEMfit`, `DSEMmodel`, or `DSEMspec`.")
  }

  fmt_chr_vec <- function(v) {
    if (is.null(v) || !length(v)) return("NULL")
    if (length(v) == 1L) sprintf('"%s"', v)
    else paste0("c(", paste(sprintf('"%s"', v), collapse = ", "), ")")
  }

  model_args <- c(
    sprintf('  syntax = "%s"', syntax),
    if (!is.null(model$id)) sprintf("  id = %s", fmt_chr_vec(model$id)) else NULL,
    if (!is.null(model$time)) sprintf('  time = "%s"', model$time) else NULL,
    if (!is.null(model$time_cluster) && identical(model$level_type, "crossclassified")) {
      sprintf('  time_cluster = "%s"', model$time_cluster)
    } else NULL,
    sprintf('  level_type = "%s"', model$level_type),
    if (!identical(model$family, "gaussian")) sprintf('  family = "%s"', model$family) else NULL,
    if (length(model$categorical)) sprintf("  categorical = %s", fmt_chr_vec(model$categorical)) else NULL,
    if (!is.null(model$tinterval)) sprintf("  tinterval = %s", format(model$tinterval)) else NULL,
    if (!identical(model$centering, "latent")) sprintf('  centering = "%s"', model$centering) else NULL
  )

  lines <- c(
    "# =============================================================================",
    sprintf("# Reproducible DSEMr Script (Hierarchy: %s | Syntax: %s)", model$level_type, syntax),
    "# Generated by DSEMr::dsem_code()",
    "# =============================================================================",
    "",
    "library(DSEMr)",
    "",
    "# 1. Specify and compile the dynamic structural equation model",
    "model_syntax <- \"",
    trimws(model_str),
    "\"",
    "",
    "model <- dsem_model(",
    "  model_syntax,",
    paste(model_args, collapse = ",\n"),
    ")",
    "",
    "# 2. Estimate via compiled Rust Bayesian MCMC backend",
    "fit <- dsem(",
    "  model = model,",
    sprintf("  data = %s,", data_name),
    sprintf("  chains = %dL,", chains),
    sprintf("  iter = %dL,", iter),
    sprintf("  warmup = %dL,", warmup),
    sprintf("  thin = %dL,", thin),
    sprintf("  seed = %dL%s", seed, if (!identical(prior_mode, "default")) "," else ""),
    if (!identical(prior_mode, "default")) sprintf('  control = list(prior_mode = "%s")', prior_mode) else NULL,
    ")"
  )

  if (isTRUE(include_outputs)) {
    lines <- c(
      lines,
      "",
      "# 3. Posterior summaries, standardization, and model fit",
      "summary(fit, standardized = TRUE, r2 = TRUE, point = \"median\")",
      "dsem_dic(fit)",
      "dsem_r2(fit)",
      "",
      "# 4. Publication APA table, LaTeX equations, and visual path diagram",
      "dsem_table(fit, format = \"markdown\")",
      "dsem_equations(fit, substitute = TRUE)",
      "dsem_diagram(fit, format = \"plot\")"
    )
  }

  if (isTRUE(include_session_info)) {
    lines <- c(lines, "", "# 5. Reproducibility environment record", "sessionInfo()")
  }

  out <- list(code = paste(lines, collapse = "\n"), syntax = syntax, hierarchy = model$level_type)
  class(out) <- c("DSEMcode", "list")
  out
}

# ---- Internal Syntax Translators ----

.dsem_spec_to_lavaan <- function(spec) {
  is_ml <- spec$level_type %in% c("twolevel", "threelevel", "crossclassified")
  lines <- character()
  if (is_ml) lines <- c(lines, "level: 1")

  # Factor measurement statements
  for (fn in names(spec$factors)) {
    lines <- c(lines, paste0("  ", fn, " =~ ", paste(spec$factors[[fn]], collapse = " + ")))
  }

  re_names <- spec$outcomes
  for (out_nm in spec$outcomes) {
    for (lg in spec$lags) {
      for (src in spec$outcomes) {
        is_r <- (isTRUE(spec$random_slopes) || (is.character(spec$random_slopes) && out_nm %in% spec$random_slopes)) && is_ml
        lbl <- if (length(spec$outcomes) == 1L && length(spec$lags) == 1L) "s" else paste0("s_", out_nm, "_", src, "_", lg)
        rhs_e <- if (spec$residual_dynamics) sprintf("residual(lag(%s, %d))", src, lg) else sprintf("lag(%s, %d)", src, lg)
        if (is_r) {
          lines <- c(lines, sprintf("  %s | %s ~ %s", lbl, out_nm, rhs_e))
          re_names <- unique(c(re_names, lbl))
        } else {
          lines <- c(lines, sprintf("  %s ~ %s", out_nm, rhs_e))
        }
      }
    }
    for (xw in spec$predictors_within) {
      is_r <- (isTRUE(spec$random_slopes) || (is.character(spec$random_slopes) && xw %in% spec$random_slopes)) && is_ml
      lbl <- paste0("s_", xw)
      if (is_r) {
        lines <- c(lines, sprintf("  %s | %s ~ %s", lbl, out_nm, xw))
        re_names <- unique(c(re_names, lbl))
      } else {
        lines <- c(lines, sprintf("  %s ~ %s", out_nm, xw))
      }
    }
    if (isTRUE(spec$random_logv) && is_ml) {
      lv_nm <- if (length(spec$outcomes) == 1L) "logv" else paste0("logv_", out_nm)
      lines <- c(lines, sprintf("  %s | %s ~~ %s", lv_nm, out_nm, out_nm))
      re_names <- unique(c(re_names, lv_nm))
    }
  }

  if (spec$level_type == "twolevel") {
    lines <- c(lines, "level: 2")
    if (length(spec$predictors_between)) {
      rhs_b <- paste(spec$predictors_between, collapse = " + ")
      for (rn in re_names) lines <- c(lines, sprintf("  %s ~ %s", rn, rhs_b))
    } else {
      lines <- c(lines, sprintf("  %s ~~ %s", spec$outcomes[1L], spec$outcomes[1L]))
    }
    if (length(spec$distal)) {
      rhs_z <- paste(re_names, collapse = " + ")
      for (zn in spec$distal) lines <- c(lines, sprintf("  %s ~ %s", zn, rhs_z))
    }
  } else if (spec$level_type == "threelevel") {
    lines <- c(lines, "level: 2")
    if (length(spec$predictors_between_l2)) {
      lines <- c(lines, sprintf("  %s ~ %s", spec$outcomes[1L], paste(spec$predictors_between_l2, collapse = " + ")))
    } else {
      lines <- c(lines, sprintf("  %s ~~ %s", spec$outcomes[1L], spec$outcomes[1L]))
    }
    lines <- c(lines, "level: 3")
    if (length(spec$predictors_between)) {
      rhs_b <- paste(spec$predictors_between, collapse = " + ")
      for (rn in re_names) lines <- c(lines, sprintf("  %s ~ %s", rn, rhs_b))
    } else {
      lines <- c(lines, sprintf("  %s ~~ %s", spec$outcomes[1L], spec$outcomes[1L]))
    }
  } else if (spec$level_type == "crossclassified") {
    lines <- c(lines, "level: 2", sprintf("  %s ~~ %s", spec$outcomes[1L], spec$outcomes[1L]),
               "level: time", sprintf("  %s ~~ %s", spec$outcomes[1L], spec$outcomes[1L]))
  }

  paste(lines, collapse = "\n")
}

.dsem_terms_to_lavaan <- function(model) {
  t <- model$terms
  lvls <- unique(t$level)
  is_ml <- model$level_type != "single"
  lines <- character()
  for (lv in lvls) {
    if (is_ml) {
      hdr <- switch(lv, within = "level: 1", between = "level: 2", between_2 = "level: 2",
                    between_3 = "level: 3", time = "level: time", paste0("level: ", lv))
      lines <- c(lines, hdr)
    }
    sub <- t[t$level == lv, , drop = FALSE]
    for (i in seq_len(nrow(sub))) {
      r <- sub[i, ]
      rhs_s <- if (r$lag > 0L) sprintf("lag(%s, %d)", r$rhs, r$lag) else r$rhs
      if (isTRUE(r$residual)) rhs_s <- sprintf("residual(%s)", rhs_s)
      pref <- if (isTRUE(r$random) && nzchar(r$label)) paste0(r$label, " | ") else ""
      lines <- c(lines, sprintf("%s%s%s %s %s", if (is_ml) "  " else "", pref, r$lhs, r$op, rhs_s))
    }
  }
  paste(lines, collapse = "\n")
}

.dsem_spec_to_mplus <- function(spec, data_file = "data.dat", title = "DSEMr Generated Mplus Input") {
  all_vars <- unique(c(
    spec$outcomes, unlist(spec$factors, use.names = FALSE),
    spec$predictors_within, spec$predictors_between_l2,
    spec$predictors_between, spec$distal,
    spec$time, spec$time_cluster, spec$id
  ))
  all_vars <- all_vars[!is.null(all_vars) & nzchar(all_vars)]

  var_lines <- c(sprintf("  NAMES = %s;", paste(all_vars, collapse = " ")))
  if (spec$level_type == "twolevel" && length(spec$id) >= 1L) {
    var_lines <- c(var_lines, sprintf("  CLUSTER = %s;", spec$id[1L]))
  } else if (spec$level_type == "threelevel" && length(spec$id) == 2L) {
    var_lines <- c(var_lines, sprintf("  CLUSTER = %s %s;", spec$id[2L], spec$id[1L]))
  } else if (spec$level_type == "crossclassified" && length(spec$id) >= 1L) {
    tc <- if (!is.null(spec$time_cluster)) spec$time_cluster else spec$time
    var_lines <- c(var_lines, sprintf("  CLUSTER = %s %s;", tc, spec$id[1L]))
  }
  if (length(spec$predictors_within)) {
    var_lines <- c(var_lines, sprintf("  WITHIN = %s;", paste(spec$predictors_within, collapse = " ")))
  }
  b_vars <- unique(c(spec$predictors_between, spec$distal))
  if (length(b_vars)) {
    var_lines <- c(var_lines, sprintf("  BETWEEN = %s;", paste(b_vars, collapse = " ")))
  }
  if (length(spec$categorical)) {
    var_lines <- c(var_lines, sprintf("  CATEGORICAL = %s;", paste(spec$categorical, collapse = " ")))
  }
  if (length(spec$censored)) {
    c_toks <- vapply(names(spec$censored), function(nm) {
      sp <- spec$censored[[nm]]
      flag <- if (is.character(sp)) {
        if (sp == "above") "a" else if (sp == "both") "ab" else "b"
      } else {
        if (is.finite(sp[1L]) && is.finite(sp[2L])) "ab" else if (is.finite(sp[2L])) "a" else "b"
      }
      sprintf("%s (%s)", nm, flag)
    }, character(1))
    var_lines <- c(var_lines, sprintf("  CENSORED = %s;", paste(c_toks, collapse = " ")))
  }
  if (!length(spec$factors)) {
    var_lines <- c(var_lines, sprintf("  LAGGED = %s(%d);", paste(spec$outcomes, collapse = " "), max(spec$lags)))
  }
  if (!is.null(spec$tinterval) && !is.null(spec$time)) {
    var_lines <- c(var_lines, sprintf("  TINTERVAL = %s(%s);", spec$time, format(spec$tinterval)))
  }

  an_type <- switch(
    spec$level_type,
    twolevel = "  TYPE = TWOLEVEL RANDOM;",
    threelevel = "  TYPE = THREELEVEL RANDOM;",
    crossclassified = "  TYPE = CROSSCLASSIFIED RANDOM;",
    NULL
  )
  an_lines <- c(
    an_type,
    "  ESTIMATOR = BAYES;",
    sprintf("  CHAINS = %d;", spec$chains),
    sprintf("  BITERATIONS = (%d);", spec$iter),
    sprintf("  BSEED = %d;", spec$seed),
    if (identical(spec$prior_mode, "penalized")) "  PRIOR = PSEM;" else NULL
  )

  mod_lines <- character()
  is_ml <- spec$level_type != "single"
  if (is_ml) mod_lines <- c(mod_lines, "  %WITHIN%")
  for (fn in names(spec$factors)) {
    mod_lines <- c(mod_lines, sprintf("    %s BY %s (&%d);", fn, paste(spec$factors[[fn]], collapse = " "), max(spec$lags)))
  }
  re_names <- spec$outcomes
  for (out_nm in spec$outcomes) {
    for (lg in spec$lags) {
      for (src in spec$outcomes) {
        is_r <- (isTRUE(spec$random_slopes) || (is.character(spec$random_slopes) && out_nm %in% spec$random_slopes)) && is_ml
        lbl <- if (length(spec$outcomes) == 1L && length(spec$lags) == 1L) "s" else paste0("s_", out_nm, "_", src, "_", lg)
        lhs_m <- if (spec$residual_dynamics) paste0(out_nm, "^") else out_nm
        rhs_m <- if (spec$residual_dynamics) paste0(src, "^", lg) else paste0(src, "&", lg)
        if (is_r) {
          mod_lines <- c(mod_lines, sprintf("    %s | %s ON %s;", lbl, lhs_m, rhs_m))
          re_names <- unique(c(re_names, lbl))
        } else {
          mod_lines <- c(mod_lines, sprintf("    %s ON %s;", lhs_m, rhs_m))
        }
      }
    }
    for (xw in spec$predictors_within) {
      is_r <- (isTRUE(spec$random_slopes) || (is.character(spec$random_slopes) && xw %in% spec$random_slopes)) && is_ml
      lbl <- paste0("s_", xw)
      if (is_r) {
        mod_lines <- c(mod_lines, sprintf("    %s | %s ON %s;", lbl, out_nm, xw))
        re_names <- unique(c(re_names, lbl))
      } else {
        mod_lines <- c(mod_lines, sprintf("    %s ON %s;", out_nm, xw))
      }
    }
    if (isTRUE(spec$random_logv) && is_ml) {
      lv_nm <- if (length(spec$outcomes) == 1L) "logv" else paste0("logv_", out_nm)
      mod_lines <- c(mod_lines, sprintf("    %s | %s;", lv_nm, out_nm))
      re_names <- unique(c(re_names, lv_nm))
    }
  }

  if (spec$level_type == "twolevel") {
    mod_lines <- c(mod_lines, "  %BETWEEN%")
    if (length(spec$predictors_between)) {
      mod_lines <- c(mod_lines, sprintf("    %s ON %s;", paste(re_names, collapse = " "), paste(spec$predictors_between, collapse = " ")))
    }
    if (length(re_names) >= 2L) {
      mod_lines <- c(mod_lines, sprintf("    %s WITH %s;", paste(re_names, collapse = " "), paste(re_names, collapse = " ")))
    }
    if (length(spec$distal)) {
      mod_lines <- c(mod_lines, sprintf("    %s ON %s;", paste(spec$distal, collapse = " "), paste(re_names, collapse = " ")))
    }
  } else if (spec$level_type == "threelevel") {
    mod_lines <- c(mod_lines, sprintf("  %%BETWEEN %s%%", spec$id[1L]), sprintf("    %s;", paste(spec$outcomes, collapse = " ")))
    if (length(spec$predictors_between_l2)) {
      mod_lines <- c(mod_lines, sprintf("    %s ON %s;", paste(spec$outcomes, collapse = " "), paste(spec$predictors_between_l2, collapse = " ")))
    }
    mod_lines <- c(mod_lines, sprintf("  %%BETWEEN %s%%", spec$id[2L]))
    if (length(spec$predictors_between)) {
      mod_lines <- c(mod_lines, sprintf("    %s ON %s;", paste(re_names, collapse = " "), paste(spec$predictors_between, collapse = " ")))
    }
    if (length(re_names) >= 2L) {
      mod_lines <- c(mod_lines, sprintf("    %s WITH %s;", paste(re_names, collapse = " "), paste(re_names, collapse = " ")))
    }
  } else if (spec$level_type == "crossclassified") {
    mod_lines <- c(
      mod_lines,
      sprintf("  %%BETWEEN %s%%", spec$id[1L]),
      sprintf("    %s;", paste(re_names, collapse = " ")),
      sprintf("  %%BETWEEN %s%%", if (!is.null(spec$time_cluster)) spec$time_cluster else spec$time),
      sprintf("    %s;", paste(spec$outcomes, collapse = " "))
    )
  }

  def_block <- if (spec$centering %in% c("groupmean", "grandmean") && length(spec$predictors_within)) {
    c("DEFINE:", sprintf("  CENTER %s (%s);", paste(spec$predictors_within, collapse = " "), toupper(spec$centering)))
  } else NULL

  paste(c(
    paste("TITLE:", title),
    sprintf("DATA:\n  FILE = %s;", data_file),
    "VARIABLE:", var_lines,
    def_block,
    "ANALYSIS:", an_lines,
    "MODEL:", mod_lines,
    "OUTPUT:\n  TECH1 TECH8 STANDARDIZED;"
  ), collapse = "\n")
}

.dsem_model_to_mplus <- function(model, fit_obj = NULL, data_file = "data.dat", title = "DSEMr Generated Mplus Input") {
  t <- model$terms
  vars <- unique(c(t$lhs, t$rhs[t$rhs != "1"], model$id, model$time, model$time_cluster))
  vars <- vars[!is.null(vars) & nzchar(vars)]
  var_lines <- c(sprintf("  NAMES = %s;", paste(vars, collapse = " ")))
  if (model$level_type == "twolevel" && length(model$id) >= 1L) {
    var_lines <- c(var_lines, sprintf("  CLUSTER = %s;", model$id[1L]))
  } else if (model$level_type == "threelevel" && length(model$id) == 2L) {
    var_lines <- c(var_lines, sprintf("  CLUSTER = %s %s;", model$id[2L], model$id[1L]))
  }
  if (length(model$categorical)) {
    var_lines <- c(var_lines, sprintf("  CATEGORICAL = %s;", paste(model$categorical, collapse = " ")))
  }
  lag_rows <- t[t$lag > 0L & !t$op == "=~", , drop = FALSE]
  if (nrow(lag_rows) && !any(t$op == "=~")) {
    var_lines <- c(var_lines, sprintf("  LAGGED = %s(%d);", paste(unique(lag_rows$rhs), collapse = " "), max(lag_rows$lag)))
  }
  if (!is.null(model$tinterval) && !is.null(model$time)) {
    var_lines <- c(var_lines, sprintf("  TINTERVAL = %s(%s);", model$time, format(model$tinterval)))
  }

  an_lines <- c(
    switch(model$level_type, twolevel = "  TYPE = TWOLEVEL RANDOM;", threelevel = "  TYPE = THREELEVEL RANDOM;",
           crossclassified = "  TYPE = CROSSCLASSIFIED RANDOM;", NULL),
    "  ESTIMATOR = BAYES;",
    if (!is.null(fit_obj)) sprintf("  CHAINS = %d;", fit_obj$sampler$chains) else "  CHAINS = 2;",
    if (!is.null(fit_obj)) sprintf("  BITERATIONS = (%d);", fit_obj$sampler$iter) else "  BITERATIONS = (2000);"
  )

  mod_lines <- character()
  is_ml <- model$level_type != "single"
  for (lv in unique(t$level)) {
    if (is_ml) {
      hdr <- switch(
        lv,
        within = "  %WITHIN%",
        between = "  %BETWEEN%",
        between_2 = sprintf("  %%BETWEEN %s%%", if (length(model$id) >= 1L) model$id[1L] else "L2"),
        between_3 = sprintf("  %%BETWEEN %s%%", if (length(model$id) >= 2L) model$id[2L] else "L3"),
        time = sprintf("  %%BETWEEN %s%%", if (!is.null(model$time_cluster)) model$time_cluster else "time"),
        paste0("  %BETWEEN ", lv, "%")
      )
      mod_lines <- c(mod_lines, hdr)
    }
    sub <- t[t$level == lv, , drop = FALSE]
    for (i in seq_len(nrow(sub))) {
      r <- sub[i, ]
      pref <- if (isTRUE(r$random) && nzchar(r$label)) paste0(r$label, " | ") else ""
      if (r$op == "=~") {
        mod_lines <- c(mod_lines, sprintf("    %s BY %s;", r$lhs, r$rhs))
      } else if (r$op == "~") {
        lhs_m <- if (isTRUE(r$residual)) paste0(r$lhs, "^") else r$lhs
        rhs_m <- if (r$lag > 0L) {
          if (isTRUE(r$residual)) paste0(r$rhs, "^", r$lag) else paste0(r$rhs, "&", r$lag)
        } else r$rhs
        mod_lines <- c(mod_lines, sprintf("    %s%s ON %s;", pref, lhs_m, rhs_m))
      } else if (r$op == "~~") {
        if (r$lhs == r$rhs) {
          mod_lines <- c(mod_lines, sprintf("    %s%s;", pref, r$lhs))
        } else {
          mod_lines <- c(mod_lines, sprintf("    %s WITH %s;", r$lhs, r$rhs))
        }
      }
    }
  }
  paste(c(
    paste("TITLE:", title),
    sprintf("DATA:\n  FILE = %s;", data_file),
    "VARIABLE:", var_lines,
    "ANALYSIS:", an_lines,
    "MODEL:", mod_lines,
    "OUTPUT:\n  TECH1 TECH8 STANDARDIZED;"
  ), collapse = "\n")
}

#' @exportS3Method base::print
print.DSEMspec <- function(x, ...) {
  cat("<DSEMspec> hierarchy:", x$level_type, "| family:", x$family, "\n")
  cat(" outcomes:", paste(x$outcomes, collapse = ", "),
      "| lags:", paste(x$lags, collapse = ", "), "\n")
  if (length(x$id)) cat(" id:", paste(x$id, collapse = ", "), "\n")
  if (!is.null(x$time)) cat(" time:", x$time, if (!is.null(x$tinterval)) paste0(" (TINTERVAL=", x$tinterval, ")") else "", "\n")
  if (length(x$predictors_within)) cat(" within predictors:", paste(x$predictors_within, collapse = ", "), "\n")
  if (length(x$predictors_between)) cat(" between predictors:", paste(x$predictors_between, collapse = ", "), "\n")
  if (length(x$distal)) cat(" distal outcomes:", paste(x$distal, collapse = ", "), "\n")
  cat(" random_slopes:", paste(x$random_slopes, collapse = ", "), "| random_logv:", x$random_logv, "\n")
  invisible(x)
}

#' @exportS3Method base::print
print.DSEMcode <- function(x, ...) {
  cat(x$code, "\n")
  invisible(x)
}

#' @exportS3Method base::as.character
as.character.DSEMcode <- function(x, ...) x$code
