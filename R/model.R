#' Compile a dynamic structural equation model
#'
#' Compiles either the `DSEMr` extension of `lavaan` syntax or standard
#' section-based SEM model scripts into a canonical `DSEMmodel`
#' intermediate representation. Supports single-level (`N=1`), two-level
#' (`TWOLEVEL RANDOM`), three-level (`THREELEVEL RANDOM` / DSEM3),
#' cross-classified (`CROSSCLASSIFIED RANDOM`), and continuous-time
#' (`continuous_time = TRUE`) dynamic structural equation models, including
#' latent variable centering, random slopes, random log innovation variances,
#' random residual covariances, residual dynamics (RDSEM), ARMA(1,1), dynamic
#' factor models (DAFS, WNFS, cross-classified CFA, and random factor
#' loadings), binary/ordinal probit links, Tobit censored outcomes, and
#' penalized shrinkage priors (PSEM).
#'
#' @param model Character string containing a model specification.
#' @param syntax Either `"auto"`, `"lavaan"`, or `"mplus"`.
#' @param id Optional cluster identifier column name (length 1 for two-level or
#'   cross-classified models; length 2 `c("level2", "level3")` for three-level
#'   DSEM3 models, ordered from Level 2 to Level 3). When `NULL` and script
#'   syntax declares `CLUSTER = ...`, cluster names are inferred automatically.
#' @param time Optional time index column name.
#' @param family Outcome family (`"gaussian"`, `"binary"`, or `"ordinal"`).
#' @param level_type Hierarchical structure: `"auto"`, `"single"`, `"twolevel"`,
#'   `"threelevel"`, or `"crossclassified"`.
#' @param time_cluster Optional occasion cluster column for cross-classified
#'   models (`TYPE = CROSSCLASSIFIED`). Defaults to `time` when `level_type =
#'   "crossclassified"`.
#' @param categorical Optional character vector of binary or ordinal variable
#'   names modeled with a probit link.
#' @param censored Optional Tobit censoring specification for floor/ceiling
#'   continuous outcomes: `NULL`, a character vector of left-censored outcome
#'   names, or a named list mapping outcome names to `"below"`, `"above"`,
#'   `"both"`, or explicit numeric bounds `c(lower, upper)`.
#' @param tinterval Optional positive numeric bin width for regular-grid
#'   `TINTERVAL` time alignment.
#' @param centering Within-cluster centering policy for covariates and outcomes:
#'   `"latent"` (default latent variable decomposition across 2 or 3 levels),
#'   `"groupmean"`, `"grandmean"`, or `"none"`.
#' @param continuous_time Logical; if `TRUE`, compiles the model as a
#'   continuous-time Ornstein-Uhlenbeck dynamic model (`CT-DSEM` / `CT-RDSEM`)
#'   using exact matrix-exponential transitions over irregular time intervals.
#' @param metadata Optional named list retained with the compiled model.
#' @return A `DSEMmodel` object.
#' @examples
#' m1 <- dsem_model("y ~ lag(y, 1)", time = "time")
#' print(m1)
#'
#' m3 <- dsem_model("
#'   level: 1
#'     s | y ~ lag(y, 1)
#'     logv | y ~~ y
#'   level: 2
#'     y ~~ y
#'   level: 3
#'     y ~ w3
#'     s ~ w3
#' ", id = c("day", "person"), time = "occasion", level_type = "threelevel")
#' print(m3)
#' @export
dsem_model <- function(model,
                       syntax = c("auto", "lavaan", "mplus"),
                       id = NULL,
                       time = NULL,
                       family = c("gaussian", "binary", "ordinal"),
                       level_type = c("auto", "single", "twolevel", "threelevel", "crossclassified"),
                       time_cluster = NULL,
                       categorical = NULL,
                       censored = NULL,
                       tinterval = NULL,
                       centering = c("latent", "groupmean", "grandmean", "none"),
                       continuous_time = FALSE,
                       metadata = list()) {
  if (!is.function(lavaan::lavaanify)) {
    .dsem_abort("The installed lavaan package does not expose its public compiler API.")
  }
  if (!is.character(model) || length(model) != 1L || is.na(model) || !nzchar(trimws(model))) {
    .dsem_abort("`model` must be one non-empty character string.")
  }
  syntax <- match.arg(syntax)
  family <- match.arg(family)
  level_type <- match.arg(level_type)
  centering_missing <- missing(centering)
  centering <- match.arg(centering)
  if (!is.null(id)) {
    if (!is.character(id) || !length(id) || length(id) > 2L || anyNA(id) || any(!nzchar(id))) {
      .dsem_abort("`id` must be NULL or one or two cluster column names.")
    }
  }
  for (field in c("time", "time_cluster")) {
    value <- get(field)
    if (!is.null(value) && (!is.character(value) || length(value) != 1L || is.na(value) || !nzchar(value))) {
      .dsem_abort("`%s` must be NULL or one column name.", field)
    }
  }
  if (!is.null(categorical) && (!is.character(categorical) || anyNA(categorical) || any(!nzchar(categorical)))) {
    .dsem_abort("`categorical` must be NULL or a character vector of variable names.")
  }
  if (!is.null(tinterval) && (!is.numeric(tinterval) || length(tinterval) != 1L || !is.finite(tinterval) || tinterval <= 0)) {
    .dsem_abort("`tinterval` must be NULL or one positive number.")
  }
  .dsem_named_controls(metadata)
  if (syntax == "auto") {
    syntax <- if (grepl("(^|\\n)\\s*(TITLE|DATA|MODEL|ANALYSIS|VARIABLE|DEFINE)\\s*:",
                        model, ignore.case = TRUE)) "mplus" else "lavaan"
  }
  parsed <- if (syntax == "mplus") {
    .dsem_parse_mplus(model)
  } else {
    .dsem_parse_lavaan(model)
  }

  # Merge Mplus-parsed structural declarations with explicit R arguments
  if (is.null(id) && length(parsed$mplus_id)) id <- parsed$mplus_id
  if (is.null(time) && !is.null(parsed$mplus_time)) time <- parsed$mplus_time
  if (is.null(time_cluster) && !is.null(parsed$mplus_time_cluster)) time_cluster <- parsed$mplus_time_cluster
  if (is.null(tinterval) && !is.null(parsed$mplus_tinterval)) tinterval <- parsed$mplus_tinterval
  cat_vars <- unique(c(categorical, parsed$mplus_categorical))
  if (length(cat_vars) && family == "gaussian") {
    family <- "binary"
  }
  cens_spec <- .dsem_normalize_censored(censored)
  if (length(parsed$mplus_censored)) {
    for (nm in names(parsed$mplus_censored)) {
      if (!nm %in% names(cens_spec)) cens_spec[[nm]] <- parsed$mplus_censored[[nm]]
    }
  }
  if (!is.null(parsed$mplus_centering) && centering_missing) {
    centering <- parsed$mplus_centering
  }
  if (level_type == "auto") {
    if (!is.null(parsed$mplus_level_type)) {
      level_type <- parsed$mplus_level_type
    } else if (!is.null(time_cluster) || any(parsed$terms$level == "time")) {
      level_type <- "crossclassified"
    } else if (length(id) == 2L || any(parsed$terms$level %in% c("between_2", "between_3"))) {
      level_type <- "threelevel"
    } else if (length(id) == 1L || any(parsed$terms$level == "between")) {
      level_type <- "twolevel"
    } else {
      level_type <- "single"
    }
  }
  if (level_type == "crossclassified" && is.null(time_cluster)) {
    time_cluster <- time
  }

  parsed$family <- family
  parsed$id <- id
  parsed$time <- time
  parsed$time_cluster <- time_cluster
  parsed$level_type <- level_type
  parsed$categorical <- cat_vars
  parsed$censored <- cens_spec
  parsed$tinterval <- tinterval
  parsed$centering <- centering
  parsed$centered_vars <- parsed$mplus_centered_vars
  parsed$continuous_time <- isTRUE(continuous_time) ||
    (!is.null(parsed$mplus_algorithm) && grepl("^MH\\s*\\(\\s*(UNIMH|MH[2-5])\\s*\\)$", parsed$mplus_algorithm))
  parsed$define_ops <- if (!is.null(parsed$mplus_define_ops)) parsed$mplus_define_ops else list()
  parsed$constraints <- if (!is.null(parsed$mplus_constraints)) parsed$mplus_constraints else character()
  parsed$wald_tests <- if (!is.null(parsed$mplus_tests)) parsed$mplus_tests else character()
  parsed$syntax <- syntax
  parsed$source <- model
  parsed$metadata <- metadata
  parsed$capabilities <- .dsem_model_capabilities(parsed)
  parsed$hash <- .dsem_model_hash(parsed)
  class(parsed) <- c("DSEMmodel", "list")
  parsed
}

.dsem_empty_terms <- function() {
  data.frame(
    level = character(), lhs = character(), op = character(),
    rhs = character(), lag = integer(), random = logical(),
    residual = logical(), label = character(),
    stringsAsFactors = FALSE
  )
}

.dsem_empty_meta <- function() {
  data.frame(
    fixed_value = numeric(), is_free = logical(),
    random_name = character(), is_logv = logical(),
    eq_label = character(),
    stringsAsFactors = FALSE
  )
}

.dsem_add_term <- function(terms, level, lhs, op, rhs, lag = 0L,
                           random = FALSE, residual = FALSE, label = "") {
  identifier <- "^[.A-Za-z][.A-Za-z0-9_]*$"
  valid_rhs <- grepl(identifier, rhs) ||
    (op == "~" && rhs == "1") ||
    (op == "|" && grepl("^[A-Za-z0-9_$]+$", rhs))
  if (!grepl(identifier, lhs) || !valid_rhs) {
    .dsem_abort("Unsupported variable name, parameter modifier, or constraint in `%s %s %s`.", lhs, op, rhs)
  }
  rbind(terms, data.frame(
    level = level, lhs = lhs, op = op, rhs = rhs,
    lag = as.integer(lag), random = isTRUE(random),
    residual = isTRUE(residual), label = as.character(label),
    stringsAsFactors = FALSE
  ))
}

.dsem_add_meta <- function(meta, fixed_value = NA_real_, is_free = TRUE,
                           random_name = "", is_logv = FALSE, eq_label = "") {
  rbind(meta, data.frame(
    fixed_value = as.double(fixed_value),
    is_free = isTRUE(is_free),
    random_name = as.character(random_name),
    is_logv = isTRUE(is_logv),
    eq_label = as.character(eq_label),
    stringsAsFactors = FALSE
  ))
}

.dsem_parse_lavaan <- function(model) {
  lines <- .dsem_lines(model)
  terms <- .dsem_empty_terms()
  meta <- .dsem_empty_meta()
  level <- "within"
  warnings <- character()

  for (line in lines) {
    if (grepl("^(level\\s*:\\s*(1|within)|%within%)$", line, ignore.case = TRUE)) {
      level <- "within"; next
    }
    if (grepl("^(level\\s*:\\s*(2|between|l2|between_2)|%between%|%between\\s+l2%)$", line, ignore.case = TRUE)) {
      level <- "between"; next
    }
    if (grepl("^(level\\s*:\\s*(3|l3|between_3)|%between\\s+l3%)$", line, ignore.case = TRUE)) {
      level <- "between_3"; next
    }
    if (grepl("^(level\\s*:\\s*time|%between\\s+time%)$", line, ignore.case = TRUE)) {
      level <- "time"; next
    }

    # Optional named random effect prefix: `s | y ~ lag(y, 1)` or `logv | y ~~ y`
    random_prefix <- ""
    work_line <- line
    if (grepl("^[.A-Za-z][.A-Za-z0-9_]*\\s*\\|\\s*.+(=~|~~|~)", work_line)) {
      bar_pos <- regexpr("\\|", work_line)
      random_prefix <- trimws(substr(work_line, 1L, bar_pos - 1L))
      work_line <- trimws(substr(work_line, bar_pos + 1L, nchar(work_line)))
    }

    if (!grepl("(=~|~~|~|\\|)", work_line)) {
      .dsem_abort("Unsupported lavaan-style statement: `%s`.", line)
    }
    op_match <- regexpr("=~|~~|~|\\|", work_line)
    op <- regmatches(work_line, op_match)
    lhs_raw <- trimws(substr(work_line, 1L, op_match - 1L))
    rhs_text <- trimws(substr(work_line, op_match + attr(op_match, "match.length"), nchar(work_line)))
    if (!nzchar(lhs_raw) || !nzchar(rhs_text)) .dsem_abort("Incomplete statement: `%s`.", line)

    lhs_res <- grepl("^residual\\((.*)\\)$", lhs_raw, ignore.case = TRUE)
    lhs <- sub("^residual\\((.*)\\)$", "\\1", lhs_raw, ignore.case = TRUE)

    rhs_parts <- trimws(unlist(strsplit(rhs_text, "\\+")))
    for (part_idx in seq_along(rhs_parts)) {
      rhs_part <- rhs_parts[part_idx]
      fixed_val <- NA_real_
      is_free <- TRUE
      explicit_free <- FALSE
      eq_lab <- ""
      # Parse lavaan pre-multiplication modifiers: NA*x, 1*x, 0.5*x, equal("a")*x
      if (grepl("\\*", rhs_part)) {
        star_pos <- regexpr("\\*", rhs_part)
        mod_str <- trimws(substr(rhs_part, 1L, star_pos - 1L))
        rhs_after <- trimws(substr(rhs_part, star_pos + 1L, nchar(rhs_part)))
        num_mod <- suppressWarnings(as.numeric(mod_str))
        if (identical(mod_str, "NA")) {
          is_free <- TRUE
          explicit_free <- TRUE
          rhs_part <- rhs_after
        } else if (!is.na(num_mod)) {
          fixed_val <- num_mod
          is_free <- FALSE
          rhs_part <- rhs_after
        } else if (grepl("^equal\\([\"']?([A-Za-z0-9_]+)[\"']?\\)$", mod_str)) {
          eq_lab <- sub("^equal\\([\"']?([A-Za-z0-9_]+)[\"']?\\)$", "\\1", mod_str)
          rhs_part <- rhs_after
        } else {
          .dsem_abort("Unsupported variable name, parameter modifier, or constraint in `%s %s %s`.", lhs, op, rhs_part)
        }
      }

      if (op == "=~" && part_idx == 1L && !explicit_free && is.na(fixed_val) && !nzchar(random_prefix)) {
        fixed_val <- 1.0
        is_free <- FALSE
      }

      random <- nzchar(random_prefix) || grepl("^random\\(", rhs_part, ignore.case = TRUE)
      rhs_part <- sub("^random\\((.*)\\)$", "\\1", rhs_part, ignore.case = TRUE)
      residual <- lhs_res || grepl("^residual\\(", rhs_part, ignore.case = TRUE)
      rhs_part <- sub("^residual\\((.*)\\)$", "\\1", rhs_part, ignore.case = TRUE)

      lag <- 0L
      if (grepl("^lag\\(", rhs_part, ignore.case = TRUE)) {
        matched <- regexec("^lag\\(\\s*([.A-Za-z][.A-Za-z0-9_]*)\\s*,\\s*([0-9]+)\\s*\\)$",
                           rhs_part, ignore.case = TRUE)
        captures <- regmatches(rhs_part, matched)[[1L]]
        if (!length(captures)) .dsem_abort("Invalid lag expression: `%s`.", rhs_part)
        rhs_part <- captures[2L]
        lag <- as.integer(captures[3L])
      }
      if (grepl("^residual\\(", rhs_part, ignore.case = TRUE)) {
        residual <- TRUE
        rhs_part <- sub("^residual\\((.*)\\)$", "\\1", rhs_part, ignore.case = TRUE)
      }

      is_lv <- (op == "~~" && lhs == rhs_part && random && level == "within")
      lbl <- if (nzchar(random_prefix)) random_prefix else eq_lab
      terms <- .dsem_add_term(terms, level, lhs, op, rhs_part, lag, random, residual, label = lbl)
      meta <- .dsem_add_meta(meta, fixed_val, is_free, random_prefix, is_lv, eq_lab)
    }
  }
  if (!nrow(terms)) .dsem_abort("No model terms were found.")
  # Normalize 3-level within/between_2 naming if between_3 is also present
  if (any(terms$level == "between_3")) {
    terms$level[terms$level == "between"] <- "between_2"
  }
  list(terms = terms, term_meta = meta, warnings = warnings)
}

.dsem_model_capabilities <- function(model) {
  terms <- model$terms
  lvl <- if (is.null(model$level_type)) "single" else model$level_type
  c(
    gaussian = identical(model$family, "gaussian") && !length(model$categorical),
    n1 = identical(lvl, "single") && is.null(model$id),
    two_level = identical(lvl, "twolevel") || (!is.null(model$id) && length(model$id) == 1L && lvl != "crossclassified"),
    three_level = identical(lvl, "threelevel") || length(model$id) == 2L,
    lagged = any(terms$lag > 0L),
    latent = any(terms$op == "=~"),
    random = any(terms$random),
    residual_dynamic = any(terms$residual),
    cross_classified = identical(lvl, "crossclassified") || any(terms$level == "time"),
    categorical = !identical(model$family, "gaussian") || length(model$categorical) > 0L,
    censored = length(model$censored) > 0L,
    continuous_time = isTRUE(model$continuous_time)
  )
}

.dsem_model_hash <- function(model) {
  semantics <- list(
    terms = model$terms, family = model$family, id = model$id,
    time = model$time, interval = .dsem_time_interval(model),
    level_type = model$level_type, categorical = model$categorical,
    censored = model$censored
  )
  if (isTRUE(model$continuous_time)) semantics$continuous_time <- TRUE
  paste0("dsem-", .dsem_stable_seed(1729L, paste(utils::capture.output(dput(semantics)), collapse = "")))
}

#' @exportS3Method base::print
print.DSEMmodel <- function(x, ...) {
  cat("<DSEMmodel>", x$hash, "\n")
  cat(" syntax:", x$syntax, " hierarchy:", x$level_type, " family:", x$family, "\n")
  cat(" terms:", nrow(x$terms), " levels:", paste(unique(x$terms$level), collapse = ", "), "\n")
  invisible(x)
}

#' Translate section-based SEM script syntax to a DSEMr model
#'
#' @param model Character string containing section-based SEM model syntax
#'   (supporting single-level, two-level, three-level, and cross-classified
#'   DSEM and RDSEM specifications).
#' @param id,time,family,... Passed to [dsem_model()].
#' @return A compiled `DSEMmodel` using the same canonical representation as
#'   `lavaan`-style input.
#' @examples
#' m <- mplus_to_dsem("
#'   ANALYSIS: TYPE = TWOLEVEL RANDOM; ESTIMATOR = BAYES;
#'   MODEL:
#'     %WITHIN%
#'     s | y ON y&1;
#'     logv | y;
#'     %BETWEEN%
#'     y s logv WITH y s logv;
#' ", id = "subject", time = "time")
#' print(m)
#' @export
mplus_to_dsem <- function(model, id = NULL, time = NULL,
                          family = c("gaussian", "binary", "ordinal"), ...) {
  dsem_model(model, syntax = "mplus", id = id, time = time,
             family = match.arg(family), ...)
}
