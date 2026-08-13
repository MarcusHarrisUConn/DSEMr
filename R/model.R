#' Compile a dynamic structural equation model
#'
#' Compiles either the DSEMr extension of lavaan syntax or the supported
#' clean-room Mplus-like subset to a common representation. Compilation does
#' not imply that every represented model family is estimable in the current
#' development release; [dsem()] performs a separate capability check.
#'
#' @param model Character string containing a model specification.
#' @param syntax Either `"lavaan"`, `"mplus"`, or `"auto"`.
#' @param id Optional cluster identifier.
#' @param time Optional time index.
#' @param family Outcome family; currently `"gaussian"`, `"binary"`, or
#'   `"ordinal"` can be represented.
#' @param metadata Optional named list retained with the compiled model.
#' @return A `DSEMmodel` object.
#' @export
dsem_model <- function(model, syntax = c("auto", "lavaan", "mplus"),
                       id = NULL, time = NULL,
                       family = c("gaussian", "binary", "ordinal"),
                       metadata = list()) {
  if (!is.function(lavaan::lavaanify)) {
    .dsem_abort("The installed lavaan package does not expose its public compiler API.")
  }
  if (!is.character(model) || length(model) != 1L || !nzchar(trimws(model))) {
    .dsem_abort("`model` must be one non-empty character string.")
  }
  syntax <- match.arg(syntax)
  family <- match.arg(family)
  if (syntax == "auto") {
    syntax <- if (grepl("(^|\\n)\\s*(MODEL|ANALYSIS|VARIABLE)\\s*:",
                        model, ignore.case = TRUE)) "mplus" else "lavaan"
  }
  parsed <- if (syntax == "mplus") {
    .dsem_parse_mplus(model)
  } else {
    .dsem_parse_lavaan(model)
  }
  parsed$family <- family
  parsed$id <- id
  parsed$time <- time
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

.dsem_add_term <- function(terms, level, lhs, op, rhs, lag = 0L,
                           random = FALSE, residual = FALSE, label = "") {
  rbind(terms, data.frame(
    level = level, lhs = lhs, op = op, rhs = rhs,
    lag = as.integer(lag), random = isTRUE(random),
    residual = isTRUE(residual), label = label,
    stringsAsFactors = FALSE
  ))
}

.dsem_parse_lavaan <- function(model) {
  lines <- .dsem_lines(model)
  terms <- .dsem_empty_terms()
  level <- "within"
  warnings <- character()
  for (line in lines) {
    if (grepl("^level\\s*:\\s*(1|within)$", line, ignore.case = TRUE) ||
        grepl("^%within%$", line, ignore.case = TRUE)) {
      level <- "within"
      next
    }
    if (grepl("^level\\s*:\\s*(2|between)$", line, ignore.case = TRUE) ||
        grepl("^%between%$", line, ignore.case = TRUE)) {
      level <- "between"
      next
    }
    if (!grepl("(=~|~~|~)", line)) {
      .dsem_abort("Unsupported lavaan-style statement: `%s`.", line)
    }
    op_match <- regexpr("=~|~~|~", line)
    op <- regmatches(line, op_match)
    lhs <- trimws(substr(line, 1L, op_match - 1L))
    rhs_text <- trimws(substr(line, op_match + attr(op_match, "match.length"), nchar(line)))
    if (!nzchar(lhs) || !nzchar(rhs_text)) .dsem_abort("Incomplete statement: `%s`.", line)
    rhs_parts <- trimws(unlist(strsplit(rhs_text, "\\+")))
    for (rhs_part in rhs_parts) {
      random <- grepl("^random\\(", rhs_part, ignore.case = TRUE)
      rhs_part <- sub("^random\\((.*)\\)$", "\\1", rhs_part, ignore.case = TRUE)
      lag <- 0L
      if (grepl("^lag\\(", rhs_part, ignore.case = TRUE)) {
        matched <- regexec("^lag\\(\\s*([.A-Za-z][.A-Za-z0-9_]*)\\s*,\\s*([0-9]+)\\s*\\)$",
                           rhs_part, ignore.case = TRUE)
        captures <- regmatches(rhs_part, matched)[[1L]]
        if (!length(captures)) .dsem_abort("Invalid lag expression: `%s`.", rhs_part)
        rhs_part <- captures[2L]
        lag <- as.integer(captures[3L])
      }
      residual <- grepl("^residual\\(", rhs_part, ignore.case = TRUE)
      rhs_part <- sub("^residual\\((.*)\\)$", "\\1", rhs_part, ignore.case = TRUE)
      terms <- .dsem_add_term(terms, level, lhs, op, rhs_part, lag, random, residual)
    }
  }
  if (!nrow(terms)) .dsem_abort("No model terms were found.")
  list(terms = terms, warnings = warnings)
}

.dsem_parse_mplus <- function(model) {
  raw <- unlist(strsplit(model, "\n", fixed = TRUE), use.names = FALSE)
  raw <- sub("!.*$", "", raw)
  text <- paste(raw, collapse = " ")
  sections <- gregexpr("[A-Za-z]+\\s*:", text, perl = TRUE)[[1L]]
  if (identical(sections, -1L)) .dsem_abort("Mplus-like syntax requires named sections.")
  section_names <- toupper(trimws(sub(":", "", regmatches(text, list(sections))[[1L]])))
  section_ends <- c(sections[-1L] - 1L, nchar(text))
  section_starts <- sections + attr(sections, "match.length")
  bodies <- mapply(substr, section_starts, section_ends, MoreArgs = list(x = text), USE.NAMES = FALSE)
  names(bodies) <- section_names
  allowed_sections <- c("TITLE", "DATA", "VARIABLE", "DEFINE", "ANALYSIS", "MODEL", "OUTPUT")
  unknown <- setdiff(section_names, allowed_sections)
  if (length(unknown)) .dsem_abort("Unsupported Mplus section(s): %s.", paste(unknown, collapse = ", "))
  if (is.null(bodies[["MODEL"]])) .dsem_abort("Mplus-like syntax requires a MODEL section.")
  model_body <- bodies[["MODEL"]]
  chunks <- unlist(strsplit(model_body, ";", fixed = TRUE), use.names = FALSE)
  terms <- .dsem_empty_terms()
  level <- "within"
  for (chunk in chunks) {
    chunk <- .dsem_trim(chunk)
    if (!nzchar(chunk)) next
    if (grepl("%WITHIN%", chunk, ignore.case = TRUE)) {
      level <- "within"
      chunk <- .dsem_trim(sub(".*%WITHIN%", "", chunk, ignore.case = TRUE))
    }
    if (grepl("%BETWEEN%", chunk, ignore.case = TRUE)) {
      level <- "between"
      chunk <- .dsem_trim(sub(".*%BETWEEN%", "", chunk, ignore.case = TRUE))
    }
    if (!nzchar(chunk)) next
    if (grepl("\\bON\\b", chunk, ignore.case = TRUE)) {
      bits <- strsplit(chunk, "(?i)\\s+ON\\s+", perl = TRUE)[[1L]]
      if (length(bits) != 2L) .dsem_abort("Ambiguous Mplus ON statement: `%s`.", chunk)
      lhs <- trimws(bits[1L])
      rhs_parts <- strsplit(trimws(bits[2L]), "\\s+")[[1L]]
      random <- grepl("^s[0-9]*\\s*\\|", lhs, ignore.case = TRUE)
      if (random) lhs <- trimws(sub("^s[0-9]*\\s*\\|", "", lhs, ignore.case = TRUE))
      for (rhs in rhs_parts) {
        lag <- 0L
        if (grepl("&[0-9]+$", rhs)) {
          lag <- as.integer(sub(".*&", "", rhs))
          rhs <- sub("&[0-9]+$", "", rhs)
        }
        terms <- .dsem_add_term(terms, level, lhs, "~", rhs, lag, random)
      }
    } else if (grepl("\\bBY\\b", chunk, ignore.case = TRUE)) {
      bits <- strsplit(chunk, "(?i)\\s+BY\\s+", perl = TRUE)[[1L]]
      if (length(bits) != 2L) .dsem_abort("Ambiguous Mplus BY statement: `%s`.", chunk)
      for (rhs in strsplit(trimws(bits[2L]), "\\s+")[[1L]]) {
        terms <- .dsem_add_term(terms, level, trimws(bits[1L]), "=~", rhs)
      }
    } else if (grepl("^(.+)\\s+WITH\\s+(.+)$", chunk, ignore.case = TRUE)) {
      bits <- strsplit(chunk, "(?i)\\s+WITH\\s+", perl = TRUE)[[1L]]
      terms <- .dsem_add_term(terms, level, trimws(bits[1L]), "~~", trimws(bits[2L]))
    } else if (grepl("^[.A-Za-z][.A-Za-z0-9_]*$", chunk)) {
      terms <- .dsem_add_term(terms, level, chunk, "~~", chunk, random = level == "between")
    } else {
      .dsem_abort("Unsupported or ambiguous Mplus MODEL statement: `%s`.", chunk)
    }
  }
  if (!nrow(terms)) .dsem_abort("No supported MODEL terms were found.")
  list(terms = terms, warnings = character(), sections = bodies)
}

.dsem_model_capabilities <- function(model) {
  terms <- model$terms
  c(
    gaussian = identical(model$family, "gaussian"),
    n1 = is.null(model$id),
    two_level = !is.null(model$id),
    lagged = any(terms$lag > 0L),
    latent = any(terms$op == "=~"),
    random = any(terms$random),
    residual_dynamic = any(terms$residual),
    cross_classified = any(terms$level == "time")
  )
}

.dsem_model_hash <- function(model) {
  paste0("dsem-", .dsem_stable_seed(1729L, paste(utils::capture.output(dput(model$terms)), collapse = "")))
}

#' @export
print.DSEMmodel <- function(x, ...) {
  cat("<DSEMmodel>", x$hash, "\n")
  cat(" syntax:", x$syntax, " family:", x$family, "\n")
  cat(" terms:", nrow(x$terms), " levels:", paste(unique(x$terms$level), collapse = ", "), "\n")
  invisible(x)
}

#' Translate supported Mplus-like syntax
#'
#' @param model Character Mplus-like model input.
#' @param id,time,family Passed to [dsem_model()].
#' @return A compiled `DSEMmodel` using the same canonical representation as
#'   lavaan-style input.
#' @export
mplus_to_dsem <- function(model, id = NULL, time = NULL,
                          family = c("gaussian", "binary", "ordinal")) {
  dsem_model(model, syntax = "mplus", id = id, time = time,
             family = match.arg(family))
}
