#' Compare competing Bayesian DSEM models using DIC and convergence criteria
#'
#' Compares two or more fitted `DSEMfit` objects using Spiegelhalter's Deviance
#' Information Criterion (`DIC`), effective number of parameters (`pD`),
#' posterior mean deviance (`D_bar`), relative `delta_dic`, DIC model weights,
#' and MCMC convergence diagnostics (`max_rhat`, `min_ess`). Models are checked
#' for comparable observation counts and outcomes to guard against invalid
#' information-criterion comparisons.
#'
#' @param ... Two or more `DSEMfit` objects (optionally named), or a single
#'   list of `DSEMfit` objects.
#' @param names Optional character vector of model labels.
#' @param sort Logical; if `TRUE` (default), models are ordered from lowest to
#'   highest `DIC` (with `NA` DIC models placed last).
#' @param rhat_threshold Numeric threshold used to flag `converged` (default
#'   `1.05`).
#' @return A `DSEMcompare` object (a data frame with class
#'   `c("DSEMcompare", "data.frame")`) containing columns `model`, `hierarchy`,
#'   `nobs`, `n_params`, `log_lik`, `d_bar`, `pd`, `dic`, `delta_dic`,
#'   `weight`, `max_rhat`, `min_ess`, `converged`, and `elapsed`.
#' @examples
#' dat <- simulate_dsem(n = 80, ar = 0.5, sigma = 0.7, seed = 15)
#' dat$x <- rnorm(80)
#' fit1 <- dsem("y ~ lag(y, 1)", dat, chains = 2, iter = 200, warmup = 100, seed = 15)
#' fit2 <- dsem("y ~ lag(y, 1) + x", dat, chains = 2, iter = 200, warmup = 100, seed = 15)
#' dsem_compare(AR1 = fit1, AR1_X = fit2)
#' @export
dsem_compare <- function(..., names = NULL, sort = TRUE, rhat_threshold = 1.05) {
  dots <- list(...)
  dot_calls <- as.list(match.call(expand.dots = FALSE)$...)
  if (length(dots) == 1L && is.list(dots[[1L]]) && !inherits(dots[[1L]], "DSEMfit")) {
    fits <- dots[[1L]]
    inferred_names <- base::names(fits)
    if (is.null(inferred_names)) inferred_names <- paste0("Model ", seq_along(fits))
  } else {
    fits <- dots
    nm_arg <- base::names(dots)
    inferred_names <- vapply(seq_along(fits), function(i) {
      if (!is.null(nm_arg) && nzchar(nm_arg[i])) {
        nm_arg[i]
      } else if (i <= length(dot_calls)) {
        paste(deparse(dot_calls[[i]]), collapse = "")
      } else {
        paste0("Model ", i)
      }
    }, character(1))
  }

  if (!length(fits)) .dsem_abort("Supply at least one `DSEMfit` object to compare.")
  bad <- !vapply(fits, inherits, logical(1), "DSEMfit")
  if (any(bad)) .dsem_abort("All objects supplied to `dsem_compare()` must be `DSEMfit` objects.")

  if (!is.null(names)) {
    if (!is.character(names) || length(names) != length(fits) || anyNA(names)) {
      .dsem_abort("`names` must be a character vector matching the number of models.")
    }
    model_names <- names
  } else {
    model_names <- make.unique(inferred_names, sep = "_")
  }

  if (!is.numeric(rhat_threshold) || length(rhat_threshold) != 1L || !is.finite(rhat_threshold) || rhat_threshold <= 1) {
    .dsem_abort("`rhat_threshold` must be a finite number greater than 1.")
  }

  rows <- lapply(seq_along(fits), function(i) {
    f <- fits[[i]]
    n_obs <- if (is.matrix(f$observed)) nrow(f$observed) else length(f$observed)
    dic_vec <- dsem_dic(f)
    rhats <- f$diagnostics$rhat
    ess_v <- if ("ess_bulk" %in% base::names(f$diagnostics)) f$diagnostics$ess_bulk else f$diagnostics$ess
    max_rh <- if (length(rhats) && any(is.finite(rhats))) max(rhats, na.rm = TRUE) else NA_real_
    min_es <- if (length(ess_v) && any(is.finite(ess_v))) min(ess_v, na.rm = TRUE) else NA_real_
    data.frame(
      model = model_names[i],
      hierarchy = f$model$level_type,
      nobs = as.integer(n_obs),
      n_params = as.integer(nrow(f$estimates)),
      log_lik = unname(dic_vec["log_lik"]),
      d_bar = unname(dic_vec["d_bar"]),
      pd = unname(dic_vec["pd"]),
      dic = unname(dic_vec["dic"]),
      delta_dic = NA_real_,
      weight = NA_real_,
      max_rhat = max_rh,
      min_ess = min_es,
      converged = is.finite(max_rh) && max_rh <= rhat_threshold,
      elapsed = as.double(f$elapsed),
      stringsAsFactors = FALSE
    )
  })
  tab <- do.call(rbind, rows)

  # Consistency checks across models
  notes <- character()
  if (length(unique(tab$nobs)) > 1L) {
    msg <- "Models differ in observation count (`nobs`); DIC differences may not be directly comparable."
    notes <- c(notes, msg)
    warning(msg, call. = FALSE)
  }
  out_sigs <- vapply(fits, function(f) {
    obs <- f$observed
    if (is.matrix(obs) && !is.null(colnames(obs))) paste(sort(colnames(obs)), collapse = ",") else "1col"
  }, character(1))
  if (length(unique(out_sigs)) > 1L) {
    msg <- "Models differ in modeled outcome columns; verify that likelihoods apply to identical outcome data."
    notes <- c(notes, msg)
    warning(msg, call. = FALSE)
  }
  if (any(!is.finite(tab$dic))) {
    notes <- c(notes, "One or more models have NA DIC (e.g., binary/ordinal probit models).")
  }

  # Compute delta_dic and Akaike/DIC model weights across models with finite DIC
  fin <- is.finite(tab$dic)
  if (any(fin)) {
    min_dic <- min(tab$dic[fin])
    tab$delta_dic[fin] <- tab$dic[fin] - min_dic
    raw_w <- exp(-0.5 * pmin(tab$delta_dic[fin], 700))
    tab$weight[fin] <- raw_w / sum(raw_w)
  }

  if (isTRUE(sort) && nrow(tab) > 1L) {
    ord <- order(!is.finite(tab$dic), tab$dic)
    tab <- tab[ord, , drop = FALSE]
  }
  rownames(tab) <- NULL
  attr(tab, "notes") <- notes
  attr(tab, "rhat_threshold") <- rhat_threshold
  class(tab) <- c("DSEMcompare", "data.frame")
  tab
}

#' @exportS3Method stats::anova
anova.DSEMfit <- function(object, ...) {
  dots <- list(...)
  if (!length(dots)) {
    .dsem_abort("Supply at least one additional `DSEMfit` object to compare via `anova()`.")
  }
  dot_calls <- as.list(match.call(expand.dots = FALSE)$...)
  obj_call <- paste(deparse(substitute(object)), collapse = "")
  nm_dots <- base::names(dots)
  nms <- c(
    obj_call,
    vapply(seq_along(dots), function(i) {
      if (!is.null(nm_dots) && nzchar(nm_dots[i])) nm_dots[i]
      else if (i <= length(dot_calls)) paste(deparse(dot_calls[[i]]), collapse = "")
      else paste0("Model ", i + 1L)
    }, character(1))
  )
  all_fits <- c(list(object), dots)
  dsem_compare(all_fits, names = make.unique(nms, sep = "_"))
}

#' @exportS3Method base::print
print.DSEMcompare <- function(x, digits = 3L, ...) {
  cat("DSEMr Bayesian Model Comparison (DIC)\n")
  disp <- as.data.frame(x)
  num_cols <- c("log_lik", "d_bar", "pd", "dic", "delta_dic", "weight", "max_rhat", "min_ess", "elapsed")
  for (cn in intersect(num_cols, names(disp))) {
    disp[[cn]] <- round(disp[[cn]], digits)
  }
  show_cols <- intersect(
    c("model", "hierarchy", "nobs", "pd", "dic", "delta_dic", "weight", "max_rhat", "converged"),
    names(disp)
  )
  print(disp[, show_cols, drop = FALSE], row.names = FALSE)
  fin <- which(is.finite(x$dic))
  if (length(fin) >= 2L) {
    best_i <- fin[which.min(x$dic[fin])]
    cat(sprintf("\nPreferred by DIC: %s (DIC = %.3f, weight = %.3f)\n",
                x$model[best_i], x$dic[best_i], x$weight[best_i]))
  }
  notes <- attr(x, "notes")
  if (length(notes)) {
    for (nt in notes) cat("Note:", nt, "\n")
  }
  invisible(x)
}
