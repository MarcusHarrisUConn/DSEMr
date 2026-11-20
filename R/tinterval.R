#' Align irregular timestamps onto a regular DSEM time grid
#'
#' Implements order-preserving time-interval discretization (`TINTERVAL`;
#' Asparouhov, Hamaker, & Muthen, 2018). Continuous or unequally spaced
#' timestamps are mapped onto an integer grid of multiples of `delta`,
#' preserving observation order within each cluster and inserting explicit `NA`
#' rows at unobserved time bins so lagged transitions correspond to uniform
#' elapsed time.
#'
#' @param data A data frame in long format.
#' @param time Name of the numeric, `Date`, or `POSIXt` time column.
#' @param delta Positive bin width (`TINTERVAL` step size).
#' @param id Optional character vector of one or two cluster identifier columns
#'   (e.g., `"person"` for two-level or `c("day", "person")` for three-level).
#' @param max_bins Maximum number of grid points permitted per series to guard
#'   against accidental micro-intervals.
#' @return A data frame augmented with integer `.dsem_bin` and aligned grid
#'   values in `time`, with `NA` entries in measurement columns at inserted
#'   empty time bins. Attribute `"tinterval_info"` records observed versus
#'   inserted row counts and maximum timestamp shift.
#' @references
#' Asparouhov, T., Hamaker, E. L., & Muthen, B. (2018). Dynamic structural
#' equation models. *Structural Equation Modeling*, 25(3), 359-388.
#' \doi{10.1080/10705511.2017.1406803}
#' @examples
#' irregular <- data.frame(
#'   id = c(1, 1, 1, 2, 2, 2),
#'   time = c(0.1, 1.05, 3.1, 0.0, 2.0, 3.0),
#'   y = c(0.5, 0.8, -0.2, 1.1, 0.4, 0.2)
#' )
#' aligned <- dsem_tinterval(irregular, time = "time", delta = 1, id = "id")
#' head(aligned)
#' @export
dsem_tinterval <- function(data, time, delta = 1, id = NULL, max_bins = 10000L) {
  if (!is.data.frame(data)) .dsem_abort("`data` must be a data frame.")
  if (!is.character(time) || length(time) != 1L || !time %in% names(data)) {
    .dsem_abort("`time` must name a column in `data`.")
  }
  if (!is.numeric(delta) || length(delta) != 1L || !is.finite(delta) || delta <= 0) {
    .dsem_abort("`delta` must be one positive finite number.")
  }
  max_bins <- .dsem_integer(max_bins, "max_bins", 4)
  if (!is.null(id)) {
    if (!is.character(id) || any(!id %in% names(data))) {
      .dsem_abort("`id` must name existing cluster column(s) in `data`.")
    }
  }
  t_raw <- data[[time]]
  if (!(is.numeric(t_raw) || inherits(t_raw, "Date") || inherits(t_raw, "POSIXt"))) {
    .dsem_abort("`time` column must be numeric, Date, or POSIXt.")
  }
  if (anyNA(t_raw) || any(!is.finite(as.numeric(t_raw)))) {
    .dsem_abort("`time` column cannot contain missing or infinite values.")
  }

  if (is.null(id)) {
    keys <- rep("1", nrow(data))
  } else {
    keys <- do.call(paste, c(lapply(id, function(col) data[[col]]), sep = "\r"))
  }
  unique_keys <- unique(keys)
  pieces <- vector("list", length(unique_keys))
  max_shift <- 0
  inserted_total <- 0L

  for (g in seq_along(unique_keys)) {
    rows <- which(keys == unique_keys[g])
    sub <- data[rows, , drop = FALSE]
    t_num <- as.numeric(sub[[time]])
    ord <- order(t_num)
    sub <- sub[ord, , drop = FALSE]
    t_num <- t_num[ord]
    t0 <- t_num[1L]
    # Order-preserving integer bin assignment (Mplus TINTERVAL rule)
    raw_bins <- as.integer(floor((t_num - t0) / delta + 0.5 + 1e-12)) + 1L
    bins <- raw_bins
    if (length(bins) > 1L) {
      for (i in 2:length(bins)) {
        if (bins[i] <= bins[i - 1L]) {
          bins[i] <- bins[i - 1L] + 1L
        }
      }
    }
    n_grid <- bins[length(bins)]
    if (n_grid > max_bins) {
      .dsem_abort(
        "Discretizing with `delta = %g` creates %d bins (> `max_bins = %d`). Increase `delta` or `max_bins`.",
        delta, n_grid, max_bins
      )
    }
    grid_times <- t0 + (seq_len(n_grid) - 1L) * delta
    shifts <- abs(t_num - (t0 + (bins - 1L) * delta)) / delta
    max_shift <- max(max_shift, shifts)

    full <- sub[rep(1L, n_grid), , drop = FALSE]
    for (nm in names(full)) {
      if (!nm %in% c(id, time)) {
        is.na(full[[nm]]) <- seq_len(n_grid)
      }
    }
    if (inherits(t_raw, "Date")) {
      full[[time]] <- as.Date(grid_times, origin = "1970-01-01")
    } else if (inherits(t_raw, "POSIXt")) {
      full[[time]] <- as.POSIXct(grid_times, origin = "1970-01-01", tz = attr(t_raw, "tzone"))
    } else {
      full[[time]] <- grid_times
    }
    full[bins, ] <- sub
    if (inherits(t_raw, "Date")) {
      full[[time]] <- as.Date(grid_times, origin = "1970-01-01")
    } else if (inherits(t_raw, "POSIXt")) {
      full[[time]] <- as.POSIXct(grid_times, origin = "1970-01-01", tz = attr(t_raw, "tzone"))
    } else {
      full[[time]] <- grid_times
    }
    full$.dsem_bin <- seq_len(n_grid)
    inserted_total <- inserted_total + (n_grid - nrow(sub))
    pieces[[g]] <- full
  }

  out <- do.call(rbind, pieces)
  rownames(out) <- NULL
  attr(out, "tinterval_info") <- list(
    delta = delta,
    observed_rows = nrow(data),
    total_grid_rows = nrow(out),
    inserted_missing_rows = inserted_total,
    max_relative_shift = max_shift
  )
  out
}
