.dsem_abort <- function(..., call. = FALSE) {
  stop(sprintf(...), call. = call.)
}

.dsem_trim <- function(x) trimws(gsub("[\r\t]", " ", x))

.dsem_lines <- function(model) {
  lines <- unlist(strsplit(model, "\n", fixed = TRUE), use.names = FALSE)
  lines <- sub("#.*$", "", lines)
  lines <- .dsem_trim(lines)
  lines[nzchar(lines)]
}

.dsem_stable_seed <- function(master_seed, task) {
  bytes <- as.integer(charToRaw(enc2utf8(as.character(task))))
  hash <- as.double(abs(as.integer(master_seed))) %% 2147483629
  for (byte in bytes) hash <- (hash * 131 + byte) %% 2147483629
  as.integer(hash + 1)
}

.dsem_version_manifest <- function() {
  list(
    DSEMr = as.character(utils::packageVersion("DSEMr")),
    R = paste(R.version$major, R.version$minor, sep = "."),
    platform = R.version$platform,
    lavaan = as.character(utils::packageVersion("lavaan"))
  )
}

.dsem_is_installed <- function() {
  "DSEMr" %in% loadedNamespaces() || requireNamespace("DSEMr", quietly = TRUE)
}

.dsem_validate_data <- function(data, id, time, variables) {
  if (!is.data.frame(data)) .dsem_abort("`data` must be a data frame.")
  needed <- unique(c(id, time, variables))
  missing <- setdiff(needed, names(data))
  if (length(missing)) {
    .dsem_abort("Data are missing required variable(s): %s.", paste(missing, collapse = ", "))
  }
  if (!is.null(id) && anyNA(data[[id]])) .dsem_abort("`id` cannot contain missing values.")
  if (!is.null(time) && anyNA(data[[time]])) .dsem_abort("`time` cannot contain missing values.")
  invisible(TRUE)
}

.dsem_lag_pairs <- function(time, interval = 1) {
  n <- length(time)
  if (n < 2L) return(list(previous = integer(), current = integer()))
  if (!is.numeric(interval) || length(interval) != 1L || !is.finite(interval) || interval <= 0) {
    .dsem_abort("`metadata$time_interval` must be one positive number.")
  }
  numeric_time <- if (inherits(time, "Date") || inherits(time, "POSIXt")) {
    as.numeric(time)
  } else {
    suppressWarnings(as.numeric(time))
  }
  if (anyNA(numeric_time)) .dsem_abort("The time index must be numeric, Date, or POSIXt.")
  delta <- diff(numeric_time)
  if (any(delta <= 0)) .dsem_abort("Time values must be unique and strictly increasing within each cluster.")
  current <- which(abs(delta - interval) <= sqrt(.Machine$double.eps) * max(1, abs(interval))) + 1L
  list(previous = current - 1L, current = current)
}

.dsem_time_interval <- function(model) {
  interval <- model$metadata$time_interval
  if (is.null(interval)) 1 else interval
}
