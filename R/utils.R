.dsem_abort <- function(..., call. = FALSE) {
  stop(sprintf(...), call. = call.)
}

.dsem_trim <- function(x) trimws(gsub("[\r\t]", " ", x))

.dsem_lines <- function(model) {
  raw <- unlist(strsplit(model, "\n", fixed = TRUE), use.names = FALSE)
  raw <- sub("#.*$", "", raw)
  parts <- unlist(strsplit(raw, ";", fixed = TRUE), use.names = FALSE)
  parts <- .dsem_trim(parts)
  parts[nzchar(parts)]
}

.dsem_stable_seed <- function(master_seed, task) {
  bytes <- as.integer(charToRaw(enc2utf8(as.character(task))))
  hash <- as.double(abs(as.integer(master_seed))) %% 2147483629
  for (byte in bytes) hash <- (hash * 131 + byte) %% 2147483629
  as.integer(hash + 1)
}

.dsem_rng_scope <- function(seed) {
  old_kind <- RNGkind()
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  old_seed <- if (had_seed) get(".Random.seed", envir = .GlobalEnv) else NULL
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  set.seed(seed)
  function() {
    do.call(RNGkind, as.list(old_kind))
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }
}

.dsem_version_manifest <- function() {
  pkg_ver <- tryCatch(
    as.character(utils::packageVersion("DSEMr")),
    error = function(e) "0.1.0"
  )
  list(
    DSEMr = pkg_ver,
    R = paste(R.version$major, R.version$minor, sep = "."),
    platform = R.version$platform,
    lavaan = as.character(utils::packageVersion("lavaan")),
    posterior = as.character(utils::packageVersion("posterior")),
    rng_kind = c("Mersenne-Twister", "Inversion", "Rejection")
  )
}

.dsem_validate_data <- function(data, id, time, variables,
                                allow_missing_vars = TRUE,
                                categorical = character()) {
  if (!is.data.frame(data)) .dsem_abort("`data` must be a data frame.")
  needed <- unique(c(id, time, variables))
  needed <- needed[nzchar(needed)]
  missing <- setdiff(needed, names(data))
  if (length(missing)) {
    .dsem_abort("Data are missing required variable(s): %s.", paste(missing, collapse = ", "))
  }
  for (id_col in id) {
    if (anyNA(data[[id_col]])) .dsem_abort("Cluster identifier `%s` cannot contain missing values.", id_col)
  }
  if (!is.null(time) && anyNA(data[[time]])) .dsem_abort("`time` cannot contain missing values.")
  for (variable in variables) {
    col <- data[[variable]]
    if (variable %in% categorical && (is.factor(col) || is.logical(col))) {
      next
    }
    if (!is.numeric(col) || any(is.infinite(col))) {
      .dsem_abort("Variable `%s` must be numeric with finite values or NA.", variable)
    }
    if (!allow_missing_vars && anyNA(col)) {
      .dsem_abort("Variable `%s` contains unexpected missing values.", variable)
    }
  }
  if (!is.null(time) && !(is.numeric(data[[time]]) || inherits(data[[time]], "Date") || inherits(data[[time]], "POSIXt"))) {
    .dsem_abort("The time index must be numeric, Date, or POSIXt.")
  }
  if (!is.null(time) && any(!is.finite(as.numeric(data[[time]])))) {
    .dsem_abort("The time index must be finite.")
  }
  invisible(TRUE)
}

.dsem_lag_pairs <- function(time, interval = 1, lag = 1L) {
  n <- length(time)
  lag <- as.integer(lag)
  if (n < 2L) return(list(previous = integer(), current = integer()))
  if (!is.numeric(interval) || length(interval) != 1L || !is.finite(interval) || interval <= 0) {
    .dsem_abort("`metadata$time_interval` must be one positive number.")
  }
  numeric_time <- if (inherits(time, "Date") || inherits(time, "POSIXt")) {
    as.numeric(time)
  } else {
    suppressWarnings(as.numeric(time))
  }
  if (any(!is.finite(numeric_time))) .dsem_abort("The time index must be finite numeric, Date, or POSIXt.")
  if (any(diff(numeric_time) <= 0)) .dsem_abort("Time values must be unique and strictly increasing within each cluster.")
  target <- lag * interval
  tol <- 1e-6 * max(1, abs(target))
  prev_idx <- integer()
  curr_idx <- integer()
  for (i in seq_len(n)) {
    diffs <- abs((numeric_time[i] - numeric_time[seq_len(i - 1L)]) - target)
    hit <- which(diffs <= tol)
    if (length(hit)) {
      prev_idx <- c(prev_idx, hit[1L])
      curr_idx <- c(curr_idx, i)
    }
  }
  list(previous = prev_idx, current = curr_idx)
}

.dsem_time_interval <- function(model) {
  interval <- model$metadata$time_interval
  if (is.null(interval)) interval <- model$tinterval
  if (is.null(interval)) 1 else interval
}

.dsem_integer <- function(value, name, minimum = 0) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
      value != floor(value) || value < minimum || value > .Machine$integer.max) {
    .dsem_abort("`%s` must be one integer between %s and %s.", name, minimum, .Machine$integer.max)
  }
  as.integer(value)
}

.dsem_named_controls <- function(control) {
  if (!is.list(control) || (length(control) &&
      (is.null(names(control)) || anyNA(names(control)) || any(!nzchar(names(control))) || anyDuplicated(names(control))))) {
    .dsem_abort("`control` must be a uniquely named list.")
  }
}

.dsem_validate_prior <- function(prior, control, allowed) {
  unknown <- setdiff(names(control), allowed)
  if (length(unknown)) .dsem_abort("Unsupported control(s): %s.", paste(unknown, collapse = ", "))
  for (name in allowed) {
    value <- prior[[name]]
    if (name == "prior_mode") {
      if (!is.character(value) || length(value) != 1L || !value %in% c("default", "mplus", "penalized")) {
        .dsem_abort("`prior_mode` must be \"default\", \"mplus\", or \"penalized\".")
      }
    } else if (name == "omega_scale") {
      if (!is.matrix(value) || !is.numeric(value) || nrow(value) != ncol(value) ||
          any(!is.finite(value)) || !isSymmetric(value) ||
          inherits(try(chol(value), silent = TRUE), "try-error")) {
        .dsem_abort("`omega_scale` must be a symmetric positive-definite matrix.")
      }
    } else {
      if (!is.numeric(value) || length(value) != 1L || !is.finite(value)) {
        .dsem_abort("`%s` must be one finite number.", name)
      }
      if (!grepl("_mean$", name) && value <= 0) .dsem_abort("`%s` must be positive.", name)
    }
  }
  prior
}

.dsem_prior_mode_int <- function(mode) {
  switch(mode, default = 0L, mplus = 1L, penalized = 2L, 0L)
}

.dsem_spd_solve <- function(a, b = NULL) {
  p <- nrow(a)
  a <- 0.5 * (a + t(a))
  for (jit in c(0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)) {
    cand <- a + diag(jit, p)
    ch <- try(chol(cand), silent = TRUE)
    if (!inherits(ch, "try-error")) {
      if (is.null(b)) return(chol2inv(ch))
      return(backsolve(ch, forwardsolve(t(ch), b)))
    }
  }
  if (is.null(b)) diag(1, p) else rep(0, p)
}

.dsem_rmvnorm_prec_R <- function(precision, info) {
  p <- nrow(precision)
  if (p == 0L) return(numeric())
  cov_m <- .dsem_spd_solve(precision)
  mu <- as.numeric(cov_m %*% info)
  ch <- try(chol(cov_m), silent = TRUE)
  if (inherits(ch, "try-error")) ch <- chol(cov_m + diag(1e-6, p))
  mu + as.numeric(t(ch) %*% stats::rnorm(p))
}

.dsem_rinv_wishart_R <- function(df, scale_mat) {
  p <- nrow(scale_mat)
  if (p == 1L) {
    val <- 1 / stats::rgamma(1L, shape = max(0.5 * df, 0.5), rate = max(0.5 * scale_mat[1L, 1L], 1e-8))
    return(matrix(val, 1L, 1L))
  }
  inv_s <- .dsem_spd_solve(scale_mat)
  w <- stats::rWishart(1L, max(df, p + 1), inv_s)[, , 1L]
  .dsem_spd_solve(w)
}

.dsem_rtruncnorm_R <- function(mean, sd, lower, upper) {
  a <- (lower - mean) / sd
  b <- (upper - mean) / sd
  pa <- stats::pnorm(a)
  pb <- stats::pnorm(b)
  if (pb - pa > 1e-10) {
    u <- stats::runif(1L, pa, pb)
    z <- stats::qnorm(min(max(u, 1e-15), 1 - 1e-15))
    return(mean + sd * min(max(z, a), b))
  }
  if (is.finite(a) && a > 0) return(mean + sd * (a + stats::rexp(1L, rate = a)))
  if (is.finite(b) && b < 0) return(mean + sd * (b - stats::rexp(1L, rate = abs(b))))
  mean
}

.dsem_encode_categorical <- function(vec) {
  if (is.logical(vec)) return(list(values = ifelse(is.na(vec), NA_real_, as.double(vec)), k = 2L))
  if (is.factor(vec)) {
    lev <- levels(vec)
    if (length(lev) < 2L) .dsem_abort("Categorical outcome requires at least two categories.")
    return(list(values = as.double(as.integer(vec) - 1L), k = length(lev)))
  }
  obs <- sort(unique(vec[!is.na(vec)]))
  if (length(obs) < 2L) .dsem_abort("Categorical outcome requires at least two observed categories.")
  mapped <- match(vec, obs) - 1L
  list(values = as.double(mapped), k = length(obs))
}
