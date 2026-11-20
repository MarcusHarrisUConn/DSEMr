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
      if (!is.character(value) || length(value) != 1L ||
          !value %in% c("default", "diffuse", "mplus", "penalized")) {
        .dsem_abort("`prior_mode` must be \"default\", \"diffuse\", \"mplus\", or \"penalized\".")
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
  switch(mode, default = 0L, diffuse = 1L, mplus = 1L, penalized = 2L, 0L)
}

.dsem_detect_shared_cov_factor <- function(model) {
  t <- model$terms
  meta <- model$term_meta
  w_regs <- .dsem_active_regs(t, meta, "within")
  outcomes <- unique(w_regs$lhs)
  if (length(outcomes) < 2L) return(NULL)

  # Case A: Ex 9.32b latent residual covariance factor `f BY y1@1 y2@1; logvf | f;`
  meas_w <- which(t$op == "=~" & t$level == "within")
  if (length(meas_w)) {
    w_facs <- unique(t$lhs[meas_w])
    for (fac in w_facs) {
      f_rows <- meas_w[t$lhs[meas_w] == fac]
      inds <- t$rhs[f_rows]
      # Must load strictly onto observed regression outcomes with fixed loadings,
      # have no dynamic regressions of its own, and carry a random log-variance
      all_outcomes <- length(inds) >= 2L && all(inds %in% outcomes)
      has_dyn_f <- any(w_regs$lhs == fac | w_regs$rhs == fac)
      all_fixed <- !is.null(meta) && all(!meta$is_free[f_rows] & !is.na(meta$fixed_value[f_rows]))
      lv_hit <- which(t$op == "~~" & t$level == "within" & t$lhs == fac & t$rhs == fac & t$random)
      if (all_outcomes && !has_dyn_f && all_fixed && length(lv_hit)) {
        loads <- stats::setNames(rep(0, length(outcomes)), outcomes)
        for (idx_r in f_rows) {
          loads[t$rhs[idx_r]] <- as.double(meta$fixed_value[idx_r])
        }
        lbl <- t$label[lv_hit[1L]]
        return(list(
          factor = fac,
          logvf_label = if (nzchar(lbl)) lbl else paste0("logv_", fac),
          loadings = loads
        ))
      }
    }
  }

  # Case B: Direct lavaan shorthand `logvf | y1 ~~ y2`
  cov_hit <- which(t$op == "~~" & t$level == "within" &
                     t$lhs != t$rhs & t$random &
                     t$lhs %in% outcomes & t$rhs %in% outcomes)
  if (length(cov_hit)) {
    loads <- stats::setNames(rep(0, length(outcomes)), outcomes)
    r0 <- cov_hit[1L]
    loads[t$lhs[r0]] <- 1.0
    loads[t$rhs[r0]] <- 1.0
    lbl <- t$label[r0]
    return(list(
      factor = character(),
      logvf_label = if (nzchar(lbl)) lbl else "logvf",
      loadings = loads
    ))
  }
  NULL
}

.dsem_apply_define <- function(data, define_ops, id = NULL) {
  if (!is.data.frame(data) || is.null(define_ops) || !length(define_ops)) return(data)
  df <- data
  id_col <- if (!is.null(id) && length(id) && id[1L] %in% names(df)) id[1L] else NULL
  for (op in define_ops) {
    if (identical(op$type, "standardize")) {
      for (v in op$vars) {
        if (v %in% names(df) && is.numeric(df[[v]])) {
          s <- stats::sd(df[[v]], na.rm = TRUE)
          if (is.finite(s) && s > 0) df[[v]] <- (df[[v]] - mean(df[[v]], na.rm = TRUE)) / s
        }
      }
    } else if (identical(op$type, "assign")) {
      expr_str <- op$expr
      # Replace Mplus-style ** power operator and common uppercase math helpers
      expr_r <- gsub("\\*\\*", "^", expr_str)
      expr_r <- gsub("\\bLOG\\s*\\(", "log(", expr_r, ignore.case = TRUE)
      expr_r <- gsub("\\bEXP\\s*\\(", "exp(", expr_r, ignore.case = TRUE)
      expr_r <- gsub("\\bSQRT\\s*\\(", "sqrt(", expr_r, ignore.case = TRUE)
      expr_r <- gsub("\\bABS\\s*\\(", "abs(", expr_r, ignore.case = TRUE)
      eval_env <- list2env(as.list(df), parent = baseenv())
      eval_env$CLUSTER_MEAN <- function(x) {
        if (is.null(id_col)) return(rep(mean(x, na.rm = TRUE), length(x)))
        stats::ave(as.double(x), df[[id_col]], FUN = function(z) mean(z, na.rm = TRUE))
      }
      eval_env$cluster_mean <- eval_env$CLUSTER_MEAN
      eval_env$MEAN <- function(x) mean(x, na.rm = TRUE)
      eval_env$mean <- eval_env$MEAN
      eval_env$SD <- function(x) stats::sd(x, na.rm = TRUE)
      eval_env$sd <- eval_env$SD
      val <- tryCatch(
        eval(parse(text = expr_r), envir = eval_env),
        error = function(e) .dsem_abort("Failed to evaluate DEFINE statement `%s = %s`: %s",
                                        op$lhs, op$expr, conditionMessage(e))
      )
      if (length(val) == 1L) val <- rep(val, nrow(df))
      df[[op$lhs]] <- as.double(val)
    }
  }
  df
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

.dsem_safe_fit_metrics <- function(dic_val, pd_val, d_bar_val = NA_real_, type = NULL) {
  dic_ok <- is.numeric(dic_val) && length(dic_val) == 1L && is.finite(dic_val)
  pd_ok <- is.numeric(pd_val) && length(pd_val) == 1L && is.finite(pd_val)
  d_bar_ok <- is.numeric(d_bar_val) && length(d_bar_val) == 1L && is.finite(d_bar_val)
  dic_out <- if (dic_ok) as.double(dic_val) else NA_real_
  pd_out <- if (pd_ok) as.double(pd_val) else NA_real_
  d_bar_out <- if (d_bar_ok) {
    as.double(d_bar_val)
  } else if (dic_ok && pd_ok) {
    as.double(dic_val - pd_val)
  } else {
    NA_real_
  }
  ll_out <- if (is.finite(d_bar_out)) -0.5 * d_bar_out else NA_real_
  out <- list(log_lik = ll_out, dic = dic_out, pd = pd_out, d_bar = d_bar_out)
  if (!is.null(type)) out$type <- type
  out
}

.dsem_active_regs <- function(terms, meta = NULL, level = "within") {
  idx <- which(terms$op == "~" & terms$rhs != "1" & terms$level == level)
  if (!length(idx)) return(terms[idx, , drop = FALSE])
  if (!is.null(meta) && nrow(meta) == nrow(terms)) {
    fixed_zero <- !meta$is_free[idx] & !is.na(meta$fixed_value[idx]) & abs(meta$fixed_value[idx]) < 1e-12
    idx <- idx[!fixed_zero]
  }
  terms[idx, , drop = FALSE]
}

.dsem_build_exog_specs <- function(regs, outcomes) {
  ex_rows <- regs[!regs$rhs %in% outcomes, , drop = FALSE]
  if (!nrow(ex_rows)) {
    return(data.frame(var = character(), lag = integer(), key = character(), stringsAsFactors = FALSE))
  }
  keys <- ifelse(ex_rows$lag > 0L, paste0("lag", ex_rows$lag, "_", ex_rows$rhs), ex_rows$rhs)
  keep <- !duplicated(keys)
  data.frame(
    var = ex_rows$rhs[keep],
    lag = as.integer(ex_rows$lag[keep]),
    key = keys[keep],
    stringsAsFactors = FALSE
  )
}

.dsem_build_exog_matrix <- function(df, exog_specs, cluster_sizes = NULL) {
  n_obs <- nrow(df)
  kx <- nrow(exog_specs)
  if (kx == 0L) return(matrix(0, n_obs, 0L))
  x_mat <- matrix(0, n_obs, kx, dimnames = list(NULL, exog_specs$key))
  if (is.null(cluster_sizes)) cluster_sizes <- n_obs
  for (k in seq_len(kx)) {
    v_raw <- as.double(df[[exog_specs$var[k]]])
    lg <- exog_specs$lag[k]
    if (lg == 0L) {
      x_mat[, k] <- v_raw
    } else {
      lagged <- rep(NA_real_, n_obs)
      pos <- 0L
      for (sz in cluster_sizes) {
        if (sz > lg) {
          src_idx <- (pos + 1L):(pos + sz - lg)
          dst_idx <- (pos + 1L + lg):(pos + sz)
          lagged[dst_idx] <- v_raw[src_idx]
        }
        # Fill boundary rows within cluster with cluster mean (never bleeding from prior cluster)
        c_slice <- v_raw[(pos + 1L):(pos + sz)]
        c_fill <- mean(c_slice, na.rm = TRUE)
        if (!is.finite(c_fill)) c_fill <- 0
        b_end <- min(lg, sz)
        lagged[(pos + 1L):(pos + b_end)] <- c_fill
        pos <- pos + sz
      }
      x_mat[, k] <- lagged
    }
  }
  x_mat
}

.dsem_normalize_censored <- function(censored) {
  if (is.null(censored) || !length(censored)) return(list())
  if (is.character(censored)) {
    if (anyNA(censored) || any(!nzchar(censored))) {
      .dsem_abort("`censored` variable names must be non-empty character strings.")
    }
    out <- as.list(rep("below", length(censored)))
    names(out) <- censored
    return(out)
  }
  if (!is.list(censored) || is.null(names(censored)) || any(!nzchar(names(censored)))) {
    .dsem_abort("`censored` must be NULL, a character vector of outcome names, or a named list of bounds/directions.")
  }
  out <- list()
  for (nm in names(censored)) {
    spec <- censored[[nm]]
    if (is.character(spec) && length(spec) == 1L) {
      dir <- tolower(trimws(spec))
      if (!dir %in% c("below", "left", "b", "above", "right", "a", "both", "interval", "ab")) {
        .dsem_abort("Unsupported censoring direction `%s` for `%s`.", spec, nm)
      }
      out[[nm]] <- if (dir %in% c("below", "left", "b")) "below" else if (dir %in% c("above", "right", "a")) "above" else "both"
    } else if (is.numeric(spec) && length(spec) == 2L && !anyNA(spec) && spec[1L] < spec[2L]) {
      out[[nm]] <- as.double(spec)
    } else {
      .dsem_abort("Censoring specification for `%s` must be \"below\", \"above\", \"both\", or a numeric vector c(lower, upper).", nm)
    }
  }
  out
}

.dsem_resolve_censoring <- function(censored, y_mat, outcomes, outcome_types) {
  d <- length(outcomes)
  lo <- rep(-Inf, d)
  hi <- rep(Inf, d)
  if (!length(censored)) return(list(outcome_types = outcome_types, lo = lo, hi = hi))
  for (j in seq_len(d)) {
    nm <- outcomes[j]
    if (!nm %in% names(censored)) next
    if (outcome_types[j] >= 2L) {
      .dsem_abort("Outcome `%s` cannot be both categorical and censored.", nm)
    }
    spec <- censored[[nm]]
    obs <- y_mat[, j]
    obs <- obs[is.finite(obs)]
    if (!length(obs)) .dsem_abort("Censored outcome `%s` has no finite observations.", nm)
    if (is.character(spec)) {
      if (identical(spec, "below")) {
        outcome_types[j] <- -1L
        lo[j] <- min(obs)
      } else if (identical(spec, "above")) {
        outcome_types[j] <- -2L
        hi[j] <- max(obs)
      } else {
        outcome_types[j] <- -3L
        lo[j] <- min(obs)
        hi[j] <- max(obs)
      }
    } else {
      has_lo <- is.finite(spec[1L])
      has_hi <- is.finite(spec[2L])
      if (has_lo && has_hi) {
        outcome_types[j] <- -3L; lo[j] <- spec[1L]; hi[j] <- spec[2L]
      } else if (has_lo) {
        outcome_types[j] <- -1L; lo[j] <- spec[1L]
      } else if (has_hi) {
        outcome_types[j] <- -2L; hi[j] <- spec[2L]
      }
    }
  }
  list(outcome_types = as.integer(outcome_types), lo = as.double(lo), hi = as.double(hi))
}
