#' Compute Mplus-compatible standardized DSEM estimates (STDYX, STDY, STD)
#'
#' Computes posterior summaries of standardized model parameters matching Mplus
#' `OUTPUT: STDYX STDY STD;`. For multilevel (`twolevel`, `threelevel`, and
#' `crossclassified`) models, within-cluster slopes are standardized within each
#' cluster using cluster-specific model-implied variances and then averaged
#' across clusters at each posterior draw following Schuurman, Ferrer,
#' de Boer-Sonnenschein, and Hamaker (2016).
#'
#' @param object A fitted `DSEMfit` object.
#' @param type Standardization convention: `"stdyx"` (standardizes by both
#'   outcome and predictor SDs), `"stdy"` (standardizes by outcome SD only), or
#'   `"std"` (standardizes latent variables only).
#' @param prob Credible interval probability (default `0.95`).
#' @return A data frame of standardized parameter summaries (`parameter`,
#'   `type`, `mean`, `sd`, `q_lower`, `q50`, `q_upper`, `p_one_tailed`).
#' @references
#' Schuurman, N. K., Ferrer, E., de Boer-Sonnenschein, M., & Hamaker, E. L.
#' (2016). How to compare cross-lagged associations in a multilevel
#' autoregressive model. *Psychological Methods*, 21(2), 206-221.
#' \doi{10.1037/met0000062}
#' @examples
#' dat <- simulate_dsem(n = 80, ar = 0.5, sigma = 0.8, seed = 10)
#' fit <- dsem("y ~ lag(y, 1)", dat, chains = 2, iter = 200, warmup = 100, seed = 10)
#' dsem_standardize(fit, type = "stdyx")
#' @export
dsem_standardize <- function(object, type = c("stdyx", "stdy", "std"), prob = 0.95) {
  if (!inherits(object, "DSEMfit")) .dsem_abort("`object` must be a `DSEMfit`.")
  type <- match.arg(type)
  if (!is.numeric(prob) || length(prob) != 1L || !is.finite(prob) || prob <= 0 || prob >= 1) {
    .dsem_abort("`prob` must be strictly between 0 and 1.")
  }
  alpha <- (1 - prob) / 2
  draws <- object$draws
  params <- dimnames(draws)[[2L]]
  n_keep <- dim(draws)[1L]
  n_chains <- dim(draws)[3L]
  n_tot <- n_keep * n_chains
  flat <- matrix(aperm(draws, c(1L, 3L, 2L)), nrow = n_tot, ncol = length(params))
  colnames(flat) <- params

  obs_y <- object$observed
  obs_sd <- if (is.matrix(obs_y)) {
    apply(obs_y, 2L, stats::sd, na.rm = TRUE)
  } else {
    stats::sd(obs_y, na.rm = TRUE)
  }

  std_list <- list()
  # Case A: Two-level / multilevel with cluster random-effect draws
  if (!is.null(object$random_effect_draws)) {
    re <- object$random_effect_draws # iter x cluster x param x chain
    re_params <- dimnames(re)[[3L]]
    g_cnt <- dim(re)[2L]
    for (p_name in re_params) {
      if (grepl("^lag[0-9]+_", p_name) || grepl("_on_", p_name)) {
        vals <- numeric(n_tot)
        idx_p <- match(p_name, re_params)
        logv_idx <- grep("^logv", re_params)
        pos <- 0L
        for (ch in seq_len(n_chains)) {
          for (it in seq_len(n_keep)) {
            pos <- pos + 1L
            slopes_c <- re[it, , idx_p, ch]
            if (type == "stdyx" && grepl("^lag1_", p_name)) {
              # For stationary AR(1), within-cluster SD_y == SD_{y,t-1} -> stdyx == phi_i!
              vals[pos] <- mean(pmin(pmax(slopes_c, -0.999), 0.999))
            } else if (length(logv_idx)) {
              sd_e <- exp(0.5 * re[it, , logv_idx[1L], ch])
              ar_idx <- grep("^lag1_", re_params)
              phi_c <- if (length(ar_idx)) pmin(abs(re[it, , ar_idx[1L], ch]), 0.98) else 0
              sd_y <- sd_e / sqrt(pmax(1 - phi_c^2, 0.02))
              scale_x <- if (type == "stdyx" && !is.null(object$within_x_sd)) object$within_x_sd[1L] else 1
              vals[pos] <- mean(slopes_c * scale_x / pmax(sd_y, 1e-6))
            } else {
              vals[pos] <- mean(slopes_c)
            }
          }
        }
        std_list[[paste0("within_std_", p_name)]] <- vals
      }
    }
  }

  # Case B: Population / single-level / latent parameters in `flat`
  for (p_name in params) {
    v <- flat[, p_name]
    if (grepl("^lag1_", p_name) || grepl("^ar1_", p_name)) {
      # For univariate AR(1), Var(y_t) = Var(y_{t-1}), so STDYX(phi) = phi * sd(y_{t-1}) / sd(y_t) = phi
      std_v <- if (type == "stdy") {
        sig_col <- intersect(c("sigma2", "factor_variance"), params)
        if (length(sig_col)) {
          sd_y <- sqrt(pmax(flat[, sig_col[1L]], 1e-8) / pmax(1 - pmin(v^2, 0.99), 0.01))
          v / sd_y
        } else v
      } else {
        v
      }
      std_list[[p_name]] <- std_v
    } else if (grepl("^loading_", p_name)) {
      item <- sub("^loading_", "", p_name)
      fvar_col <- intersect(c("factor_variance", grep("^psi_", params, value = TRUE)), params)
      rvar_col <- paste0("residual_variance_", item)
      sd_eta <- if (length(fvar_col)) {
        ar_col <- grep("^ar1_", params, value = TRUE)
        phi_sq <- if (length(ar_col)) pmin(flat[, ar_col[1L]]^2, 0.98) else 0
        sqrt(pmax(flat[, fvar_col[1L]], 1e-8) / pmax(1 - phi_sq, 0.02))
      } else {
        rep(1, n_tot)
      }
      sd_item <- if (rvar_col %in% params) {
        sqrt(v^2 * sd_eta^2 + pmax(flat[, rvar_col], 1e-8))
      } else {
        sqrt(v^2 * sd_eta^2 + 1)
      }
      std_list[[p_name]] <- switch(
        type,
        stdyx = v * sd_eta / pmax(sd_item, 1e-8),
        stdy = v / pmax(sd_item, 1e-8),
        std = v * sd_eta
      )
    } else if (!grepl("^(sigma2|tau_|cov_|omega_|residual_variance_|factor_variance|\\(Intercept\\)|intercept_)", p_name)) {
      # Covariate or cross-lagged slope
      if (type == "std") {
        std_list[[p_name]] <- v
      } else {
        x_sd <- if (!is.null(object$data) && p_name %in% names(object$data)) {
          stats::sd(object$data[[p_name]], na.rm = TRUE)
        } else 1
        y_sd <- if (length(obs_sd) == 1L && is.finite(obs_sd) && obs_sd > 0) obs_sd else 1
        mult <- if (type == "stdyx") x_sd / y_sd else 1 / y_sd
        std_list[[p_name]] <- v * mult
      }
    }
  }

  if (!length(std_list)) {
    for (p_name in params) std_list[[p_name]] <- flat[, p_name]
  }

  rows <- lapply(names(std_list), function(nm) {
    vals <- std_list[[nm]]
    p_one <- min(mean(vals > 0, na.rm = TRUE), mean(vals < 0, na.rm = TRUE))
    data.frame(
      parameter = nm,
      type = toupper(type),
      mean = mean(vals, na.rm = TRUE),
      sd = stats::sd(vals, na.rm = TRUE),
      q_lower = stats::quantile(vals, alpha, names = FALSE, na.rm = TRUE),
      q50 = stats::quantile(vals, 0.5, names = FALSE, na.rm = TRUE),
      q_upper = stats::quantile(vals, 1 - alpha, names = FALSE, na.rm = TRUE),
      p_one_tailed = p_one,
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}
