use crate::linalg::{
    chol_lower_jitter, eye, idx, mat_t_mul, quad_form, rinv_wishart, rmvnorm_precision, spd_invert,
    spd_logdet, symmetrize,
};
use crate::rng::{log_std_norm_cdf, Rng};
use std::f64::consts::PI;

#[allow(clippy::too_many_arguments)]
pub struct VarN1Output {
    pub draws: Vec<f64>, // n_keep x n_params (col-major)
    pub param_count: usize,
    pub fitted_mean: Vec<f64>, // n_time x d (col-major)
    pub dic: f64,
    pub pd: f64,
    pub d_bar: f64,
}

/// General N=1 Multivariate VAR(p) / Covariate / RDSEM / Probit / Tobit Censored Gibbs sampler.
/// Supports:
/// - d >= 1 outcomes, max_lag >= 1
/// - kx >= 0 exogenous covariates
/// - residual_dynamics (RDSEM: y_t = alpha + B_x x_t + u_t, u_t = sum_l Phi_l u_{t-l} + e_t)
/// - outcome_types: `0` = continuous uncensored, `>= 2` = K-category ordinal/binary probit
///   (values `0..K-1`), `-1` = left-censored (below `censor_lo`), `-2` = right-censored
///   (above `censor_hi`), `-3` = interval/two-sided censored at `[censor_lo, censor_hi]`
/// - missing values (NaN) imputed in Gibbs on the regular / TINTERVAL grid
/// - prior_mode: 0 = weakly informative proper, 1 = diffuse flat/IG(-1,0), 2 = PSEM shrinkage (Laplace/Lasso)
#[allow(clippy::too_many_arguments)]
pub fn mcmc_var_n1(
    y_in: &[f64], // n_time x d (col-major)
    x_in: &[f64], // n_time x kx (col-major)
    n_time: usize,
    d: usize,
    kx: usize,
    lags: &[usize],        // active lag orders, e.g. [1] or [1, 2]
    outcome_types: &[i32], // 0 = cont, >=2 = K-cat probit, -1/-2/-3 = censored
    censor_lo: &[f64],     // length d (lower censoring thresholds)
    censor_hi: &[f64],     // length d (upper censoring thresholds)
    residual_dynamics: bool,
    prior_mode: i32, // 0 = default, 1 = diffuse, 2 = penalized (PSEM)
    beta_sd: f64,
    sigma_shape: f64,
    sigma_scale: f64,
    iter: usize,
    warmup: usize,
    thin: usize,
    seed: u64,
) -> VarN1Output {
    let mut rng = Rng::from_seed(seed);
    let max_lag = lags.iter().copied().max().unwrap_or(1).max(1);
    let n_lags = lags.len();
    let is_cat: Vec<bool> = outcome_types.iter().map(|&k| k >= 2).collect();
    let is_cens: Vec<bool> = outcome_types.iter().map(|&k| k < 0).collect();
    let any_cens = is_cens.iter().any(|&b| b);

    // Initialize working continuous latent/imputed y_work (n_time x d)
    // cens_flag: 0 = observed uncensored, -1 = left-censored (<= lo), 1 = right-censored (>= hi)
    let mut y_work = y_in.to_vec();
    let mut missing_mask = vec![false; n_time * d];
    let mut cens_flag = vec![0i8; n_time * d];
    for col in 0..d {
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for t in 0..n_time {
            let v = y_in[idx(t, col, n_time)];
            if v.is_finite() {
                sum += v;
                cnt += 1;
            } else {
                missing_mask[idx(t, col, n_time)] = true;
            }
        }
        let mean = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
        let lo_c = censor_lo.get(col).copied().unwrap_or(f64::NEG_INFINITY);
        let hi_c = censor_hi.get(col).copied().unwrap_or(f64::INFINITY);
        let ot = outcome_types[col];
        for t in 0..n_time {
            let cell = idx(t, col, n_time);
            if missing_mask[cell] {
                y_work[cell] = if is_cat[col] { 0.0 } else { mean };
            } else if is_cat[col] {
                let k_cat = outcome_types[col].max(2) as f64;
                let cat = y_in[cell];
                y_work[cell] = ((cat + 0.5) / k_cat - 0.5) * 2.0;
            } else if ot < 0 {
                let v = y_in[cell];
                if (ot == -1 || ot == -3) && lo_c.is_finite() && v <= lo_c + 1e-12 {
                    cens_flag[cell] = -1;
                    y_work[cell] = lo_c - 0.25;
                } else if (ot == -2 || ot == -3) && hi_c.is_finite() && v >= hi_c - 1e-12 {
                    cens_flag[cell] = 1;
                    y_work[cell] = hi_c + 0.25;
                }
            }
        }
    }

    // Thresholds for ordinal outcomes (K >= 3): tau[col] has K-1 thresholds.
    // For binary (K=2), fix threshold tau_1 = -alpha_j (equivalently fix intercept free & threshold=0 on y*,
    // Mplus reports threshold = -intercept!). For K >= 3, fix tau_1 = 0, free intercept, sample tau_2..tau_{K-1}.
    let mut extra_thresholds: Vec<Vec<f64>> = (0..d)
        .map(|col| {
            let k = outcome_types[col];
            if k >= 3 {
                (0..(k as usize - 1)).map(|c| c as f64 * 0.8).collect()
            } else {
                Vec::new()
            }
        })
        .collect();

    // Parameters:
    // If !residual_dynamics:
    //   For each outcome j in 0..d: design row at t >= max_lag is [1, y_{t-l1}, ..., y_{t-lL}, x_t]
    //   p_reg = 1 + n_lags * d + kx. B is p_reg x d.
    // If residual_dynamics (RDSEM):
    //   Structural eq: y_t = alpha + B_x x_t + u_t  (p_struct = 1 + kx, B_s is p_struct x d)
    //   Residual eq:   u_t = sum_l Phi_l u_{t-l} + e_t (p_dyn = n_lags * d, Phi is p_dyn x d)
    //   Total regression coeffs per outcome = 1 + n_lags * d + kx (same count & ordering: [intercept, lags..., x...])!
    let p_reg = 1 + n_lags * d + kx;
    let mut beta_mat = vec![0.0; p_reg * d]; // col-major p_reg x d
    for j in 0..d {
        // Initialize self-AR1 around 0.2
        if n_lags >= 1 {
            beta_mat[idx(1 + j, j, p_reg)] = 0.2;
        }
    }
    let mut sigma = eye(d, 1.0);
    // PSEM local shrinkage variances tau2_shrink for non-intercept coefficients
    let mut shrink_tau2: Vec<f64> = vec![1.0_f64; p_reg * d];
    let mut lambda_lasso = 1.0_f64;

    let n_cov_params = d * (d + 1) / 2;
    let n_extra_thresh: usize = extra_thresholds
        .iter()
        .map(|v| v.len().saturating_sub(1))
        .sum();
    let n_params = d * p_reg + n_cov_params + n_extra_thresh;

    let keep_indices: Vec<usize> = ((warmup + 1)..=iter)
        .filter(|i| (i - warmup - 1) % thin == 0)
        .collect();
    let n_keep = keep_indices.len();
    let mut draws = vec![0.0; n_keep * n_params];
    let mut fitted_accum = vec![0.0; n_time * d];
    let mut beta_accum = vec![0.0; p_reg * d];
    let mut sigma_accum = vec![0.0; d * d];
    let mut deviance_sum = 0.0;
    let mut keep_pos = 0usize;

    let n_eff = n_time.saturating_sub(max_lag);

    for step in 1..=iter {
        let sigma_inv = spd_invert(&sigma, d);

        // 1. Probit / Tobit censored data augmentation & missing-data Gibbs imputation
        for t in 0..n_time {
            for j in 0..d {
                let cell = idx(t, j, n_time);
                let is_miss = missing_mask[cell];
                let c_flag = cens_flag[cell];
                if !is_cat[j] && !is_miss && c_flag == 0 {
                    continue;
                }
                let cond_var = (1.0 / sigma_inv[idx(j, j, d)].max(1e-8)).clamp(1e-6, 100.0);
                let cond_sd = if is_cat[j] { 1.0 } else { cond_var.sqrt() };

                // Conditional mean of y_{t,j} from current-time equation (conditioning on y_{t,-j} if d > 1)
                let mut mu_tj = predict_var_cell(
                    &y_work,
                    x_in,
                    &beta_mat,
                    n_time,
                    d,
                    kx,
                    lags,
                    max_lag,
                    p_reg,
                    residual_dynamics,
                    t,
                    j,
                );
                if d > 1 && t >= max_lag {
                    let s_jj = sigma_inv[idx(j, j, d)].max(1e-8);
                    for k in 0..d {
                        if k != j {
                            let s_jk = sigma_inv[idx(j, k, d)];
                            if s_jk.abs() > 1e-14 {
                                let mu_tk = predict_var_cell(
                                    &y_work,
                                    x_in,
                                    &beta_mat,
                                    n_time,
                                    d,
                                    kx,
                                    lags,
                                    max_lag,
                                    p_reg,
                                    residual_dynamics,
                                    t,
                                    k,
                                );
                                mu_tj -= (s_jk / s_jj) * (y_work[idx(t, k, n_time)] - mu_tk);
                            }
                        }
                    }
                }

                // Combine current-time Gaussian likelihood with future lag appearances t + lag < n_time
                let mut prec = 1.0 / (cond_sd * cond_sd);
                let mut info = mu_tj * prec;
                if !residual_dynamics && (is_miss || c_flag != 0) {
                    for (l_idx, &lag_val) in lags.iter().enumerate() {
                        let t_fut = t + lag_val;
                        if t_fut >= max_lag && t_fut < n_time {
                            let reg_row = 1 + l_idx * d + j;
                            for m_out in 0..d {
                                let phi_mj = beta_mat[idx(reg_row, m_out, p_reg)];
                                if phi_mj.abs() > 1e-12 {
                                    let sig_mm = sigma[idx(m_out, m_out, d)].max(1e-6);
                                    let mu_fut = predict_var_cell(
                                        &y_work, x_in, &beta_mat, n_time, d, kx, lags, max_lag,
                                        p_reg, false, t_fut, m_out,
                                    );
                                    let r_without = y_work[idx(t_fut, m_out, n_time)]
                                        - (mu_fut - phi_mj * y_work[cell]);
                                    prec += (phi_mj * phi_mj) / sig_mm;
                                    info += phi_mj * r_without / sig_mm;
                                }
                            }
                        }
                    }
                }
                let post_sd = (1.0 / prec.max(1e-8)).sqrt();
                let post_mean = info / prec.max(1e-8);

                if is_cat[j] && !is_miss {
                    let obs_cat = y_in[cell].round() as i32;
                    let k_cat = outcome_types[j];
                    let (lo, hi) = if k_cat == 2 {
                        if obs_cat <= 0 {
                            (f64::NEG_INFINITY, 0.0)
                        } else {
                            (0.0, f64::INFINITY)
                        }
                    } else {
                        let taus = &extra_thresholds[j];
                        let c = (obs_cat.max(0) as usize).min(k_cat as usize - 1);
                        let lower = if c == 0 {
                            f64::NEG_INFINITY
                        } else {
                            taus[c - 1]
                        };
                        let upper = if c + 1 >= k_cat as usize {
                            f64::INFINITY
                        } else {
                            taus[c]
                        };
                        (lower, upper)
                    };
                    y_work[cell] = rng.rtruncnorm(mu_tj, 1.0, lo, hi);
                } else if c_flag != 0 {
                    let (lo, hi) = if c_flag < 0 {
                        (f64::NEG_INFINITY, censor_lo.get(j).copied().unwrap_or(0.0))
                    } else {
                        (censor_hi.get(j).copied().unwrap_or(0.0), f64::INFINITY)
                    };
                    y_work[cell] = rng.rtruncnorm(post_mean, post_sd, lo, hi);
                } else if is_miss {
                    y_work[cell] = post_mean + post_sd * rng.rnorm();
                }
            }
        }

        // Update ordinal extra thresholds (for K >= 3) via uniform-window MH
        for j in 0..d {
            if outcome_types[j] >= 3 {
                let k_cat = outcome_types[j] as usize;
                for c in 1..(k_cat - 1) {
                    let mut lo_bound = extra_thresholds[j][c - 1] + 1e-4;
                    let mut hi_bound = if c + 1 < k_cat - 1 {
                        extra_thresholds[j][c + 1] - 1e-4
                    } else {
                        10.0
                    };
                    for t in 0..n_time {
                        if !missing_mask[idx(t, j, n_time)] {
                            let obs = y_in[idx(t, j, n_time)].round() as usize;
                            let ystar = y_work[idx(t, j, n_time)];
                            if obs == c {
                                lo_bound = lo_bound.max(ystar);
                            } else if obs == c + 1 {
                                hi_bound = hi_bound.min(ystar);
                            }
                        }
                    }
                    if hi_bound > lo_bound {
                        extra_thresholds[j][c] = lo_bound + rng.runif() * (hi_bound - lo_bound);
                    }
                }
            }
        }

        // 2. Update regression coefficients B (equation-by-equation or SUR block)
        if !residual_dynamics {
            // Build design matrix Z (n_eff x p_reg)
            let mut z_mat = vec![0.0; n_eff * p_reg];
            let mut y_eff = vec![0.0; n_eff * d];
            for r in 0..n_eff {
                let t = r + max_lag;
                z_mat[idx(r, 0, n_eff)] = 1.0;
                for (l_idx, &lag_val) in lags.iter().enumerate() {
                    for col in 0..d {
                        z_mat[idx(r, 1 + l_idx * d + col, n_eff)] =
                            y_work[idx(t - lag_val, col, n_time)];
                    }
                }
                for col in 0..kx {
                    z_mat[idx(r, 1 + n_lags * d + col, n_eff)] = x_in[idx(t, col, n_time)];
                }
                for col in 0..d {
                    y_eff[idx(r, col, n_eff)] = y_work[idx(t, col, n_time)];
                }
            }
            let ztz = mat_t_mul(&z_mat, &z_mat, p_reg, n_eff, p_reg);

            // Conditional equation-by-equation SUR Gibbs update (exact under correlated Sigma!)
            for j in 0..d {
                let s_jj = sigma_inv[idx(j, j, d)].max(1e-8);
                // Adjusted response y_tilde_j = y_j + (1/s_jj) sum_{k != j} s_{jk} (y_k - Z beta_k)
                let mut y_adj = vec![0.0; n_eff];
                for r in 0..n_eff {
                    let mut adj = y_eff[idx(r, j, n_eff)];
                    for k in 0..d {
                        if k != j {
                            let s_jk = sigma_inv[idx(j, k, d)];
                            if s_jk.abs() > 1e-14 {
                                let mut fit_rk = 0.0;
                                for p_i in 0..p_reg {
                                    fit_rk +=
                                        z_mat[idx(r, p_i, n_eff)] * beta_mat[idx(p_i, k, p_reg)];
                                }
                                adj += (s_jk / s_jj) * (y_eff[idx(r, k, n_eff)] - fit_rk);
                            }
                        }
                    }
                    y_adj[r] = adj;
                }
                let zty_adj = mat_t_mul(&z_mat, &y_adj, p_reg, n_eff, 1);
                let mut prec = vec![0.0; p_reg * p_reg];
                for i in 0..(p_reg * p_reg) {
                    prec[i] = ztz[i] * s_jj;
                }
                for p_i in 0..p_reg {
                    let prior_prec = match prior_mode {
                        1 => 1e-10, // Mplus improper flat N(0, inf) stabilized at 1e-10
                        2 => {
                            if p_i == 0 {
                                1.0 / (beta_sd * beta_sd)
                            } else {
                                1.0 / shrink_tau2[idx(p_i, j, p_reg)].max(1e-6)
                            }
                        }
                        _ => 1.0 / (beta_sd * beta_sd),
                    };
                    prec[idx(p_i, p_i, p_reg)] += prior_prec;
                }
                let info: Vec<f64> = zty_adj.iter().map(|v| v * s_jj).collect();
                let b_j = rmvnorm_precision(&mut rng, &prec, &info, p_reg);
                for p_i in 0..p_reg {
                    beta_mat[idx(p_i, j, p_reg)] = b_j[p_i];
                }
            }
        } else {
            // RDSEM two-block conditional GLS + dynamic update:
            // Block A: update structural [alpha_j, beta_xj] given Phi via Cochrane-Orcutt whitened regression
            let p_s = 1 + kx;
            for j in 0..d {
                let sig_jj = sigma[idx(j, j, d)].max(1e-6);
                let mut x_star = vec![0.0; n_eff * p_s];
                let mut y_star = vec![0.0; n_eff];
                for r in 0..n_eff {
                    let t = r + max_lag;
                    let mut y_wh = y_work[idx(t, j, n_time)];
                    let mut c0_wh = 1.0;
                    let mut x_wh = vec![0.0; kx];
                    for col in 0..kx {
                        x_wh[col] = x_in[idx(t, col, n_time)];
                    }
                    for (l_idx, &lag_val) in lags.iter().enumerate() {
                        // Self-residual lag whitening + cross-residual subtraction
                        for m_col in 0..d {
                            let phi = beta_mat[idx(1 + l_idx * d + m_col, j, p_reg)];
                            if m_col == j {
                                y_wh -= phi * y_work[idx(t - lag_val, j, n_time)];
                                c0_wh -= phi;
                                for col in 0..kx {
                                    x_wh[col] -= phi * x_in[idx(t - lag_val, col, n_time)];
                                }
                            } else {
                                let mut u_prev = y_work[idx(t - lag_val, m_col, n_time)]
                                    - beta_mat[idx(0, m_col, p_reg)];
                                for col in 0..kx {
                                    u_prev -= beta_mat[idx(1 + n_lags * d + col, m_col, p_reg)]
                                        * x_in[idx(t - lag_val, col, n_time)];
                                }
                                y_wh -= phi * u_prev;
                            }
                        }
                    }
                    x_star[idx(r, 0, n_eff)] = c0_wh;
                    for col in 0..kx {
                        x_star[idx(r, 1 + col, n_eff)] = x_wh[col];
                    }
                    y_star[r] = y_wh;
                }
                let xtx = mat_t_mul(&x_star, &x_star, p_s, n_eff, p_s);
                let xty = mat_t_mul(&x_star, &y_star, p_s, n_eff, 1);
                let mut prec = vec![0.0; p_s * p_s];
                for i in 0..(p_s * p_s) {
                    prec[i] = xtx[i] / sig_jj;
                }
                let p_prec = if prior_mode == 1 {
                    1e-10
                } else {
                    1.0 / (beta_sd * beta_sd)
                };
                for i in 0..p_s {
                    prec[idx(i, i, p_s)] += p_prec;
                }
                let info: Vec<f64> = xty.iter().map(|v| v / sig_jj).collect();
                let b_s = rmvnorm_precision(&mut rng, &prec, &info, p_s);
                beta_mat[idx(0, j, p_reg)] = b_s[0];
                for col in 0..kx {
                    beta_mat[idx(1 + n_lags * d + col, j, p_reg)] = b_s[1 + col];
                }
            }

            // Block B: compute structural residuals u_{t,j} and update dynamic Phi
            let mut u_mat = vec![0.0; n_time * d];
            for t in 0..n_time {
                for j in 0..d {
                    let mut u_tj = y_work[idx(t, j, n_time)] - beta_mat[idx(0, j, p_reg)];
                    for col in 0..kx {
                        u_tj -= beta_mat[idx(1 + n_lags * d + col, j, p_reg)]
                            * x_in[idx(t, col, n_time)];
                    }
                    u_mat[idx(t, j, n_time)] = u_tj;
                }
            }
            let p_d = n_lags * d;
            let mut u_lag = vec![0.0; n_eff * p_d];
            for r in 0..n_eff {
                let t = r + max_lag;
                for (l_idx, &lag_val) in lags.iter().enumerate() {
                    for col in 0..d {
                        u_lag[idx(r, l_idx * d + col, n_eff)] =
                            u_mat[idx(t - lag_val, col, n_time)];
                    }
                }
            }
            let utu = mat_t_mul(&u_lag, &u_lag, p_d, n_eff, p_d);
            for j in 0..d {
                let sig_jj = sigma[idx(j, j, d)].max(1e-6);
                let u_curr: Vec<f64> = (0..n_eff)
                    .map(|r| u_mat[idx(r + max_lag, j, n_time)])
                    .collect();
                let uty = mat_t_mul(&u_lag, &u_curr, p_d, n_eff, 1);
                let mut prec = vec![0.0; p_d * p_d];
                for i in 0..(p_d * p_d) {
                    prec[i] = utu[i] / sig_jj;
                }
                for i in 0..p_d {
                    let p_prec = if prior_mode == 2 {
                        1.0 / shrink_tau2[idx(1 + i, j, p_reg)].max(1e-6)
                    } else if prior_mode == 1 {
                        1e-10
                    } else {
                        1.0 / (beta_sd * beta_sd)
                    };
                    prec[idx(i, i, p_d)] += p_prec;
                }
                let info: Vec<f64> = uty.iter().map(|v| v / sig_jj).collect();
                let phi_j = rmvnorm_precision(&mut rng, &prec, &info, p_d);
                for i in 0..p_d {
                    beta_mat[idx(1 + i, j, p_reg)] = phi_j[i].clamp(-0.995, 0.995);
                }
            }
        }

        // Update PSEM Bayesian Lasso shrinkage parameters if prior_mode == 2
        if prior_mode == 2 {
            let mut sum_tau = 0.0;
            let mut cnt_s = 0usize;
            for j in 0..d {
                for p_i in 1..p_reg {
                    let b = beta_mat[idx(p_i, j, p_reg)].abs().max(1e-6);
                    // Inverse-Gaussian mean mu = sqrt(lambda^2 / b^2), shape = lambda^2
                    let mu_ig = (lambda_lasso / b).clamp(1e-4, 1e4);
                    let lam_ig = (lambda_lasso * lambda_lasso).max(1e-4);
                    let v = rng.rnorm();
                    let y_ig = v * v;
                    let x_ig = mu_ig + (mu_ig * mu_ig * y_ig) / (2.0 * lam_ig)
                        - (mu_ig / (2.0 * lam_ig))
                            * (4.0 * mu_ig * lam_ig * y_ig + mu_ig * mu_ig * y_ig * y_ig).sqrt();
                    let inv_tau2 = if rng.runif() <= mu_ig / (mu_ig + x_ig) {
                        x_ig
                    } else {
                        mu_ig * mu_ig / x_ig.max(1e-8)
                    };
                    let t2 = (1.0 / inv_tau2.max(1e-6)).clamp(1e-5, 100.0);
                    shrink_tau2[idx(p_i, j, p_reg)] = t2;
                    sum_tau += t2;
                    cnt_s += 1;
                }
            }
            if cnt_s > 0 {
                let l2 = rng.rgamma(1.0 + cnt_s as f64, 1.0 + 0.5 * sum_tau);
                lambda_lasso = l2.sqrt().clamp(0.05, 50.0);
            }
        }

        // 3. Compute residuals E (n_eff x d) and update Sigma
        let mut e_mat = vec![0.0; n_eff * d];
        for r in 0..n_eff {
            let t = r + max_lag;
            for j in 0..d {
                let pred = predict_var_cell(
                    &y_work,
                    x_in,
                    &beta_mat,
                    n_time,
                    d,
                    kx,
                    lags,
                    max_lag,
                    p_reg,
                    residual_dynamics,
                    t,
                    j,
                );
                e_mat[idx(r, j, n_eff)] = y_work[idx(t, j, n_time)] - pred;
            }
        }

        if d == 1 {
            if is_cat[0] {
                sigma[0] = 1.0;
            } else {
                let rss: f64 = e_mat.iter().map(|v| v * v).sum();
                let (sh, sc) = if prior_mode == 1 {
                    // Mplus IG(-1, 0): posterior shape = n_eff/2 - 1, scale = rss/2
                    ((0.5 * n_eff as f64 - 1.0).max(0.5), 0.5 * rss.max(1e-8))
                } else {
                    (sigma_shape + 0.5 * n_eff as f64, sigma_scale + 0.5 * rss)
                };
                sigma[0] = rng.rinv_gamma(sh, sc);
            }
        } else {
            let ete = mat_t_mul(&e_mat, &e_mat, d, n_eff, d);
            let mut scale_psi = eye(
                d,
                if prior_mode == 1 {
                    1e-6
                } else {
                    2.0 * sigma_scale.max(1e-3)
                },
            );
            for i in 0..(d * d) {
                scale_psi[i] += ete[i];
            }
            let df_post = if prior_mode == 1 {
                (n_eff as f64 - d as f64 - 1.0).max(d as f64 + 1.0)
            } else {
                (d as f64 + 2.0) + n_eff as f64
            };
            sigma = rinv_wishart(&mut rng, df_post, &scale_psi, d);
            for j in 0..d {
                if is_cat[j] {
                    // Rescale categorical latent variance to 1
                    let scale_j = sigma[idx(j, j, d)].max(1e-8).sqrt();
                    for k in 0..d {
                        sigma[idx(j, k, d)] /= scale_j;
                        sigma[idx(k, j, d)] /= scale_j;
                    }
                    sigma[idx(j, j, d)] = 1.0;
                }
            }
            symmetrize(&mut sigma, d);
        }

        if keep_pos < n_keep && step == keep_indices[keep_pos] {
            let mut p_cursor = 0usize;
            for j in 0..d {
                for p_i in 0..p_reg {
                    let val = if p_i == 0 && is_cat[j] {
                        // Report Mplus threshold tau_1 = -intercept!
                        -beta_mat[idx(0, j, p_reg)]
                    } else {
                        beta_mat[idx(p_i, j, p_reg)]
                    };
                    draws[idx(keep_pos, p_cursor, n_keep)] = val;
                    p_cursor += 1;
                }
            }
            for c in 0..d {
                for r in c..d {
                    draws[idx(keep_pos, p_cursor, n_keep)] = sigma[idx(r, c, d)];
                    p_cursor += 1;
                }
            }
            for j in 0..d {
                if outcome_types[j] >= 3 {
                    let k_cat = outcome_types[j] as usize;
                    let int_j = beta_mat[idx(0, j, p_reg)];
                    for &tau_c in extra_thresholds[j].iter().take(k_cat - 1).skip(1) {
                        draws[idx(keep_pos, p_cursor, n_keep)] = tau_c - int_j;
                        p_cursor += 1;
                    }
                }
            }

            // Accumulate posterior means for DIC & fitted values
            for i in 0..(p_reg * d) {
                beta_accum[i] += beta_mat[i];
            }
            for i in 0..(d * d) {
                sigma_accum[i] += sigma[i];
            }
            for t in 0..n_time {
                for j in 0..d {
                    let fv = if t >= max_lag {
                        predict_var_cell(
                            &y_work,
                            x_in,
                            &beta_mat,
                            n_time,
                            d,
                            kx,
                            lags,
                            max_lag,
                            p_reg,
                            residual_dynamics,
                            t,
                            j,
                        )
                    } else {
                        beta_mat[idx(0, j, p_reg)]
                    };
                    fitted_accum[idx(t, j, n_time)] += fv;
                }
            }

            let ll = if any_cens && d == 1 {
                let sd0 = sigma[0].max(1e-8).sqrt();
                let lo0 = censor_lo.first().copied().unwrap_or(f64::NEG_INFINITY);
                let hi0 = censor_hi.first().copied().unwrap_or(f64::INFINITY);
                let mut lsum = 0.0;
                for (r, &e_r) in e_mat.iter().enumerate().take(n_eff) {
                    let t = r + max_lag;
                    if missing_mask[t] {
                        continue;
                    }
                    let pred = y_work[t] - e_r;
                    match cens_flag[t] {
                        -1 => lsum += log_std_norm_cdf((lo0 - pred) / sd0),
                        1 => lsum += log_std_norm_cdf((pred - hi0) / sd0),
                        _ => {
                            let err = y_in[t] - pred;
                            lsum +=
                                -0.5 * ((2.0 * PI * sd0 * sd0).ln() + (err * err) / (sd0 * sd0));
                        }
                    }
                }
                lsum
            } else {
                var_loglik(&e_mat, &sigma, n_eff, d)
            };
            deviance_sum += -2.0 * ll;
            keep_pos += 1;
        }
    }

    let inv_keep = 1.0 / (n_keep.max(1) as f64);
    let fitted_mean: Vec<f64> = fitted_accum.into_iter().map(|v| v * inv_keep).collect();
    let beta_bar: Vec<f64> = beta_accum.into_iter().map(|v| v * inv_keep).collect();
    let sigma_bar: Vec<f64> = sigma_accum.into_iter().map(|v| v * inv_keep).collect();
    let mut e_bar = vec![0.0; n_eff * d];
    for r in 0..n_eff {
        let t = r + max_lag;
        for j in 0..d {
            let pred = predict_var_cell(
                &y_work,
                x_in,
                &beta_bar,
                n_time,
                d,
                kx,
                lags,
                max_lag,
                p_reg,
                residual_dynamics,
                t,
                j,
            );
            e_bar[idx(r, j, n_eff)] = y_work[idx(t, j, n_time)] - pred;
        }
    }
    let has_continuous = is_cat.iter().any(|&b| !b);
    let (dic, pd, d_bar) = if any_cens && d == 1 {
        let db = deviance_sum * inv_keep;
        let sd0 = sigma_bar[0].max(1e-8).sqrt();
        let lo0 = censor_lo.first().copied().unwrap_or(f64::NEG_INFINITY);
        let hi0 = censor_hi.first().copied().unwrap_or(f64::INFINITY);
        let mut lhat = 0.0;
        for r in 0..n_eff {
            let t = r + max_lag;
            if missing_mask[t] {
                continue;
            }
            let pred = fitted_mean[t];
            match cens_flag[t] {
                -1 => lhat += log_std_norm_cdf((lo0 - pred) / sd0),
                1 => lhat += log_std_norm_cdf((pred - hi0) / sd0),
                _ => {
                    let err = y_in[t] - pred;
                    lhat += -0.5 * ((2.0 * PI * sd0 * sd0).ln() + (err * err) / (sd0 * sd0));
                }
            }
        }
        let d_hat = -2.0 * lhat;
        let pd_v = (db - d_hat).max(0.0);
        (db + pd_v, pd_v, db)
    } else if has_continuous && !any_cens {
        let db = deviance_sum * inv_keep;
        let d_hat = -2.0 * var_loglik(&e_bar, &sigma_bar, n_eff, d);
        let pd_v = (db - d_hat).max(0.0);
        (db + pd_v, pd_v, db)
    } else {
        (f64::NAN, f64::NAN, f64::NAN)
    };

    VarN1Output {
        draws,
        param_count: n_params,
        fitted_mean,
        dic,
        pd,
        d_bar,
    }
}

#[allow(clippy::too_many_arguments)]
fn predict_var_cell(
    y: &[f64],
    x: &[f64],
    beta: &[f64],
    n_time: usize,
    d: usize,
    kx: usize,
    lags: &[usize],
    max_lag: usize,
    p_reg: usize,
    residual_dynamics: bool,
    t: usize,
    j: usize,
) -> f64 {
    let n_lags = lags.len();
    let mut structural = beta[idx(0, j, p_reg)];
    for col in 0..kx {
        structural += beta[idx(1 + n_lags * d + col, j, p_reg)] * x[idx(t, col, n_time)];
    }
    if t < max_lag {
        return structural;
    }
    if !residual_dynamics {
        for (l_idx, &lag_val) in lags.iter().enumerate() {
            for col in 0..d {
                structural +=
                    beta[idx(1 + l_idx * d + col, j, p_reg)] * y[idx(t - lag_val, col, n_time)];
            }
        }
        structural
    } else {
        for (l_idx, &lag_val) in lags.iter().enumerate() {
            for col in 0..d {
                let mut u_prev = y[idx(t - lag_val, col, n_time)] - beta[idx(0, col, p_reg)];
                for x_col in 0..kx {
                    u_prev -= beta[idx(1 + n_lags * d + x_col, col, p_reg)]
                        * x[idx(t - lag_val, x_col, n_time)];
                }
                structural += beta[idx(1 + l_idx * d + col, j, p_reg)] * u_prev;
            }
        }
        structural
    }
}

fn var_loglik(e_mat: &[f64], sigma: &[f64], n_eff: usize, d: usize) -> f64 {
    let l = chol_lower_jitter(sigma, d);
    let logdet = spd_logdet(&l, d);
    let sig_inv = spd_invert(sigma, d);
    let mut ll = 0.0;
    let mut row = vec![0.0; d];
    for r in 0..n_eff {
        for j in 0..d {
            row[j] = e_mat[idx(r, j, n_eff)];
        }
        let qf = quad_form(&row, &sig_inv, d);
        ll += -0.5 * ((d as f64) * (2.0 * PI).ln() + logdet + qf);
    }
    ll
}
