use crate::linalg::{
    chol_lower_jitter, eye, idx, mat_t_mul, mat_vec, quad_form, rinv_wishart, rmvnorm_precision,
    spd_invert, spd_logdet, symmetrize,
};
use crate::rng::Rng;
use std::f64::consts::PI;

pub struct TwoLevelOutput {
    pub draws: Vec<f64>, // n_keep x n_pop_params (col-major)
    pub param_count: usize,
    pub re_draws: Vec<f64>, // n_keep x n_clusters x q_re (flattened)
    pub q_re: usize,
    pub fitted_mean: Vec<f64>, // n_obs x d
    pub dic: f64,
    pub pd: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn mcmc_twolevel(
    y_in: &[f64], // n_obs x d (col-major, contiguous by cluster, sorted by regular grid)
    x_in: &[f64], // n_obs x kx (col-major, time-varying covariates)
    cluster_sizes: &[usize], // length g_clusters, sum = n_obs
    w_in: &[f64], // g_clusters x kw (between exogenous predictors)
    z_in: &[f64], // g_clusters x kz (between endogenous distal outcomes, e.g. z ON y s logv)
    z_on_re_mask: &[i32], // q_re x kz (1 if z_m regresses on theta_q)
    z_on_w_mask: &[i32], // kw x kz (1 if z_m regresses on w_k)
    re_on_w_mask: &[i32], // kw x q_re (1 if theta_q regresses on w_k)
    d: usize,
    kx: usize,
    kw: usize,
    kz: usize,
    lags: &[usize],
    slope_is_random: &[i32], // length d * (n_lags * d + kx): 1 if random slope, 0 if fixed
    has_random_logv: &[i32], // length d: 1 if log(sigma_j^2) is random across clusters
    latent_center_y: bool,
    latent_center_x: &[i32], // length kx: 1 if x_k is latently centered per cluster
    residual_dynamics: bool,
    outcome_types: &[i32], // 0 = continuous, K >= 2 = K categories
    prior_mode: i32,
    beta_sd: f64,
    omega_df_add: f64,
    omega_scale_diag: f64,
    iter: usize,
    warmup: usize,
    thin: usize,
    seed: u64,
) -> TwoLevelOutput {
    let mut rng = Rng::from_seed(seed);
    let g_cnt = cluster_sizes.len();
    let n_obs: usize = cluster_sizes.iter().sum();
    let max_lag = lags.iter().copied().max().unwrap_or(1).max(1);
    let n_lags = lags.len();
    let n_slopes_per_out = n_lags * d + kx;
    let is_cat: Vec<bool> = outcome_types.iter().map(|&k| k >= 2).collect();

    // Cluster row offsets
    let mut offsets = Vec::with_capacity(g_cnt + 1);
    offsets.push(0usize);
    for &sz in cluster_sizes {
        offsets.push(offsets.last().unwrap() + sz);
    }

    // Map random effects theta_i: [mu_{i,0..d}, random_slopes..., logv_{i,...}]
    let mut slope_re_idx = vec![-1i32; d * n_slopes_per_out];
    let mut fixed_slope_idx = vec![-1i32; d * n_slopes_per_out];
    let mut n_rand_slopes = 0usize;
    let mut n_fixed_slopes = 0usize;
    for s in 0..(d * n_slopes_per_out) {
        if slope_is_random[s] != 0 {
            slope_re_idx[s] = (d + n_rand_slopes) as i32;
            n_rand_slopes += 1;
        } else {
            fixed_slope_idx[s] = n_fixed_slopes as i32;
            n_fixed_slopes += 1;
        }
    }
    let mut logv_re_idx = vec![-1i32; d];
    let mut n_rand_logv = 0usize;
    let mut n_fixed_sig = 0usize;
    for j in 0..d {
        if has_random_logv[j] != 0 && !is_cat[j] {
            logv_re_idx[j] = (d + n_rand_slopes + n_rand_logv) as i32;
            n_rand_logv += 1;
        } else if !is_cat[j] {
            n_fixed_sig += 1;
        }
    }
    let q_re = d + n_rand_slopes + n_rand_logv;

    // Initialize y_work and missingness
    let mut y_work = y_in.to_vec();
    let mut miss_y = vec![false; n_obs * d];
    for j in 0..d {
        let mut s = 0.0;
        let mut c = 0usize;
        for r in 0..n_obs {
            let v = y_in[idx(r, j, n_obs)];
            if v.is_finite() {
                s += v;
                c += 1;
            } else {
                miss_y[idx(r, j, n_obs)] = true;
            }
        }
        let m = if c > 0 { s / c as f64 } else { 0.0 };
        for r in 0..n_obs {
            if miss_y[idx(r, j, n_obs)] {
                y_work[idx(r, j, n_obs)] = if is_cat[j] { 0.0 } else { m };
            } else if is_cat[j] {
                let kc = outcome_types[j].max(2) as f64;
                y_work[idx(r, j, n_obs)] = ((y_in[idx(r, j, n_obs)] + 0.5) / kc - 0.5) * 2.0;
            }
        }
    }

    // Latent cluster means for x if requested
    let mut x_between: Vec<f64> = vec![0.0_f64; g_cnt * kx];
    let mut x_var_within: Vec<f64> = vec![1.0_f64; kx];
    let mut x_var_between: Vec<f64> = vec![1.0_f64; kx];
    let mut x_grand_mean: Vec<f64> = vec![0.0_f64; kx];
    for k in 0..kx {
        let mut all_s = 0.0;
        let mut all_c = 0usize;
        for g in 0..g_cnt {
            let st = offsets[g];
            let en = offsets[g + 1];
            let mut gs = 0.0;
            let mut gc = 0usize;
            for r in st..en {
                let v = x_in[idx(r, k, n_obs)];
                if v.is_finite() {
                    gs += v;
                    gc += 1;
                    all_s += v;
                    all_c += 1;
                }
            }
            x_between[idx(g, k, g_cnt)] = if gc > 0 { gs / gc as f64 } else { 0.0 };
        }
        x_grand_mean[k] = if all_c > 0 { all_s / all_c as f64 } else { 0.0 };
    }

    // Initialize cluster random effects theta (g_cnt x q_re), population gamma ((1+kw) x q_re), Omega (q_re x q_re)
    let mut theta = vec![0.0; g_cnt * q_re];
    for g in 0..g_cnt {
        let st = offsets[g];
        let en = offsets[g + 1];
        for j in 0..d {
            let mut s = 0.0;
            for r in st..en {
                s += y_work[idx(r, j, n_obs)];
            }
            theta[idx(g, j, g_cnt)] = s / ((en - st).max(1) as f64);
            // Initialize self-AR1 slope around 0.25 if random
            if n_lags >= 1 {
                let s_pos = j * n_slopes_per_out + j;
                if slope_re_idx[s_pos] >= 0 {
                    theta[idx(g, slope_re_idx[s_pos] as usize, g_cnt)] = 0.25;
                }
            }
        }
    }
    let p_w = 1 + kw;
    let mut gamma_b = vec![0.0; p_w * q_re]; // row 0 = intercept, rows 1..=kw = w effects
    for q in 0..q_re {
        let m: f64 = (0..g_cnt).map(|g| theta[idx(g, q, g_cnt)]).sum::<f64>() / (g_cnt as f64);
        gamma_b[idx(0, q, p_w)] = m;
    }
    let mut omega_b = eye(q_re, 0.1);
    let mut fixed_slopes = vec![0.0; n_fixed_slopes];
    let mut fixed_sigma2 = vec![1.0; d];

    // Distal outcome parameters: for each m in 0..kz, predictors [1, selected theta, selected w]
    let p_z = 1 + q_re + kw;
    let mut beta_z: Vec<f64> = vec![0.0_f64; p_z * kz];
    let mut sigma2_z: Vec<f64> = vec![0.5_f64; kz];
    let mut mh_step_logv = vec![0.25; g_cnt * n_rand_logv.max(1)];
    let mut mh_accept_logv = vec![0usize; g_cnt * n_rand_logv.max(1)];

    // Count population parameters reported in draws:
    // 1. gamma_b intercepts (q_re)
    // 2. active gamma_b slopes on w
    // 3. fixed within slopes (n_fixed_slopes)
    // 4. lower triangle of omega_b (q_re*(q_re+1)/2)
    // 5. fixed within sigma2 (for continuous outcomes without random logv)
    // 6. distal z intercepts, active z slopes on theta & w, and sigma2_z
    let n_re_w_active: usize = re_on_w_mask.iter().filter(|&&v| v != 0).count();
    let n_z_re_active: usize = z_on_re_mask.iter().filter(|&&v| v != 0).count();
    let n_z_w_active: usize = z_on_w_mask.iter().filter(|&&v| v != 0).count();
    let n_omega_cov = q_re * (q_re + 1) / 2;
    let n_params = q_re
        + n_re_w_active
        + n_fixed_slopes
        + n_omega_cov
        + n_fixed_sig
        + kz
        + n_z_re_active
        + n_z_w_active
        + kz;

    let keep_indices: Vec<usize> = ((warmup + 1)..=iter)
        .filter(|i| (i - warmup - 1) % thin == 0)
        .collect();
    let n_keep = keep_indices.len();
    let mut draws = vec![0.0; n_keep * n_params];
    let mut re_draws = vec![0.0; n_keep * g_cnt * q_re];
    let mut fitted_accum = vec![0.0; n_obs * d];
    let mut deviance_sum = 0.0;
    let mut pred_bar_accum = vec![0.0; n_obs * d];
    let mut sig_bar_accum = vec![0.0; g_cnt * d];
    let mut keep_pos = 0usize;

    for step in 1..=iter {
        let omega_inv = spd_invert(&omega_b, q_re);

        // Optional latent centering update for exogenous covariates x_k
        for k in 0..kx {
            if latent_center_x[k] != 0 {
                let mut ss_w = 0.0;
                let mut cnt_w = 0usize;
                for g in 0..g_cnt {
                    let st = offsets[g];
                    let en = offsets[g + 1];
                    let mut sum_x = 0.0;
                    let mut cnt = 0usize;
                    for r in st..en {
                        let v = x_in[idx(r, k, n_obs)];
                        if v.is_finite() {
                            sum_x += v;
                            cnt += 1;
                        }
                    }
                    let prec =
                        (cnt as f64) / x_var_within[k].max(1e-6) + 1.0 / x_var_between[k].max(1e-6);
                    let info = sum_x / x_var_within[k].max(1e-6)
                        + x_grand_mean[k] / x_var_between[k].max(1e-6);
                    let xb = info / prec + (1.0 / prec).sqrt() * rng.rnorm();
                    x_between[idx(g, k, g_cnt)] = xb;
                    for r in st..en {
                        let v = x_in[idx(r, k, n_obs)];
                        if v.is_finite() {
                            let d_x = v - xb;
                            ss_w += d_x * d_x;
                            cnt_w += 1;
                        }
                    }
                }
                x_var_within[k] = rng.rinv_gamma(0.5 * cnt_w as f64 + 0.01, 0.5 * ss_w + 0.01);
                let mean_xb: f64 =
                    (0..g_cnt).map(|g| x_between[idx(g, k, g_cnt)]).sum::<f64>() / g_cnt as f64;
                x_grand_mean[k] = mean_xb + (x_var_between[k] / g_cnt as f64).sqrt() * rng.rnorm();
                let ss_b: f64 = (0..g_cnt)
                    .map(|g| {
                        let df = x_between[idx(g, k, g_cnt)] - x_grand_mean[k];
                        df * df
                    })
                    .sum();
                x_var_between[k] = rng.rinv_gamma(0.5 * g_cnt as f64 + 0.01, 0.5 * ss_b + 0.01);
            }
        }

        // 1. Per-cluster updates: missing/probit y, linear random effects (mu_i, s_i), and MH for logv_i
        for g in 0..g_cnt {
            let st = offsets[g];
            let en = offsets[g + 1];
            let tg = en - st;
            if tg <= max_lag {
                continue;
            }
            let n_eff_g = tg - max_lag;

            // Prior mean for theta_g from gamma_b and w_g
            let mut mu_prior_g = vec![0.0; q_re];
            for q in 0..q_re {
                let mut m = gamma_b[idx(0, q, p_w)];
                for k in 0..kw {
                    if re_on_w_mask[idx(k, q, kw)] != 0 {
                        m += gamma_b[idx(1 + k, q, p_w)] * w_in[idx(g, k, g_cnt)];
                    }
                }
                mu_prior_g[q] = m;
            }

            // Current cluster variances sigma2_{g,j}
            let mut sig2_g = vec![1.0; d];
            for j in 0..d {
                sig2_g[j] = if is_cat[j] {
                    1.0
                } else if logv_re_idx[j] >= 0 {
                    theta[idx(g, logv_re_idx[j] as usize, g_cnt)]
                        .clamp(-12.0, 12.0)
                        .exp()
                } else {
                    fixed_sigma2[j]
                };
            }

            // Helper closure values: extract cluster slopes B_g (n_slopes_per_out x d)
            let mut slopes_g = vec![0.0; n_slopes_per_out * d];
            for j in 0..d {
                for s in 0..n_slopes_per_out {
                    let flat = j * n_slopes_per_out + s;
                    slopes_g[idx(s, j, n_slopes_per_out)] = if slope_re_idx[flat] >= 0 {
                        theta[idx(g, slope_re_idx[flat] as usize, g_cnt)]
                    } else {
                        fixed_slopes[fixed_slope_idx[flat] as usize]
                    };
                }
            }

            // 1a. Probit / missing data imputation inside cluster g
            for t_loc in 0..tg {
                let r_abs = st + t_loc;
                for j in 0..d {
                    let mu_gj = theta[idx(g, j, g_cnt)];
                    let pred = cluster_cell_mean(
                        &y_work,
                        x_in,
                        &x_between,
                        latent_center_x,
                        g,
                        g_cnt,
                        st,
                        t_loc,
                        j,
                        d,
                        kx,
                        lags,
                        max_lag,
                        mu_gj,
                        &theta,
                        &slopes_g,
                        n_slopes_per_out,
                        latent_center_y,
                        residual_dynamics,
                        n_obs,
                    );
                    if is_cat[j] && !miss_y[idx(r_abs, j, n_obs)] {
                        let obs_c = y_in[idx(r_abs, j, n_obs)].round() as i32;
                        let (lo, hi) = if obs_c <= 0 {
                            (f64::NEG_INFINITY, 0.0)
                        } else {
                            (0.0, f64::INFINITY)
                        };
                        y_work[idx(r_abs, j, n_obs)] = rng.rtruncnorm(pred, 1.0, lo, hi);
                    } else if miss_y[idx(r_abs, j, n_obs)] {
                        let mut prec = 1.0 / sig2_g[j].max(1e-6);
                        let mut info = pred * prec;
                        if !residual_dynamics && n_lags >= 1 {
                            for (l_idx, &lag_v) in lags.iter().enumerate() {
                                let t_f = t_loc + lag_v;
                                if t_f >= max_lag && t_f < tg {
                                    let phi = slopes_g[idx(l_idx * d + j, j, n_slopes_per_out)];
                                    let pred_f = cluster_cell_mean(
                                        &y_work,
                                        x_in,
                                        &x_between,
                                        latent_center_x,
                                        g,
                                        g_cnt,
                                        st,
                                        t_f,
                                        j,
                                        d,
                                        kx,
                                        lags,
                                        max_lag,
                                        mu_gj,
                                        &theta,
                                        &slopes_g,
                                        n_slopes_per_out,
                                        latent_center_y,
                                        false,
                                        n_obs,
                                    );
                                    let rem = y_work[idx(st + t_f, j, n_obs)]
                                        - (pred_f - phi * y_work[idx(r_abs, j, n_obs)]);
                                    prec += (phi * phi) / sig2_g[j].max(1e-6);
                                    info += phi * rem / sig2_g[j].max(1e-6);
                                }
                            }
                        }
                        y_work[idx(r_abs, j, n_obs)] =
                            info / prec + (1.0 / prec).sqrt() * rng.rnorm();
                    }
                }
            }

            // 1b. Update cluster intercepts mu_{g,0..d} conditional on slopes, Omega_B, and distal z_g!
            // Then update cluster random slopes conditional on mu_g, Omega_B, and distal z_g!
            let q_lin = d + n_rand_slopes;
            // Start with prior precision & info on linear subblock [0..q_lin] conditional on logv subblock
            let mut lin_prec = vec![0.0; q_lin * q_lin];
            let mut lin_info = vec![0.0; q_lin];
            for c in 0..q_lin {
                for r in 0..q_lin {
                    lin_prec[idx(r, c, q_lin)] = omega_inv[idx(r, c, q_re)];
                }
            }
            for r in 0..q_lin {
                let mut rhs = 0.0;
                for c in 0..q_lin {
                    rhs += omega_inv[idx(r, c, q_re)] * mu_prior_g[c];
                }
                // Cross-precision contribution from current logv components (q_lin..q_re)
                for c_lv in q_lin..q_re {
                    let diff_lv = theta[idx(g, c_lv, g_cnt)] - mu_prior_g[c_lv];
                    rhs -= omega_inv[idx(r, c_lv, q_re)] * diff_lv;
                }
                lin_info[r] = rhs;
            }
            // Add distal z_g likelihood contributions to linear random effects!
            for m_z in 0..kz {
                let z_val = z_in[idx(g, m_z, g_cnt)];
                if !z_val.is_finite() {
                    continue;
                }
                let sz2 = sigma2_z[m_z].max(1e-6);
                let mut rest = z_val - beta_z[idx(0, m_z, p_z)];
                for k_w in 0..kw {
                    if z_on_w_mask[idx(k_w, m_z, kw)] != 0 {
                        rest -= beta_z[idx(1 + q_re + k_w, m_z, p_z)] * w_in[idx(g, k_w, g_cnt)];
                    }
                }
                for c_lv in q_lin..q_re {
                    if z_on_re_mask[idx(c_lv, m_z, q_re)] != 0 {
                        rest -= beta_z[idx(1 + c_lv, m_z, p_z)] * theta[idx(g, c_lv, g_cnt)];
                    }
                }
                for r in 0..q_lin {
                    if z_on_re_mask[idx(r, m_z, q_re)] != 0 {
                        let br = beta_z[idx(1 + r, m_z, p_z)];
                        lin_info[r] += br * rest / sz2;
                        for c in 0..q_lin {
                            if z_on_re_mask[idx(c, m_z, q_re)] != 0 {
                                let bc = beta_z[idx(1 + c, m_z, p_z)];
                                lin_prec[idx(r, c, q_lin)] += br * bc / sz2;
                            }
                        }
                    }
                }
            }

            // First: Two-step alternating conditional update within the linear block:
            // Step i: update intercepts mu_{g,0..d} given current slopes_g (exact under latent centering!)
            {
                let mut p_mu = vec![0.0; d * d];
                let mut b_mu = vec![0.0; d];
                for c in 0..d {
                    for r in 0..d {
                        p_mu[idx(r, c, d)] = lin_prec[idx(r, c, q_lin)];
                    }
                    let mut rhs = lin_info[c];
                    for c_s in d..q_lin {
                        rhs -= lin_prec[idx(c, c_s, q_lin)] * theta[idx(g, c_s, g_cnt)];
                    }
                    b_mu[c] = rhs;
                }
                for j in 0..d {
                    let s2 = sig2_g[j].max(1e-6);
                    // Stationary initial observation contribution if latent_center_y && d == 1 && n_lags == 1
                    if latent_center_y && !residual_dynamics && n_lags == 1 && d == 1 {
                        let phi = slopes_g[0].clamp(-0.98, 0.98);
                        let w0 = (1.0 - phi * phi).max(0.02) / s2;
                        p_mu[0] += w0;
                        b_mu[0] += w0 * y_work[idx(st, 0, n_obs)];
                    }
                    for r_loc in 0..n_eff_g {
                        let t_loc = r_loc + max_lag;
                        // Coefficient on mu_{g,j} in y_{g,t,j}:
                        // Under latent centering (!residual_dynamics): (1 - sum_l phi_{self,l}) * mu_j
                        // Under RDSEM or conditional intercept: 1.0 * mu_j (with cross-effects accounted)
                        let mut c_mult = 1.0;
                        let mut rem = y_work[idx(st + t_loc, j, n_obs)];
                        for k_x in 0..kx {
                            let x_val = if latent_center_x[k_x] != 0 {
                                x_in[idx(st + t_loc, k_x, n_obs)] - x_between[idx(g, k_x, g_cnt)]
                            } else {
                                x_in[idx(st + t_loc, k_x, n_obs)]
                            };
                            rem -= slopes_g[idx(n_lags * d + k_x, j, n_slopes_per_out)] * x_val;
                        }
                        if !residual_dynamics {
                            for (l_idx, &lag_v) in lags.iter().enumerate() {
                                for m_col in 0..d {
                                    let phi = slopes_g[idx(l_idx * d + m_col, j, n_slopes_per_out)];
                                    let y_lag = y_work[idx(st + t_loc - lag_v, m_col, n_obs)];
                                    rem -= phi * y_lag;
                                    if latent_center_y {
                                        if m_col == j {
                                            c_mult -= phi;
                                        } else {
                                            rem += phi * theta[idx(g, m_col, g_cnt)];
                                        }
                                    }
                                }
                            }
                        } else {
                            for (l_idx, &lag_v) in lags.iter().enumerate() {
                                for m_col in 0..d {
                                    let phi = slopes_g[idx(l_idx * d + m_col, j, n_slopes_per_out)];
                                    let mut u_lag = y_work[idx(st + t_loc - lag_v, m_col, n_obs)];
                                    for k_x in 0..kx {
                                        let x_l = if latent_center_x[k_x] != 0 {
                                            x_in[idx(st + t_loc - lag_v, k_x, n_obs)]
                                                - x_between[idx(g, k_x, g_cnt)]
                                        } else {
                                            x_in[idx(st + t_loc - lag_v, k_x, n_obs)]
                                        };
                                        u_lag -= slopes_g
                                            [idx(n_lags * d + k_x, m_col, n_slopes_per_out)]
                                            * x_l;
                                    }
                                    if m_col == j {
                                        rem -= phi * u_lag;
                                        c_mult -= phi;
                                    } else {
                                        rem -= phi * (u_lag - theta[idx(g, m_col, g_cnt)]);
                                    }
                                }
                            }
                        }
                        p_mu[idx(j, j, d)] += (c_mult * c_mult) / s2;
                        b_mu[j] += c_mult * rem / s2;
                    }
                }
                let mu_draw = rmvnorm_precision(&mut rng, &p_mu, &b_mu, d);
                for j in 0..d {
                    theta[idx(g, j, g_cnt)] = mu_draw[j];
                }
            }

            // Step ii: update random slopes theta[g, d..q_lin] given updated mu_{g,0..d}
            if n_rand_slopes > 0 {
                let mut p_sl = vec![0.0; n_rand_slopes * n_rand_slopes];
                let mut b_sl = vec![0.0; n_rand_slopes];
                for c in 0..n_rand_slopes {
                    for r in 0..n_rand_slopes {
                        p_sl[idx(r, c, n_rand_slopes)] = lin_prec[idx(d + r, d + c, q_lin)];
                    }
                    let mut rhs = lin_info[d + c];
                    for j in 0..d {
                        rhs -= lin_prec[idx(d + c, j, q_lin)] * theta[idx(g, j, g_cnt)];
                    }
                    b_sl[c] = rhs;
                }
                for j in 0..d {
                    let s2 = sig2_g[j].max(1e-6);
                    let mu_gj = theta[idx(g, j, g_cnt)];
                    for r_loc in 0..n_eff_g {
                        let t_loc = r_loc + max_lag;
                        let mut y_dev = y_work[idx(st + t_loc, j, n_obs)] - mu_gj;
                        let mut reg_vec = vec![0.0; n_slopes_per_out];
                        for (l_idx, &lag_v) in lags.iter().enumerate() {
                            for m_col in 0..d {
                                let y_l = y_work[idx(st + t_loc - lag_v, m_col, n_obs)];
                                reg_vec[l_idx * d + m_col] = if residual_dynamics {
                                    let mut u_l = y_l - theta[idx(g, m_col, g_cnt)];
                                    for k_x in 0..kx {
                                        let x_l = if latent_center_x[k_x] != 0 {
                                            x_in[idx(st + t_loc - lag_v, k_x, n_obs)]
                                                - x_between[idx(g, k_x, g_cnt)]
                                        } else {
                                            x_in[idx(st + t_loc - lag_v, k_x, n_obs)]
                                        };
                                        u_l -= slopes_g
                                            [idx(n_lags * d + k_x, m_col, n_slopes_per_out)]
                                            * x_l;
                                    }
                                    u_l
                                } else if latent_center_y {
                                    y_l - theta[idx(g, m_col, g_cnt)]
                                } else {
                                    y_l
                                };
                            }
                        }
                        for k_x in 0..kx {
                            reg_vec[n_lags * d + k_x] = if latent_center_x[k_x] != 0 {
                                x_in[idx(st + t_loc, k_x, n_obs)] - x_between[idx(g, k_x, g_cnt)]
                            } else {
                                x_in[idx(st + t_loc, k_x, n_obs)]
                            };
                        }
                        // Subtract fixed-slope contributions
                        let mut active_rs: Vec<(usize, f64)> = Vec::new();
                        for (s, &rv_s) in reg_vec.iter().enumerate().take(n_slopes_per_out) {
                            let flat = j * n_slopes_per_out + s;
                            if slope_re_idx[flat] >= 0 {
                                let loc_idx = slope_re_idx[flat] as usize - d;
                                active_rs.push((loc_idx, rv_s));
                            } else {
                                y_dev -= fixed_slopes[fixed_slope_idx[flat] as usize] * rv_s;
                            }
                        }
                        for &(ri, xv_r) in &active_rs {
                            b_sl[ri] += xv_r * y_dev / s2;
                            for &(ci, xv_c) in &active_rs {
                                p_sl[idx(ri, ci, n_rand_slopes)] += xv_r * xv_c / s2;
                            }
                        }
                    }
                }
                let sl_draw = rmvnorm_precision(&mut rng, &p_sl, &b_sl, n_rand_slopes);
                for s in 0..n_rand_slopes {
                    theta[idx(g, d + s, g_cnt)] = sl_draw[s];
                }
                // Refresh slopes_g
                for j in 0..d {
                    for s in 0..n_slopes_per_out {
                        let flat = j * n_slopes_per_out + s;
                        if slope_re_idx[flat] >= 0 {
                            slopes_g[idx(s, j, n_slopes_per_out)] =
                                theta[idx(g, slope_re_idx[flat] as usize, g_cnt)];
                        }
                    }
                }
            }

            // Step iii: Metropolis-Hastings update for random log residual variances logv_{g,j}
            for j in 0..d {
                if logv_re_idx[j] >= 0 {
                    let q_idx = logv_re_idx[j] as usize;
                    let lv_slot = q_idx - (d + n_rand_slopes);
                    let mh_i = g * n_rand_logv + lv_slot;
                    let cur_lv = theta[idx(g, q_idx, g_cnt)];
                    let prop_lv = (cur_lv + mh_step_logv[mh_i] * rng.rnorm()).clamp(-10.0, 10.0);

                    // Compute cluster residual sum of squares for outcome j
                    let mut rss_j = 0.0;
                    let mu_gj = theta[idx(g, j, g_cnt)];
                    for r_loc in 0..n_eff_g {
                        let t_loc = r_loc + max_lag;
                        let pred = cluster_cell_mean(
                            &y_work,
                            x_in,
                            &x_between,
                            latent_center_x,
                            g,
                            g_cnt,
                            st,
                            t_loc,
                            j,
                            d,
                            kx,
                            lags,
                            max_lag,
                            mu_gj,
                            &theta,
                            &slopes_g,
                            n_slopes_per_out,
                            latent_center_y,
                            residual_dynamics,
                            n_obs,
                        );
                        let err = y_work[idx(st + t_loc, j, n_obs)] - pred;
                        rss_j += err * err;
                    }
                    let mut n_cnt = n_eff_g as f64;
                    if latent_center_y && !residual_dynamics && n_lags == 1 && d == 1 {
                        let phi = slopes_g[0].clamp(-0.98, 0.98);
                        let e0 = y_work[idx(st, 0, n_obs)] - mu_gj;
                        rss_j += (1.0 - phi * phi).max(0.02) * e0 * e0;
                        n_cnt += 1.0;
                    }

                    let logpost = |lv_val: f64, th_slice: &[f64]| -> f64 {
                        let mut lp = -0.5 * n_cnt * lv_val - 0.5 * (-lv_val).exp() * rss_j;
                        let mut diff = vec![0.0; q_re];
                        for q in 0..q_re {
                            let v = if q == q_idx { lv_val } else { th_slice[q] };
                            diff[q] = v - mu_prior_g[q];
                        }
                        lp -= 0.5 * quad_form(&diff, &omega_inv, q_re);
                        for m_z in 0..kz {
                            let zv = z_in[idx(g, m_z, g_cnt)];
                            if zv.is_finite() && z_on_re_mask[idx(q_idx, m_z, q_re)] != 0 {
                                let mut zp = beta_z[idx(0, m_z, p_z)];
                                for q in 0..q_re {
                                    if z_on_re_mask[idx(q, m_z, q_re)] != 0 {
                                        let v = if q == q_idx { lv_val } else { th_slice[q] };
                                        zp += beta_z[idx(1 + q, m_z, p_z)] * v;
                                    }
                                }
                                for kw_i in 0..kw {
                                    if z_on_w_mask[idx(kw_i, m_z, kw)] != 0 {
                                        zp += beta_z[idx(1 + q_re + kw_i, m_z, p_z)]
                                            * w_in[idx(g, kw_i, g_cnt)];
                                    }
                                }
                                let ez = zv - zp;
                                lp -= 0.5 * ez * ez / sigma2_z[m_z].max(1e-6);
                            }
                        }
                        lp
                    };

                    let cur_th: Vec<f64> = (0..q_re).map(|q| theta[idx(g, q, g_cnt)]).collect();
                    let log_acc = logpost(prop_lv, &cur_th) - logpost(cur_lv, &cur_th);
                    if rng.runif().ln() < log_acc {
                        theta[idx(g, q_idx, g_cnt)] = prop_lv;
                        mh_accept_logv[mh_i] += 1;
                    }
                    if step <= warmup && step % 25 == 0 {
                        let rate = mh_accept_logv[mh_i] as f64 / 25.0;
                        mh_accept_logv[mh_i] = 0;
                        if rate > 0.55 {
                            mh_step_logv[mh_i] = (mh_step_logv[mh_i] * 1.18).min(2.5);
                        } else if rate < 0.30 {
                            mh_step_logv[mh_i] = (mh_step_logv[mh_i] * 0.84).max(0.02);
                        }
                    }
                }
            }
        }

        // 2. Update fixed within-level slopes (if any)
        if n_fixed_slopes > 0 {
            let mut p_fs = eye(
                n_fixed_slopes,
                if prior_mode == 1 {
                    1e-10
                } else {
                    1.0 / (beta_sd * beta_sd)
                },
            );
            let mut b_fs = vec![0.0; n_fixed_slopes];
            for g in 0..g_cnt {
                let st = offsets[g];
                let en = offsets[g + 1];
                let tg = en - st;
                if tg <= max_lag {
                    continue;
                }
                for j in 0..d {
                    let s2 = if is_cat[j] {
                        1.0
                    } else if logv_re_idx[j] >= 0 {
                        theta[idx(g, logv_re_idx[j] as usize, g_cnt)]
                            .clamp(-12.0, 12.0)
                            .exp()
                    } else {
                        fixed_sigma2[j].max(1e-6)
                    };
                    let mu_gj = theta[idx(g, j, g_cnt)];
                    for t_loc in max_lag..tg {
                        let mut y_dev = y_work[idx(st + t_loc, j, n_obs)] - mu_gj;
                        let mut active: Vec<(usize, f64)> = Vec::new();
                        for s in 0..n_slopes_per_out {
                            let xv = if s < n_lags * d {
                                let l_idx = s / d;
                                let m_col = s % d;
                                let yl = y_work[idx(st + t_loc - lags[l_idx], m_col, n_obs)];
                                if latent_center_y {
                                    yl - theta[idx(g, m_col, g_cnt)]
                                } else {
                                    yl
                                }
                            } else {
                                let k_x = s - n_lags * d;
                                if latent_center_x[k_x] != 0 {
                                    x_in[idx(st + t_loc, k_x, n_obs)]
                                        - x_between[idx(g, k_x, g_cnt)]
                                } else {
                                    x_in[idx(st + t_loc, k_x, n_obs)]
                                }
                            };
                            let flat = j * n_slopes_per_out + s;
                            if slope_re_idx[flat] >= 0 {
                                y_dev -= theta[idx(g, slope_re_idx[flat] as usize, g_cnt)] * xv;
                            } else {
                                active.push((fixed_slope_idx[flat] as usize, xv));
                            }
                        }
                        for &(fi, x_f) in &active {
                            b_fs[fi] += x_f * y_dev / s2;
                            for &(fj, x_fj) in &active {
                                p_fs[idx(fi, fj, n_fixed_slopes)] += x_f * x_fj / s2;
                            }
                        }
                    }
                }
            }
            fixed_slopes = rmvnorm_precision(&mut rng, &p_fs, &b_fs, n_fixed_slopes);
        }

        // 3. Update fixed within-level residual variances (for outcomes without random logv)
        for j in 0..d {
            if logv_re_idx[j] < 0 && !is_cat[j] {
                let mut rss = 0.0;
                let mut cnt = 0usize;
                for g in 0..g_cnt {
                    let st = offsets[g];
                    let en = offsets[g + 1];
                    let tg = en - st;
                    if tg <= max_lag {
                        continue;
                    }
                    let mut slopes_g = vec![0.0; n_slopes_per_out * d];
                    for jj in 0..d {
                        for s in 0..n_slopes_per_out {
                            let fl = jj * n_slopes_per_out + s;
                            slopes_g[idx(s, jj, n_slopes_per_out)] = if slope_re_idx[fl] >= 0 {
                                theta[idx(g, slope_re_idx[fl] as usize, g_cnt)]
                            } else {
                                fixed_slopes[fixed_slope_idx[fl] as usize]
                            };
                        }
                    }
                    let mu_gj = theta[idx(g, j, g_cnt)];
                    for t_loc in max_lag..tg {
                        let pred = cluster_cell_mean(
                            &y_work,
                            x_in,
                            &x_between,
                            latent_center_x,
                            g,
                            g_cnt,
                            st,
                            t_loc,
                            j,
                            d,
                            kx,
                            lags,
                            max_lag,
                            mu_gj,
                            &theta,
                            &slopes_g,
                            n_slopes_per_out,
                            latent_center_y,
                            residual_dynamics,
                            n_obs,
                        );
                        let err = y_work[idx(st + t_loc, j, n_obs)] - pred;
                        rss += err * err;
                        cnt += 1;
                    }
                }
                let (sh, sc) = if prior_mode == 1 {
                    ((0.5 * cnt as f64 - 1.0).max(0.5), 0.5 * rss.max(1e-8))
                } else {
                    (0.001 + 0.5 * cnt as f64, 0.001 + 0.5 * rss)
                };
                fixed_sigma2[j] = rng.rinv_gamma(sh, sc);
            }
        }

        // 4. Update Between-Level structural regressions gamma_b ((1+kw) x q_re) and Omega_B!
        // Use exact equation-by-equation SUR Gibbs conditional on Omega_B so sparse masks on w are respected!
        {
            for q in 0..q_re {
                let o_qq = omega_inv[idx(q, q, q_re)].max(1e-8);
                // Active columns in [1, w_0..w_{kw-1}]
                let mut active_cols = vec![0usize];
                for k in 0..kw {
                    if re_on_w_mask[idx(k, q, kw)] != 0 {
                        active_cols.push(1 + k);
                    }
                }
                let pa = active_cols.len();
                let mut w_des = vec![0.0; g_cnt * pa];
                let mut y_adj = vec![0.0; g_cnt];
                for g in 0..g_cnt {
                    for (ci, &col_id) in active_cols.iter().enumerate() {
                        w_des[idx(g, ci, g_cnt)] = if col_id == 0 {
                            1.0
                        } else {
                            w_in[idx(g, col_id - 1, g_cnt)]
                        };
                    }
                    let mut adj = theta[idx(g, q, g_cnt)];
                    for q2 in 0..q_re {
                        if q2 != q {
                            let o_qq2 = omega_inv[idx(q, q2, q_re)];
                            if o_qq2.abs() > 1e-14 {
                                let mut m_q2 = gamma_b[idx(0, q2, p_w)];
                                for k in 0..kw {
                                    if re_on_w_mask[idx(k, q2, kw)] != 0 {
                                        m_q2 +=
                                            gamma_b[idx(1 + k, q2, p_w)] * w_in[idx(g, k, g_cnt)];
                                    }
                                }
                                adj += (o_qq2 / o_qq) * (theta[idx(g, q2, g_cnt)] - m_q2);
                            }
                        }
                    }
                    y_adj[g] = adj;
                }
                let wtw = mat_t_mul(&w_des, &w_des, pa, g_cnt, pa);
                let wty = mat_t_mul(&w_des, &y_adj, pa, g_cnt, 1);
                let mut prec = vec![0.0; pa * pa];
                for i in 0..(pa * pa) {
                    prec[i] = wtw[i] * o_qq;
                }
                let p_pr = if prior_mode == 1 {
                    1e-10
                } else {
                    1.0 / (beta_sd * beta_sd)
                };
                for i in 0..pa {
                    prec[idx(i, i, pa)] += p_pr;
                }
                let info: Vec<f64> = wty.iter().map(|v| v * o_qq).collect();
                let g_draw = rmvnorm_precision(&mut rng, &prec, &info, pa);
                for k in 0..p_w {
                    gamma_b[idx(k, q, p_w)] = 0.0;
                }
                for (ci, &col_id) in active_cols.iter().enumerate() {
                    gamma_b[idx(col_id, q, p_w)] = g_draw[ci];
                }
            }

            // Update Omega_B via Inverse-Wishart on between residuals zeta_g = theta_g - W_g gamma_b
            let mut s_mat = eye(
                q_re,
                if prior_mode == 1 {
                    1e-5
                } else {
                    omega_scale_diag.max(1e-4)
                },
            );
            for g in 0..g_cnt {
                let mut z_g = vec![0.0; q_re];
                for q in 0..q_re {
                    let mut m = gamma_b[idx(0, q, p_w)];
                    for k in 0..kw {
                        if re_on_w_mask[idx(k, q, kw)] != 0 {
                            m += gamma_b[idx(1 + k, q, p_w)] * w_in[idx(g, k, g_cnt)];
                        }
                    }
                    z_g[q] = theta[idx(g, q, g_cnt)] - m;
                }
                for c in 0..q_re {
                    for r in 0..q_re {
                        s_mat[idx(r, c, q_re)] += z_g[r] * z_g[c];
                    }
                }
            }
            let df_post = if prior_mode == 1 {
                // Mplus IW(-q-1, 0) improper or minimally proper df = g_cnt - q_re - 1 (clamped >= q_re + 1)
                (g_cnt as f64 - q_re as f64 - 1.0).max(q_re as f64 + 1.0)
            } else {
                (q_re as f64 + omega_df_add.max(1.0)) + g_cnt as f64
            };
            omega_b = rinv_wishart(&mut rng, df_post, &s_mat, q_re);
        }

        // 5. Update Between-Level distal outcome equations z_m (e.g. z ON y s logv w)
        for m_z in 0..kz {
            let mut active = vec![0usize];
            for q in 0..q_re {
                if z_on_re_mask[idx(q, m_z, q_re)] != 0 {
                    active.push(1 + q);
                }
            }
            for k in 0..kw {
                if z_on_w_mask[idx(k, m_z, kw)] != 0 {
                    active.push(1 + q_re + k);
                }
            }
            let pa = active.len();
            let mut x_z = Vec::new();
            let mut y_z = Vec::new();
            for g in 0..g_cnt {
                let zv = z_in[idx(g, m_z, g_cnt)];
                if zv.is_finite() {
                    y_z.push(zv);
                    for &col_id in &active {
                        let v = if col_id == 0 {
                            1.0
                        } else if col_id <= q_re {
                            theta[idx(g, col_id - 1, g_cnt)]
                        } else {
                            w_in[idx(g, col_id - 1 - q_re, g_cnt)]
                        };
                        x_z.push(v);
                    }
                }
            }
            let nz = y_z.len();
            if nz > pa {
                // Re-layout row-major temp into col-major nz x pa
                let mut xz_cm = vec![0.0; nz * pa];
                for r in 0..nz {
                    for c in 0..pa {
                        xz_cm[idx(r, c, nz)] = x_z[r * pa + c];
                    }
                }
                let xtx = mat_t_mul(&xz_cm, &xz_cm, pa, nz, pa);
                let xty = mat_t_mul(&xz_cm, &y_z, pa, nz, 1);
                let sz2 = sigma2_z[m_z].max(1e-6);
                let mut prec = vec![0.0; pa * pa];
                for i in 0..(pa * pa) {
                    prec[i] = xtx[i] / sz2;
                }
                let p_pr = if prior_mode == 1 {
                    1e-10
                } else {
                    1.0 / (beta_sd * beta_sd)
                };
                for i in 0..pa {
                    prec[idx(i, i, pa)] += p_pr;
                }
                let info: Vec<f64> = xty.iter().map(|v| v / sz2).collect();
                let bz = rmvnorm_precision(&mut rng, &prec, &info, pa);
                for c in 0..p_z {
                    beta_z[idx(c, m_z, p_z)] = 0.0;
                }
                for (ci, &col_id) in active.iter().enumerate() {
                    beta_z[idx(col_id, m_z, p_z)] = bz[ci];
                }
                let fit_z = mat_vec(&xz_cm, &bz, nz, pa);
                let rss_z: f64 = y_z
                    .iter()
                    .zip(fit_z)
                    .map(|(yi, fi)| (yi - fi) * (yi - fi))
                    .sum();
                let (sh, sc) = if prior_mode == 1 {
                    ((0.5 * nz as f64 - 1.0).max(0.5), 0.5 * rss_z.max(1e-8))
                } else {
                    (0.001 + 0.5 * nz as f64, 0.001 + 0.5 * rss_z)
                };
                sigma2_z[m_z] = rng.rinv_gamma(sh, sc);
            }
        }

        // Store retained draw
        if keep_pos < n_keep && step == keep_indices[keep_pos] {
            let mut cur = 0usize;
            for q in 0..q_re {
                let val = if q < d && is_cat[q] {
                    -gamma_b[idx(0, q, p_w)]
                } else {
                    gamma_b[idx(0, q, p_w)]
                };
                draws[idx(keep_pos, cur, n_keep)] = val;
                cur += 1;
            }
            for q in 0..q_re {
                for k in 0..kw {
                    if re_on_w_mask[idx(k, q, kw)] != 0 {
                        draws[idx(keep_pos, cur, n_keep)] = gamma_b[idx(1 + k, q, p_w)];
                        cur += 1;
                    }
                }
            }
            for &fs in &fixed_slopes {
                draws[idx(keep_pos, cur, n_keep)] = fs;
                cur += 1;
            }
            for c in 0..q_re {
                for r in c..q_re {
                    draws[idx(keep_pos, cur, n_keep)] = omega_b[idx(r, c, q_re)];
                    cur += 1;
                }
            }
            for j in 0..d {
                if logv_re_idx[j] < 0 && !is_cat[j] {
                    draws[idx(keep_pos, cur, n_keep)] = fixed_sigma2[j];
                    cur += 1;
                }
            }
            for m_z in 0..kz {
                draws[idx(keep_pos, cur, n_keep)] = beta_z[idx(0, m_z, p_z)];
                cur += 1;
                for q in 0..q_re {
                    if z_on_re_mask[idx(q, m_z, q_re)] != 0 {
                        draws[idx(keep_pos, cur, n_keep)] = beta_z[idx(1 + q, m_z, p_z)];
                        cur += 1;
                    }
                }
                for k in 0..kw {
                    if z_on_w_mask[idx(k, m_z, kw)] != 0 {
                        draws[idx(keep_pos, cur, n_keep)] = beta_z[idx(1 + q_re + k, m_z, p_z)];
                        cur += 1;
                    }
                }
                draws[idx(keep_pos, cur, n_keep)] = sigma2_z[m_z];
                cur += 1;
            }

            // Store cluster random effects
            for q in 0..q_re {
                for g in 0..g_cnt {
                    re_draws[keep_pos + n_keep * (g + g_cnt * q)] = theta[idx(g, q, g_cnt)];
                }
            }

            // Accumulate conditional log-likelihood & fitted values
            let mut ll_iter = 0.0;
            for g in 0..g_cnt {
                let st = offsets[g];
                let en = offsets[g + 1];
                let tg = en - st;
                let mut slopes_g = vec![0.0; n_slopes_per_out * d];
                for jj in 0..d {
                    for s in 0..n_slopes_per_out {
                        let fl = jj * n_slopes_per_out + s;
                        slopes_g[idx(s, jj, n_slopes_per_out)] = if slope_re_idx[fl] >= 0 {
                            theta[idx(g, slope_re_idx[fl] as usize, g_cnt)]
                        } else {
                            fixed_slopes[fixed_slope_idx[fl] as usize]
                        };
                    }
                }
                for j in 0..d {
                    let s2 = if is_cat[j] {
                        1.0
                    } else if logv_re_idx[j] >= 0 {
                        theta[idx(g, logv_re_idx[j] as usize, g_cnt)]
                            .clamp(-12.0, 12.0)
                            .exp()
                    } else {
                        fixed_sigma2[j].max(1e-6)
                    };
                    sig_bar_accum[idx(g, j, g_cnt)] += s2;
                    let mu_gj = theta[idx(g, j, g_cnt)];
                    for t_loc in 0..tg {
                        let fv = if t_loc >= max_lag {
                            cluster_cell_mean(
                                &y_work,
                                x_in,
                                &x_between,
                                latent_center_x,
                                g,
                                g_cnt,
                                st,
                                t_loc,
                                j,
                                d,
                                kx,
                                lags,
                                max_lag,
                                mu_gj,
                                &theta,
                                &slopes_g,
                                n_slopes_per_out,
                                latent_center_y,
                                residual_dynamics,
                                n_obs,
                            )
                        } else {
                            mu_gj
                        };
                        fitted_accum[idx(st + t_loc, j, n_obs)] += fv;
                        pred_bar_accum[idx(st + t_loc, j, n_obs)] += fv;
                        if t_loc >= max_lag && !miss_y[idx(st + t_loc, j, n_obs)] {
                            let err = y_work[idx(st + t_loc, j, n_obs)] - fv;
                            ll_iter += -0.5 * ((2.0 * PI * s2).ln() + err * err / s2);
                        }
                    }
                }
            }
            deviance_sum += -2.0 * ll_iter;
            keep_pos += 1;
        }
    }

    let inv_keep = 1.0 / (n_keep.max(1) as f64);
    let fitted_mean: Vec<f64> = fitted_accum.into_iter().map(|v| v * inv_keep).collect();
    let d_bar = deviance_sum * inv_keep;
    let mut ll_hat = 0.0;
    for g in 0..g_cnt {
        let st = offsets[g];
        let en = offsets[g + 1];
        let tg = en - st;
        for j in 0..d {
            let s2 = (sig_bar_accum[idx(g, j, g_cnt)] * inv_keep).max(1e-6);
            for t_loc in max_lag..tg {
                if !miss_y[idx(st + t_loc, j, n_obs)] {
                    let fv = pred_bar_accum[idx(st + t_loc, j, n_obs)] * inv_keep;
                    let err = y_work[idx(st + t_loc, j, n_obs)] - fv;
                    ll_hat += -0.5 * ((2.0 * PI * s2).ln() + err * err / s2);
                }
            }
        }
    }
    let d_hat = -2.0 * ll_hat;
    let pd = (d_bar - d_hat).max(0.0);
    let dic = d_bar + pd;
    let _ = (
        spd_logdet(&chol_lower_jitter(&omega_b, q_re), q_re),
        symmetrize,
    );

    TwoLevelOutput {
        draws,
        param_count: n_params,
        re_draws,
        q_re,
        fitted_mean,
        dic,
        pd,
    }
}

#[allow(clippy::too_many_arguments)]
fn cluster_cell_mean(
    y_work: &[f64],
    x_in: &[f64],
    x_between: &[f64],
    latent_center_x: &[i32],
    g: usize,
    g_cnt: usize,
    st: usize,
    t_loc: usize,
    j: usize,
    d: usize,
    kx: usize,
    lags: &[usize],
    max_lag: usize,
    mu_gj: f64,
    theta: &[f64],
    slopes_g: &[f64],
    n_slopes_per_out: usize,
    latent_center_y: bool,
    residual_dynamics: bool,
    n_obs: usize,
) -> f64 {
    let n_lags = lags.len();
    let mut pred = mu_gj;
    for k_x in 0..kx {
        let xv = if latent_center_x[k_x] != 0 {
            x_in[idx(st + t_loc, k_x, n_obs)] - x_between[idx(g, k_x, g_cnt)]
        } else {
            x_in[idx(st + t_loc, k_x, n_obs)]
        };
        pred += slopes_g[idx(n_lags * d + k_x, j, n_slopes_per_out)] * xv;
    }
    if t_loc < max_lag {
        return pred;
    }
    for (l_idx, &lag_v) in lags.iter().enumerate() {
        for m_col in 0..d {
            let phi = slopes_g[idx(l_idx * d + m_col, j, n_slopes_per_out)];
            let yl = y_work[idx(st + t_loc - lag_v, m_col, n_obs)];
            if residual_dynamics {
                let mut u_l = yl - theta[idx(g, m_col, g_cnt)];
                for k_x in 0..kx {
                    let xl = if latent_center_x[k_x] != 0 {
                        x_in[idx(st + t_loc - lag_v, k_x, n_obs)] - x_between[idx(g, k_x, g_cnt)]
                    } else {
                        x_in[idx(st + t_loc - lag_v, k_x, n_obs)]
                    };
                    u_l -= slopes_g[idx(n_lags * d + k_x, m_col, n_slopes_per_out)] * xl;
                }
                pred += phi * u_l;
            } else if latent_center_y {
                pred += phi * (yl - theta[idx(g, m_col, g_cnt)]);
            } else {
                pred += phi * yl;
            }
        }
    }
    pred
}
