use crate::linalg::{
    chol_lower_jitter, eye, idx, mat_t_mul, mat_vec, quad_form, rinv_wishart, rmvnorm_precision,
    spd_invert, spd_logdet, symmetrize,
};
use crate::rng::{log_std_norm_cdf, Rng};
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
    outcome_types: &[i32], // 0 = cont, >= 2 = K-cat probit, -1/-2/-3 = Tobit censored
    censor_lo: &[f64],     // length d
    censor_hi: &[f64],     // length d
    shared_cov_loadings: &[f64], // length d: non-zero entries define shared random residual covariance factor f (Ex 9.32b)
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
    let is_cens: Vec<bool> = outcome_types.iter().map(|&k| k < 0).collect();
    let any_cat = is_cat.iter().any(|&b| b);
    let any_cens = is_cens.iter().any(|&b| b);
    let has_random_logvf = d > 1
        && !any_cat
        && !any_cens
        && shared_cov_loadings.len() >= d
        && shared_cov_loadings.iter().take(d).any(|&v| v.abs() > 1e-12);
    let cov_lam: Vec<f64> = if has_random_logvf {
        shared_cov_loadings[..d].to_vec()
    } else {
        vec![0.0; d]
    };

    // Cluster row offsets
    let mut offsets = Vec::with_capacity(g_cnt + 1);
    offsets.push(0usize);
    for &sz in cluster_sizes {
        offsets.push(offsets.last().unwrap() + sz);
    }

    // Map random effects theta_i: [mu_{i,0..d}, random_slopes..., logv_{i,...}, optional logvf_i]
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
    let logvf_re_idx: i32 = if has_random_logvf {
        (d + n_rand_slopes + n_rand_logv) as i32
    } else {
        -1
    };
    // Use full unstructured d x d within innovation covariance Sigma_W for multivariate continuous models without random logv/logvf
    let use_full_sigma_w = d > 1 && n_rand_logv == 0 && !has_random_logvf && !any_cat && !any_cens;
    let n_within_cov_params = if use_full_sigma_w {
        d * (d + 1) / 2
    } else {
        n_fixed_sig
    };
    let q_re = d + n_rand_slopes + n_rand_logv + usize::from(has_random_logvf);

    // Precompute Block A (intercepts + random exogenous slopes) vs Block B (random AR slopes)
    // for joint linear Gibbs when !residual_dynamics (eliminates intercept-trend correlation ridge)
    let mut exog_rand_slots: Vec<(usize, usize, usize)> = Vec::new(); // (re_pos, out_j, k_x)
    let mut ar_rand_slots: Vec<(usize, usize, usize, usize)> = Vec::new(); // (re_pos, out_j, l_idx, m_col)
    for j in 0..d {
        for s in 0..n_slopes_per_out {
            let flat = j * n_slopes_per_out + s;
            if slope_re_idx[flat] >= 0 {
                let rp = slope_re_idx[flat] as usize;
                if s < n_lags * d {
                    ar_rand_slots.push((rp, j, s / d, s % d));
                } else {
                    exog_rand_slots.push((rp, j, s - n_lags * d));
                }
            }
        }
    }

    // Extra thresholds for ordinal outcomes (K >= 3): tau_1 = 0 absorbed into -mean_y, tau_2..tau_{K-1} free
    let mut extra_thresholds: Vec<Vec<f64>> = (0..d)
        .map(|j| {
            let k = outcome_types[j];
            if k >= 3 {
                (0..(k as usize - 1)).map(|c| c as f64 * 0.8).collect()
            } else {
                Vec::new()
            }
        })
        .collect();
    let n_extra_thresh: usize = extra_thresholds
        .iter()
        .map(|v| v.len().saturating_sub(1))
        .sum();

    // Shared latent residual shock f_{gt} for random residual covariance models (Ex 9.32b)
    let mut f_shock: Vec<f64> = if has_random_logvf {
        vec![0.0; n_obs]
    } else {
        Vec::new()
    };

    // Initialize y_work, missingness, and censoring flags
    let mut y_work = y_in.to_vec();
    let mut miss_y = vec![false; n_obs * d];
    let mut cens_flag = vec![0i8; n_obs * d];
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
        let lo_c = censor_lo.get(j).copied().unwrap_or(f64::NEG_INFINITY);
        let hi_c = censor_hi.get(j).copied().unwrap_or(f64::INFINITY);
        let ot = outcome_types[j];
        for r in 0..n_obs {
            let cell = idx(r, j, n_obs);
            if miss_y[cell] {
                y_work[cell] = if is_cat[j] { 0.0 } else { m };
            } else if is_cat[j] {
                let kc = outcome_types[j].max(2) as f64;
                y_work[cell] = ((y_in[cell] + 0.5) / kc - 0.5) * 2.0;
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
        if logvf_re_idx >= 0 {
            theta[idx(g, logvf_re_idx as usize, g_cnt)] = -0.7;
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
    let mut sigma_w = eye(d, 1.0);

    // Distal outcome parameters: for each m in 0..kz, predictors [1, selected theta, selected w]
    let p_z = 1 + q_re + kw;
    let mut beta_z: Vec<f64> = vec![0.0_f64; p_z * kz];
    let mut sigma2_z: Vec<f64> = vec![0.5_f64; kz];

    // PSEM Bayesian Lasso scale-mixture shrinkage states (prior_mode == 2)
    let mut psem_tau2_fs = vec![1.0_f64; n_fixed_slopes];
    let mut psem_tau2_gb = vec![1.0_f64; p_w * q_re];
    let mut psem_tau2_bz = vec![1.0_f64; p_z * kz];
    let mut psem_lambda2 = 1.0_f64;

    // Count population parameters reported in draws:
    // 1. gamma_b intercepts (q_re)
    // 2. active gamma_b slopes on w
    // 3. fixed within slopes (n_fixed_slopes)
    // 4. lower triangle of omega_b (q_re*(q_re+1)/2)
    // 5. fixed within sigma2 (or lower triangle of full Sigma_W when use_full_sigma_w)
    // 6. distal z intercepts, active z slopes on theta & w, and sigma2_z
    // 7. extra ordinal thresholds (tau_2..tau_{K-1})
    let n_re_w_active: usize = re_on_w_mask.iter().filter(|&&v| v != 0).count();
    let n_z_re_active: usize = z_on_re_mask.iter().filter(|&&v| v != 0).count();
    let n_z_w_active: usize = z_on_w_mask.iter().filter(|&&v| v != 0).count();
    let n_omega_cov = q_re * (q_re + 1) / 2;
    let n_params = q_re
        + n_re_w_active
        + n_fixed_slopes
        + n_omega_cov
        + n_within_cov_params
        + kz
        + n_z_re_active
        + n_z_w_active
        + kz
        + n_extra_thresh;

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
    let mut sig_vf_accum = vec![0.0; if has_random_logvf { g_cnt } else { 0 }];
    let mut sigma_w_accum = vec![0.0; d * d];
    let mut keep_pos = 0usize;

    for step in 1..=iter {
        let omega_inv = spd_invert(&omega_b, q_re);
        let sigma_w_inv = if use_full_sigma_w {
            spd_invert(&sigma_w, d)
        } else {
            Vec::new()
        };

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

            // 1a. Probit / Tobit censored / missing data imputation inside cluster g
            for t_loc in 0..tg {
                let r_abs = st + t_loc;
                for j in 0..d {
                    let cell = idx(r_abs, j, n_obs);
                    let is_m = miss_y[cell];
                    let c_fl = cens_flag[cell];
                    if !is_cat[j] && !is_m && c_fl == 0 {
                        continue;
                    }
                    let mu_gj = theta[idx(g, j, g_cnt)];
                    let mut pred = cluster_cell_mean(
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
                    let cond_var_j = if is_cat[j] {
                        1.0
                    } else if use_full_sigma_w && t_loc >= max_lag {
                        let s_jj = sigma_w_inv[idx(j, j, d)].max(1e-8);
                        for k in 0..d {
                            if k != j {
                                let s_jk = sigma_w_inv[idx(j, k, d)];
                                if s_jk.abs() > 1e-14 {
                                    let pred_k = cluster_cell_mean(
                                        &y_work,
                                        x_in,
                                        &x_between,
                                        latent_center_x,
                                        g,
                                        g_cnt,
                                        st,
                                        t_loc,
                                        k,
                                        d,
                                        kx,
                                        lags,
                                        max_lag,
                                        theta[idx(g, k, g_cnt)],
                                        &theta,
                                        &slopes_g,
                                        n_slopes_per_out,
                                        latent_center_y,
                                        residual_dynamics,
                                        n_obs,
                                    );
                                    pred -= (s_jk / s_jj) * (y_work[idx(r_abs, k, n_obs)] - pred_k);
                                }
                            }
                        }
                        (1.0 / s_jj).clamp(1e-6, 100.0)
                    } else {
                        sig2_g[j].max(1e-6)
                    };

                    if is_cat[j] || is_m || c_fl != 0 {
                        let mut prec = 1.0 / cond_var_j;
                        let mut info = pred * prec;
                        if !residual_dynamics && n_lags >= 1 {
                            for (l_idx, &lag_v) in lags.iter().enumerate() {
                                let t_f = t_loc + lag_v;
                                if t_f >= max_lag && t_f < tg {
                                    for m_out in 0..d {
                                        let phi =
                                            slopes_g[idx(l_idx * d + j, m_out, n_slopes_per_out)];
                                        if phi.abs() > 1e-12 {
                                            let s2_m = sig2_g[m_out].max(1e-6);
                                            let pred_f = cluster_cell_mean(
                                                &y_work,
                                                x_in,
                                                &x_between,
                                                latent_center_x,
                                                g,
                                                g_cnt,
                                                st,
                                                t_f,
                                                m_out,
                                                d,
                                                kx,
                                                lags,
                                                max_lag,
                                                theta[idx(g, m_out, g_cnt)],
                                                &theta,
                                                &slopes_g,
                                                n_slopes_per_out,
                                                latent_center_y,
                                                false,
                                                n_obs,
                                            ) + if has_random_logvf {
                                                cov_lam[m_out] * f_shock[st + t_f]
                                            } else {
                                                0.0
                                            };
                                            let rem = y_work[idx(st + t_f, m_out, n_obs)]
                                                - (pred_f - phi * y_work[cell]);
                                            prec += (phi * phi) / s2_m;
                                            info += phi * rem / s2_m;
                                        }
                                    }
                                }
                            }
                        }
                        let post_mean = info / prec.max(1e-8);
                        let post_sd = (1.0 / prec.max(1e-8)).sqrt();
                        if is_cat[j] && !is_m {
                            let obs_c = y_in[cell].round() as i32;
                            let k_cat = outcome_types[j];
                            let (lo, hi) = if k_cat == 2 {
                                if obs_c <= 0 {
                                    (f64::NEG_INFINITY, 0.0)
                                } else {
                                    (0.0, f64::INFINITY)
                                }
                            } else {
                                let taus = &extra_thresholds[j];
                                let c = (obs_c.max(0) as usize).min(k_cat as usize - 1);
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
                            y_work[cell] = rng.rtruncnorm(post_mean, post_sd, lo, hi);
                        } else if c_fl != 0 {
                            let (lo, hi) = if c_fl < 0 {
                                (f64::NEG_INFINITY, censor_lo.get(j).copied().unwrap_or(0.0))
                            } else {
                                (censor_hi.get(j).copied().unwrap_or(0.0), f64::INFINITY)
                            };
                            y_work[cell] = rng.rtruncnorm(post_mean, post_sd, lo, hi);
                        } else {
                            y_work[cell] = post_mean + post_sd * rng.rnorm();
                        }
                    }
                }
            }

            // 1a-ii. Shared latent within-level residual shock f_{gt} for random residual covariance (Ex 9.32b)
            if has_random_logvf {
                let vf_g = theta[idx(g, logvf_re_idx as usize, g_cnt)]
                    .clamp(-12.0, 12.0)
                    .exp();
                let mut p_f = 1.0 / vf_g;
                for j in 0..d {
                    let lj = cov_lam[j];
                    if lj.abs() > 1e-12 {
                        p_f += (lj * lj) / sig2_g[j].max(1e-6);
                    }
                }
                let sd_f = (1.0 / p_f.max(1e-8)).sqrt();
                for t_loc in max_lag..tg {
                    let r_abs = st + t_loc;
                    let mut h_f = 0.0;
                    for j in 0..d {
                        let lj = cov_lam[j];
                        if lj.abs() > 1e-12 {
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
                                theta[idx(g, j, g_cnt)],
                                &theta,
                                &slopes_g,
                                n_slopes_per_out,
                                latent_center_y,
                                residual_dynamics,
                                n_obs,
                            );
                            h_f += lj * (y_work[idx(r_abs, j, n_obs)] - pred) / sig2_g[j].max(1e-6);
                        }
                    }
                    f_shock[r_abs] = h_f / p_f + sd_f * rng.rnorm();
                }
            }

            // 1b. Update cluster linear random effects (mu_g and random slopes) conditional on Omega_B & distal z_g!
            let q_lin = d + n_rand_slopes;
            // Start with prior precision & info on linear subblock [0..q_lin] conditional on logv/logvf subblock
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
                // Cross-precision contribution from current logv/logvf components (q_lin..q_re)
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

            let use_joint_mu_exog = !residual_dynamics && !exog_rand_slots.is_empty();
            if use_joint_mu_exog {
                // Exact Joint Linear Block A: (mu_{g,0..d}, random exogenous slopes) | AR slopes
                // Eliminates intercept-trend correlation ridge (e.g. Ex 9.37 with uncentered time = 1..100)!
                let n_a = d + exog_rand_slots.len();
                let mut block_a_re: Vec<usize> = Vec::with_capacity(n_a);
                for j in 0..d {
                    block_a_re.push(j);
                }
                for &(rp, _, _) in &exog_rand_slots {
                    block_a_re.push(rp);
                }
                let mut p_a = vec![0.0; n_a * n_a];
                let mut b_a = vec![0.0; n_a];
                for (ca, &rc) in block_a_re.iter().enumerate() {
                    for (ra, &rr) in block_a_re.iter().enumerate() {
                        p_a[idx(ra, ca, n_a)] = lin_prec[idx(rr, rc, q_lin)];
                    }
                    let mut rhs = lin_info[rc];
                    for &(rb, _, _, _) in &ar_rand_slots {
                        rhs -= lin_prec[idx(rc, rb, q_lin)] * theta[idx(g, rb, g_cnt)];
                    }
                    b_a[ca] = rhs;
                }
                let mut c_mat = eye(d, 1.0);
                if latent_center_y {
                    for j in 0..d {
                        for l_idx in 0..n_lags {
                            for m_col in 0..d {
                                c_mat[idx(j, m_col, d)] -=
                                    slopes_g[idx(l_idx * d + m_col, j, n_slopes_per_out)];
                            }
                        }
                    }
                }
                let mut a_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); d];
                let mut r_vec = vec![0.0; d];
                for r_loc in 0..n_eff_g {
                    let t_loc = r_loc + max_lag;
                    for j in 0..d {
                        a_rows[j].clear();
                        let mut rem = y_work[idx(st + t_loc, j, n_obs)];
                        if has_random_logvf {
                            rem -= cov_lam[j] * f_shock[st + t_loc];
                        }
                        for (l_idx, &lag_v) in lags.iter().enumerate() {
                            for m_col in 0..d {
                                let phi = slopes_g[idx(l_idx * d + m_col, j, n_slopes_per_out)];
                                rem -= phi * y_work[idx(st + t_loc - lag_v, m_col, n_obs)];
                            }
                        }
                        for k_x in 0..kx {
                            let flat = j * n_slopes_per_out + n_lags * d + k_x;
                            if slope_re_idx[flat] < 0 {
                                let xv = if latent_center_x[k_x] != 0 {
                                    x_in[idx(st + t_loc, k_x, n_obs)]
                                        - x_between[idx(g, k_x, g_cnt)]
                                } else {
                                    x_in[idx(st + t_loc, k_x, n_obs)]
                                };
                                rem -= fixed_slopes[fixed_slope_idx[flat] as usize] * xv;
                            }
                        }
                        for m_col in 0..d {
                            let cm = c_mat[idx(j, m_col, d)];
                            if cm.abs() > 1e-14 {
                                a_rows[j].push((m_col, cm));
                            }
                        }
                        for (ei, &(_, out_j, k_x)) in exog_rand_slots.iter().enumerate() {
                            if out_j == j {
                                let xv = if latent_center_x[k_x] != 0 {
                                    x_in[idx(st + t_loc, k_x, n_obs)]
                                        - x_between[idx(g, k_x, g_cnt)]
                                } else {
                                    x_in[idx(st + t_loc, k_x, n_obs)]
                                };
                                if xv.abs() > 1e-14 {
                                    a_rows[j].push((d + ei, xv));
                                }
                            }
                        }
                        r_vec[j] = rem;
                    }
                    if use_full_sigma_w {
                        for j in 0..d {
                            for k in 0..d {
                                let w_jk = sigma_w_inv[idx(j, k, d)];
                                if w_jk.abs() < 1e-14 {
                                    continue;
                                }
                                let rk = r_vec[k];
                                for &(ra, xa) in &a_rows[j] {
                                    b_a[ra] += w_jk * xa * rk;
                                    for &(ca, xc) in &a_rows[k] {
                                        p_a[idx(ra, ca, n_a)] += w_jk * xa * xc;
                                    }
                                }
                            }
                        }
                    } else {
                        for j in 0..d {
                            let inv_s2 = 1.0 / sig2_g[j].max(1e-6);
                            let rj = r_vec[j];
                            for &(ra, xa) in &a_rows[j] {
                                b_a[ra] += xa * rj * inv_s2;
                                for &(ca, xc) in &a_rows[j] {
                                    p_a[idx(ra, ca, n_a)] += xa * xc * inv_s2;
                                }
                            }
                        }
                    }
                }
                let a_draw = rmvnorm_precision(&mut rng, &p_a, &b_a, n_a);
                for (ai, &rp) in block_a_re.iter().enumerate() {
                    theta[idx(g, rp, g_cnt)] = a_draw[ai];
                }
                for &(_, out_j, k_x) in &exog_rand_slots {
                    let flat = out_j * n_slopes_per_out + n_lags * d + k_x;
                    slopes_g[idx(n_lags * d + k_x, out_j, n_slopes_per_out)] =
                        theta[idx(g, slope_re_idx[flat] as usize, g_cnt)];
                }

                // Block B: Update random AR slopes conditional on updated (mu_g, random exogenous slopes)
                if !ar_rand_slots.is_empty() {
                    let n_b = ar_rand_slots.len();
                    let mut p_b = vec![0.0; n_b * n_b];
                    let mut b_b = vec![0.0; n_b];
                    for (cb, &(rc, _, _, _)) in ar_rand_slots.iter().enumerate() {
                        for (rb, &(rr, _, _, _)) in ar_rand_slots.iter().enumerate() {
                            p_b[idx(rb, cb, n_b)] = lin_prec[idx(rr, rc, q_lin)];
                        }
                        let mut rhs = lin_info[rc];
                        for &ra in &block_a_re {
                            rhs -= lin_prec[idx(rc, ra, q_lin)] * theta[idx(g, ra, g_cnt)];
                        }
                        b_b[cb] = rhs;
                    }
                    let mut b_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); d];
                    let mut y_rem = vec![0.0; d];
                    for r_loc in 0..n_eff_g {
                        let t_loc = r_loc + max_lag;
                        for j in 0..d {
                            b_rows[j].clear();
                            let mut yd =
                                y_work[idx(st + t_loc, j, n_obs)] - theta[idx(g, j, g_cnt)];
                            if has_random_logvf {
                                yd -= cov_lam[j] * f_shock[st + t_loc];
                            }
                            for k_x in 0..kx {
                                let xv = if latent_center_x[k_x] != 0 {
                                    x_in[idx(st + t_loc, k_x, n_obs)]
                                        - x_between[idx(g, k_x, g_cnt)]
                                } else {
                                    x_in[idx(st + t_loc, k_x, n_obs)]
                                };
                                yd -= slopes_g[idx(n_lags * d + k_x, j, n_slopes_per_out)] * xv;
                            }
                            for (l_idx, &lag_v) in lags.iter().enumerate() {
                                for m_col in 0..d {
                                    let flat = j * n_slopes_per_out + l_idx * d + m_col;
                                    let yl = y_work[idx(st + t_loc - lag_v, m_col, n_obs)];
                                    let reg_v = if latent_center_y {
                                        yl - theta[idx(g, m_col, g_cnt)]
                                    } else {
                                        yl
                                    };
                                    if slope_re_idx[flat] < 0 {
                                        yd -= fixed_slopes[fixed_slope_idx[flat] as usize] * reg_v;
                                    }
                                }
                            }
                            for (bi, &(_, out_j, l_idx, m_col)) in ar_rand_slots.iter().enumerate()
                            {
                                if out_j == j {
                                    let yl = y_work[idx(st + t_loc - lags[l_idx], m_col, n_obs)];
                                    let reg_v = if latent_center_y {
                                        yl - theta[idx(g, m_col, g_cnt)]
                                    } else {
                                        yl
                                    };
                                    b_rows[j].push((bi, reg_v));
                                }
                            }
                            y_rem[j] = yd;
                        }
                        if use_full_sigma_w {
                            for j in 0..d {
                                for k in 0..d {
                                    let w_jk = sigma_w_inv[idx(j, k, d)];
                                    if w_jk.abs() < 1e-14 {
                                        continue;
                                    }
                                    let yk = y_rem[k];
                                    for &(rb, xb) in &b_rows[j] {
                                        b_b[rb] += w_jk * xb * yk;
                                        for &(cb, xc) in &b_rows[k] {
                                            p_b[idx(rb, cb, n_b)] += w_jk * xb * xc;
                                        }
                                    }
                                }
                            }
                        } else {
                            for j in 0..d {
                                let inv_s2 = 1.0 / sig2_g[j].max(1e-6);
                                let yj = y_rem[j];
                                for &(rb, xb) in &b_rows[j] {
                                    b_b[rb] += xb * yj * inv_s2;
                                    for &(cb, xc) in &b_rows[j] {
                                        p_b[idx(rb, cb, n_b)] += xb * xc * inv_s2;
                                    }
                                }
                            }
                        }
                    }
                    let b_draw = rmvnorm_precision(&mut rng, &p_b, &b_b, n_b);
                    for (bi, &(rp, out_j, l_idx, m_col)) in ar_rand_slots.iter().enumerate() {
                        theta[idx(g, rp, g_cnt)] = b_draw[bi];
                        slopes_g[idx(l_idx * d + m_col, out_j, n_slopes_per_out)] = b_draw[bi];
                    }
                }
            } else {
                // Two-step alternating conditional update within the linear block (DSEM without random exog or RDSEM)
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
                    if use_full_sigma_w {
                        // Exact multivariate joint intercept update: e_{gt} = r_{gt} - C_g * mu_g, e ~ N(0, Sigma_W)
                        let mut c_mat = eye(d, 1.0);
                        if latent_center_y || residual_dynamics {
                            for j in 0..d {
                                for (l_idx, _) in lags.iter().enumerate() {
                                    for m_col in 0..d {
                                        let phi =
                                            slopes_g[idx(l_idx * d + m_col, j, n_slopes_per_out)];
                                        c_mat[idx(j, m_col, d)] -= phi;
                                    }
                                }
                            }
                        }
                        let mut r_sum = vec![0.0; d];
                        for r_loc in 0..n_eff_g {
                            let t_loc = r_loc + max_lag;
                            for j in 0..d {
                                let mut rem = y_work[idx(st + t_loc, j, n_obs)];
                                for k_x in 0..kx {
                                    let x_val = if latent_center_x[k_x] != 0 {
                                        x_in[idx(st + t_loc, k_x, n_obs)]
                                            - x_between[idx(g, k_x, g_cnt)]
                                    } else {
                                        x_in[idx(st + t_loc, k_x, n_obs)]
                                    };
                                    rem -= slopes_g[idx(n_lags * d + k_x, j, n_slopes_per_out)]
                                        * x_val;
                                }
                                for (l_idx, &lag_v) in lags.iter().enumerate() {
                                    for m_col in 0..d {
                                        let phi =
                                            slopes_g[idx(l_idx * d + m_col, j, n_slopes_per_out)];
                                        let mut y_l = y_work[idx(st + t_loc - lag_v, m_col, n_obs)];
                                        if residual_dynamics {
                                            for k_x in 0..kx {
                                                let x_l = if latent_center_x[k_x] != 0 {
                                                    x_in[idx(st + t_loc - lag_v, k_x, n_obs)]
                                                        - x_between[idx(g, k_x, g_cnt)]
                                                } else {
                                                    x_in[idx(st + t_loc - lag_v, k_x, n_obs)]
                                                };
                                                y_l -= slopes_g[idx(
                                                    n_lags * d + k_x,
                                                    m_col,
                                                    n_slopes_per_out,
                                                )] * x_l;
                                            }
                                        }
                                        rem -= phi * y_l;
                                    }
                                }
                                r_sum[j] += rem;
                            }
                        }
                        let mut sw_c = vec![0.0; d * d];
                        for col in 0..d {
                            for row in 0..d {
                                let mut s = 0.0;
                                for k in 0..d {
                                    s += sigma_w_inv[idx(row, k, d)] * c_mat[idx(k, col, d)];
                                }
                                sw_c[idx(row, col, d)] = s;
                            }
                        }
                        let ct_sw_c = mat_t_mul(&c_mat, &sw_c, d, d, d);
                        let sw_r = mat_vec(&sigma_w_inv, &r_sum, d, d);
                        let ct_sw_r = mat_t_mul(&c_mat, &sw_r, d, d, 1);
                        let nf = n_eff_g as f64;
                        for c in 0..d {
                            b_mu[c] += ct_sw_r[c];
                            for r in 0..d {
                                p_mu[idx(r, c, d)] += nf * ct_sw_c[idx(r, c, d)];
                            }
                        }
                    } else {
                        for j in 0..d {
                            let s2 = sig2_g[j].max(1e-6);
                            // Exact stationary initial observation contribution for short-T univariate AR(1) DSEM2 & RDSEM2 (Asparouhov 2026)
                            if d == 1
                                && n_lags == 1
                                && ((latent_center_y && !residual_dynamics && kx == 0)
                                    || residual_dynamics)
                            {
                                let phi = slopes_g[0].clamp(-0.98, 0.98);
                                let w0 = (1.0 - phi * phi).max(0.02) / s2;
                                let mut y0_adj = y_work[idx(st, 0, n_obs)];
                                if residual_dynamics {
                                    for k_x in 0..kx {
                                        let x0 = if latent_center_x[k_x] != 0 {
                                            x_in[idx(st, k_x, n_obs)]
                                                - x_between[idx(g, k_x, g_cnt)]
                                        } else {
                                            x_in[idx(st, k_x, n_obs)]
                                        };
                                        y0_adj -= slopes_g
                                            [idx(n_lags * d + k_x, 0, n_slopes_per_out)]
                                            * x0;
                                    }
                                }
                                p_mu[0] += w0;
                                b_mu[0] += w0 * y0_adj;
                            }
                            for r_loc in 0..n_eff_g {
                                let t_loc = r_loc + max_lag;
                                let mut c_mult = 1.0;
                                let mut rem = y_work[idx(st + t_loc, j, n_obs)];
                                if has_random_logvf {
                                    rem -= cov_lam[j] * f_shock[st + t_loc];
                                }
                                for k_x in 0..kx {
                                    let x_val = if latent_center_x[k_x] != 0 {
                                        x_in[idx(st + t_loc, k_x, n_obs)]
                                            - x_between[idx(g, k_x, g_cnt)]
                                    } else {
                                        x_in[idx(st + t_loc, k_x, n_obs)]
                                    };
                                    rem -= slopes_g[idx(n_lags * d + k_x, j, n_slopes_per_out)]
                                        * x_val;
                                }
                                if !residual_dynamics {
                                    for (l_idx, &lag_v) in lags.iter().enumerate() {
                                        for m_col in 0..d {
                                            let phi = slopes_g
                                                [idx(l_idx * d + m_col, j, n_slopes_per_out)];
                                            let y_lag =
                                                y_work[idx(st + t_loc - lag_v, m_col, n_obs)];
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
                                            let phi = slopes_g
                                                [idx(l_idx * d + m_col, j, n_slopes_per_out)];
                                            let mut u_lag =
                                                y_work[idx(st + t_loc - lag_v, m_col, n_obs)];
                                            for k_x in 0..kx {
                                                let x_l = if latent_center_x[k_x] != 0 {
                                                    x_in[idx(st + t_loc - lag_v, k_x, n_obs)]
                                                        - x_between[idx(g, k_x, g_cnt)]
                                                } else {
                                                    x_in[idx(st + t_loc - lag_v, k_x, n_obs)]
                                                };
                                                u_lag -= slopes_g[idx(
                                                    n_lags * d + k_x,
                                                    m_col,
                                                    n_slopes_per_out,
                                                )] * x_l;
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
                    let mut reg_vec = vec![0.0; n_slopes_per_out];
                    let mut y_dev_vec = vec![0.0; d];
                    let mut active_by_j: Vec<Vec<(usize, f64)>> = vec![Vec::new(); d];
                    // Stationary initial-state contribution for RDSEM2 random exogenous slopes (Asparouhov 2026)
                    if residual_dynamics && d == 1 && n_lags == 1 && kx > 0 {
                        let phi = slopes_g[0].clamp(-0.98, 0.98);
                        let w0 = (1.0 - phi * phi).max(0.02) / sig2_g[0].max(1e-6);
                        let mut y0_d = y_work[idx(st, 0, n_obs)] - theta[idx(g, 0, g_cnt)];
                        let mut act0: Vec<(usize, f64)> = Vec::new();
                        for k_x in 0..kx {
                            let x0 = if latent_center_x[k_x] != 0 {
                                x_in[idx(st, k_x, n_obs)] - x_between[idx(g, k_x, g_cnt)]
                            } else {
                                x_in[idx(st, k_x, n_obs)]
                            };
                            let flat = n_lags * d + k_x;
                            if slope_re_idx[flat] >= 0 {
                                act0.push((slope_re_idx[flat] as usize - d, x0));
                            } else {
                                y0_d -= fixed_slopes[fixed_slope_idx[flat] as usize] * x0;
                            }
                        }
                        for &(ri, xr) in &act0 {
                            b_sl[ri] += w0 * xr * y0_d;
                            for &(ci, xc) in &act0 {
                                p_sl[idx(ri, ci, n_rand_slopes)] += w0 * xr * xc;
                            }
                        }
                    }
                    for r_loc in 0..n_eff_g {
                        let t_loc = r_loc + max_lag;
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
                        for j in 0..d {
                            active_by_j[j].clear();
                            let mut yd =
                                y_work[idx(st + t_loc, j, n_obs)] - theta[idx(g, j, g_cnt)];
                            if has_random_logvf {
                                yd -= cov_lam[j] * f_shock[st + t_loc];
                            }
                            for (s, &rv_s) in reg_vec.iter().enumerate().take(n_slopes_per_out) {
                                let flat = j * n_slopes_per_out + s;
                                if slope_re_idx[flat] >= 0 {
                                    active_by_j[j].push((slope_re_idx[flat] as usize - d, rv_s));
                                } else {
                                    yd -= fixed_slopes[fixed_slope_idx[flat] as usize] * rv_s;
                                }
                            }
                            y_dev_vec[j] = yd;
                        }
                        if use_full_sigma_w {
                            for j in 0..d {
                                for k in 0..d {
                                    let w_jk = sigma_w_inv[idx(j, k, d)];
                                    if w_jk.abs() < 1e-14 {
                                        continue;
                                    }
                                    let ydk = y_dev_vec[k];
                                    for &(ri, xv_r) in &active_by_j[j] {
                                        b_sl[ri] += w_jk * xv_r * ydk;
                                        for &(ci, xv_c) in &active_by_j[k] {
                                            p_sl[idx(ri, ci, n_rand_slopes)] += w_jk * xv_r * xv_c;
                                        }
                                    }
                                }
                            }
                        } else {
                            for j in 0..d {
                                let inv_s2 = 1.0 / sig2_g[j].max(1e-6);
                                let ydj = y_dev_vec[j];
                                for &(ri, xv_r) in &active_by_j[j] {
                                    b_sl[ri] += xv_r * ydj * inv_s2;
                                    for &(ci, xv_c) in &active_by_j[j] {
                                        p_sl[idx(ri, ci, n_rand_slopes)] += xv_r * xv_c * inv_s2;
                                    }
                                }
                            }
                        }
                    }
                    let sl_draw = rmvnorm_precision(&mut rng, &p_sl, &b_sl, n_rand_slopes);
                    // Exact stationary initial-state MH step for short-T univariate AR(1) DSEM2 & RDSEM2 (Asparouhov 2026)
                    let use_stat_mh = d == 1
                        && n_lags == 1
                        && slope_re_idx[0] >= 0
                        && ((latent_center_y
                            && !residual_dynamics
                            && kx == 0
                            && n_rand_slopes == 1)
                            || residual_dynamics);
                    if use_stat_mh {
                        // First commit any random exogenous slopes in RDSEM2, then MH-check the AR(1) slope
                        for (s, &sv) in sl_draw.iter().enumerate().take(n_rand_slopes) {
                            let re_pos = d + s;
                            if re_pos != slope_re_idx[0] as usize {
                                theta[idx(g, re_pos, g_cnt)] = sv;
                            }
                        }
                        let ar_slot = slope_re_idx[0] as usize;
                        let phi_cur = theta[idx(g, ar_slot, g_cnt)];
                        let phi_prop = sl_draw[ar_slot - d];
                        if phi_prop.abs() < 0.995 {
                            if phi_cur.abs() >= 0.995 {
                                theta[idx(g, ar_slot, g_cnt)] = phi_prop;
                            } else {
                                let mut e0 = y_work[idx(st, 0, n_obs)] - theta[idx(g, 0, g_cnt)];
                                if residual_dynamics {
                                    for k_x in 0..kx {
                                        let flat_k = n_lags * d + k_x;
                                        let bk = if slope_re_idx[flat_k] >= 0 {
                                            theta[idx(g, slope_re_idx[flat_k] as usize, g_cnt)]
                                        } else {
                                            fixed_slopes[fixed_slope_idx[flat_k] as usize]
                                        };
                                        let x0 = if latent_center_x[k_x] != 0 {
                                            x_in[idx(st, k_x, n_obs)]
                                                - x_between[idx(g, k_x, g_cnt)]
                                        } else {
                                            x_in[idx(st, k_x, n_obs)]
                                        };
                                        e0 -= bk * x0;
                                    }
                                }
                                let s2 = sig2_g[0].max(1e-6);
                                let log_acc = 0.5
                                    * ((1.0 - phi_prop * phi_prop) / (1.0 - phi_cur * phi_cur))
                                        .ln()
                                    + 0.5 * (phi_prop * phi_prop - phi_cur * phi_cur) * e0 * e0
                                        / s2;
                                if log_acc >= 0.0 || rng.runif().ln() < log_acc {
                                    theta[idx(g, ar_slot, g_cnt)] = phi_prop;
                                }
                            }
                        }
                    } else {
                        for s in 0..n_rand_slopes {
                            theta[idx(g, d + s, g_cnt)] = sl_draw[s];
                        }
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
            }

            // Step iii: Strictly-concave Newton-Laplace Independence MH for random log residual variances logv_{g,j} and optional logvf_g
            for j in 0..d {
                if logv_re_idx[j] >= 0 {
                    let q_idx = logv_re_idx[j] as usize;
                    let mut cur_lv = theta[idx(g, q_idx, g_cnt)];

                    // Compute cluster residual sum of squares for outcome j (constant across logv proposals)
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
                        ) + if has_random_logvf {
                            cov_lam[j] * f_shock[st + t_loc]
                        } else {
                            0.0
                        };
                        let err = y_work[idx(st + t_loc, j, n_obs)] - pred;
                        rss_j += err * err;
                    }
                    let mut n_cnt = n_eff_g as f64;
                    if d == 1
                        && n_lags == 1
                        && ((latent_center_y && !residual_dynamics && kx == 0) || residual_dynamics)
                    {
                        let phi = slopes_g[0].clamp(-0.98, 0.98);
                        let mut e0 = y_work[idx(st, 0, n_obs)] - mu_gj;
                        if residual_dynamics {
                            for k_x in 0..kx {
                                let x0 = if latent_center_x[k_x] != 0 {
                                    x_in[idx(st, k_x, n_obs)] - x_between[idx(g, k_x, g_cnt)]
                                } else {
                                    x_in[idx(st, k_x, n_obs)]
                                };
                                e0 -= slopes_g[idx(n_lags * d + k_x, 0, n_slopes_per_out)] * x0;
                            }
                        }
                        rss_j += (1.0 - phi * phi).max(0.02) * e0 * e0;
                        n_cnt += 1.0;
                    }

                    // Collapse exact Gaussian conditional prior (Omega_B^{-1} + distal z likelihood) into (p_tot, m_prior)
                    let mut p_tot = omega_inv[idx(q_idx, q_idx, q_re)].max(1e-8);
                    let mut info_tot = p_tot * mu_prior_g[q_idx];
                    for r in 0..q_re {
                        if r != q_idx {
                            info_tot -= omega_inv[idx(q_idx, r, q_re)]
                                * (theta[idx(g, r, g_cnt)] - mu_prior_g[r]);
                        }
                    }
                    for m_z in 0..kz {
                        let zv = z_in[idx(g, m_z, g_cnt)];
                        if zv.is_finite() && z_on_re_mask[idx(q_idx, m_z, q_re)] != 0 {
                            let sz2 = sigma2_z[m_z].max(1e-6);
                            let bq = beta_z[idx(1 + q_idx, m_z, p_z)];
                            let mut rz = zv - beta_z[idx(0, m_z, p_z)];
                            for r in 0..q_re {
                                if r != q_idx && z_on_re_mask[idx(r, m_z, q_re)] != 0 {
                                    rz -= beta_z[idx(1 + r, m_z, p_z)] * theta[idx(g, r, g_cnt)];
                                }
                            }
                            for kw_i in 0..kw {
                                if z_on_w_mask[idx(kw_i, m_z, kw)] != 0 {
                                    rz -= beta_z[idx(1 + q_re + kw_i, m_z, p_z)]
                                        * w_in[idx(g, kw_i, g_cnt)];
                                }
                            }
                            p_tot += (bq * bq) / sz2;
                            info_tot += (bq * rz) / sz2;
                        }
                    }
                    let m_prior = (info_tot / p_tot).clamp(-10.0, 10.0);

                    let mut v_mode = m_prior.clamp(-8.0, 8.0);
                    for _ in 0..6 {
                        let exp_mv = (-v_mode).exp().min(1e12);
                        let g1 = -0.5 * n_cnt + 0.5 * exp_mv * rss_j - p_tot * (v_mode - m_prior);
                        let neg_g2 = (0.5 * exp_mv * rss_j + p_tot).max(1e-6);
                        let delta = (g1 / neg_g2).clamp(-2.0, 2.0);
                        v_mode = (v_mode + delta).clamp(-10.0, 10.0);
                        if delta.abs() < 1e-6 {
                            break;
                        }
                    }
                    let hess_mode = (0.5 * (-v_mode).exp().min(1e12) * rss_j + p_tot).max(1e-6);
                    let prop_sd = (1.25 / hess_mode.sqrt()).clamp(0.01, 2.5);

                    let logpost = |v: f64| -> f64 {
                        -0.5 * n_cnt * v
                            - 0.5 * (-v).exp().min(1e12) * rss_j
                            - 0.5 * p_tot * (v - m_prior) * (v - m_prior)
                    };
                    let logq = |v: f64| -> f64 {
                        let z = (v - v_mode) / prop_sd;
                        -0.5 * z * z
                    };

                    for _ in 0..2 {
                        let prop_lv = (v_mode + prop_sd * rng.rnorm()).clamp(-10.0, 10.0);
                        let log_acc =
                            (logpost(prop_lv) - logq(prop_lv)) - (logpost(cur_lv) - logq(cur_lv));
                        if rng.runif().ln() < log_acc {
                            cur_lv = prop_lv;
                        }
                    }
                    theta[idx(g, q_idx, g_cnt)] = cur_lv;
                }
            }

            // Step iii-b: Newton-Laplace Independence MH for random log residual covariance factor variance logvf_g (Ex 9.32b)
            if logvf_re_idx >= 0 {
                let q_idx = logvf_re_idx as usize;
                let mut cur_lv = theta[idx(g, q_idx, g_cnt)];
                let mut rss_f = 0.0;
                for r_loc in 0..n_eff_g {
                    let fv = f_shock[st + r_loc + max_lag];
                    rss_f += fv * fv;
                }
                let n_cnt = n_eff_g as f64;
                let mut p_tot = omega_inv[idx(q_idx, q_idx, q_re)].max(1e-8);
                let mut info_tot = p_tot * mu_prior_g[q_idx];
                for r in 0..q_re {
                    if r != q_idx {
                        info_tot -= omega_inv[idx(q_idx, r, q_re)]
                            * (theta[idx(g, r, g_cnt)] - mu_prior_g[r]);
                    }
                }
                for m_z in 0..kz {
                    let zv = z_in[idx(g, m_z, g_cnt)];
                    if zv.is_finite() && z_on_re_mask[idx(q_idx, m_z, q_re)] != 0 {
                        let sz2 = sigma2_z[m_z].max(1e-6);
                        let bq = beta_z[idx(1 + q_idx, m_z, p_z)];
                        let mut rz = zv - beta_z[idx(0, m_z, p_z)];
                        for r in 0..q_re {
                            if r != q_idx && z_on_re_mask[idx(r, m_z, q_re)] != 0 {
                                rz -= beta_z[idx(1 + r, m_z, p_z)] * theta[idx(g, r, g_cnt)];
                            }
                        }
                        for kw_i in 0..kw {
                            if z_on_w_mask[idx(kw_i, m_z, kw)] != 0 {
                                rz -= beta_z[idx(1 + q_re + kw_i, m_z, p_z)]
                                    * w_in[idx(g, kw_i, g_cnt)];
                            }
                        }
                        p_tot += (bq * bq) / sz2;
                        info_tot += (bq * rz) / sz2;
                    }
                }
                let m_prior = (info_tot / p_tot).clamp(-10.0, 10.0);
                let mut v_mode = m_prior.clamp(-8.0, 8.0);
                for _ in 0..6 {
                    let exp_mv = (-v_mode).exp().min(1e12);
                    let g1 = -0.5 * n_cnt + 0.5 * exp_mv * rss_f - p_tot * (v_mode - m_prior);
                    let neg_g2 = (0.5 * exp_mv * rss_f + p_tot).max(1e-6);
                    let delta = (g1 / neg_g2).clamp(-2.0, 2.0);
                    v_mode = (v_mode + delta).clamp(-10.0, 10.0);
                    if delta.abs() < 1e-6 {
                        break;
                    }
                }
                let hess_mode = (0.5 * (-v_mode).exp().min(1e12) * rss_f + p_tot).max(1e-6);
                let prop_sd = (1.25 / hess_mode.sqrt()).clamp(0.01, 2.5);
                let logpost = |v: f64| -> f64 {
                    -0.5 * n_cnt * v
                        - 0.5 * (-v).exp().min(1e12) * rss_f
                        - 0.5 * p_tot * (v - m_prior) * (v - m_prior)
                };
                let logq = |v: f64| -> f64 {
                    let z = (v - v_mode) / prop_sd;
                    -0.5 * z * z
                };
                for _ in 0..2 {
                    let prop_lv = (v_mode + prop_sd * rng.rnorm()).clamp(-10.0, 10.0);
                    let log_acc =
                        (logpost(prop_lv) - logq(prop_lv)) - (logpost(cur_lv) - logq(cur_lv));
                    if rng.runif().ln() < log_acc {
                        cur_lv = prop_lv;
                    }
                }
                theta[idx(g, q_idx, g_cnt)] = cur_lv;
            }
        }

        // 1c. Update ordinal extra thresholds (K >= 3) across all clusters & occasions
        for j in 0..d {
            if outcome_types[j] >= 3 {
                let k_cat = outcome_types[j] as usize;
                for c in 1..(k_cat - 1) {
                    let mut lo_b = extra_thresholds[j][c - 1] + 1e-4;
                    let mut hi_b = if c + 1 < k_cat - 1 {
                        extra_thresholds[j][c + 1] - 1e-4
                    } else {
                        10.0
                    };
                    for r in 0..n_obs {
                        let cell = idx(r, j, n_obs);
                        if !miss_y[cell] {
                            let obs = y_in[cell].round() as usize;
                            let yst = y_work[cell];
                            if obs == c {
                                lo_b = lo_b.max(yst);
                            } else if obs == c + 1 {
                                hi_b = hi_b.min(yst);
                            }
                        }
                    }
                    if hi_b > lo_b {
                        extra_thresholds[j][c] = lo_b + rng.runif() * (hi_b - lo_b);
                    }
                }
            }
        }

        // 2. Update fixed within-level slopes (if any)
        if n_fixed_slopes > 0 {
            let mut p_fs = vec![0.0; n_fixed_slopes * n_fixed_slopes];
            for fi in 0..n_fixed_slopes {
                p_fs[idx(fi, fi, n_fixed_slopes)] = if prior_mode == 2 {
                    1.0 / psem_tau2_fs[fi].max(1e-6)
                } else if prior_mode == 1 {
                    1e-10
                } else {
                    1.0 / (beta_sd * beta_sd)
                };
            }
            let mut b_fs = vec![0.0; n_fixed_slopes];
            let mut active_by_j: Vec<Vec<(usize, f64)>> = vec![Vec::new(); d];
            let mut y_dev_vec = vec![0.0; d];
            for g in 0..g_cnt {
                let st = offsets[g];
                let en = offsets[g + 1];
                let tg = en - st;
                if tg <= max_lag {
                    continue;
                }
                for t_loc in max_lag..tg {
                    for j in 0..d {
                        active_by_j[j].clear();
                        let mu_gj = theta[idx(g, j, g_cnt)];
                        let mut y_dev = y_work[idx(st + t_loc, j, n_obs)] - mu_gj;
                        if has_random_logvf {
                            y_dev -= cov_lam[j] * f_shock[st + t_loc];
                        }
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
                                active_by_j[j].push((fixed_slope_idx[flat] as usize, xv));
                            }
                        }
                        y_dev_vec[j] = y_dev;
                    }
                    if use_full_sigma_w {
                        for j in 0..d {
                            for k in 0..d {
                                let w_jk = sigma_w_inv[idx(j, k, d)];
                                if w_jk.abs() < 1e-14 {
                                    continue;
                                }
                                let ydk = y_dev_vec[k];
                                for &(fi, x_f) in &active_by_j[j] {
                                    b_fs[fi] += w_jk * x_f * ydk;
                                    for &(fj, x_fj) in &active_by_j[k] {
                                        p_fs[idx(fi, fj, n_fixed_slopes)] += w_jk * x_f * x_fj;
                                    }
                                }
                            }
                        }
                    } else {
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
                            let ydj = y_dev_vec[j];
                            for &(fi, x_f) in &active_by_j[j] {
                                b_fs[fi] += x_f * ydj / s2;
                                for &(fj, x_fj) in &active_by_j[j] {
                                    p_fs[idx(fi, fj, n_fixed_slopes)] += x_f * x_fj / s2;
                                }
                            }
                        }
                    }
                }
            }
            let fs_prop = rmvnorm_precision(&mut rng, &p_fs, &b_fs, n_fixed_slopes);
            // Stationary initial-state MH check when fixed AR(1) is sole slope in short-T univariate AR(1)
            if d == 1
                && n_lags == 1
                && n_fixed_slopes == 1
                && fixed_slope_idx[0] == 0
                && ((latent_center_y && !residual_dynamics && kx == 0) || residual_dynamics)
            {
                let phi_c = fixed_slopes[0];
                let phi_p = fs_prop[0];
                if phi_p.abs() < 0.995 {
                    if phi_c.abs() >= 0.995 {
                        fixed_slopes[0] = phi_p;
                    } else {
                        let mut log_acc = 0.0;
                        for g in 0..g_cnt {
                            let st = offsets[g];
                            let s2 = if logv_re_idx[0] >= 0 {
                                theta[idx(g, logv_re_idx[0] as usize, g_cnt)]
                                    .clamp(-12.0, 12.0)
                                    .exp()
                            } else {
                                fixed_sigma2[0].max(1e-6)
                            };
                            let mut e0 = y_work[idx(st, 0, n_obs)] - theta[idx(g, 0, g_cnt)];
                            if residual_dynamics {
                                for k_x in 0..kx {
                                    let flat_k = n_lags * d + k_x;
                                    let bk = if slope_re_idx[flat_k] >= 0 {
                                        theta[idx(g, slope_re_idx[flat_k] as usize, g_cnt)]
                                    } else {
                                        fixed_slopes[fixed_slope_idx[flat_k] as usize]
                                    };
                                    let x0 = if latent_center_x[k_x] != 0 {
                                        x_in[idx(st, k_x, n_obs)] - x_between[idx(g, k_x, g_cnt)]
                                    } else {
                                        x_in[idx(st, k_x, n_obs)]
                                    };
                                    e0 -= bk * x0;
                                }
                            }
                            log_acc += 0.5 * ((1.0 - phi_p * phi_p) / (1.0 - phi_c * phi_c)).ln()
                                + 0.5 * (phi_p * phi_p - phi_c * phi_c) * e0 * e0 / s2;
                        }
                        if log_acc >= 0.0 || rng.runif().ln() < log_acc {
                            fixed_slopes[0] = phi_p;
                        }
                    }
                }
            } else {
                fixed_slopes = fs_prop;
            }
        }

        // 3. Update fixed within-level residual variances / unstructured covariance Sigma_W
        if use_full_sigma_w {
            let mut s_mat = eye(d, if prior_mode == 1 { 1e-6 } else { 0.002 });
            let mut cnt = 0usize;
            let mut e_vec = vec![0.0; d];
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
                for t_loc in max_lag..tg {
                    for j in 0..d {
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
                            theta[idx(g, j, g_cnt)],
                            &theta,
                            &slopes_g,
                            n_slopes_per_out,
                            latent_center_y,
                            residual_dynamics,
                            n_obs,
                        );
                        e_vec[j] = y_work[idx(st + t_loc, j, n_obs)] - pred;
                    }
                    for c in 0..d {
                        for r in 0..d {
                            s_mat[idx(r, c, d)] += e_vec[r] * e_vec[c];
                        }
                    }
                    cnt += 1;
                }
            }
            let df_post = if prior_mode == 1 {
                (cnt as f64 - d as f64 - 1.0).max(d as f64 + 1.0)
            } else {
                (d as f64 + 2.0) + cnt as f64
            };
            sigma_w = rinv_wishart(&mut rng, df_post, &s_mat, d);
            symmetrize(&mut sigma_w, d);
            for j in 0..d {
                fixed_sigma2[j] = sigma_w[idx(j, j, d)].max(1e-6);
            }
        } else {
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
                        if d == 1
                            && n_lags == 1
                            && ((latent_center_y && !residual_dynamics && kx == 0)
                                || residual_dynamics)
                        {
                            let phi = slopes_g[0].clamp(-0.98, 0.98);
                            let mut e0 = y_work[idx(st, 0, n_obs)] - mu_gj;
                            if residual_dynamics {
                                for k_x in 0..kx {
                                    let x0 = if latent_center_x[k_x] != 0 {
                                        x_in[idx(st, k_x, n_obs)] - x_between[idx(g, k_x, g_cnt)]
                                    } else {
                                        x_in[idx(st, k_x, n_obs)]
                                    };
                                    e0 -= slopes_g[idx(n_lags * d + k_x, 0, n_slopes_per_out)] * x0;
                                }
                            }
                            rss += (1.0 - phi * phi).max(0.02) * e0 * e0;
                            cnt += 1;
                        }
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
                            ) + if has_random_logvf {
                                cov_lam[j] * f_shock[st + t_loc]
                            } else {
                                0.0
                            };
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
                let p_pr_base = if prior_mode == 1 {
                    1e-10
                } else {
                    1.0 / (beta_sd * beta_sd)
                };
                for (ci, &col_id) in active_cols.iter().enumerate() {
                    let p_i = if prior_mode == 2 && col_id > 0 {
                        1.0 / psem_tau2_gb[idx(col_id, q, p_w)].max(1e-6)
                    } else {
                        p_pr_base
                    };
                    prec[idx(ci, ci, pa)] += p_i;
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
            let psi0_diag = if prior_mode == 1 {
                1e-5
            } else {
                omega_scale_diag.max(1e-4)
            };
            let mut s_mat = eye(q_re, psi0_diag);
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
            let nu0_iw = if prior_mode == 1 {
                -(q_re as f64 + 1.0)
            } else {
                q_re as f64 + omega_df_add.max(1.0)
            };
            let df_post = (g_cnt as f64 + nu0_iw).max(q_re as f64 + 1.0);
            omega_b = rinv_wishart(&mut rng, df_post, &s_mat, q_re);

            // 4b. Exact Parameter-Expanded (PX) Scale Group Move on random log-variances (logv_j & logvf)
            // Jointly rescales (zeta_{g,q} -> a * zeta_{g,q}, Omega_B -> A * Omega_B * A) to break centered Gibbs funnels!
            if n_rand_logv > 0 || logvf_re_idx >= 0 {
                let mut om_inv_px = spd_invert(&omega_b, q_re);
                let mut px_targets: Vec<(usize, Option<usize>)> = Vec::new();
                for (j, &lv_idx) in logv_re_idx.iter().enumerate().take(d) {
                    if lv_idx >= 0 {
                        px_targets.push((lv_idx as usize, Some(j)));
                    }
                }
                if logvf_re_idx >= 0 {
                    px_targets.push((logvf_re_idx as usize, None));
                }
                for (q_lv, j_opt) in px_targets {
                    // Precompute (m_prior_g, zeta_gq, n_cnt_g, rss_g) for cluster g = 0..g_cnt
                    let mut m_vec = vec![0.0; g_cnt];
                    let mut z_vec = vec![0.0; g_cnt];
                    let mut n_vec = vec![0.0; g_cnt];
                    let mut rss_vec = vec![0.0; g_cnt];
                    for g in 0..g_cnt {
                        let st = offsets[g];
                        let en = offsets[g + 1];
                        let tg = en - st;
                        let mut mq = gamma_b[idx(0, q_lv, p_w)];
                        for k in 0..kw {
                            if re_on_w_mask[idx(k, q_lv, kw)] != 0 {
                                mq += gamma_b[idx(1 + k, q_lv, p_w)] * w_in[idx(g, k, g_cnt)];
                            }
                        }
                        m_vec[g] = mq;
                        z_vec[g] = theta[idx(g, q_lv, g_cnt)] - mq;
                        if tg <= max_lag {
                            continue;
                        }
                        let n_eff_g = tg - max_lag;
                        if let Some(j) = j_opt {
                            let mut slopes_g = vec![0.0; n_slopes_per_out * d];
                            for jj in 0..d {
                                for s in 0..n_slopes_per_out {
                                    let fl = jj * n_slopes_per_out + s;
                                    slopes_g[idx(s, jj, n_slopes_per_out)] =
                                        if slope_re_idx[fl] >= 0 {
                                            theta[idx(g, slope_re_idx[fl] as usize, g_cnt)]
                                        } else {
                                            fixed_slopes[fixed_slope_idx[fl] as usize]
                                        };
                                }
                            }
                            let mu_gj = theta[idx(g, j, g_cnt)];
                            let mut rss_j = 0.0;
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
                                ) + if has_random_logvf {
                                    cov_lam[j] * f_shock[st + t_loc]
                                } else {
                                    0.0
                                };
                                let err = y_work[idx(st + t_loc, j, n_obs)] - pred;
                                rss_j += err * err;
                            }
                            let mut n_cnt = n_eff_g as f64;
                            if d == 1
                                && n_lags == 1
                                && ((latent_center_y && !residual_dynamics && kx == 0)
                                    || residual_dynamics)
                            {
                                let phi = slopes_g[0].clamp(-0.98, 0.98);
                                let mut e0 = y_work[idx(st, 0, n_obs)] - mu_gj;
                                if residual_dynamics {
                                    for k_x in 0..kx {
                                        let x0 = if latent_center_x[k_x] != 0 {
                                            x_in[idx(st, k_x, n_obs)]
                                                - x_between[idx(g, k_x, g_cnt)]
                                        } else {
                                            x_in[idx(st, k_x, n_obs)]
                                        };
                                        e0 -= slopes_g[idx(n_lags * d + k_x, 0, n_slopes_per_out)]
                                            * x0;
                                    }
                                }
                                rss_j += (1.0 - phi * phi).max(0.02) * e0 * e0;
                                n_cnt += 1.0;
                            }
                            n_vec[g] = n_cnt;
                            rss_vec[g] = rss_j;
                        } else {
                            let mut rss_f = 0.0;
                            for r_loc in 0..n_eff_g {
                                let fv = f_shock[st + r_loc + max_lag];
                                rss_f += fv * fv;
                            }
                            n_vec[g] = n_eff_g as f64;
                            rss_vec[g] = rss_f;
                        }
                    }

                    // 3 fast O(G) PX scale proposals
                    for _ in 0..3 {
                        let delta = 0.18 * rng.rnorm();
                        let a = delta.exp();
                        let inv_a2 = (-2.0 * delta).exp();
                        // Check bounds on proposed theta_{g,q}
                        let mut valid = true;
                        let mut log_lik_diff = 0.0;
                        for g in 0..g_cnt {
                            let cur_v = m_vec[g] + z_vec[g];
                            let prop_v = m_vec[g] + a * z_vec[g];
                            if prop_v.abs() > 10.0 {
                                valid = false;
                                break;
                            }
                            if n_vec[g] > 0.0 {
                                log_lik_diff += -0.5 * n_vec[g] * (prop_v - cur_v)
                                    - 0.5
                                        * ((-prop_v).exp().min(1e12) - (-cur_v).exp().min(1e12))
                                        * rss_vec[g];
                            }
                            // Distal z likelihood contribution when z depends on logv
                            for m_z in 0..kz {
                                let zv = z_in[idx(g, m_z, g_cnt)];
                                if zv.is_finite() && z_on_re_mask[idx(q_lv, m_z, q_re)] != 0 {
                                    let sz2 = sigma2_z[m_z].max(1e-6);
                                    let bq = beta_z[idx(1 + q_lv, m_z, p_z)];
                                    let mut r_cur = zv - beta_z[idx(0, m_z, p_z)];
                                    for r in 0..q_re {
                                        if z_on_re_mask[idx(r, m_z, q_re)] != 0 {
                                            r_cur -= beta_z[idx(1 + r, m_z, p_z)]
                                                * theta[idx(g, r, g_cnt)];
                                        }
                                    }
                                    for kw_i in 0..kw {
                                        if z_on_w_mask[idx(kw_i, m_z, kw)] != 0 {
                                            r_cur -= beta_z[idx(1 + q_re + kw_i, m_z, p_z)]
                                                * w_in[idx(g, kw_i, g_cnt)];
                                        }
                                    }
                                    let r_prop = r_cur - bq * (prop_v - cur_v);
                                    log_lik_diff += -0.5 * (r_prop * r_prop - r_cur * r_cur) / sz2;
                                }
                            }
                        }
                        if !valid {
                            continue;
                        }
                        // Exact IW prior + Jacobian term: -nu0 * delta - 0.5 * psi0_diag * (a^{-2} - 1) * Omega_inv[q,q]
                        let om_inv_qq = om_inv_px[idx(q_lv, q_lv, q_re)];
                        let log_prior_jac =
                            -nu0_iw * delta - 0.5 * psi0_diag * (inv_a2 - 1.0) * om_inv_qq;
                        let log_acc = log_lik_diff + log_prior_jac;
                        if log_acc.is_finite() && (log_acc >= 0.0 || rng.runif().ln() < log_acc) {
                            for g in 0..g_cnt {
                                z_vec[g] *= a;
                                theta[idx(g, q_lv, g_cnt)] = m_vec[g] + z_vec[g];
                            }
                            let inv_a = 1.0 / a;
                            for r in 0..q_re {
                                omega_b[idx(r, q_lv, q_re)] *= a;
                                omega_b[idx(q_lv, r, q_re)] *= a;
                                om_inv_px[idx(r, q_lv, q_re)] *= inv_a;
                                om_inv_px[idx(q_lv, r, q_re)] *= inv_a;
                            }
                        }
                    }
                }
            }
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
                let p_pr_base = if prior_mode == 1 {
                    1e-10
                } else {
                    1.0 / (beta_sd * beta_sd)
                };
                for (ci, &col_id) in active.iter().enumerate() {
                    let p_i = if prior_mode == 2 && col_id > 0 {
                        1.0 / psem_tau2_bz[idx(col_id, m_z, p_z)].max(1e-6)
                    } else {
                        p_pr_base
                    };
                    prec[idx(ci, ci, pa)] += p_i;
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

        // 5b. Update PSEM Bayesian Lasso scale-mixture shrinkage states (prior_mode == 2)
        if prior_mode == 2 {
            let mut sum_tau2 = 0.0;
            let mut n_pen = 0usize;
            for fi in 0..n_fixed_slopes {
                let b_abs = fixed_slopes[fi].abs().max(1e-5);
                let inv_t2 = rng.rinv_gaussian(psem_lambda2.sqrt() / b_abs, psem_lambda2);
                psem_tau2_fs[fi] = (1.0 / inv_t2).clamp(1e-5, 1e4);
                sum_tau2 += psem_tau2_fs[fi];
                n_pen += 1;
            }
            for q in 0..q_re {
                for k in 0..kw {
                    if re_on_w_mask[idx(k, q, kw)] != 0 {
                        let slot = idx(1 + k, q, p_w);
                        let b_abs = gamma_b[slot].abs().max(1e-5);
                        let inv_t2 = rng.rinv_gaussian(psem_lambda2.sqrt() / b_abs, psem_lambda2);
                        psem_tau2_gb[slot] = (1.0 / inv_t2).clamp(1e-5, 1e4);
                        sum_tau2 += psem_tau2_gb[slot];
                        n_pen += 1;
                    }
                }
            }
            for m_z in 0..kz {
                for q in 0..q_re {
                    if z_on_re_mask[idx(q, m_z, q_re)] != 0 {
                        let slot = idx(1 + q, m_z, p_z);
                        let b_abs = beta_z[slot].abs().max(1e-5);
                        let inv_t2 = rng.rinv_gaussian(psem_lambda2.sqrt() / b_abs, psem_lambda2);
                        psem_tau2_bz[slot] = (1.0 / inv_t2).clamp(1e-5, 1e4);
                        sum_tau2 += psem_tau2_bz[slot];
                        n_pen += 1;
                    }
                }
                for k in 0..kw {
                    if z_on_w_mask[idx(k, m_z, kw)] != 0 {
                        let slot = idx(1 + q_re + k, m_z, p_z);
                        let b_abs = beta_z[slot].abs().max(1e-5);
                        let inv_t2 = rng.rinv_gaussian(psem_lambda2.sqrt() / b_abs, psem_lambda2);
                        psem_tau2_bz[slot] = (1.0 / inv_t2).clamp(1e-5, 1e4);
                        sum_tau2 += psem_tau2_bz[slot];
                        n_pen += 1;
                    }
                }
            }
            if n_pen > 0 {
                psem_lambda2 = rng
                    .rgamma(1.0 + n_pen as f64, 1.0 + 0.5 * sum_tau2)
                    .clamp(1e-3, 1e3);
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
            if use_full_sigma_w {
                for c in 0..d {
                    for r in c..d {
                        draws[idx(keep_pos, cur, n_keep)] = sigma_w[idx(r, c, d)];
                        cur += 1;
                    }
                }
                for i in 0..(d * d) {
                    sigma_w_accum[i] += sigma_w[i];
                }
            } else {
                for j in 0..d {
                    if logv_re_idx[j] < 0 && !is_cat[j] {
                        draws[idx(keep_pos, cur, n_keep)] = fixed_sigma2[j];
                        cur += 1;
                    }
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
            for j in 0..d {
                if outcome_types[j] >= 3 {
                    let taus = &extra_thresholds[j];
                    for &tau_c in taus.iter().skip(1) {
                        draws[idx(keep_pos, cur, n_keep)] = tau_c - gamma_b[idx(0, j, p_w)];
                        cur += 1;
                    }
                }
            }

            // Store cluster random effects
            for q in 0..q_re {
                for g in 0..g_cnt {
                    re_draws[keep_pos + n_keep * (g + g_cnt * q)] = theta[idx(g, q, g_cnt)];
                }
            }

            // Accumulate conditional log-likelihood & fitted values
            let mut ll_iter = 0.0;
            let (sw_logdet, sw_inv_cur) = if use_full_sigma_w {
                let l_sw = chol_lower_jitter(&sigma_w, d);
                (spd_logdet(&l_sw, d), spd_invert(&sigma_w, d))
            } else {
                (0.0, Vec::new())
            };
            let mut err_row = vec![0.0; d];
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
                let mut s2_cur_g = vec![1.0; d];
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
                    s2_cur_g[j] = s2;
                    sig_bar_accum[idx(g, j, g_cnt)] += s2;
                }
                let vf_cur_g = if has_random_logvf {
                    let vf = theta[idx(g, logvf_re_idx as usize, g_cnt)]
                        .clamp(-12.0, 12.0)
                        .exp();
                    sig_vf_accum[g] += vf;
                    vf
                } else {
                    0.0
                };
                // Rank-1 Woodbury quantities for Sigma_{Wg} = diag(s2_cur_g) + vf_cur_g * lam * lam^T
                let (wb_logdet, wb_denom) = if has_random_logvf {
                    let mut sum_l2_d = 0.0;
                    let mut logdet_d = 0.0;
                    for j in 0..d {
                        sum_l2_d += (cov_lam[j] * cov_lam[j]) / s2_cur_g[j];
                        logdet_d += s2_cur_g[j].ln();
                    }
                    let denom = (1.0 + vf_cur_g * sum_l2_d).max(1e-12);
                    (logdet_d + denom.ln(), denom)
                } else {
                    (0.0, 1.0)
                };
                for t_loc in 0..tg {
                    let mut all_obs = true;
                    for j in 0..d {
                        let mu_gj = theta[idx(g, j, g_cnt)];
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
                        let cell = idx(st + t_loc, j, n_obs);
                        fitted_accum[cell] += fv;
                        pred_bar_accum[cell] += fv;
                        if t_loc >= max_lag {
                            if miss_y[cell] {
                                all_obs = false;
                            } else if !use_full_sigma_w && !has_random_logvf && !is_cat[j] {
                                let s2 = s2_cur_g[j];
                                let sd_j = s2.sqrt();
                                match cens_flag[cell] {
                                    -1 => {
                                        let lo_j = censor_lo.get(j).copied().unwrap_or(0.0);
                                        ll_iter += log_std_norm_cdf((lo_j - fv) / sd_j);
                                    }
                                    1 => {
                                        let hi_j = censor_hi.get(j).copied().unwrap_or(0.0);
                                        ll_iter += log_std_norm_cdf((fv - hi_j) / sd_j);
                                    }
                                    _ => {
                                        let err = y_in[cell] - fv;
                                        ll_iter += -0.5 * ((2.0 * PI * s2).ln() + err * err / s2);
                                    }
                                }
                            } else if use_full_sigma_w || has_random_logvf {
                                err_row[j] = y_work[cell] - fv;
                            }
                        }
                    }
                    if use_full_sigma_w && t_loc >= max_lag && all_obs {
                        let qf = quad_form(&err_row, &sw_inv_cur, d);
                        ll_iter += -0.5 * ((d as f64) * (2.0 * PI).ln() + sw_logdet + qf);
                    } else if has_random_logvf && t_loc >= max_lag && all_obs {
                        let mut q_diag = 0.0;
                        let mut l_d_e = 0.0;
                        for j in 0..d {
                            q_diag += (err_row[j] * err_row[j]) / s2_cur_g[j];
                            l_d_e += cov_lam[j] * err_row[j] / s2_cur_g[j];
                        }
                        let qf = q_diag - (vf_cur_g / wb_denom) * l_d_e * l_d_e;
                        ll_iter += -0.5 * ((d as f64) * (2.0 * PI).ln() + wb_logdet + qf);
                    }
                }
            }
            deviance_sum += -2.0 * ll_iter;
            keep_pos += 1;
        }
    }

    let inv_keep = 1.0 / (n_keep.max(1) as f64);
    let fitted_mean: Vec<f64> = fitted_accum.into_iter().map(|v| v * inv_keep).collect();
    let has_continuous = is_cat.iter().any(|&b| !b);
    let (dic, pd) = if use_full_sigma_w {
        let d_bar = deviance_sum * inv_keep;
        let sw_bar: Vec<f64> = sigma_w_accum.into_iter().map(|v| v * inv_keep).collect();
        let l_sw = chol_lower_jitter(&sw_bar, d);
        let sw_logdet = spd_logdet(&l_sw, d);
        let sw_inv_bar = spd_invert(&sw_bar, d);
        let mut ll_hat = 0.0;
        let mut err_row = vec![0.0; d];
        for g in 0..g_cnt {
            let st = offsets[g];
            let en = offsets[g + 1];
            let tg = en - st;
            for t_loc in max_lag..tg {
                let mut all_obs = true;
                for (j, err_j) in err_row.iter_mut().enumerate().take(d) {
                    let cell = idx(st + t_loc, j, n_obs);
                    if miss_y[cell] {
                        all_obs = false;
                        break;
                    }
                    *err_j = y_in[cell] - pred_bar_accum[cell] * inv_keep;
                }
                if all_obs {
                    let qf = quad_form(&err_row, &sw_inv_bar, d);
                    ll_hat += -0.5 * ((d as f64) * (2.0 * PI).ln() + sw_logdet + qf);
                }
            }
        }
        let d_hat = -2.0 * ll_hat;
        let pd_v = (d_bar - d_hat).max(0.0);
        (d_bar + pd_v, pd_v)
    } else if has_random_logvf {
        let d_bar = deviance_sum * inv_keep;
        let mut ll_hat = 0.0;
        let mut err_row = vec![0.0; d];
        for g in 0..g_cnt {
            let st = offsets[g];
            let en = offsets[g + 1];
            let tg = en - st;
            let s2_bar: Vec<f64> = (0..d)
                .map(|j| (sig_bar_accum[idx(g, j, g_cnt)] * inv_keep).max(1e-6))
                .collect();
            let vf_bar = (sig_vf_accum[g] * inv_keep).max(1e-8);
            let mut sum_l2_d = 0.0;
            let mut logdet_d = 0.0;
            for j in 0..d {
                sum_l2_d += (cov_lam[j] * cov_lam[j]) / s2_bar[j];
                logdet_d += s2_bar[j].ln();
            }
            let denom = (1.0 + vf_bar * sum_l2_d).max(1e-12);
            let wb_logdet = logdet_d + denom.ln();
            for t_loc in max_lag..tg {
                let mut all_obs = true;
                for (j, err_j) in err_row.iter_mut().enumerate().take(d) {
                    let cell = idx(st + t_loc, j, n_obs);
                    if miss_y[cell] {
                        all_obs = false;
                        break;
                    }
                    *err_j = y_in[cell] - pred_bar_accum[cell] * inv_keep;
                }
                if all_obs {
                    let mut q_diag = 0.0;
                    let mut l_d_e = 0.0;
                    for j in 0..d {
                        q_diag += (err_row[j] * err_row[j]) / s2_bar[j];
                        l_d_e += cov_lam[j] * err_row[j] / s2_bar[j];
                    }
                    let qf = q_diag - (vf_bar / denom) * l_d_e * l_d_e;
                    ll_hat += -0.5 * ((d as f64) * (2.0 * PI).ln() + wb_logdet + qf);
                }
            }
        }
        let d_hat = -2.0 * ll_hat;
        let pd_v = (d_bar - d_hat).max(0.0);
        (d_bar + pd_v, pd_v)
    } else if has_continuous {
        let d_bar = deviance_sum * inv_keep;
        let mut ll_hat = 0.0;
        for g in 0..g_cnt {
            let st = offsets[g];
            let en = offsets[g + 1];
            let tg = en - st;
            for j in 0..d {
                if !is_cat[j] {
                    let s2 = (sig_bar_accum[idx(g, j, g_cnt)] * inv_keep).max(1e-6);
                    let sd_j = s2.sqrt();
                    for t_loc in max_lag..tg {
                        let cell = idx(st + t_loc, j, n_obs);
                        if !miss_y[cell] {
                            let fv = pred_bar_accum[cell] * inv_keep;
                            match cens_flag[cell] {
                                -1 => {
                                    let lo_j = censor_lo.get(j).copied().unwrap_or(0.0);
                                    ll_hat += log_std_norm_cdf((lo_j - fv) / sd_j);
                                }
                                1 => {
                                    let hi_j = censor_hi.get(j).copied().unwrap_or(0.0);
                                    ll_hat += log_std_norm_cdf((fv - hi_j) / sd_j);
                                }
                                _ => {
                                    let err = y_in[cell] - fv;
                                    ll_hat += -0.5 * ((2.0 * PI * s2).ln() + err * err / s2);
                                }
                            }
                        }
                    }
                }
            }
        }
        let d_hat = -2.0 * ll_hat;
        let pd_v = (d_bar - d_hat).max(0.0);
        (d_bar + pd_v, pd_v)
    } else {
        (f64::NAN, f64::NAN)
    };
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
