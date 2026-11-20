use crate::linalg::{eye, idx, mat_t_mul, rinv_wishart, rmvnorm_precision, spd_invert};
use crate::rng::Rng;
use std::f64::consts::PI;

pub struct CrossClassifiedOutput {
    pub draws: Vec<f64>,
    pub param_count: usize,
    pub id_draws: Vec<f64>,   // n_keep x n_id x q_id
    pub time_draws: Vec<f64>, // n_keep x n_time x q_time
    pub q_id: usize,
    pub q_time: usize,
    pub fitted_mean: Vec<f64>,
    pub dic: f64,
    pub pd: f64,
}

/// Cross-Classified DSEM (Subject x Time two-way random intercepts, selective random slopes
/// across subjects and/or occasions, selective subject- and time-level predictor regressions,
/// subject random logv via Newton-Laplace Independence MH + PX scale move, and PSEM shrinkage).
#[allow(clippy::too_many_arguments)]
pub fn mcmc_crossclassified(
    y_grid: &[f64],          // (n_id * n_time) x d, ordered by id g=0..n_id, t=0..n_time
    x_grid: &[f64],          // (n_id * n_time) x kx
    w_id: &[f64],            // n_id x kw_id
    w_time: &[f64],          // n_time x kw_time
    slope_rand_id: &[i32],   // d * n_sl
    slope_rand_time: &[i32], // d * n_sl
    has_random_logv: &[i32], // d
    re_id_on_w_mask: &[i32], // kw_id x q_id
    re_time_on_w_mask: &[i32], // kw_time x q_time
    n_id: usize,
    n_time: usize,
    d: usize,
    kx: usize,
    kw_id: usize,
    kw_time: usize,
    lags: &[usize],
    prior_mode: i32,
    beta_sd: f64,
    iter: usize,
    warmup: usize,
    thin: usize,
    seed: u64,
) -> CrossClassifiedOutput {
    let mut rng = Rng::from_seed(seed);
    let n_obs = n_id * n_time;
    let max_lag = lags.iter().copied().max().unwrap_or(1).max(1);
    let n_lags = lags.len();
    let n_sl = n_lags * d + kx;
    let n_sl_tot = d * n_sl;

    // Map per-slope random indices for subject (id), occasion (time), and fixed slopes
    let mut sl_id_idx = vec![-1i32; n_sl_tot];
    let mut sl_time_idx = vec![-1i32; n_sl_tot];
    let mut sl_fixed_idx = vec![-1i32; n_sl_tot];
    let mut n_rs_id = 0usize;
    let mut n_rs_time = 0usize;
    let mut n_fixed_sl = 0usize;
    for flat in 0..n_sl_tot {
        let r_id = slope_rand_id.get(flat).copied().unwrap_or(0) != 0;
        let r_tm = slope_rand_time.get(flat).copied().unwrap_or(0) != 0;
        if r_id {
            sl_id_idx[flat] = (d + n_rs_id) as i32;
            n_rs_id += 1;
        }
        if r_tm {
            sl_time_idx[flat] = (d + n_rs_time) as i32;
            n_rs_time += 1;
        }
        if !r_id {
            // Slopes without a subject random mean hold their grand mean in fixed_slopes
            sl_fixed_idx[flat] = n_fixed_sl as i32;
            n_fixed_sl += 1;
        }
    }

    let mut lv_id_idx = vec![-1i32; d];
    let mut n_lv = 0usize;
    let mut n_fixed_sig = 0usize;
    for (j, lv_slot) in lv_id_idx.iter_mut().enumerate().take(d) {
        if has_random_logv.get(j).copied().unwrap_or(0) != 0 {
            *lv_slot = (d + n_rs_id + n_lv) as i32;
            n_lv += 1;
        } else {
            n_fixed_sig += 1;
        }
    }

    let q_id = d + n_rs_id + n_lv;
    let q_time = d + n_rs_time;
    let q_lin_id = d + n_rs_id;

    let mut y_work = y_grid.to_vec();
    let mut miss_y = vec![false; n_obs * d];
    for j in 0..d {
        let mut s = 0.0;
        let mut c = 0usize;
        for r in 0..n_obs {
            let v = y_grid[idx(r, j, n_obs)];
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
                y_work[idx(r, j, n_obs)] = m;
            }
        }
    }

    let mut gamma0 = vec![0.0; q_id];
    let mut fixed_slopes = vec![0.0; n_fixed_sl];
    for j in 0..d {
        gamma0[j] = (0..n_obs).map(|r| y_work[idx(r, j, n_obs)]).sum::<f64>() / n_obs as f64;
        if n_lags >= 1 {
            let self_flat = j * n_sl + j;
            if sl_id_idx[self_flat] >= 0 {
                gamma0[sl_id_idx[self_flat] as usize] = 0.25;
            } else if sl_fixed_idx[self_flat] >= 0 {
                fixed_slopes[sl_fixed_idx[self_flat] as usize] = 0.25;
            }
        }
    }
    let mut gamma_w_id = vec![0.0; kw_id * q_id];
    let mut gamma_w_time = vec![0.0; kw_time * q_time];
    let mut u_id = vec![0.0; n_id * q_id]; // centered subject deviations
    let mut v_time = vec![0.0; n_time * q_time]; // centered occasion deviations
    let mut fixed_sigma2 = vec![1.0_f64; d];
    let mut omega_id = eye(q_id, 0.1);
    let mut omega_time = eye(q_time, 0.1);

    // PSEM Bayesian Lasso scale-mixture states (prior_mode == 2)
    let mut psem_tau2_fsl = vec![1.0_f64; n_fixed_sl];
    let mut psem_tau2_wid = vec![1.0_f64; kw_id * q_id];
    let mut psem_tau2_wt = vec![1.0_f64; kw_time * q_time];
    let mut psem_lambda2 = 1.0_f64;

    let n_wid_active: usize = (0..q_id)
        .flat_map(|q| (0..kw_id).map(move |k| (k, q)))
        .filter(|&(k, q)| re_id_on_w_mask.get(idx(k, q, kw_id)).copied().unwrap_or(1) != 0)
        .count();
    let n_wt_active: usize = (0..q_time)
        .flat_map(|q| (0..kw_time).map(move |k| (k, q)))
        .filter(|&(k, q)| re_time_on_w_mask.get(idx(k, q, kw_time)).copied().unwrap_or(1) != 0)
        .count();

    let n_om_id = q_id * (q_id + 1) / 2;
    let n_om_t = q_time * (q_time + 1) / 2;
    let n_params =
        q_id + n_wid_active + n_wt_active + n_fixed_sl + n_om_id + n_om_t + n_fixed_sig;

    let keep_indices: Vec<usize> = ((warmup + 1)..=iter)
        .filter(|i| (i - warmup - 1) % thin == 0)
        .collect();
    let n_keep = keep_indices.len();
    let mut draws = vec![0.0; n_keep * n_params];
    let mut id_draws = vec![0.0; n_keep * n_id * q_id];
    let mut time_draws = vec![0.0; n_keep * n_time * q_time];
    let mut fitted_accum = vec![0.0; n_obs * d];
    let mut sig_accum = vec![0.0; n_id * d];
    let mut deviance_sum = 0.0;
    let mut keep_pos = 0usize;

    // Precompute index list of time-random slopes
    let time_sl_flats: Vec<(usize, usize)> = (0..n_sl_tot)
        .filter_map(|flat| {
            if sl_time_idx[flat] >= 0 {
                Some((flat, sl_time_idx[flat] as usize))
            } else {
                None
            }
        })
        .collect();
    let id_sl_flats: Vec<(usize, usize)> = (0..n_sl_tot)
        .filter_map(|flat| {
            if sl_id_idx[flat] >= 0 {
                Some((flat, sl_id_idx[flat] as usize))
            } else {
                None
            }
        })
        .collect();

    for step in 1..=iter {
        let om_id_inv = spd_invert(&omega_id, q_id);
        let om_t_inv = spd_invert(&omega_time, q_time);

        let build_th_id = |g: usize, g0: &[f64], gw: &[f64], uid: &[f64]| -> Vec<f64> {
            let mut th = g0.to_vec();
            for q in 0..q_id {
                for k in 0..kw_id {
                    if re_id_on_w_mask.get(idx(k, q, kw_id)).copied().unwrap_or(1) != 0 {
                        th[q] += gw[idx(k, q, kw_id)] * w_id[idx(g, k, n_id)];
                    }
                }
                th[q] += uid[idx(g, q, n_id)];
            }
            th
        };

        let build_vt_tot = |t: usize, gwt: &[f64], vt: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; q_time];
            for q in 0..q_time {
                let mut v = vt[idx(t, q, n_time)];
                for k in 0..kw_time {
                    if re_time_on_w_mask.get(idx(k, q, kw_time)).copied().unwrap_or(1) != 0 {
                        v += gwt[idx(k, q, kw_time)] * w_time[idx(t, k, n_time)];
                    }
                }
                out[q] = v;
            }
            out
        };

        // Precompute full occasion effects v_tot[t, 0..q_time] for fast lookup during subject updates
        let mut vt_tot = vec![0.0; n_time * q_time];
        for t in 0..n_time {
            let v_t = build_vt_tot(t, &gamma_w_time, &v_time);
            for q in 0..q_time {
                vt_tot[idx(t, q, n_time)] = v_t[q];
            }
        }

        let cell_slope =
            |flat: usize, th_g: &[f64], t: usize, vt_all: &[f64], f_sl: &[f64]| -> f64 {
                let mut s = if sl_id_idx[flat] >= 0 {
                    th_g[sl_id_idx[flat] as usize]
                } else {
                    f_sl[sl_fixed_idx[flat] as usize]
                };
                if sl_time_idx[flat] >= 0 {
                    s += vt_all[idx(t, sl_time_idx[flat] as usize, n_time)];
                }
                s
            };

        let cell_mean = |y_w: &[f64],
                         st: usize,
                         t: usize,
                         j: usize,
                         th_g: &[f64],
                         vt_all: &[f64],
                         f_sl: &[f64]|
         -> f64 {
            let mut pr = th_g[j] + vt_all[idx(t, j, n_time)];
            for k in 0..kx {
                let flat = j * n_sl + n_lags * d + k;
                pr += cell_slope(flat, th_g, t, vt_all, f_sl) * x_grid[idx(st + t, k, n_obs)];
            }
            if t < max_lag {
                return pr;
            }
            for (li, &lv) in lags.iter().enumerate() {
                for mc in 0..d {
                    let flat = j * n_sl + li * d + mc;
                    let phi = cell_slope(flat, th_g, t, vt_all, f_sl);
                    let dev = y_w[idx(st + t - lv, mc, n_obs)]
                        - th_g[mc]
                        - vt_all[idx(t - lv, mc, n_time)];
                    pr += phi * dev;
                }
            }
            pr
        };

        // 1. Impute missing cells & update subject random effects u_id[g, .]
        for g in 0..n_id {
            let st = g * n_time;
            let th_cur = build_th_id(g, &gamma0, &gamma_w_id, &u_id);
            let sig2_g: Vec<f64> = (0..d)
                .map(|j| {
                    if lv_id_idx[j] >= 0 {
                        th_cur[lv_id_idx[j] as usize].clamp(-10.0, 10.0).exp()
                    } else {
                        fixed_sigma2[j].max(1e-6)
                    }
                })
                .collect();

            for t in 0..n_time {
                let r_abs = st + t;
                for j in 0..d {
                    if miss_y[idx(r_abs, j, n_obs)] {
                        let fv = cell_mean(&y_work, st, t, j, &th_cur, &vt_tot, &fixed_slopes);
                        y_work[idx(r_abs, j, n_obs)] = fv + sig2_g[j].sqrt() * rng.rnorm();
                    }
                }
            }

            // 1a. Update subject intercepts u_id[g, 0..d]
            {
                let mut pr = vec![0.0; d * d];
                let mut inf = vec![0.0; d];
                for ca in 0..d {
                    for ra in 0..d {
                        pr[idx(ra, ca, d)] = om_id_inv[idx(ra, ca, q_id)];
                    }
                    for q_o in d..q_id {
                        inf[ca] -= om_id_inv[idx(ca, q_o, q_id)] * u_id[idx(g, q_o, n_id)];
                    }
                }
                let mut base_mu = vec![0.0; d];
                for j in 0..d {
                    base_mu[j] = gamma0[j];
                    for k in 0..kw_id {
                        if re_id_on_w_mask.get(idx(k, j, kw_id)).copied().unwrap_or(1) != 0 {
                            base_mu[j] += gamma_w_id[idx(k, j, kw_id)] * w_id[idx(g, k, n_id)];
                        }
                    }
                }
                for j in 0..d {
                    let s2 = sig2_g[j];
                    for t in max_lag..n_time {
                        let mut cm = 1.0;
                        let mut rem =
                            y_work[idx(st + t, j, n_obs)] - vt_tot[idx(t, j, n_time)] - base_mu[j];
                        for k in 0..kx {
                            let flat = j * n_sl + n_lags * d + k;
                            let sl = cell_slope(flat, &th_cur, t, &vt_tot, &fixed_slopes);
                            rem -= sl * x_grid[idx(st + t, k, n_obs)];
                        }
                        for (li, &lv) in lags.iter().enumerate() {
                            for mc in 0..d {
                                let flat = j * n_sl + li * d + mc;
                                let phi = cell_slope(flat, &th_cur, t, &vt_tot, &fixed_slopes);
                                let yl = y_work[idx(st + t - lv, mc, n_obs)]
                                    - vt_tot[idx(t - lv, mc, n_time)];
                                if mc == j {
                                    cm -= phi;
                                    rem -= phi * (yl - base_mu[j]);
                                } else {
                                    rem -= phi * (yl - (base_mu[mc] + u_id[idx(g, mc, n_id)]));
                                }
                            }
                        }
                        pr[idx(j, j, d)] += cm * cm / s2;
                        inf[j] += cm * rem / s2;
                    }
                }
                let u_mu = rmvnorm_precision(&mut rng, &pr, &inf, d);
                for j in 0..d {
                    u_id[idx(g, j, n_id)] = u_mu[j];
                }
            }

            // 1b. Update subject random slopes u_id[g, d..q_lin_id]
            if n_rs_id > 0 {
                let mut pr = vec![0.0; n_rs_id * n_rs_id];
                let mut inf = vec![0.0; n_rs_id];
                for ca in 0..n_rs_id {
                    for ra in 0..n_rs_id {
                        pr[idx(ra, ca, n_rs_id)] = om_id_inv[idx(d + ra, d + ca, q_id)];
                    }
                    for q_o in 0..q_id {
                        if q_o < d || q_o >= q_lin_id {
                            inf[ca] -= om_id_inv[idx(d + ca, q_o, q_id)] * u_id[idx(g, q_o, n_id)];
                        }
                    }
                }
                let th_now = build_th_id(g, &gamma0, &gamma_w_id, &u_id);
                let mut rv = vec![0.0; n_sl];
                for j in 0..d {
                    let s2 = sig2_g[j];
                    for t in max_lag..n_time {
                        let mut y_dev =
                            y_work[idx(st + t, j, n_obs)] - th_now[j] - vt_tot[idx(t, j, n_time)];
                        for (li, &lv) in lags.iter().enumerate() {
                            for mc in 0..d {
                                rv[li * d + mc] = y_work[idx(st + t - lv, mc, n_obs)]
                                    - th_now[mc]
                                    - vt_tot[idx(t - lv, mc, n_time)];
                            }
                        }
                        for k in 0..kx {
                            rv[n_lags * d + k] = x_grid[idx(st + t, k, n_obs)];
                        }
                        let mut active: Vec<(usize, f64)> = Vec::new();
                        for (s, &rv_s) in rv.iter().enumerate().take(n_sl) {
                            let flat = j * n_sl + s;
                            let sl_full = cell_slope(flat, &th_now, t, &vt_tot, &fixed_slopes);
                            if sl_id_idx[flat] >= 0 {
                                let q_s = sl_id_idx[flat] as usize;
                                let u_cur = u_id[idx(g, q_s, n_id)];
                                y_dev -= (sl_full - u_cur) * rv_s;
                                active.push((q_s - d, rv_s));
                            } else {
                                y_dev -= sl_full * rv_s;
                            }
                        }
                        for &(ra, xa) in &active {
                            inf[ra] += xa * y_dev / s2;
                            for &(cb, xb) in &active {
                                pr[idx(ra, cb, n_rs_id)] += xa * xb / s2;
                            }
                        }
                    }
                }
                let u_sl = rmvnorm_precision(&mut rng, &pr, &inf, n_rs_id);
                for s in 0..n_rs_id {
                    u_id[idx(g, d + s, n_id)] = u_sl[s];
                }
            }

            // 1c. Update subject random log-variances via strictly-concave Newton-Laplace Independence MH
            if n_lv > 0 {
                let th_now = build_th_id(g, &gamma0, &gamma_w_id, &u_id);
                for j in 0..d {
                    if lv_id_idx[j] < 0 {
                        continue;
                    }
                    let q_lv = lv_id_idx[j] as usize;
                    let mut cur_u = u_id[idx(g, q_lv, n_id)];
                    let base_lv = th_now[q_lv] - cur_u;
                    let mut rss = 0.0;
                    for t in max_lag..n_time {
                        let fv = cell_mean(&y_work, st, t, j, &th_now, &vt_tot, &fixed_slopes);
                        let e = y_work[idx(st + t, j, n_obs)] - fv;
                        rss += e * e;
                    }
                    let ne = (n_time - max_lag) as f64;
                    let p_tot = om_id_inv[idx(q_lv, q_lv, q_id)].max(1e-8);
                    let mut info_tot = 0.0;
                    for q_o in 0..q_id {
                        if q_o != q_lv {
                            info_tot -= om_id_inv[idx(q_lv, q_o, q_id)] * u_id[idx(g, q_o, n_id)];
                        }
                    }
                    let m_prior = (info_tot / p_tot).clamp(-8.0, 8.0);
                    let mut u_mode = m_prior;
                    for _ in 0..6 {
                        let exp_m = (-(base_lv + u_mode)).exp().min(1e12);
                        let g1 = -0.5 * ne + 0.5 * exp_m * rss - p_tot * (u_mode - m_prior);
                        let neg_g2 = (0.5 * exp_m * rss + p_tot).max(1e-6);
                        let delta = (g1 / neg_g2).clamp(-2.0, 2.0);
                        u_mode = (u_mode + delta).clamp(-8.0, 8.0);
                        if delta.abs() < 1e-6 {
                            break;
                        }
                    }
                    let hess =
                        (0.5 * (-(base_lv + u_mode)).exp().min(1e12) * rss + p_tot).max(1e-6);
                    let prop_sd = (1.25 / hess.sqrt()).clamp(0.01, 2.5);
                    let lp_fn = |uv: f64| -> f64 {
                        let lv = base_lv + uv;
                        -0.5 * ne * lv
                            - 0.5 * (-lv).exp().min(1e12) * rss
                            - 0.5 * p_tot * (uv - m_prior) * (uv - m_prior)
                    };
                    let logq = |uv: f64| -> f64 {
                        let z = (uv - u_mode) / prop_sd;
                        -0.5 * z * z
                    };
                    for _ in 0..2 {
                        let prop_u = (u_mode + prop_sd * rng.rnorm()).clamp(-8.0, 8.0);
                        let log_acc = (lp_fn(prop_u) - logq(prop_u)) - (lp_fn(cur_u) - logq(cur_u));
                        if rng.runif().ln() < log_acc {
                            cur_u = prop_u;
                        }
                    }
                    u_id[idx(g, q_lv, n_id)] = cur_u;
                }
            }
        }

        // Precompute updated th_id for all subjects
        let th_all: Vec<Vec<f64>> = (0..n_id)
            .map(|g| build_th_id(g, &gamma0, &gamma_w_id, &u_id))
            .collect();

        // 2. Update occasion random effects v_time[t, 0..q_time]
        //    2a. Occasion intercepts v_time[t, 0..d]
        for t in 0..n_time {
            let mut pr = vec![0.0; d * d];
            let mut inf = vec![0.0; d];
            for ca in 0..d {
                for ra in 0..d {
                    pr[idx(ra, ca, d)] = om_t_inv[idx(ra, ca, q_time)];
                }
                for q_o in d..q_time {
                    inf[ca] -= om_t_inv[idx(ca, q_o, q_time)] * v_time[idx(t, q_o, n_time)];
                }
            }
            for (g, th_g) in th_all.iter().enumerate().take(n_id) {
                let st = g * n_time;
                for j in 0..d {
                    let s2 = if lv_id_idx[j] >= 0 {
                        th_g[lv_id_idx[j] as usize].clamp(-10.0, 10.0).exp()
                    } else {
                        fixed_sigma2[j].max(1e-6)
                    };
                    if t >= max_lag {
                        let fv_cur = cell_mean(&y_work, st, t, j, th_g, &vt_tot, &fixed_slopes);
                        let fv_no_vt = fv_cur - v_time[idx(t, j, n_time)];
                        let r = y_work[idx(st + t, j, n_obs)] - fv_no_vt;
                        pr[idx(j, j, d)] += 1.0 / s2;
                        inf[j] += r / s2;
                    } else {
                        let mut base_t = th_g[j];
                        for k in 0..kw_time {
                            if re_time_on_w_mask.get(idx(k, j, kw_time)).copied().unwrap_or(1) != 0 {
                                base_t += gamma_w_time[idx(k, j, kw_time)] * w_time[idx(t, k, n_time)];
                            }
                        }
                        let r = y_work[idx(st + t, j, n_obs)] - base_t;
                        pr[idx(j, j, d)] += 1.0 / s2;
                        inf[j] += r / s2;
                    }
                    // Future lag appearances of v_time[t, j]
                    for (li, &lv) in lags.iter().enumerate() {
                        let tf = t + lv;
                        if tf >= max_lag && tf < n_time {
                            for m_out in 0..d {
                                let flat = m_out * n_sl + li * d + j;
                                let phi = cell_slope(flat, th_g, tf, &vt_tot, &fixed_slopes);
                                if phi.abs() > 1e-12 {
                                    let s2_m = if lv_id_idx[m_out] >= 0 {
                                        th_g[lv_id_idx[m_out] as usize].clamp(-10.0, 10.0).exp()
                                    } else {
                                        fixed_sigma2[m_out].max(1e-6)
                                    };
                                    let fv_cur = cell_mean(
                                        &y_work,
                                        st,
                                        tf,
                                        m_out,
                                        th_g,
                                        &vt_tot,
                                        &fixed_slopes,
                                    );
                                    let fv_at_zero = fv_cur + phi * v_time[idx(t, j, n_time)];
                                    let r_f = y_work[idx(st + tf, m_out, n_obs)] - fv_at_zero;
                                    pr[idx(j, j, d)] += (phi * phi) / s2_m;
                                    inf[j] += (-phi) * r_f / s2_m;
                                }
                            }
                        }
                    }
                }
            }
            let vt = rmvnorm_precision(&mut rng, &pr, &inf, d);
            for j in 0..d {
                let delta = vt[j] - v_time[idx(t, j, n_time)];
                v_time[idx(t, j, n_time)] = vt[j];
                vt_tot[idx(t, j, n_time)] += delta;
            }
        }

        //    2b. Occasion-varying random slopes v_time[t, d..q_time]
        if n_rs_time > 0 {
            let mut rv = vec![0.0; n_sl];
            for t in max_lag..n_time {
                let mut pr = vec![0.0; n_rs_time * n_rs_time];
                let mut inf = vec![0.0; n_rs_time];
                for ca in 0..n_rs_time {
                    for ra in 0..n_rs_time {
                        pr[idx(ra, ca, n_rs_time)] = om_t_inv[idx(d + ra, d + ca, q_time)];
                    }
                    for j in 0..d {
                        inf[ca] -= om_t_inv[idx(d + ca, j, q_time)] * v_time[idx(t, j, n_time)];
                    }
                }
                for (g, th_g) in th_all.iter().enumerate().take(n_id) {
                    let st = g * n_time;
                    for (li, &lv) in lags.iter().enumerate() {
                        for mc in 0..d {
                            rv[li * d + mc] = y_work[idx(st + t - lv, mc, n_obs)]
                                - th_g[mc]
                                - vt_tot[idx(t - lv, mc, n_time)];
                        }
                    }
                    for k in 0..kx {
                        rv[n_lags * d + k] = x_grid[idx(st + t, k, n_obs)];
                    }
                    for j in 0..d {
                        let s2 = if lv_id_idx[j] >= 0 {
                            th_g[lv_id_idx[j] as usize].clamp(-10.0, 10.0).exp()
                        } else {
                            fixed_sigma2[j].max(1e-6)
                        };
                        let mut y_dev =
                            y_work[idx(st + t, j, n_obs)] - th_g[j] - vt_tot[idx(t, j, n_time)];
                        let mut active: Vec<(usize, f64)> = Vec::new();
                        for (s, &rv_s) in rv.iter().enumerate().take(n_sl) {
                            let flat = j * n_sl + s;
                            let sl_full = cell_slope(flat, th_g, t, &vt_tot, &fixed_slopes);
                            if sl_time_idx[flat] >= 0 {
                                let q_t = sl_time_idx[flat] as usize;
                                let v_cur = v_time[idx(t, q_t, n_time)];
                                y_dev -= (sl_full - v_cur) * rv_s;
                                active.push((q_t - d, rv_s));
                            } else {
                                y_dev -= sl_full * rv_s;
                            }
                        }
                        for &(ra, xa) in &active {
                            inf[ra] += xa * y_dev / s2;
                            for &(cb, xb) in &active {
                                pr[idx(ra, cb, n_rs_time)] += xa * xb / s2;
                            }
                        }
                    }
                }
                let vt_sl = rmvnorm_precision(&mut rng, &pr, &inf, n_rs_time);
                for s in 0..n_rs_time {
                    let delta = vt_sl[s] - v_time[idx(t, d + s, n_time)];
                    v_time[idx(t, d + s, n_time)] = vt_sl[s];
                    vt_tot[idx(t, d + s, n_time)] += delta;
                }
            }
        }

        // Center v_time to identify grand means
        for j in 0..d {
            let mv: f64 =
                (0..n_time).map(|t| v_time[idx(t, j, n_time)]).sum::<f64>() / n_time as f64;
            for t in 0..n_time {
                v_time[idx(t, j, n_time)] -= mv;
                vt_tot[idx(t, j, n_time)] -= mv;
            }
            gamma0[j] += mv;
        }
        for &(flat, q_t) in &time_sl_flats {
            let mv: f64 = (0..n_time)
                .map(|t| v_time[idx(t, q_t, n_time)])
                .sum::<f64>()
                / n_time as f64;
            for t in 0..n_time {
                v_time[idx(t, q_t, n_time)] -= mv;
                vt_tot[idx(t, q_t, n_time)] -= mv;
            }
            if sl_id_idx[flat] >= 0 {
                gamma0[sl_id_idx[flat] as usize] += mv;
            } else {
                fixed_slopes[sl_fixed_idx[flat] as usize] += mv;
            }
        }

        // 3. Update fixed slopes & fixed sigma2
        let p_pr_base = if prior_mode == 1 {
            1e-10
        } else {
            1.0 / (beta_sd * beta_sd)
        };
        if n_fixed_sl > 0 {
            let mut pr = vec![0.0; n_fixed_sl * n_fixed_sl];
            for fi in 0..n_fixed_sl {
                pr[idx(fi, fi, n_fixed_sl)] = if prior_mode == 2 {
                    1.0 / psem_tau2_fsl[fi].max(1e-6)
                } else {
                    p_pr_base
                };
            }
            let mut inf = vec![0.0; n_fixed_sl];
            let mut rv = vec![0.0; n_sl];
            for g in 0..n_id {
                let st = g * n_time;
                let th_g = build_th_id(g, &gamma0, &gamma_w_id, &u_id);
                for j in 0..d {
                    let s2 = if lv_id_idx[j] >= 0 {
                        th_g[lv_id_idx[j] as usize].clamp(-10.0, 10.0).exp()
                    } else {
                        fixed_sigma2[j].max(1e-6)
                    };
                    for t in max_lag..n_time {
                        let mut y_dev =
                            y_work[idx(st + t, j, n_obs)] - th_g[j] - vt_tot[idx(t, j, n_time)];
                        for (li, &lv) in lags.iter().enumerate() {
                            for mc in 0..d {
                                rv[li * d + mc] = y_work[idx(st + t - lv, mc, n_obs)]
                                    - th_g[mc]
                                    - vt_tot[idx(t - lv, mc, n_time)];
                            }
                        }
                        for k in 0..kx {
                            rv[n_lags * d + k] = x_grid[idx(st + t, k, n_obs)];
                        }
                        let mut active: Vec<(usize, f64)> = Vec::new();
                        for (s, &rv_s) in rv.iter().enumerate().take(n_sl) {
                            let flat = j * n_sl + s;
                            if sl_fixed_idx[flat] >= 0 {
                                let fi = sl_fixed_idx[flat] as usize;
                                // Subtract occasion random component if also time-varying
                                if sl_time_idx[flat] >= 0 {
                                    y_dev -=
                                        vt_tot[idx(t, sl_time_idx[flat] as usize, n_time)] * rv_s;
                                }
                                active.push((fi, rv_s));
                            } else {
                                let sl_val = cell_slope(flat, &th_g, t, &vt_tot, &fixed_slopes);
                                y_dev -= sl_val * rv_s;
                            }
                        }
                        for &(fa, xa) in &active {
                            inf[fa] += xa * y_dev / s2;
                            for &(fb, xb) in &active {
                                pr[idx(fa, fb, n_fixed_sl)] += xa * xb / s2;
                            }
                        }
                    }
                }
            }
            fixed_slopes = rmvnorm_precision(&mut rng, &pr, &inf, n_fixed_sl);
        }

        for j in 0..d {
            if lv_id_idx[j] < 0 {
                let mut rss = 0.0;
                let mut cnt = 0usize;
                for g in 0..n_id {
                    let st = g * n_time;
                    let th_g = build_th_id(g, &gamma0, &gamma_w_id, &u_id);
                    for t in max_lag..n_time {
                        let fv = cell_mean(&y_work, st, t, j, &th_g, &vt_tot, &fixed_slopes);
                        let e = y_work[idx(st + t, j, n_obs)] - fv;
                        rss += e * e;
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

        // 4. Update gamma0, selective gamma_w_id, selective gamma_w_time, PSEM Lasso, omega_id, omega_time
        for q in 0..q_id {
            let o_qq = om_id_inv[idx(q, q, q_id)].max(1e-8);
            let mut active_k: Vec<usize> = Vec::new();
            for k in 0..kw_id {
                if re_id_on_w_mask.get(idx(k, q, kw_id)).copied().unwrap_or(1) != 0 {
                    active_k.push(k);
                }
            }
            let pw_id = 1 + active_k.len();
            let mut wd = vec![0.0; n_id * pw_id];
            let mut yv = vec![0.0; n_id];
            for g in 0..n_id {
                wd[idx(g, 0, n_id)] = 1.0;
                let mut tot = gamma0[q] + u_id[idx(g, q, n_id)];
                for (ai, &k) in active_k.iter().enumerate() {
                    let wv = w_id[idx(g, k, n_id)];
                    wd[idx(g, 1 + ai, n_id)] = wv;
                    tot += gamma_w_id[idx(k, q, kw_id)] * wv;
                }
                for q2 in 0..q_id {
                    if q2 != q {
                        let o_q2 = om_id_inv[idx(q, q2, q_id)];
                        if o_q2.abs() > 1e-14 {
                            tot += (o_q2 / o_qq) * u_id[idx(g, q2, n_id)];
                        }
                    }
                }
                yv[g] = tot;
            }
            let wtw = mat_t_mul(&wd, &wd, pw_id, n_id, pw_id);
            let wty = mat_t_mul(&wd, &yv, pw_id, n_id, 1);
            let mut pr = vec![0.0; pw_id * pw_id];
            for i in 0..(pw_id * pw_id) {
                pr[i] = wtw[i] * o_qq;
            }
            pr[idx(0, 0, pw_id)] += p_pr_base;
            for (ai, &k) in active_k.iter().enumerate() {
                let pk = if prior_mode == 2 {
                    1.0 / psem_tau2_wid[idx(k, q, kw_id)].max(1e-6)
                } else {
                    p_pr_base
                };
                pr[idx(1 + ai, 1 + ai, pw_id)] += pk;
            }
            let inf: Vec<f64> = wty.iter().map(|v| v * o_qq).collect();
            let b = rmvnorm_precision(&mut rng, &pr, &inf, pw_id);
            let old_g0 = gamma0[q];
            let old_gw: Vec<f64> = (0..kw_id).map(|k| gamma_w_id[idx(k, q, kw_id)]).collect();
            gamma0[q] = b[0];
            for k in 0..kw_id {
                gamma_w_id[idx(k, q, kw_id)] = 0.0;
            }
            for (ai, &k) in active_k.iter().enumerate() {
                gamma_w_id[idx(k, q, kw_id)] = b[1 + ai];
            }
            for g in 0..n_id {
                let mut dl = old_g0 - gamma0[q];
                for &k in &active_k {
                    dl += (old_gw[k] - gamma_w_id[idx(k, q, kw_id)]) * w_id[idx(g, k, n_id)];
                }
                u_id[idx(g, q, n_id)] += dl;
            }
        }

        if kw_time > 0 {
            for q in 0..q_time {
                let mut active_k: Vec<usize> = Vec::new();
                for k in 0..kw_time {
                    if re_time_on_w_mask.get(idx(k, q, kw_time)).copied().unwrap_or(1) != 0 {
                        active_k.push(k);
                    }
                }
                if active_k.is_empty() {
                    for k in 0..kw_time {
                        gamma_w_time[idx(k, q, kw_time)] = 0.0;
                    }
                    continue;
                }
                let pa = active_k.len();
                let o_qq = om_t_inv[idx(q, q, q_time)].max(1e-8);
                let mut wt = vec![0.0; n_time * pa];
                let mut yv = vec![0.0; n_time];
                for t in 0..n_time {
                    let mut tot = v_time[idx(t, q, n_time)];
                    for (ai, &k) in active_k.iter().enumerate() {
                        let wv = w_time[idx(t, k, n_time)];
                        wt[idx(t, ai, n_time)] = wv;
                        tot += gamma_w_time[idx(k, q, kw_time)] * wv;
                    }
                    for q2 in 0..q_time {
                        if q2 != q {
                            let o_q2 = om_t_inv[idx(q, q2, q_time)];
                            if o_q2.abs() > 1e-14 {
                                tot += (o_q2 / o_qq) * v_time[idx(t, q2, n_time)];
                            }
                        }
                    }
                    yv[t] = tot;
                }
                let wtw = mat_t_mul(&wt, &wt, pa, n_time, pa);
                let wty = mat_t_mul(&wt, &yv, pa, n_time, 1);
                let mut pr = vec![0.0; pa * pa];
                for i in 0..(pa * pa) {
                    pr[i] = wtw[i] * o_qq;
                }
                for (ai, &k) in active_k.iter().enumerate() {
                    let pk = if prior_mode == 2 {
                        1.0 / psem_tau2_wt[idx(k, q, kw_time)].max(1e-6)
                    } else {
                        p_pr_base
                    };
                    pr[idx(ai, ai, pa)] += pk;
                }
                let inf: Vec<f64> = wty.iter().map(|v| v * o_qq).collect();
                let b = rmvnorm_precision(&mut rng, &pr, &inf, pa);
                let old_gw: Vec<f64> = (0..kw_time)
                    .map(|k| gamma_w_time[idx(k, q, kw_time)])
                    .collect();
                for k in 0..kw_time {
                    gamma_w_time[idx(k, q, kw_time)] = 0.0;
                }
                for (ai, &k) in active_k.iter().enumerate() {
                    gamma_w_time[idx(k, q, kw_time)] = b[ai];
                }
                for t in 0..n_time {
                    let mut dl = 0.0;
                    for &k in &active_k {
                        dl += (old_gw[k] - gamma_w_time[idx(k, q, kw_time)])
                            * w_time[idx(t, k, n_time)];
                    }
                    v_time[idx(t, q, n_time)] += dl;
                }
            }
        }

        // Update PSEM Bayesian Lasso hyperparameters (prior_mode == 2)
        if prior_mode == 2 {
            let mut sum_tau2 = 0.0;
            let mut n_pen = 0usize;
            for fi in 0..n_fixed_sl {
                let b = fixed_slopes[fi].abs().max(1e-6);
                let inv_t2 = rng.rinv_gaussian(psem_lambda2.sqrt() / b, psem_lambda2);
                psem_tau2_fsl[fi] = (1.0 / inv_t2.max(1e-8)).clamp(1e-6, 1e6);
                sum_tau2 += psem_tau2_fsl[fi];
                n_pen += 1;
            }
            for q in 0..q_id {
                for k in 0..kw_id {
                    if re_id_on_w_mask.get(idx(k, q, kw_id)).copied().unwrap_or(1) != 0 {
                        let b = gamma_w_id[idx(k, q, kw_id)].abs().max(1e-6);
                        let inv_t2 = rng.rinv_gaussian(psem_lambda2.sqrt() / b, psem_lambda2);
                        psem_tau2_wid[idx(k, q, kw_id)] = (1.0 / inv_t2.max(1e-8)).clamp(1e-6, 1e6);
                        sum_tau2 += psem_tau2_wid[idx(k, q, kw_id)];
                        n_pen += 1;
                    }
                }
            }
            for q in 0..q_time {
                for k in 0..kw_time {
                    if re_time_on_w_mask.get(idx(k, q, kw_time)).copied().unwrap_or(1) != 0 {
                        let b = gamma_w_time[idx(k, q, kw_time)].abs().max(1e-6);
                        let inv_t2 = rng.rinv_gaussian(psem_lambda2.sqrt() / b, psem_lambda2);
                        psem_tau2_wt[idx(k, q, kw_time)] = (1.0 / inv_t2.max(1e-8)).clamp(1e-6, 1e6);
                        sum_tau2 += psem_tau2_wt[idx(k, q, kw_time)];
                        n_pen += 1;
                    }
                }
            }
            if n_pen > 0 {
                psem_lambda2 = rng
                    .rgamma(1.0 + n_pen as f64, 1.0 + 0.5 * sum_tau2)
                    .clamp(1e-4, 1e4);
            }
        }

        let psi0_diag = if prior_mode == 1 { 1e-5 } else { 0.05 };
        let nu0_id = if prior_mode == 1 {
            -(q_id as f64 + 1.0)
        } else {
            q_id as f64 + 2.0
        };
        let nu0_t = if prior_mode == 1 {
            -(q_time as f64 + 1.0)
        } else {
            q_time as f64 + 2.0
        };
        let mut sid = eye(q_id, psi0_diag);
        for g in 0..n_id {
            for ca in 0..q_id {
                for ra in 0..q_id {
                    sid[idx(ra, ca, q_id)] += u_id[idx(g, ra, n_id)] * u_id[idx(g, ca, n_id)];
                }
            }
        }
        let df_id = (n_id as f64 + nu0_id).max(q_id as f64 + 1.0);
        omega_id = rinv_wishart(&mut rng, df_id, &sid, q_id);

        // Exact PX Scale Group Move on subject random log-variances in omega_id!
        if n_lv > 0 {
            let mut om_id_inv_px = spd_invert(&omega_id, q_id);
            for t in 0..n_time {
                let v_t = build_vt_tot(t, &gamma_w_time, &v_time);
                for q in 0..q_time {
                    vt_tot[idx(t, q, n_time)] = v_t[q];
                }
            }
            for j in 0..d {
                if lv_id_idx[j] < 0 {
                    continue;
                }
                let q_lv = lv_id_idx[j] as usize;
                let ne = (n_time - max_lag) as f64;
                let mut base_v = vec![0.0; n_id];
                let mut u_v = vec![0.0; n_id];
                let mut rss_v = vec![0.0; n_id];
                for g in 0..n_id {
                    let st = g * n_time;
                    let th_g = build_th_id(g, &gamma0, &gamma_w_id, &u_id);
                    u_v[g] = u_id[idx(g, q_lv, n_id)];
                    base_v[g] = th_g[q_lv] - u_v[g];
                    let mut rss = 0.0;
                    for t in max_lag..n_time {
                        let fv = cell_mean(&y_work, st, t, j, &th_g, &vt_tot, &fixed_slopes);
                        let e = y_work[idx(st + t, j, n_obs)] - fv;
                        rss += e * e;
                    }
                    rss_v[g] = rss;
                }
                for _ in 0..3 {
                    let delta = 0.18 * rng.rnorm();
                    let a = delta.exp();
                    let inv_a2 = (-2.0 * delta).exp();
                    let mut valid = true;
                    let mut ll_diff = 0.0;
                    for g in 0..n_id {
                        let cur_lv = base_v[g] + u_v[g];
                        let prop_lv = base_v[g] + a * u_v[g];
                        if prop_lv.abs() > 10.0 {
                            valid = false;
                            break;
                        }
                        ll_diff += -0.5 * ne * (prop_lv - cur_lv)
                            - 0.5
                                * ((-prop_lv).exp().min(1e12) - (-cur_lv).exp().min(1e12))
                                * rss_v[g];
                    }
                    if !valid {
                        continue;
                    }
                    let om_qq = om_id_inv_px[idx(q_lv, q_lv, q_id)];
                    let log_acc =
                        ll_diff - nu0_id * delta - 0.5 * psi0_diag * (inv_a2 - 1.0) * om_qq;
                    if log_acc.is_finite() && (log_acc >= 0.0 || rng.runif().ln() < log_acc) {
                        for g in 0..n_id {
                            u_v[g] *= a;
                            u_id[idx(g, q_lv, n_id)] = u_v[g];
                        }
                        let inv_a = 1.0 / a;
                        for r in 0..q_id {
                            omega_id[idx(r, q_lv, q_id)] *= a;
                            omega_id[idx(q_lv, r, q_id)] *= a;
                            om_id_inv_px[idx(r, q_lv, q_id)] *= inv_a;
                            om_id_inv_px[idx(q_lv, r, q_id)] *= inv_a;
                        }
                    }
                }
            }
        }

        let mut st_m = eye(q_time, psi0_diag);
        for t in 0..n_time {
            for ca in 0..q_time {
                for ra in 0..q_time {
                    st_m[idx(ra, ca, q_time)] +=
                        v_time[idx(t, ra, n_time)] * v_time[idx(t, ca, n_time)];
                }
            }
        }
        let df_t = (n_time as f64 + nu0_t).max(q_time as f64 + 1.0);
        omega_time = rinv_wishart(&mut rng, df_t, &st_m, q_time);

        let _ = &id_sl_flats;

        if keep_pos < n_keep && step == keep_indices[keep_pos] {
            // Refresh vt_tot after gamma_w_time update
            for t in 0..n_time {
                let v_t = build_vt_tot(t, &gamma_w_time, &v_time);
                for q in 0..q_time {
                    vt_tot[idx(t, q, n_time)] = v_t[q];
                }
            }

            let mut cur = 0usize;
            for &g0 in &gamma0 {
                draws[idx(keep_pos, cur, n_keep)] = g0;
                cur += 1;
            }
            for q in 0..q_id {
                for k in 0..kw_id {
                    if re_id_on_w_mask.get(idx(k, q, kw_id)).copied().unwrap_or(1) != 0 {
                        draws[idx(keep_pos, cur, n_keep)] = gamma_w_id[idx(k, q, kw_id)];
                        cur += 1;
                    }
                }
            }
            for q in 0..q_time {
                for k in 0..kw_time {
                    if re_time_on_w_mask.get(idx(k, q, kw_time)).copied().unwrap_or(1) != 0 {
                        draws[idx(keep_pos, cur, n_keep)] = gamma_w_time[idx(k, q, kw_time)];
                        cur += 1;
                    }
                }
            }
            for &fs in &fixed_slopes {
                draws[idx(keep_pos, cur, n_keep)] = fs;
                cur += 1;
            }
            for ca in 0..q_id {
                for ra in ca..q_id {
                    draws[idx(keep_pos, cur, n_keep)] = omega_id[idx(ra, ca, q_id)];
                    cur += 1;
                }
            }
            for ca in 0..q_time {
                for ra in ca..q_time {
                    draws[idx(keep_pos, cur, n_keep)] = omega_time[idx(ra, ca, q_time)];
                    cur += 1;
                }
            }
            for j in 0..d {
                if lv_id_idx[j] < 0 {
                    draws[idx(keep_pos, cur, n_keep)] = fixed_sigma2[j];
                    cur += 1;
                }
            }

            for q in 0..q_id {
                for g in 0..n_id {
                    let mut tot = gamma0[q] + u_id[idx(g, q, n_id)];
                    for k in 0..kw_id {
                        tot += gamma_w_id[idx(k, q, kw_id)] * w_id[idx(g, k, n_id)];
                    }
                    id_draws[keep_pos + n_keep * (g + n_id * q)] = tot;
                }
            }
            for q in 0..q_time {
                for t in 0..n_time {
                    time_draws[keep_pos + n_keep * (t + n_time * q)] = vt_tot[idx(t, q, n_time)];
                }
            }

            let mut ll = 0.0;
            for g in 0..n_id {
                let st = g * n_time;
                let th_g = build_th_id(g, &gamma0, &gamma_w_id, &u_id);
                for j in 0..d {
                    let s2 = if lv_id_idx[j] >= 0 {
                        th_g[lv_id_idx[j] as usize].clamp(-10.0, 10.0).exp()
                    } else {
                        fixed_sigma2[j].max(1e-6)
                    };
                    sig_accum[idx(g, j, n_id)] += s2;
                    for t in 0..n_time {
                        let fv = if t >= max_lag {
                            cell_mean(&y_work, st, t, j, &th_g, &vt_tot, &fixed_slopes)
                        } else {
                            th_g[j] + vt_tot[idx(t, j, n_time)]
                        };
                        fitted_accum[idx(st + t, j, n_obs)] += fv;
                        if t >= max_lag && !miss_y[idx(st + t, j, n_obs)] {
                            let e = y_work[idx(st + t, j, n_obs)] - fv;
                            ll += -0.5 * ((2.0 * PI * s2).ln() + e * e / s2);
                        }
                    }
                }
            }
            deviance_sum += -2.0 * ll;
            keep_pos += 1;
        }
    }

    let inv_keep = 1.0 / (n_keep.max(1) as f64);
    let fitted_mean: Vec<f64> = fitted_accum.into_iter().map(|v| v * inv_keep).collect();
    let d_bar = deviance_sum * inv_keep;
    let mut ll_hat = 0.0;
    for g in 0..n_id {
        let st = g * n_time;
        for j in 0..d {
            let s2 = (sig_accum[idx(g, j, n_id)] * inv_keep).max(1e-6);
            for t in max_lag..n_time {
                if !miss_y[idx(st + t, j, n_obs)] {
                    let e = y_work[idx(st + t, j, n_obs)] - fitted_mean[idx(st + t, j, n_obs)];
                    ll_hat += -0.5 * ((2.0 * PI * s2).ln() + e * e / s2);
                }
            }
        }
    }
    let pd = (d_bar - (-2.0 * ll_hat)).max(0.0);
    let dic = d_bar + pd;

    CrossClassifiedOutput {
        draws,
        param_count: n_params,
        id_draws,
        time_draws,
        q_id,
        q_time,
        fitted_mean,
        dic,
        pd,
    }
}
