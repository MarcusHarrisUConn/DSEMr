use crate::linalg::{eye, idx, mat_t_mul, quad_form, rinv_wishart, rmvnorm_precision, spd_invert};
use crate::rng::Rng;
use std::f64::consts::PI;

pub struct CrossClassifiedOutput {
    pub draws: Vec<f64>,
    pub param_count: usize,
    pub id_draws: Vec<f64>,   // n_keep x n_id x q_id
    pub time_draws: Vec<f64>, // n_keep x n_time x d
    pub fitted_mean: Vec<f64>,
    pub dic: f64,
    pub pd: f64,
}

/// Mplus 9.38-9.40 Cross-Classified DSEM (Subject x Time two-way random effects + within dynamics).
#[allow(clippy::too_many_arguments)]
pub fn mcmc_crossclassified(
    y_grid: &[f64], // (n_id * n_time) x d, ordered by id g=0..n_id, t=0..n_time
    x_grid: &[f64], // (n_id * n_time) x kx
    w_id: &[f64],   // n_id x kw
    n_id: usize,
    n_time: usize,
    d: usize,
    kx: usize,
    kw: usize,
    lags: &[usize],
    has_random_slopes: bool,
    has_random_logv: bool,
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

    let n_rs = if has_random_slopes { d * n_sl } else { 0 };
    let n_lv = if has_random_logv { d } else { 0 };
    let q_id = d + n_rs + n_lv;

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
    for j in 0..d {
        gamma0[j] = (0..n_obs).map(|r| y_work[idx(r, j, n_obs)]).sum::<f64>() / n_obs as f64;
        if has_random_slopes && n_lags >= 1 {
            gamma0[d + j * n_sl + j] = 0.25;
        }
    }
    let mut gamma_w = vec![0.0; kw * q_id];
    let mut u_id = vec![0.0; n_id * q_id]; // centered subject deviations
    let mut v_time = vec![0.0; n_time * d]; // zero-mean time shocks
    let mut fixed_slopes = vec![0.0; if has_random_slopes { 0 } else { d * n_sl }];
    if !has_random_slopes && n_lags >= 1 {
        for j in 0..d {
            fixed_slopes[j * n_sl + j] = 0.25;
        }
    }
    let mut fixed_sigma2: Vec<f64> = vec![1.0_f64; if has_random_logv { 0 } else { d }];
    let mut omega_id = eye(q_id, 0.1);
    let mut omega_time = eye(d, 0.1);

    let n_om_id = q_id * (q_id + 1) / 2;
    let n_om_t = d * (d + 1) / 2;
    let n_params = q_id + q_id * kw + fixed_slopes.len() + n_om_id + n_om_t + fixed_sigma2.len();

    let keep_indices: Vec<usize> = ((warmup + 1)..=iter)
        .filter(|i| (i - warmup - 1) % thin == 0)
        .collect();
    let n_keep = keep_indices.len();
    let mut draws = vec![0.0; n_keep * n_params];
    let mut id_draws = vec![0.0; n_keep * n_id * q_id];
    let mut time_draws = vec![0.0; n_keep * n_time * d];
    let mut fitted_accum = vec![0.0; n_obs * d];
    let mut sig_accum = vec![0.0; n_id * d];
    let mut deviance_sum = 0.0;
    let mut keep_pos = 0usize;

    for step in 1..=iter {
        let om_id_inv = spd_invert(&omega_id, q_id);
        let om_t_inv = spd_invert(&omega_time, d);

        let build_th = |g: usize, g0: &[f64], gw: &[f64], uid: &[f64]| -> Vec<f64> {
            let mut th = g0.to_vec();
            for q in 0..q_id {
                for k in 0..kw {
                    th[q] += gw[idx(k, q, kw)] * w_id[idx(g, k, n_id)];
                }
                th[q] += uid[idx(g, q, n_id)];
            }
            th
        };

        // 1. Impute missing cells & update subject random effects u_id[g, .]
        let q_lin = d + n_rs;
        for g in 0..n_id {
            let st = g * n_time;
            let th_cur = build_th(g, &gamma0, &gamma_w, &u_id);
            let slopes_g: Vec<f64> = if has_random_slopes {
                th_cur[d..(d + n_rs)].to_vec()
            } else {
                fixed_slopes.clone()
            };
            let sig2_g: Vec<f64> = (0..d)
                .map(|j| {
                    if has_random_logv {
                        th_cur[d + n_rs + j].clamp(-10.0, 10.0).exp()
                    } else {
                        fixed_sigma2[j].max(1e-6)
                    }
                })
                .collect();

            for t in 0..n_time {
                let r_abs = st + t;
                for j in 0..d {
                    if miss_y[idx(r_abs, j, n_obs)] {
                        let fv = cc_cell_mean(
                            &y_work, x_grid, &v_time, st, t, j, d, kx, lags, max_lag, &th_cur,
                            &slopes_g, n_time, n_obs,
                        );
                        y_work[idx(r_abs, j, n_obs)] = fv + sig2_g[j].sqrt() * rng.rnorm();
                    }
                }
            }

            // Update u_id[g, 0..d] (subject intercepts)
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
                    for k in 0..kw {
                        base_mu[j] += gamma_w[idx(k, j, kw)] * w_id[idx(g, k, n_id)];
                    }
                }
                for j in 0..d {
                    let s2 = sig2_g[j];
                    let mut cm = 1.0;
                    for (li, _) in lags.iter().enumerate() {
                        cm -= slopes_g[j * n_sl + li * d + j];
                    }
                    for t in max_lag..n_time {
                        let mut rem = y_work[idx(st + t, j, n_obs)]
                            - v_time[idx(t, j, n_time)]
                            - cm * base_mu[j];
                        for k in 0..kx {
                            rem -=
                                slopes_g[j * n_sl + n_lags * d + k] * x_grid[idx(st + t, k, n_obs)];
                        }
                        for (li, &lv) in lags.iter().enumerate() {
                            for mc in 0..d {
                                let phi = slopes_g[j * n_sl + li * d + mc];
                                let yl = y_work[idx(st + t - lv, mc, n_obs)]
                                    - v_time[idx(t - lv, mc, n_time)];
                                rem -= phi * yl;
                                if mc != j {
                                    rem += phi * (base_mu[mc] + u_id[idx(g, mc, n_id)]);
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

            // Update subject random slopes u_id[g, d..q_lin] if enabled
            if has_random_slopes {
                let mut pr = vec![0.0; n_rs * n_rs];
                let mut inf = vec![0.0; n_rs];
                for ca in 0..n_rs {
                    for ra in 0..n_rs {
                        pr[idx(ra, ca, n_rs)] = om_id_inv[idx(d + ra, d + ca, q_id)];
                    }
                    for q_o in 0..q_id {
                        if q_o < d || q_o >= q_lin {
                            inf[ca] -= om_id_inv[idx(d + ca, q_o, q_id)] * u_id[idx(g, q_o, n_id)];
                        }
                    }
                }
                let th_now = build_th(g, &gamma0, &gamma_w, &u_id);
                let mut base_sl = vec![0.0; n_rs];
                for s in 0..n_rs {
                    base_sl[s] = gamma0[d + s];
                    for k in 0..kw {
                        base_sl[s] += gamma_w[idx(k, d + s, kw)] * w_id[idx(g, k, n_id)];
                    }
                }
                for j in 0..d {
                    let s2 = sig2_g[j];
                    for t in max_lag..n_time {
                        let mut y_dev =
                            y_work[idx(st + t, j, n_obs)] - th_now[j] - v_time[idx(t, j, n_time)];
                        let mut rv = vec![0.0; n_sl];
                        for (li, &lv) in lags.iter().enumerate() {
                            for mc in 0..d {
                                rv[li * d + mc] = y_work[idx(st + t - lv, mc, n_obs)]
                                    - th_now[mc]
                                    - v_time[idx(t - lv, mc, n_time)];
                            }
                        }
                        for k in 0..kx {
                            rv[n_lags * d + k] = x_grid[idx(st + t, k, n_obs)];
                        }
                        for s in 0..n_sl {
                            y_dev -= base_sl[j * n_sl + s] * rv[s];
                        }
                        for a in 0..n_sl {
                            let ga = j * n_sl + a;
                            inf[ga] += rv[a] * y_dev / s2;
                            for b in 0..n_sl {
                                let gb = j * n_sl + b;
                                pr[idx(ga, gb, n_rs)] += rv[a] * rv[b] / s2;
                            }
                        }
                    }
                }
                let u_sl = rmvnorm_precision(&mut rng, &pr, &inf, n_rs);
                for s in 0..n_rs {
                    u_id[idx(g, d + s, n_id)] = u_sl[s];
                }
            }

            // Update subject random log-variances via MH
            if has_random_logv {
                let th_now = build_th(g, &gamma0, &gamma_w, &u_id);
                let sl_now = if has_random_slopes {
                    th_now[d..(d + n_rs)].to_vec()
                } else {
                    fixed_slopes.clone()
                };
                for j in 0..d {
                    let q_lv = d + n_rs + j;
                    let cur_u = u_id[idx(g, q_lv, n_id)];
                    let prop_u = (cur_u + 0.22 * rng.rnorm()).clamp(-8.0, 8.0);
                    let base_lv = th_now[q_lv] - cur_u;
                    let mut rss = 0.0;
                    for t in max_lag..n_time {
                        let fv = cc_cell_mean(
                            &y_work, x_grid, &v_time, st, t, j, d, kx, lags, max_lag, &th_now,
                            &sl_now, n_time, n_obs,
                        );
                        let e = y_work[idx(st + t, j, n_obs)] - fv;
                        rss += e * e;
                    }
                    let ne = (n_time - max_lag) as f64;
                    let lp = |uv: f64| -> f64 {
                        let lv = base_lv + uv;
                        let mut uv_vec: Vec<f64> =
                            (0..q_id).map(|q| u_id[idx(g, q, n_id)]).collect();
                        uv_vec[q_lv] = uv;
                        -0.5 * ne * lv
                            - 0.5 * (-lv).exp() * rss
                            - 0.5 * quad_form(&uv_vec, &om_id_inv, q_id)
                    };
                    if rng.runif().ln() < lp(prop_u) - lp(cur_u) {
                        u_id[idx(g, q_lv, n_id)] = prop_u;
                    }
                }
            }
        }

        // 2. Update occasion random effects v_time[t, 0..d] sequentially across t = 0..n_time
        for t in 0..n_time {
            let mut pr = om_t_inv.clone();
            let mut inf = vec![0.0; d];
            for g in 0..n_id {
                let st = g * n_time;
                let th_g = build_th(g, &gamma0, &gamma_w, &u_id);
                let sl_g: Vec<f64> = if has_random_slopes {
                    th_g[d..(d + n_rs)].to_vec()
                } else {
                    fixed_slopes.clone()
                };
                for j in 0..d {
                    let s2 = if has_random_logv {
                        th_g[d + n_rs + j].clamp(-10.0, 10.0).exp()
                    } else {
                        fixed_sigma2[j].max(1e-6)
                    };
                    if t >= max_lag {
                        let mut fv_no_vt = cc_cell_mean(
                            &y_work, x_grid, &v_time, st, t, j, d, kx, lags, max_lag, &th_g, &sl_g,
                            n_time, n_obs,
                        );
                        fv_no_vt -= v_time[idx(t, j, n_time)];
                        let r = y_work[idx(st + t, j, n_obs)] - fv_no_vt;
                        pr[idx(j, j, d)] += 1.0 / s2;
                        inf[j] += r / s2;
                    } else {
                        let r = y_work[idx(st + t, j, n_obs)] - th_g[j];
                        pr[idx(j, j, d)] += 1.0 / s2;
                        inf[j] += r / s2;
                    }
                    // Future lag appearances of v_time[t, j] (since y^{(w)}_{it} = y_{it} - mu_i - v_t)
                    for (li, &lv) in lags.iter().enumerate() {
                        let tf = t + lv;
                        if tf >= max_lag && tf < n_time {
                            let phi = sl_g[j * n_sl + li * d + j];
                            // In y_{tf}, pred has + phi * (y_t - mu_i - v_t) -> derivative wrt v_t is -phi!
                            let fv_cur = cc_cell_mean(
                                &y_work, x_grid, &v_time, st, tf, j, d, kx, lags, max_lag, &th_g,
                                &sl_g, n_time, n_obs,
                            );
                            let fv_at_zero = fv_cur + phi * v_time[idx(t, j, n_time)];
                            let r_f = y_work[idx(st + tf, j, n_obs)] - fv_at_zero;
                            pr[idx(j, j, d)] += (phi * phi) / s2;
                            inf[j] += (-phi) * r_f / s2;
                        }
                    }
                }
            }
            let vt = rmvnorm_precision(&mut rng, &pr, &inf, d);
            for j in 0..d {
                v_time[idx(t, j, n_time)] = vt[j];
            }
        }
        // Center v_time to identify grand intercept mu
        for j in 0..d {
            let mv: f64 =
                (0..n_time).map(|t| v_time[idx(t, j, n_time)]).sum::<f64>() / n_time as f64;
            for t in 0..n_time {
                v_time[idx(t, j, n_time)] -= mv;
            }
            gamma0[j] += mv;
        }

        // 3. Update fixed slopes & fixed sigma2 (if not random)
        let p_pr = if prior_mode == 1 {
            1e-10
        } else {
            1.0 / (beta_sd * beta_sd)
        };
        if !has_random_slopes {
            let nf = d * n_sl;
            let mut pr = eye(nf, p_pr);
            let mut inf = vec![0.0; nf];
            for g in 0..n_id {
                let st = g * n_time;
                let th_g = build_th(g, &gamma0, &gamma_w, &u_id);
                for j in 0..d {
                    let s2 = if has_random_logv {
                        th_g[d + j].clamp(-10.0, 10.0).exp()
                    } else {
                        fixed_sigma2[j].max(1e-6)
                    };
                    for t in max_lag..n_time {
                        let y_dev =
                            y_work[idx(st + t, j, n_obs)] - th_g[j] - v_time[idx(t, j, n_time)];
                        let mut rv = vec![0.0; n_sl];
                        for (li, &lv) in lags.iter().enumerate() {
                            for mc in 0..d {
                                rv[li * d + mc] = y_work[idx(st + t - lv, mc, n_obs)]
                                    - th_g[mc]
                                    - v_time[idx(t - lv, mc, n_time)];
                            }
                        }
                        for k in 0..kx {
                            rv[n_lags * d + k] = x_grid[idx(st + t, k, n_obs)];
                        }
                        for a in 0..n_sl {
                            let ga = j * n_sl + a;
                            inf[ga] += rv[a] * y_dev / s2;
                            for b in 0..n_sl {
                                pr[idx(ga, j * n_sl + b, nf)] += rv[a] * rv[b] / s2;
                            }
                        }
                    }
                }
            }
            fixed_slopes = rmvnorm_precision(&mut rng, &pr, &inf, nf);
        }
        if !has_random_logv {
            for j in 0..d {
                let mut rss = 0.0;
                let mut cnt = 0usize;
                for g in 0..n_id {
                    let st = g * n_time;
                    let th_g = build_th(g, &gamma0, &gamma_w, &u_id);
                    let sl_g = if has_random_slopes {
                        th_g[d..(d + n_rs)].to_vec()
                    } else {
                        fixed_slopes.clone()
                    };
                    for t in max_lag..n_time {
                        let fv = cc_cell_mean(
                            &y_work, x_grid, &v_time, st, t, j, d, kx, lags, max_lag, &th_g, &sl_g,
                            n_time, n_obs,
                        );
                        let e = y_work[idx(st + t, j, n_obs)] - fv;
                        rss += e * e;
                        cnt += 1;
                    }
                }
                fixed_sigma2[j] = rng.rinv_gamma(0.001 + 0.5 * cnt as f64, 0.001 + 0.5 * rss);
            }
        }

        // 4. Update gamma0, gamma_w, omega_id, omega_time
        let pw = 1 + kw;
        for q in 0..q_id {
            let vq = omega_id[idx(q, q, q_id)].max(1e-6);
            let mut wd = vec![0.0; n_id * pw];
            let mut yv = vec![0.0; n_id];
            for g in 0..n_id {
                wd[idx(g, 0, n_id)] = 1.0;
                let mut tot = gamma0[q] + u_id[idx(g, q, n_id)];
                for k in 0..kw {
                    wd[idx(g, 1 + k, n_id)] = w_id[idx(g, k, n_id)];
                    tot += gamma_w[idx(k, q, kw)] * w_id[idx(g, k, n_id)];
                }
                yv[g] = tot;
            }
            let wtw = mat_t_mul(&wd, &wd, pw, n_id, pw);
            let wty = mat_t_mul(&wd, &yv, pw, n_id, 1);
            let mut pr = vec![0.0; pw * pw];
            for i in 0..(pw * pw) {
                pr[i] = wtw[i] / vq;
            }
            for i in 0..pw {
                pr[idx(i, i, pw)] += p_pr;
            }
            let inf: Vec<f64> = wty.iter().map(|v| v / vq).collect();
            let b = rmvnorm_precision(&mut rng, &pr, &inf, pw);
            let old_g0 = gamma0[q];
            let old_gw: Vec<f64> = (0..kw).map(|k| gamma_w[idx(k, q, kw)]).collect();
            gamma0[q] = b[0];
            for k in 0..kw {
                gamma_w[idx(k, q, kw)] = b[1 + k];
            }
            for g in 0..n_id {
                let mut dl = old_g0 - gamma0[q];
                for k in 0..kw {
                    dl += (old_gw[k] - gamma_w[idx(k, q, kw)]) * w_id[idx(g, k, n_id)];
                }
                u_id[idx(g, q, n_id)] += dl;
            }
        }
        let mut sid = eye(q_id, 0.05);
        for g in 0..n_id {
            for ca in 0..q_id {
                for ra in 0..q_id {
                    sid[idx(ra, ca, q_id)] += u_id[idx(g, ra, n_id)] * u_id[idx(g, ca, n_id)];
                }
            }
        }
        omega_id = rinv_wishart(&mut rng, (q_id as f64 + 2.0) + n_id as f64, &sid, q_id);

        let mut st_m = eye(d, 0.05);
        for t in 0..n_time {
            for ca in 0..d {
                for ra in 0..d {
                    st_m[idx(ra, ca, d)] += v_time[idx(t, ra, n_time)] * v_time[idx(t, ca, n_time)];
                }
            }
        }
        omega_time = rinv_wishart(&mut rng, (d as f64 + 2.0) + n_time as f64, &st_m, d);

        if keep_pos < n_keep && step == keep_indices[keep_pos] {
            let mut cur = 0usize;
            for &g0 in &gamma0 {
                draws[idx(keep_pos, cur, n_keep)] = g0;
                cur += 1;
            }
            for q in 0..q_id {
                for k in 0..kw {
                    draws[idx(keep_pos, cur, n_keep)] = gamma_w[idx(k, q, kw)];
                    cur += 1;
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
            for ca in 0..d {
                for ra in ca..d {
                    draws[idx(keep_pos, cur, n_keep)] = omega_time[idx(ra, ca, d)];
                    cur += 1;
                }
            }
            for &s2 in &fixed_sigma2 {
                draws[idx(keep_pos, cur, n_keep)] = s2;
                cur += 1;
            }
            for q in 0..q_id {
                for g in 0..n_id {
                    id_draws[keep_pos + n_keep * (g + n_id * q)] =
                        gamma0[q] + u_id[idx(g, q, n_id)];
                }
            }
            for j in 0..d {
                for t in 0..n_time {
                    time_draws[keep_pos + n_keep * (t + n_time * j)] = v_time[idx(t, j, n_time)];
                }
            }

            let mut ll = 0.0;
            for g in 0..n_id {
                let st = g * n_time;
                let th_g = build_th(g, &gamma0, &gamma_w, &u_id);
                let sl_g = if has_random_slopes {
                    th_g[d..(d + n_rs)].to_vec()
                } else {
                    fixed_slopes.clone()
                };
                for j in 0..d {
                    let s2 = if has_random_logv {
                        th_g[d + n_rs + j].clamp(-10.0, 10.0).exp()
                    } else {
                        fixed_sigma2[j].max(1e-6)
                    };
                    sig_accum[idx(g, j, n_id)] += s2;
                    for t in 0..n_time {
                        let fv = if t >= max_lag {
                            cc_cell_mean(
                                &y_work, x_grid, &v_time, st, t, j, d, kx, lags, max_lag, &th_g,
                                &sl_g, n_time, n_obs,
                            )
                        } else {
                            th_g[j] + v_time[idx(t, j, n_time)]
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
        fitted_mean,
        dic,
        pd,
    }
}

#[allow(clippy::too_many_arguments)]
fn cc_cell_mean(
    y: &[f64],
    x: &[f64],
    v_time: &[f64],
    st: usize,
    t: usize,
    j: usize,
    d: usize,
    kx: usize,
    lags: &[usize],
    max_lag: usize,
    th: &[f64],
    sl: &[f64],
    n_time: usize,
    n_obs: usize,
) -> f64 {
    let n_lags = lags.len();
    let n_sl = n_lags * d + kx;
    let mut pr = th[j] + v_time[idx(t, j, n_time)];
    for k in 0..kx {
        pr += sl[j * n_sl + n_lags * d + k] * x[idx(st + t, k, n_obs)];
    }
    if t < max_lag {
        return pr;
    }
    for (li, &lv) in lags.iter().enumerate() {
        for mc in 0..d {
            let phi = sl[j * n_sl + li * d + mc];
            let dev = y[idx(st + t - lv, mc, n_obs)] - th[mc] - v_time[idx(t - lv, mc, n_time)];
            pr += phi * dev;
        }
    }
    pr
}
