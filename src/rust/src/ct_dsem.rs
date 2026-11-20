use crate::linalg::{eye, idx, mat_mul, mat_mul_t, rinv_wishart, rmvnorm_precision, spd_invert};
use crate::rng::Rng;
use std::f64::consts::PI;

pub struct CtDsemOutput {
    pub draws: Vec<f64>,
    pub param_count: usize,
    pub re_draws: Vec<f64>,    // n_keep x g_cnt x q_b (if is_twolevel)
    pub fitted_mean: Vec<f64>, // n_obs x d
    pub dic: f64,
    pub pd: f64,
}

/// Continuous-Time DSEM / CT-RDSEM (Ornstein-Uhlenbeck exact matrix-exponential transition):
///   y_{g,i} = mu_g + X_{g,i} beta + eta_{g,i}
///   d eta(t) = A_g eta(t) dt + G_g dW(t),  Q_g = G_g G_g^T
/// Exact discrete transition over arbitrary irregular interval dt = t_{g,i} - t_{g,i-1} > 0:
///   Phi(dt) = exp(A dt),   Q(dt) = Sigma_inf - Phi(dt) Sigma_inf Phi(dt)^T
/// where A Sigma_inf + Sigma_inf A^T + Q = 0 (exact continuous-time Lyapunov equation).
/// Supports N=1 (d >= 1) and Two-Level (d >= 1, with optional univariate random drift & log-diffusion).
#[allow(clippy::too_many_arguments)]
pub fn mcmc_ct_dsem(
    y_in: &[f64],            // n_obs x d (contiguous by cluster, ordered by time)
    x_in: &[f64],            // n_obs x kx
    time_in: &[f64],         // n_obs (strictly increasing within cluster)
    cluster_sizes: &[usize], // length g_cnt
    d: usize,
    kx: usize,
    random_drift: bool, // supported when is_twolevel && d == 1
    random_logv: bool,  // supported when is_twolevel && d == 1
    prior_mode: i32,    // 0 = weakly_informative, 1 = diffuse
    beta_sd: f64,
    iter: usize,
    warmup: usize,
    thin: usize,
    seed: u64,
) -> CtDsemOutput {
    let mut rng = Rng::from_seed(seed);
    let g_cnt = cluster_sizes.len();
    let is_twolevel = g_cnt > 1;
    let n_obs: usize = cluster_sizes.iter().sum();
    let use_rand_drift = is_twolevel && d == 1 && random_drift;
    let use_rand_logv = is_twolevel && d == 1 && random_logv;

    let mut offsets = Vec::with_capacity(g_cnt + 1);
    offsets.push(0usize);
    for &sz in cluster_sizes {
        offsets.push(offsets.last().unwrap() + sz);
    }

    // Compute time increments dt[r] (for t_loc >= 1)
    let mut dt_vec = vec![1.0_f64; n_obs];
    for g in 0..g_cnt {
        let st = offsets[g];
        let en = offsets[g + 1];
        for r in (st + 1)..en {
            let dtv = time_in[r] - time_in[r - 1];
            dt_vec[r] = if dtv.is_finite() && dtv > 1e-8 {
                dtv
            } else {
                1.0
            };
        }
    }

    // Handle missing y (initialize to column mean; skip missing rows or impute conditionally)
    let mut y_work = y_in.to_vec();
    let mut miss_y = vec![false; n_obs * d];
    let mut any_miss = false;
    let mut col_means = vec![0.0; d];
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
                any_miss = true;
            }
        }
        let m = if c > 0 { s / c as f64 } else { 0.0 };
        col_means[j] = m;
        for r in 0..n_obs {
            if miss_y[idx(r, j, n_obs)] {
                y_work[idx(r, j, n_obs)] = m;
            }
        }
    }

    // Parameters
    let mut mu_pop = col_means.clone();
    let mut drift_a = vec![0.0_f64; d * d];
    for j in 0..d {
        drift_a[idx(j, j, d)] = -0.8;
    }
    let mut beta_x = vec![0.0_f64; d * kx]; // row j, col k -> idx(j, k, d)
    // Parameterize diffusion Q = L_q L_q^T where diag(L_q) = exp(l_diag), lower-tri = l_off
    let mut lq_diag = vec![0.0_f64; d]; // log(L_jj)
    let mut lq_off = vec![0.0_f64; d * (d - 1) / 2];

    let mut q_mat = build_spd_from_chol(&lq_diag, &lq_off, d);

    // Two-level random effects layout:
    //   [mu_{g,0..d}, drift_g (if use_rand_drift), logq_g (if use_rand_logv)]
    let q_b = if is_twolevel {
        d + usize::from(use_rand_drift) + usize::from(use_rand_logv)
    } else {
        0
    };
    let drift_slot = if use_rand_drift { Some(d) } else { None };
    let logv_slot = if use_rand_logv {
        Some(d + usize::from(use_rand_drift))
    } else {
        None
    };

    let mut theta_b = vec![0.0_f64; g_cnt * q_b.max(1)];
    if is_twolevel {
        for g in 0..g_cnt {
            let st = offsets[g];
            let en = offsets[g + 1];
            let tg = (en - st).max(1) as f64;
            for j in 0..d {
                let mg = (st..en).map(|r| y_work[idx(r, j, n_obs)]).sum::<f64>() / tg;
                theta_b[idx(g, j, g_cnt)] = mg;
            }
            if let Some(ds) = drift_slot {
                theta_b[idx(g, ds, g_cnt)] = -0.8;
            }
            if let Some(ls) = logv_slot {
                theta_b[idx(g, ls, g_cnt)] = 0.0;
            }
        }
    }
    let mut logq_pop = 0.0_f64;
    let mut omega_b = eye(q_b, 0.2);
    if let Some(ds) = drift_slot {
        omega_b[idx(ds, ds, q_b)] = 0.1;
    }
    if let Some(ls) = logv_slot {
        omega_b[idx(ls, ls, q_b)] = 0.1;
    }

    // Output parameter order:
    //   1. Intercept means: mu_pop[0..d]
    //   2. Drift matrix A: row-major (for j in 0..d, for k in 0..d: A[j, k])
    //   3. Exogenous slopes beta_x (for j in 0..d, for k in 0..kx: beta[j, k])
    //   4. Diffusion covariance Q lower triangle (or logq_pop if use_rand_logv)
    //   5. If is_twolevel: Omega_B lower triangle
    let n_q_out = if use_rand_logv { 1 } else { d * (d + 1) / 2 };
    let n_om_b = if is_twolevel { q_b * (q_b + 1) / 2 } else { 0 };
    let n_params = d + d * d + d * kx + n_q_out + n_om_b;

    let keep_indices: Vec<usize> = ((warmup + 1)..=iter)
        .filter(|i| (i - warmup - 1) % thin == 0)
        .collect();
    let n_keep = keep_indices.len();
    let mut draws = vec![0.0; n_keep * n_params];
    let mut re_draws = vec![0.0; if is_twolevel { n_keep * g_cnt * q_b } else { 0 }];
    let mut fitted_accum = vec![0.0; n_obs * d];
    let mut q_dt_accum = vec![0.0; n_obs * d * d];
    let mut deviance_sum = 0.0;
    let mut keep_pos = 0usize;

    let p_pr = if prior_mode == 1 {
        1e-10
    } else {
        1.0 / (beta_sd * beta_sd)
    };
    let mut mh_scale_a = 0.08_f64;
    let mut mh_scale_q = 0.08_f64;
    let mut mh_scale_g = 0.15_f64;

    for step in 1..=iter {
        let om_b_inv = if is_twolevel {
            spd_invert(&omega_b, q_b)
        } else {
            Vec::new()
        };

        // 1. Impute any missing y_{g,i} conditional on exact CT transition
        if any_miss {
            for g in 0..g_cnt {
                let st = offsets[g];
                let en = offsets[g + 1];
                let tg = en - st;
                let has_m = (st..en).any(|r| (0..d).any(|j| miss_y[idx(r, j, n_obs)]));
                if !has_m || tg == 0 {
                    continue;
                }
                let a_g = if let Some(ds) = drift_slot {
                    vec![theta_b[idx(g, ds, g_cnt)].min(-1e-4)]
                } else {
                    drift_a.clone()
                };
                let q_g = if let Some(ls) = logv_slot {
                    vec![theta_b[idx(g, ls, g_cnt)].clamp(-8.0, 8.0).exp()]
                } else {
                    q_mat.clone()
                };
                let Some(sig_inf) = solve_lyapunov_ct(&a_g, &q_g, d) else {
                    continue;
                };
                for t_loc in 0..tg {
                    let r = st + t_loc;
                    let any_j_m = (0..d).any(|j| miss_y[idx(r, j, n_obs)]);
                    if !any_j_m {
                        continue;
                    }
                    let mut mean_r = vec![0.0; d];
                    for j in 0..d {
                        mean_r[j] = if is_twolevel {
                            theta_b[idx(g, j, g_cnt)]
                        } else {
                            mu_pop[j]
                        };
                        for k in 0..kx {
                            mean_r[j] += beta_x[idx(j, k, d)] * x_in[idx(r, k, n_obs)];
                        }
                    }
                    let q_cov = if t_loc == 0 {
                        sig_inf.clone()
                    } else {
                        let phi = mat_exp_scaled(&a_g, dt_vec[r], d);
                        let mut prev_dev = vec![0.0; d];
                        for j in 0..d {
                            let mut mj = if is_twolevel {
                                theta_b[idx(g, j, g_cnt)]
                            } else {
                                mu_pop[j]
                            };
                            for k in 0..kx {
                                mj += beta_x[idx(j, k, d)] * x_in[idx(r - 1, k, n_obs)];
                            }
                            prev_dev[j] = y_work[idx(r - 1, j, n_obs)] - mj;
                        }
                        for j in 0..d {
                            for k in 0..d {
                                mean_r[j] += phi[idx(j, k, d)] * prev_dev[k];
                            }
                        }
                        ct_step_cov(&phi, &sig_inf, d)
                    };
                    for j in 0..d {
                        if miss_y[idx(r, j, n_obs)] {
                            y_work[idx(r, j, n_obs)] =
                                mean_r[j] + q_cov[idx(j, j, d)].max(1e-6).sqrt() * rng.rnorm();
                        }
                    }
                }
            }
        }

        // 2. Exact Conjugate Gaussian Gibbs Update for Intercepts (mu_g if two-level, else mu_pop)
        if is_twolevel {
            for g in 0..g_cnt {
                let st = offsets[g];
                let en = offsets[g + 1];
                let tg = en - st;
                let a_g = if let Some(ds) = drift_slot {
                    vec![theta_b[idx(g, ds, g_cnt)].min(-1e-4)]
                } else {
                    drift_a.clone()
                };
                let q_g = if let Some(ls) = logv_slot {
                    vec![theta_b[idx(g, ls, g_cnt)].clamp(-8.0, 8.0).exp()]
                } else {
                    q_mat.clone()
                };
                let Some(sig_inf) = solve_lyapunov_ct(&a_g, &q_g, d) else {
                    continue;
                };
                let sig_inf_inv = spd_invert(&sig_inf, d);

                let mut pr = vec![0.0; d * d];
                let mut inf = vec![0.0; d];
                for ca in 0..d {
                    for ra in 0..d {
                        pr[idx(ra, ca, d)] = om_b_inv[idx(ra, ca, q_b)];
                    }
                    let mut rhs = 0.0;
                    for ra in 0..d {
                        rhs += om_b_inv[idx(ca, ra, q_b)] * mu_pop[ra];
                    }
                    if let Some(ds) = drift_slot {
                        rhs -= om_b_inv[idx(ca, ds, q_b)]
                            * (theta_b[idx(g, ds, g_cnt)] - drift_a[0]);
                    }
                    if let Some(ls) = logv_slot {
                        rhs -= om_b_inv[idx(ca, ls, q_b)]
                            * (theta_b[idx(g, ls, g_cnt)] - logq_pop);
                    }
                    inf[ca] = rhs;
                }

                // t_loc == 0 stationary contribution
                if tg > 0 {
                    let mut r0 = vec![0.0; d];
                    for j in 0..d {
                        r0[j] = y_work[idx(st, j, n_obs)];
                        for k in 0..kx {
                            r0[j] -= beta_x[idx(j, k, d)] * x_in[idx(st, k, n_obs)];
                        }
                    }
                    for ca in 0..d {
                        for ra in 0..d {
                            let w = sig_inf_inv[idx(ra, ca, d)];
                            pr[idx(ra, ca, d)] += w;
                            inf[ra] += w * r0[ca];
                        }
                    }
                }
                // t_loc >= 1 transitions: (y_t - X_t beta) - Phi (y_{t-1} - X_{t-1} beta) = (I - Phi) mu_g + e_t
                for t_loc in 1..tg {
                    let r = st + t_loc;
                    let phi = mat_exp_scaled(&a_g, dt_vec[r], d);
                    let q_dt = ct_step_cov(&phi, &sig_inf, d);
                    let q_inv = spd_invert(&q_dt, d);
                    let mut c_mat = eye(d, 1.0);
                    for i in 0..(d * d) {
                        c_mat[i] -= phi[i];
                    }
                    let mut y_cur = vec![0.0; d];
                    let mut y_prv = vec![0.0; d];
                    for j in 0..d {
                        y_cur[j] = y_work[idx(r, j, n_obs)];
                        y_prv[j] = y_work[idx(r - 1, j, n_obs)];
                        for k in 0..kx {
                            y_cur[j] -= beta_x[idx(j, k, d)] * x_in[idx(r, k, n_obs)];
                            y_prv[j] -= beta_x[idx(j, k, d)] * x_in[idx(r - 1, k, n_obs)];
                        }
                    }
                    let mut rem = y_cur;
                    for j in 0..d {
                        for k in 0..d {
                            rem[j] -= phi[idx(j, k, d)] * y_prv[k];
                        }
                    }
                    // Add C^T Q_inv C to pr and C^T Q_inv rem to inf
                    let mut qi_c = vec![0.0; d * d];
                    let mut qi_r = vec![0.0; d];
                    for ca in 0..d {
                        for ra in 0..d {
                            let mut s = 0.0;
                            for k in 0..d {
                                s += q_inv[idx(ra, k, d)] * c_mat[idx(k, ca, d)];
                            }
                            qi_c[idx(ra, ca, d)] = s;
                        }
                        let mut sr = 0.0;
                        for k in 0..d {
                            sr += q_inv[idx(ca, k, d)] * rem[k];
                        }
                        qi_r[ca] = sr;
                    }
                    for ca in 0..d {
                        let mut si = 0.0;
                        for k in 0..d {
                            si += c_mat[idx(k, ca, d)] * qi_r[k];
                        }
                        inf[ca] += si;
                        for ra in 0..d {
                            let mut sp = 0.0;
                            for k in 0..d {
                                sp += c_mat[idx(k, ra, d)] * qi_c[idx(k, ca, d)];
                            }
                            pr[idx(ra, ca, d)] += sp;
                        }
                    }
                }
                let mu_draw = rmvnorm_precision(&mut rng, &pr, &inf, d);
                for j in 0..d {
                    theta_b[idx(g, j, g_cnt)] = mu_draw[j];
                }
            }
        } else {
            // N=1 exact conjugate Gaussian draw of mu_pop
            if let Some(sig_inf) = solve_lyapunov_ct(&drift_a, &q_mat, d) {
                let sig_inf_inv = spd_invert(&sig_inf, d);
                let mut pr = eye(d, p_pr);
                let mut inf = vec![0.0; d];
                if n_obs > 0 {
                    let mut r0 = vec![0.0; d];
                    for j in 0..d {
                        r0[j] = y_work[idx(0, j, n_obs)];
                        for k in 0..kx {
                            r0[j] -= beta_x[idx(j, k, d)] * x_in[idx(0, k, n_obs)];
                        }
                    }
                    for ca in 0..d {
                        for ra in 0..d {
                            let w = sig_inf_inv[idx(ra, ca, d)];
                            pr[idx(ra, ca, d)] += w;
                            inf[ra] += w * r0[ca];
                        }
                    }
                }
                for r in 1..n_obs {
                    let phi = mat_exp_scaled(&drift_a, dt_vec[r], d);
                    let q_dt = ct_step_cov(&phi, &sig_inf, d);
                    let q_inv = spd_invert(&q_dt, d);
                    let mut c_mat = eye(d, 1.0);
                    for i in 0..(d * d) {
                        c_mat[i] -= phi[i];
                    }
                    let mut y_cur = vec![0.0; d];
                    let mut y_prv = vec![0.0; d];
                    for j in 0..d {
                        y_cur[j] = y_work[idx(r, j, n_obs)];
                        y_prv[j] = y_work[idx(r - 1, j, n_obs)];
                        for k in 0..kx {
                            y_cur[j] -= beta_x[idx(j, k, d)] * x_in[idx(r, k, n_obs)];
                            y_prv[j] -= beta_x[idx(j, k, d)] * x_in[idx(r - 1, k, n_obs)];
                        }
                    }
                    let mut rem = y_cur;
                    for j in 0..d {
                        for k in 0..d {
                            rem[j] -= phi[idx(j, k, d)] * y_prv[k];
                        }
                    }
                    let mut qi_c = vec![0.0; d * d];
                    let mut qi_r = vec![0.0; d];
                    for ca in 0..d {
                        for ra in 0..d {
                            let mut s = 0.0;
                            for k in 0..d {
                                s += q_inv[idx(ra, k, d)] * c_mat[idx(k, ca, d)];
                            }
                            qi_c[idx(ra, ca, d)] = s;
                        }
                        let mut sr = 0.0;
                        for k in 0..d {
                            sr += q_inv[idx(ca, k, d)] * rem[k];
                        }
                        qi_r[ca] = sr;
                    }
                    for ca in 0..d {
                        let mut si = 0.0;
                        for k in 0..d {
                            si += c_mat[idx(k, ca, d)] * qi_r[k];
                        }
                        inf[ca] += si;
                        for ra in 0..d {
                            let mut sp = 0.0;
                            for k in 0..d {
                                sp += c_mat[idx(k, ra, d)] * qi_c[idx(k, ca, d)];
                            }
                            pr[idx(ra, ca, d)] += sp;
                        }
                    }
                }
                mu_pop = rmvnorm_precision(&mut rng, &pr, &inf, d);
            }
        }

        // 3. Exact Conjugate Gaussian Gibbs Update for Exogenous Slopes beta_x (if kx > 0, CT-RDSEM)
        if kx > 0 {
            let n_bx = d * kx;
            let mut pr = eye(n_bx, p_pr);
            let mut inf = vec![0.0; n_bx];
            for g in 0..g_cnt {
                let st = offsets[g];
                let en = offsets[g + 1];
                let tg = en - st;
                let a_g = if let Some(ds) = drift_slot {
                    vec![theta_b[idx(g, ds, g_cnt)].min(-1e-4)]
                } else {
                    drift_a.clone()
                };
                let q_g = if let Some(ls) = logv_slot {
                    vec![theta_b[idx(g, ls, g_cnt)].clamp(-8.0, 8.0).exp()]
                } else {
                    q_mat.clone()
                };
                let Some(sig_inf) = solve_lyapunov_ct(&a_g, &q_g, d) else {
                    continue;
                };
                let sig_inf_inv = spd_invert(&sig_inf, d);
                let mu_g: Vec<f64> = (0..d)
                    .map(|j| {
                        if is_twolevel {
                            theta_b[idx(g, j, g_cnt)]
                        } else {
                            mu_pop[j]
                        }
                    })
                    .collect();

                // t_loc == 0
                if tg > 0 {
                    let mut y0 = vec![0.0; d];
                    for j in 0..d {
                        y0[j] = y_work[idx(st, j, n_obs)] - mu_g[j];
                    }
                    for j in 0..d {
                        for k in 0..kx {
                            let xjk = x_in[idx(st, k, n_obs)];
                            let p_idx = j * kx + k;
                            for j2 in 0..d {
                                let w = sig_inf_inv[idx(j, j2, d)];
                                inf[p_idx] += w * xjk * y0[j2];
                                for k2 in 0..kx {
                                    pr[idx(p_idx, j2 * kx + k2, n_bx)] +=
                                        w * xjk * x_in[idx(st, k2, n_obs)];
                                }
                            }
                        }
                    }
                }
                // t_loc >= 1: (y_t - mu_g) - Phi (y_{t-1} - mu_g) = X_t beta - Phi X_{t-1} beta + e_t
                for t_loc in 1..tg {
                    let r = st + t_loc;
                    let phi = mat_exp_scaled(&a_g, dt_vec[r], d);
                    let q_dt = ct_step_cov(&phi, &sig_inf, d);
                    let q_inv = spd_invert(&q_dt, d);
                    let mut rem = vec![0.0; d];
                    for j in 0..d {
                        rem[j] = y_work[idx(r, j, n_obs)] - mu_g[j];
                        for m_c in 0..d {
                            rem[j] -= phi[idx(j, m_c, d)] * (y_work[idx(r - 1, m_c, n_obs)] - mu_g[m_c]);
                        }
                    }
                    // Design matrix H (d x n_bx): row j, col (m_c * kx + k) = delta_{j,m_c} x_t[k] - phi[j, m_c] x_{t-1}[k]
                    let mut h_mat = vec![0.0; d * n_bx];
                    for j in 0..d {
                        for m_c in 0..d {
                            let ph = phi[idx(j, m_c, d)];
                            for k in 0..kx {
                                let col = m_c * kx + k;
                                let mut v = -ph * x_in[idx(r - 1, k, n_obs)];
                                if j == m_c {
                                    v += x_in[idx(r, k, n_obs)];
                                }
                                h_mat[idx(j, col, d)] = v;
                            }
                        }
                    }
                    for a in 0..n_bx {
                        for j in 0..d {
                            let hja = h_mat[idx(j, a, d)];
                            if hja.abs() < 1e-14 {
                                continue;
                            }
                            for j2 in 0..d {
                                let w = q_inv[idx(j, j2, d)];
                                inf[a] += hja * w * rem[j2];
                                for b in 0..n_bx {
                                    let hj2b = h_mat[idx(j2, b, d)];
                                    if hj2b.abs() > 1e-14 {
                                        pr[idx(a, b, n_bx)] += hja * w * hj2b;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let bx_draw = rmvnorm_precision(&mut rng, &pr, &inf, n_bx);
            for j in 0..d {
                for k in 0..kx {
                    beta_x[idx(j, k, d)] = bx_draw[j * kx + k];
                }
            }
        }

        // 4. Update Cluster Random Drift a_g and/or Random Log-Diffusion logq_g (if univariate two-level)
        if use_rand_drift || use_rand_logv {
            let mut acc_cnt = 0usize;
            let mut prop_cnt = 0usize;
            for g in 0..g_cnt {
                let st = offsets[g];
                let en = offsets[g + 1];
                let tg = en - st;
                let mug = theta_b[idx(g, 0, g_cnt)];
                let mut cur_a = if let Some(ds) = drift_slot {
                    theta_b[idx(g, ds, g_cnt)]
                } else {
                    drift_a[0]
                };
                let mut cur_lq = if let Some(ls) = logv_slot {
                    theta_b[idx(g, ls, g_cnt)]
                } else {
                    2.0 * lq_diag[0]
                };
                let prior_lp = |av: f64, lqv: f64| -> f64 {
                    let mut z = vec![0.0; q_b];
                    z[0] = mug - mu_pop[0];
                    if let Some(ds) = drift_slot {
                        z[ds] = av - drift_a[0];
                    }
                    if let Some(ls) = logv_slot {
                        z[ls] = lqv - logq_pop;
                    }
                    let mut qf = 0.0;
                    for ca in 0..q_b {
                        for ra in 0..q_b {
                            qf += z[ra] * om_b_inv[idx(ra, ca, q_b)] * z[ca];
                        }
                    }
                    -0.5 * qf
                };
                let mut cur_lp =
                    ct_1d_cluster_loglik(&y_work, x_in, &dt_vec, st, tg, n_obs, kx, mug, &beta_x, cur_a, cur_lq.exp())
                        + prior_lp(cur_a, cur_lq);
                for _ in 0..2 {
                    let prop_a = if use_rand_drift {
                        cur_a + mh_scale_g * rng.rnorm()
                    } else {
                        cur_a
                    };
                    let prop_lq = if use_rand_logv {
                        cur_lq + mh_scale_g * rng.rnorm()
                    } else {
                        cur_lq
                    };
                    prop_cnt += 1;
                    if prop_a < -1e-4 && prop_a > -20.0 && prop_lq > -8.0 && prop_lq < 8.0 {
                        let prop_lp = ct_1d_cluster_loglik(
                            &y_work,
                            x_in,
                            &dt_vec,
                            st,
                            tg,
                            n_obs,
                            kx,
                            mug,
                            &beta_x,
                            prop_a,
                            prop_lq.exp(),
                        ) + prior_lp(prop_a, prop_lq);
                        let acc = prop_lp - cur_lp;
                        if acc.is_finite() && (acc >= 0.0 || rng.runif().ln() < acc) {
                            cur_a = prop_a;
                            cur_lq = prop_lq;
                            cur_lp = prop_lp;
                            acc_cnt += 1;
                        }
                    }
                }
                if let Some(ds) = drift_slot {
                    theta_b[idx(g, ds, g_cnt)] = cur_a;
                }
                if let Some(ls) = logv_slot {
                    theta_b[idx(g, ls, g_cnt)] = cur_lq;
                }
            }
            if step <= warmup && prop_cnt > 0 {
                let rate = acc_cnt as f64 / prop_cnt as f64;
                let eta = 0.05 / (1.0 + step as f64 / 100.0);
                mh_scale_g = (mh_scale_g * (eta * (rate - 0.30)).exp()).clamp(0.01, 1.5);
            }
        }

        // 5. Update Global Drift Matrix A (if !use_rand_drift) and Global Diffusion Q (if !use_rand_logv) via MH
        if !use_rand_drift || !use_rand_logv {
            let eval_global_ll = |a_m: &[f64], q_m: &[f64]| -> Option<f64> {
                let mut total_ll = 0.0;
                if d == 1 {
                    let q_fixed = q_m[0];
                    let a_fixed = a_m[0];
                    if !(-25.0..-1e-4).contains(&a_fixed) || q_fixed <= 1e-8 {
                        return None;
                    }
                    for g in 0..g_cnt {
                        let st = offsets[g];
                        let tg = offsets[g + 1] - st;
                        let mug = if is_twolevel {
                            theta_b[idx(g, 0, g_cnt)]
                        } else {
                            mu_pop[0]
                        };
                        let ag = if let Some(ds) = drift_slot {
                            theta_b[idx(g, ds, g_cnt)]
                        } else {
                            a_fixed
                        };
                        let qg = if let Some(ls) = logv_slot {
                            theta_b[idx(g, ls, g_cnt)].clamp(-8.0, 8.0).exp()
                        } else {
                            q_fixed
                        };
                        let ll = ct_1d_cluster_loglik(
                            &y_work, x_in, &dt_vec, st, tg, n_obs, kx, mug, &beta_x, ag, qg,
                        );
                        if !ll.is_finite() {
                            return None;
                        }
                        total_ll += ll;
                    }
                    return Some(total_ll);
                }
                let sig_inf = solve_lyapunov_ct(a_m, q_m, d)?;
                let sig_inf_inv = spd_invert(&sig_inf, d);
                let sig_inf_ld = log_det_spd(&sig_inf, d)?;
                let log2pi = (2.0 * PI).ln();
                let mut prev_dev = vec![0.0; d];
                let mut cur_dev = vec![0.0; d];
                let mut err = vec![0.0; d];
                for g in 0..g_cnt {
                    let st = offsets[g];
                    let tg = offsets[g + 1] - st;
                    if tg == 0 {
                        continue;
                    }
                    let mu_g: Vec<f64> = (0..d)
                        .map(|j| {
                            if is_twolevel {
                                theta_b[idx(g, j, g_cnt)]
                            } else {
                                mu_pop[j]
                            }
                        })
                        .collect();
                    for j in 0..d {
                        let mut v = y_work[idx(st, j, n_obs)] - mu_g[j];
                        for k in 0..kx {
                            v -= beta_x[idx(j, k, d)] * x_in[idx(st, k, n_obs)];
                        }
                        prev_dev[j] = v;
                    }
                    let mut q0 = 0.0;
                    for ca in 0..d {
                        for ra in 0..d {
                            q0 += prev_dev[ra] * sig_inf_inv[idx(ra, ca, d)] * prev_dev[ca];
                        }
                    }
                    total_ll += -0.5 * (d as f64 * log2pi + sig_inf_ld + q0);
                    for t_loc in 1..tg {
                        let r = st + t_loc;
                        let phi = mat_exp_scaled(a_m, dt_vec[r], d);
                        let q_dt = ct_step_cov(&phi, &sig_inf, d);
                        let q_ld = log_det_spd(&q_dt, d)?;
                        let q_inv = spd_invert(&q_dt, d);
                        for j in 0..d {
                            let mut v = y_work[idx(r, j, n_obs)] - mu_g[j];
                            for k in 0..kx {
                                v -= beta_x[idx(j, k, d)] * x_in[idx(r, k, n_obs)];
                            }
                            cur_dev[j] = v;
                        }
                        for j in 0..d {
                            let mut pred = 0.0;
                            for k in 0..d {
                                pred += phi[idx(j, k, d)] * prev_dev[k];
                            }
                            err[j] = cur_dev[j] - pred;
                        }
                        let mut quad = 0.0;
                        for ca in 0..d {
                            for ra in 0..d {
                                quad += err[ra] * q_inv[idx(ra, ca, d)] * err[ca];
                            }
                        }
                        total_ll += -0.5 * (d as f64 * log2pi + q_ld + quad);
                        prev_dev.copy_from_slice(&cur_dev);
                    }
                }
                if total_ll.is_finite() {
                    Some(total_ll)
                } else {
                    None
                }
            };

            let prior_a = |a_m: &[f64]| -> f64 {
                let mut lp = 0.0;
                for &v in a_m {
                    lp += -0.5 * p_pr * v * v;
                }
                lp
            };
            // Jacobian of Q = L L^T with diag(L_jj) = exp(l_j): |dQ / d(l_diag, l_off)| = 2^d prod_j exp((d - j + 1) l_j)
            let prior_q_jac = |ld: &[f64], lo: &[f64]| -> f64 {
                let mut lp = 0.0;
                for (j, &ld_j) in ld.iter().enumerate().take(d) {
                    let jac_pow = (d - j + 1) as f64;
                    lp += jac_pow * ld_j;
                    if prior_mode != 1 {
                        lp += -0.5 * 0.1 * ld_j * ld_j;
                    }
                }
                for &v in lo {
                    if prior_mode != 1 {
                        lp += -0.5 * 0.1 * v * v;
                    }
                }
                lp
            };

            let mut cur_ll = eval_global_ll(&drift_a, &q_mat).unwrap_or(f64::NEG_INFINITY);

            // 5a. MH update of Drift Matrix A (element-wise + joint block)
            if !use_rand_drift {
                let mut acc_a = 0usize;
                let mut prop_a_cnt = 0usize;
                for _ in 0..2 {
                    let mut cand_a = drift_a.clone();
                    for v in &mut cand_a {
                        *v += mh_scale_a * rng.rnorm();
                    }
                    let diag_ok = (0..d).all(|j| cand_a[idx(j, j, d)] < -1e-4 && cand_a[idx(j, j, d)] > -25.0);
                    let off_ok = cand_a.iter().all(|v| v.abs() < 20.0);
                    prop_a_cnt += 1;
                    if diag_ok && off_ok {
                        if let Some(prop_ll) = eval_global_ll(&cand_a, &q_mat) {
                            let acc = (prop_ll + prior_a(&cand_a)) - (cur_ll + prior_a(&drift_a));
                            if acc.is_finite() && (acc >= 0.0 || rng.runif().ln() < acc) {
                                drift_a = cand_a;
                                cur_ll = prop_ll;
                                acc_a += 1;
                            }
                        }
                    }
                }
                if step <= warmup && prop_a_cnt > 0 {
                    let rate = acc_a as f64 / prop_a_cnt as f64;
                    let eta = 0.05 / (1.0 + step as f64 / 100.0);
                    mh_scale_a = (mh_scale_a * (eta * (rate - 0.30)).exp()).clamp(0.005, 1.0);
                }
            }

            // 5b. MH update of Diffusion Matrix Q via Cholesky (lq_diag, lq_off) + joint (A, Q) ridge move
            if !use_rand_logv {
                let mut acc_q = 0usize;
                let mut prop_q_cnt = 0usize;
                for mv in 0..2 {
                    let mut cand_ld = lq_diag.clone();
                    let mut cand_lo = lq_off.clone();
                    let mut cand_a = drift_a.clone();
                    for v in &mut cand_ld {
                        *v += mh_scale_q * rng.rnorm();
                    }
                    for v in &mut cand_lo {
                        *v += mh_scale_q * rng.rnorm();
                    }
                    // Joint correlated (A, Q) move on second step to traverse the stationary variance ridge Q / (-2A) = const!
                    if mv == 1 && !use_rand_drift {
                        for j in 0..d {
                            let dl = cand_ld[j] - lq_diag[j];
                            cand_a[idx(j, j, d)] = drift_a[idx(j, j, d)] * (2.0 * dl).exp();
                        }
                    }
                    let ld_ok = cand_ld.iter().all(|&v| v > -6.0 && v < 6.0);
                    let a_ok = (0..d).all(|j| cand_a[idx(j, j, d)] < -1e-4 && cand_a[idx(j, j, d)] > -25.0);
                    prop_q_cnt += 1;
                    if ld_ok && a_ok {
                        let cand_q = build_spd_from_chol(&cand_ld, &cand_lo, d);
                        if let Some(prop_ll) = eval_global_ll(&cand_a, &cand_q) {
                            // If mv == 1 (deterministic rescaling A_jj' = A_jj * exp(2 dl)), Jacobian adds sum_j 2 dl_j
                            let jac_a = if mv == 1 && !use_rand_drift {
                                (0..d).map(|j| 2.0 * (cand_ld[j] - lq_diag[j])).sum::<f64>()
                            } else {
                                0.0
                            };
                            let acc = (prop_ll
                                + prior_a(&cand_a)
                                + prior_q_jac(&cand_ld, &cand_lo))
                                - (cur_ll + prior_a(&drift_a) + prior_q_jac(&lq_diag, &lq_off))
                                + jac_a;
                            if acc.is_finite() && (acc >= 0.0 || rng.runif().ln() < acc) {
                                lq_diag = cand_ld;
                                lq_off = cand_lo;
                                drift_a = cand_a;
                                q_mat = cand_q;
                                cur_ll = prop_ll;
                                acc_q += 1;
                            }
                        }
                    }
                }
                if step <= warmup && prop_q_cnt > 0 {
                    let rate = acc_q as f64 / prop_q_cnt as f64;
                    let eta = 0.05 / (1.0 + step as f64 / 100.0);
                    mh_scale_q = (mh_scale_q * (eta * (rate - 0.30)).exp()).clamp(0.005, 1.0);
                }
            }
        }

        // 6. Update Two-Level Population Means & Omega_B
        if is_twolevel {
            let mut pr_m = vec![0.0; q_b * q_b];
            let mut inf_m = vec![0.0; q_b];
            let mut sum_th = vec![0.0; q_b];
            for g in 0..g_cnt {
                for q in 0..q_b {
                    sum_th[q] += theta_b[idx(g, q, g_cnt)];
                }
            }
            for ca in 0..q_b {
                for ra in 0..q_b {
                    pr_m[idx(ra, ca, q_b)] = g_cnt as f64 * om_b_inv[idx(ra, ca, q_b)];
                    inf_m[ra] += om_b_inv[idx(ra, ca, q_b)] * sum_th[ca];
                }
                pr_m[idx(ca, ca, q_b)] += p_pr;
            }
            let m_draw = rmvnorm_precision(&mut rng, &pr_m, &inf_m, q_b);
            mu_pop[..d].copy_from_slice(&m_draw[..d]);
            if let Some(ds) = drift_slot {
                if m_draw[ds] < -1e-4 && m_draw[ds] > -20.0 {
                    drift_a[0] = m_draw[ds];
                }
            }
            if let Some(ls) = logv_slot {
                logq_pop = m_draw[ls].clamp(-8.0, 8.0);
            }

            let psi0_d = if prior_mode == 1 { 1e-5 } else { 0.05 };
            let mut sb = eye(q_b, psi0_d);
            let mut mean_vec = vec![0.0; q_b];
            mean_vec[..d].copy_from_slice(&mu_pop[..d]);
            if let Some(ds) = drift_slot {
                mean_vec[ds] = drift_a[0];
            }
            if let Some(ls) = logv_slot {
                mean_vec[ls] = logq_pop;
            }
            for g in 0..g_cnt {
                for ca in 0..q_b {
                    let dc = theta_b[idx(g, ca, g_cnt)] - mean_vec[ca];
                    for ra in 0..q_b {
                        let dr = theta_b[idx(g, ra, g_cnt)] - mean_vec[ra];
                        sb[idx(ra, ca, q_b)] += dr * dc;
                    }
                }
            }
            let df_b = if prior_mode == 1 {
                (g_cnt as f64 - q_b as f64 - 1.0).max(q_b as f64 + 1.0)
            } else {
                g_cnt as f64 + q_b as f64 + 2.0
            };
            omega_b = rinv_wishart(&mut rng, df_b, &sb, q_b);
        }

        // 7. Store retained draw & exact DIC
        if keep_pos < n_keep && step == keep_indices[keep_pos] {
            let mut cur = 0usize;
            for &v in &mu_pop {
                draws[idx(keep_pos, cur, n_keep)] = v;
                cur += 1;
            }
            for j in 0..d {
                for k in 0..d {
                    draws[idx(keep_pos, cur, n_keep)] = drift_a[idx(j, k, d)];
                    cur += 1;
                }
            }
            for j in 0..d {
                for k in 0..kx {
                    draws[idx(keep_pos, cur, n_keep)] = beta_x[idx(j, k, d)];
                    cur += 1;
                }
            }
            if use_rand_logv {
                draws[idx(keep_pos, cur, n_keep)] = logq_pop;
                cur += 1;
            } else {
                for ca in 0..d {
                    for ra in ca..d {
                        draws[idx(keep_pos, cur, n_keep)] = q_mat[idx(ra, ca, d)];
                        cur += 1;
                    }
                }
            }
            if is_twolevel {
                for ca in 0..q_b {
                    for ra in ca..q_b {
                        draws[idx(keep_pos, cur, n_keep)] = omega_b[idx(ra, ca, q_b)];
                        cur += 1;
                    }
                }
                for q in 0..q_b {
                    for g in 0..g_cnt {
                        re_draws[keep_pos + n_keep * (g + g_cnt * q)] = theta_b[idx(g, q, g_cnt)];
                    }
                }
            }

            // Compute conditional log-likelihood & fitted values
            let mut ll = 0.0;
            let log2pi = (2.0 * PI).ln();
            for g in 0..g_cnt {
                let st = offsets[g];
                let tg = offsets[g + 1] - st;
                let a_g = if let Some(ds) = drift_slot {
                    vec![theta_b[idx(g, ds, g_cnt)].min(-1e-4)]
                } else {
                    drift_a.clone()
                };
                let q_g = if let Some(ls) = logv_slot {
                    vec![theta_b[idx(g, ls, g_cnt)].clamp(-8.0, 8.0).exp()]
                } else {
                    q_mat.clone()
                };
                let Some(sig_inf) = solve_lyapunov_ct(&a_g, &q_g, d) else {
                    continue;
                };
                let mu_g: Vec<f64> = (0..d)
                    .map(|j| {
                        if is_twolevel {
                            theta_b[idx(g, j, g_cnt)]
                        } else {
                            mu_pop[j]
                        }
                    })
                    .collect();
                for t_loc in 0..tg {
                    let r = st + t_loc;
                    let mut fv = vec![0.0; d];
                    for j in 0..d {
                        fv[j] = mu_g[j];
                        for k in 0..kx {
                            fv[j] += beta_x[idx(j, k, d)] * x_in[idx(r, k, n_obs)];
                        }
                    }
                    let q_cov = if t_loc == 0 {
                        sig_inf.clone()
                    } else {
                        let phi = mat_exp_scaled(&a_g, dt_vec[r], d);
                        for j in 0..d {
                            for k in 0..d {
                                let mut prev_m = mu_g[k];
                                for kk in 0..kx {
                                    prev_m += beta_x[idx(k, kk, d)] * x_in[idx(r - 1, kk, n_obs)];
                                }
                                fv[j] += phi[idx(j, k, d)] * (y_work[idx(r - 1, k, n_obs)] - prev_m);
                            }
                        }
                        ct_step_cov(&phi, &sig_inf, d)
                    };
                    for j in 0..d {
                        fitted_accum[idx(r, j, n_obs)] += fv[j];
                    }
                    for i in 0..(d * d) {
                        q_dt_accum[r * d * d + i] += q_cov[i];
                    }
                    let all_obs = (0..d).all(|j| !miss_y[idx(r, j, n_obs)]);
                    if all_obs {
                        if let Some(ld) = log_det_spd(&q_cov, d) {
                            let qi = spd_invert(&q_cov, d);
                            let mut quad = 0.0;
                            for ca in 0..d {
                                let ec = y_work[idx(r, ca, n_obs)] - fv[ca];
                                for ra in 0..d {
                                    let er = y_work[idx(r, ra, n_obs)] - fv[ra];
                                    quad += er * qi[idx(ra, ca, d)] * ec;
                                }
                            }
                            ll += -0.5 * (d as f64 * log2pi + ld + quad);
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
    let log2pi = (2.0 * PI).ln();
    for r in 0..n_obs {
        let all_obs = (0..d).all(|j| !miss_y[idx(r, j, n_obs)]);
        if all_obs {
            let q_bar: Vec<f64> = (0..(d * d))
                .map(|i| q_dt_accum[r * d * d + i] * inv_keep)
                .collect();
            if let Some(ld) = log_det_spd(&q_bar, d) {
                let qi = spd_invert(&q_bar, d);
                let mut quad = 0.0;
                for ca in 0..d {
                    let ec = y_work[idx(r, ca, n_obs)] - fitted_mean[idx(r, ca, n_obs)];
                    for ra in 0..d {
                        let er = y_work[idx(r, ra, n_obs)] - fitted_mean[idx(r, ra, n_obs)];
                        quad += er * qi[idx(ra, ca, d)] * ec;
                    }
                }
                ll_hat += -0.5 * (d as f64 * log2pi + ld + quad);
            }
        }
    }
    let pd = (d_bar - (-2.0 * ll_hat)).max(0.0);
    let dic = d_bar + pd;

    CtDsemOutput {
        draws,
        param_count: n_params,
        re_draws,
        fitted_mean,
        dic,
        pd,
    }
}

/// Exact scalar 1D Ornstein-Uhlenbeck log-likelihood for one cluster.
#[allow(clippy::too_many_arguments)]
fn ct_1d_cluster_loglik(
    y: &[f64],
    x: &[f64],
    dt: &[f64],
    st: usize,
    tg: usize,
    n_obs: usize,
    kx: usize,
    mu: f64,
    beta_x: &[f64],
    a: f64,
    q: f64,
) -> f64 {
    if tg == 0 || a >= -1e-6 || q <= 1e-10 {
        return f64::NEG_INFINITY;
    }
    let sig_inf = q / (-2.0 * a);
    let log2pi = (2.0 * PI).ln();
    let mut prev_dev = y[idx(st, 0, n_obs)] - mu;
    for k in 0..kx {
        prev_dev -= beta_x[idx(0, k, 1)] * x[idx(st, k, n_obs)];
    }
    let mut ll = -0.5 * (log2pi + sig_inf.ln() + prev_dev * prev_dev / sig_inf);
    for t_loc in 1..tg {
        let r = st + t_loc;
        let phi = (a * dt[r]).exp();
        let v_dt = (sig_inf * (1.0 - phi * phi)).max(1e-10);
        let mut cur_dev = y[idx(r, 0, n_obs)] - mu;
        for k in 0..kx {
            cur_dev -= beta_x[idx(0, k, 1)] * x[idx(r, k, n_obs)];
        }
        let err = cur_dev - phi * prev_dev;
        ll += -0.5 * (log2pi + v_dt.ln() + err * err / v_dt);
        prev_dev = cur_dev;
    }
    ll
}

fn build_spd_from_chol(ld: &[f64], lo: &[f64], d: usize) -> Vec<f64> {
    let mut l = vec![0.0; d * d];
    let mut k = 0usize;
    for col in 0..d {
        l[idx(col, col, d)] = ld[col].exp();
        for row in (col + 1)..d {
            l[idx(row, col, d)] = lo[k];
            k += 1;
        }
    }
    mat_mul_t(&l, &l, d, d, d)
}

/// Matrix exponential exp(A * dt) via scaling and squaring + [6/6] Pade approximation (Higham 2005).
pub fn mat_exp_scaled(a: &[f64], dt: f64, d: usize) -> Vec<f64> {
    if d == 1 {
        return vec![(a[0] * dt).exp()];
    }
    let mut m: Vec<f64> = a.iter().map(|&v| v * dt).collect();
    // Infinity norm
    let mut norm = 0.0_f64;
    for r in 0..d {
        let s: f64 = (0..d).map(|c| m[idx(r, c, d)].abs()).sum();
        if s > norm {
            norm = s;
        }
    }
    let s_pow = if norm > 0.5 {
        (norm / 0.5).log2().ceil().max(0.0) as u32
    } else {
        0
    };
    let scale = 2.0_f64.powi(-(s_pow as i32));
    for v in &mut m {
        *v *= scale;
    }

    // [6/6] Pade coefficients b_k = (12-k)! 6! / (12! k! (6-k)!)
    let b = [1.0, 0.5, 5.0 / 44.0, 1.0 / 66.0, 1.0 / 792.0, 1.0 / 15840.0, 1.0 / 665280.0];
    let id = eye(d, 1.0);
    let m2 = mat_mul(&m, &m, d, d, d);
    let m4 = mat_mul(&m2, &m2, d, d, d);
    let m6 = mat_mul(&m4, &m2, d, d, d);

    let mut u_inner = vec![0.0; d * d];
    let mut v_mat = vec![0.0; d * d];
    for i in 0..(d * d) {
        u_inner[i] = b[1] * id[i] + b[3] * m2[i] + b[5] * m4[i];
        v_mat[i] = b[0] * id[i] + b[2] * m2[i] + b[4] * m4[i] + b[6] * m6[i];
    }
    let u_mat = mat_mul(&m, &u_inner, d, d, d);

    let mut num = vec![0.0; d * d];
    let mut den = vec![0.0; d * d];
    for i in 0..(d * d) {
        num[i] = v_mat[i] + u_mat[i];
        den[i] = v_mat[i] - u_mat[i];
    }
    let mut res = solve_mat_system(&den, &num, d).unwrap_or_else(|| eye(d, 1.0));
    for _ in 0..s_pow {
        res = mat_mul(&res, &res, d, d, d);
    }
    res
}

/// Solve continuous-time Lyapunov equation A Sigma + Sigma A^T = -Q via exact Kronecker system
/// (I (x) A + A (x) I) vec(Sigma) = -vec(Q). Returns Some(Sigma) iff Sigma is strictly SPD
/// (by Lyapunov's theorem, this certifies A is strictly Hurwitz-stable for SPD Q!).
pub fn solve_lyapunov_ct(a: &[f64], q: &[f64], d: usize) -> Option<Vec<f64>> {
    if d == 1 {
        if a[0] >= -1e-6 || q[0] <= 1e-10 {
            return None;
        }
        return Some(vec![q[0] / (-2.0 * a[0])]);
    }
    let n = d * d;
    let mut k_mat = vec![0.0; n * n];
    let mut rhs = vec![0.0; n];
    // Column-major vec(X): entry (r, c) is at index c * d + r
    // (A X)_{r, c} = sum_k A_{r, k} X_{k, c} -> row (c*d + r), col (c*d + k) += A_{r, k}
    // (X A^T)_{r, c} = sum_k X_{r, k} A_{c, k} -> row (c*d + r), col (k*d + r) += A_{c, k}
    for c in 0..d {
        for r in 0..d {
            let row_id = c * d + r;
            rhs[row_id] = -q[idx(r, c, d)];
            for k in 0..d {
                k_mat[idx(row_id, c * d + k, n)] += a[idx(r, k, d)];
                k_mat[idx(row_id, k * d + r, n)] += a[idx(c, k, d)];
            }
        }
    }
    let mut sig = solve_linear_system(&k_mat, &rhs, n)?;
    // Symmetrize & verify SPD via Cholesky
    for c in 0..d {
        for r in (c + 1)..d {
            let avg = 0.5 * (sig[idx(r, c, d)] + sig[idx(c, r, d)]);
            sig[idx(r, c, d)] = avg;
            sig[idx(c, r, d)] = avg;
        }
    }
    log_det_spd(&sig, d)?;
    Some(sig)
}

fn ct_step_cov(phi: &[f64], sig_inf: &[f64], d: usize) -> Vec<f64> {
    if d == 1 {
        return vec![(sig_inf[0] * (1.0 - phi[0] * phi[0])).max(1e-10)];
    }
    let ps = mat_mul(phi, sig_inf, d, d, d);
    let psp = mat_mul_t(&ps, phi, d, d, d);
    let mut out = vec![0.0; d * d];
    for c in 0..d {
        for r in 0..d {
            out[idx(r, c, d)] = sig_inf[idx(r, c, d)] - psp[idx(r, c, d)];
        }
    }
    for c in 0..d {
        for r in (c + 1)..d {
            let avg = 0.5 * (out[idx(r, c, d)] + out[idx(c, r, d)]);
            out[idx(r, c, d)] = avg;
            out[idx(c, r, d)] = avg;
        }
        out[idx(c, c, d)] = out[idx(c, c, d)].max(1e-9);
    }
    out
}

fn log_det_spd(a: &[f64], n: usize) -> Option<f64> {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut s = a[idx(j, j, n)];
        for k in 0..j {
            s -= l[idx(j, k, n)] * l[idx(j, k, n)];
        }
        if !s.is_finite() || s <= 1e-12 {
            return None;
        }
        let d_val = s.sqrt();
        l[idx(j, j, n)] = d_val;
        for i in (j + 1)..n {
            let mut si = a[idx(i, j, n)];
            for k in 0..j {
                si -= l[idx(i, k, n)] * l[idx(j, k, n)];
            }
            l[idx(i, j, n)] = si / d_val;
        }
    }
    Some(2.0 * (0..n).map(|j| l[idx(j, j, n)].ln()).sum::<f64>())
}

fn solve_linear_system(a_in: &[f64], b_in: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = a_in.to_vec();
    let mut b = b_in.to_vec();
    for k in 0..n {
        let mut piv = k;
        let mut max_v = a[idx(k, k, n)].abs();
        for i in (k + 1)..n {
            let v = a[idx(i, k, n)].abs();
            if v > max_v {
                max_v = v;
                piv = i;
            }
        }
        if !max_v.is_finite() || max_v < 1e-12 {
            return None;
        }
        if piv != k {
            for j in k..n {
                let tmp = a[idx(k, j, n)];
                a[idx(k, j, n)] = a[idx(piv, j, n)];
                a[idx(piv, j, n)] = tmp;
            }
            b.swap(k, piv);
        }
        let diag = a[idx(k, k, n)];
        for i in (k + 1)..n {
            let f = a[idx(i, k, n)] / diag;
            a[idx(i, k, n)] = 0.0;
            for j in (k + 1)..n {
                a[idx(i, j, n)] -= f * a[idx(k, j, n)];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[idx(i, j, n)] * x[j];
        }
        x[i] = s / a[idx(i, i, n)];
        if !x[i].is_finite() {
            return None;
        }
    }
    Some(x)
}

fn solve_mat_system(a_in: &[f64], b_in: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut out = vec![0.0; n * n];
    for c in 0..n {
        let col_b: Vec<f64> = (0..n).map(|r| b_in[idx(r, c, n)]).collect();
        let col_x = solve_linear_system(a_in, &col_b, n)?;
        for r in 0..n {
            out[idx(r, c, n)] = col_x[r];
        }
    }
    Some(out)
}
