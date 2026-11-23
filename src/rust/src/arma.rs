use crate::linalg::{eye, idx, rinv_wishart, rmvnorm_precision, spd_invert};
use crate::rng::Rng;
use std::f64::consts::PI;

pub struct ArmaOutput {
    pub draws: Vec<f64>,
    pub param_count: usize,
    pub latent_draws: Vec<f64>, // n_keep x n_obs (if store_latent)
    pub re_draws: Vec<f64>,     // n_keep x g_cnt x q_b (if is_twolevel)
    pub fitted_mean: Vec<f64>,  // n_obs
    pub dic: f64,
    pub pd: f64,
}

/// Exact collapsed tridiagonal-Cholesky Gibbs/MH sampler for N=1 and Two-Level ARMA(1,1) DSEM:
///   y_{gt} = mu_g + y^w_{gt}
///   y^w_{gt} = s_g y^w_{g,t-1} + e_{gt} + theta * e_{g,t-1} + eps_{gt}
///   e_{gt} ~ N(0, sigma_e^2),  eps_{gt} ~ N(0, theta_y)
/// Integrates out all latent innovations e_{gt} analytically via exact O(T) Toeplitz tridiagonal
/// Cholesky when updating (mu_g, s_g, theta, sigma_e^2), eliminating Gibbs slow-mixing completely.
#[allow(clippy::too_many_arguments)]
pub fn mcmc_arma11(
    y_in: &[f64],            // length n_obs (contiguous by cluster)
    cluster_sizes: &[usize], // length g_cnt (1 for N=1, >=2 for two-level)
    random_ar: bool,         // true for random AR(1) slope s_g (e.g. Ex 9.33b)
    theta_y_fixed: f64,      // fixed observation variance (e.g. 0.01)
    prior_mode: i32,         // 0 = weakly_informative, 1 = diffuse
    ar_bound: f64,
    store_latent: bool,
    iter: usize,
    warmup: usize,
    thin: usize,
    seed: u64,
) -> ArmaOutput {
    let mut rng = Rng::from_seed(seed);
    let g_cnt = cluster_sizes.len();
    let is_twolevel = g_cnt > 1;
    let n_obs: usize = cluster_sizes.iter().sum();
    let theta_y = theta_y_fixed.max(1e-6);
    let ma_bound = ar_bound.min(0.98);

    let mut offsets = Vec::with_capacity(g_cnt + 1);
    offsets.push(0usize);
    for &sz in cluster_sizes {
        offsets.push(offsets.last().unwrap() + sz);
    }

    // Detect & initialize missing observations
    let mut y_work = y_in.to_vec();
    let mut miss_y = vec![false; n_obs];
    let mut any_miss = false;
    let mut y_sum = 0.0;
    let mut y_cnt = 0usize;
    for r in 0..n_obs {
        if y_in[r].is_finite() {
            y_sum += y_in[r];
            y_cnt += 1;
        } else {
            miss_y[r] = true;
            any_miss = true;
        }
    }
    let y_gm = if y_cnt > 0 { y_sum / y_cnt as f64 } else { 0.0 };
    for r in 0..n_obs {
        if miss_y[r] {
            y_work[r] = y_gm;
        }
    }

    // Parameters
    let mut theta_ma = -0.05_f64;
    let mut sigma2_e = 1.0_f64;
    let mut mu_pop = y_gm;
    let mut s_pop = 0.25_f64;
    let mut mu_g = vec![y_gm; g_cnt];
    let mut s_g = vec![0.25_f64; g_cnt];
    if is_twolevel {
        for g in 0..g_cnt {
            let st = offsets[g];
            let en = offsets[g + 1];
            let tg = en - st;
            if tg > 0 {
                let m_g = (st..en).map(|r| y_work[r]).sum::<f64>() / tg as f64;
                mu_g[g] = m_g;
            }
        }
    }
    let q_b = if !is_twolevel {
        0
    } else if random_ar {
        2
    } else {
        1
    };
    let mut omega_b = eye(q_b, 0.2);
    if q_b == 2 {
        omega_b[idx(1, 1, 2)] = 0.03;
    }

    // Parameter layout in `draws`:
    //   0: ma1 (theta_ma)
    //   1: var_e (sigma2_e)
    //   if !is_twolevel:
    //     2: ar1 (s_pop)
    //     3: mean_y (mu_pop)
    //   if is_twolevel && random_ar:
    //     2: mean_y (mu_pop)
    //     3: mean_s (s_pop)
    //     4: var_y (omega_b[0,0])
    //     5: cov_y_s (omega_b[1,0])
    //     6: var_s (omega_b[1,1])
    //   if is_twolevel && !random_ar:
    //     2: ar1 (s_pop)
    //     3: mean_y (mu_pop)
    //     4: var_y (omega_b[0,0])
    let n_params = if !is_twolevel {
        4
    } else if random_ar {
        7
    } else {
        5
    };

    let keep_indices: Vec<usize> = ((warmup + 1)..=iter)
        .filter(|i| (i - warmup - 1) % thin == 0)
        .collect();
    let n_keep = keep_indices.len();
    let mut draws = vec![0.0; n_keep * n_params];
    let mut latent_draws = vec![0.0; if store_latent { n_keep * n_obs } else { 0 }];
    let mut re_draws = vec![0.0; if is_twolevel { n_keep * g_cnt * q_b } else { 0 }];
    let mut fitted_accum = vec![0.0; n_obs];
    let mut mu_g_accum = vec![0.0; g_cnt];
    let mut s_g_accum = vec![0.0; g_cnt];
    let mut theta_ma_accum = 0.0;
    let mut sigma2_e_accum = 0.0;
    let mut deviance_sum = 0.0;
    let mut keep_pos = 0usize;

    let max_t = cluster_sizes.iter().copied().max().unwrap_or(1);
    let mut chol_l = vec![0.0; max_t];
    let mut chol_m = vec![0.0; max_t];
    let p_pr = if prior_mode == 1 { 1e-10 } else { 0.01 };

    let mut e_buf = vec![0.0; max_t];

    for step in 1..=iter {
        let om_b_inv = if is_twolevel {
            spd_invert(&omega_b, q_b)
        } else {
            Vec::new()
        };

        // Factorize tridiagonal Toeplitz Sigma_z for current (theta_ma, sigma2_e)
        factor_toeplitz_tridiag(
            (1.0 + theta_ma * theta_ma) * sigma2_e + theta_y,
            theta_ma * sigma2_e,
            max_t.saturating_sub(1),
            &mut chol_l,
            &mut chol_m,
        );

        // 1. Impute any missing y_{gt} conditional on sampled latent innovations e_{gt}
        if any_miss {
            for g in 0..g_cnt {
                let st = offsets[g];
                let en = offsets[g + 1];
                let tg = en - st;
                if tg < 2 {
                    continue;
                }
                let has_m = (st..en).any(|r| miss_y[r]);
                if !has_m {
                    continue;
                }
                let mug = mu_g[g];
                let sg = if is_twolevel && random_ar {
                    s_g[g]
                } else {
                    s_pop
                };
                sample_e_given_y(
                    &mut rng,
                    &y_work[st..en],
                    mug,
                    sg,
                    theta_ma,
                    sigma2_e,
                    theta_y,
                    &mut e_buf[..tg],
                );
                for t_loc in 0..tg {
                    let r = st + t_loc;
                    if miss_y[r] {
                        let pred = if t_loc == 0 {
                            mug
                        } else {
                            mug + sg * (y_work[r - 1] - mug)
                                + e_buf[t_loc]
                                + theta_ma * e_buf[t_loc - 1]
                        };
                        let sd = if t_loc == 0 {
                            (sigma2_e / (1.0 - sg * sg).max(0.05)).sqrt()
                        } else {
                            theta_y.sqrt()
                        };
                        y_work[r] = pred + sd * rng.rnorm();
                    }
                }
            }
        }

        // 2. Exact Collapsed Gibbs updates for cluster random effects (mu_g, s_g) (or global mu_pop, s_pop)
        if is_twolevel {
            for g in 0..g_cnt {
                let st = offsets[g];
                let en = offsets[g + 1];
                let tg = en - st;
                if tg < 2 {
                    continue;
                }
                let n = tg - 1;
                let sg = if random_ar { s_g[g] } else { s_pop };

                // 2a. Exact Gaussian draw of mu_g | y_g, s_g, theta_ma, sigma2_e (integrating out e_g!)
                let mut pr_mu = om_b_inv[idx(0, 0, q_b)];
                let mut inf_mu = pr_mu * mu_pop;
                if random_ar {
                    inf_mu -= om_b_inv[idx(0, 1, q_b)] * (s_g[g] - s_pop);
                }
                let c_const = 1.0 - sg;
                let mut ur_prev = 0.0;
                let mut uc_prev = 0.0;
                for i in 0..n {
                    let ri = y_work[st + i + 1] - sg * y_work[st + i];
                    let (ur, uc) = if i == 0 {
                        (ri / chol_l[0], c_const / chol_l[0])
                    } else {
                        (
                            (ri - chol_m[i] * ur_prev) / chol_l[i],
                            (c_const - chol_m[i] * uc_prev) / chol_l[i],
                        )
                    };
                    pr_mu += uc * uc;
                    inf_mu += uc * ur;
                    ur_prev = ur;
                    uc_prev = uc;
                }
                mu_g[g] = inf_mu / pr_mu.max(1e-10) + (1.0 / pr_mu.max(1e-10)).sqrt() * rng.rnorm();

                // 2b. Exact truncated Gaussian draw of s_g | y_g, mu_g, theta_ma, sigma2_e (integrating out e_g!)
                if random_ar {
                    let mug = mu_g[g];
                    let mut pr_s = om_b_inv[idx(1, 1, 2)];
                    let mut inf_s = pr_s * s_pop - om_b_inv[idx(1, 0, 2)] * (mug - mu_pop);
                    let mut ur_p = 0.0;
                    let mut uc_p = 0.0;
                    for i in 0..n {
                        let ri = y_work[st + i + 1] - mug;
                        let ci = y_work[st + i] - mug;
                        let (ur, uc) = if i == 0 {
                            (ri / chol_l[0], ci / chol_l[0])
                        } else {
                            (
                                (ri - chol_m[i] * ur_p) / chol_l[i],
                                (ci - chol_m[i] * uc_p) / chol_l[i],
                            )
                        };
                        pr_s += uc * uc;
                        inf_s += uc * ur;
                        ur_p = ur;
                        uc_p = uc;
                    }
                    let m_s = inf_s / pr_s.max(1e-10);
                    let sd_s = (1.0 / pr_s.max(1e-10)).sqrt();
                    s_g[g] = rng.rtruncnorm(m_s, sd_s, -ar_bound, ar_bound);
                }
            }

            // Fixed global AR slope s_pop if !random_ar
            if !random_ar {
                let mut pr_s = p_pr;
                let mut inf_s = 0.0;
                for g in 0..g_cnt {
                    let st = offsets[g];
                    let tg = offsets[g + 1] - st;
                    if tg < 2 {
                        continue;
                    }
                    let n = tg - 1;
                    let mug = mu_g[g];
                    let mut ur_p = 0.0;
                    let mut uc_p = 0.0;
                    for i in 0..n {
                        let ri = y_work[st + i + 1] - mug;
                        let ci = y_work[st + i] - mug;
                        let (ur, uc) = if i == 0 {
                            (ri / chol_l[0], ci / chol_l[0])
                        } else {
                            (
                                (ri - chol_m[i] * ur_p) / chol_l[i],
                                (ci - chol_m[i] * uc_p) / chol_l[i],
                            )
                        };
                        pr_s += uc * uc;
                        inf_s += uc * ur;
                        ur_p = ur;
                        uc_p = uc;
                    }
                }
                let m_s = inf_s / pr_s.max(1e-10);
                let sd_s = (1.0 / pr_s.max(1e-10)).sqrt();
                s_pop = rng.rtruncnorm(m_s, sd_s, -ar_bound, ar_bound);
                s_g.fill(s_pop);
            }

            // 3. Exact SUR update of (mu_pop, s_pop) and Omega_B
            if random_ar {
                let mut pr_m = vec![0.0; 4];
                let mut inf_m = vec![0.0; 2];
                for ca in 0..2 {
                    for ra in 0..2 {
                        pr_m[idx(ra, ca, 2)] = g_cnt as f64 * om_b_inv[idx(ra, ca, 2)];
                    }
                    pr_m[idx(ca, ca, 2)] += p_pr;
                }
                let mut sum_mu = 0.0;
                let mut sum_s = 0.0;
                for g in 0..g_cnt {
                    sum_mu += mu_g[g];
                    sum_s += s_g[g];
                }
                inf_m[0] = om_b_inv[idx(0, 0, 2)] * sum_mu + om_b_inv[idx(0, 1, 2)] * sum_s;
                inf_m[1] = om_b_inv[idx(1, 0, 2)] * sum_mu + om_b_inv[idx(1, 1, 2)] * sum_s;
                let m_draw = rmvnorm_precision(&mut rng, &pr_m, &inf_m, 2);
                mu_pop = m_draw[0];
                s_pop = m_draw[1].clamp(-ar_bound, ar_bound);

                let psi0_d = if prior_mode == 1 { 1e-5 } else { 0.05 };
                let mut sb = eye(2, psi0_d);
                for g in 0..g_cnt {
                    let d0 = mu_g[g] - mu_pop;
                    let d1 = s_g[g] - s_pop;
                    sb[idx(0, 0, 2)] += d0 * d0;
                    sb[idx(1, 0, 2)] += d1 * d0;
                    sb[idx(0, 1, 2)] += d0 * d1;
                    sb[idx(1, 1, 2)] += d1 * d1;
                }
                let df_b = if prior_mode == 1 {
                    (g_cnt as f64 - 3.0).max(3.0)
                } else {
                    g_cnt as f64 + 4.0
                };
                omega_b = rinv_wishart(&mut rng, df_b, &sb, 2);
            } else {
                let o00 = om_b_inv[0].max(1e-8);
                let pr_m = g_cnt as f64 * o00 + p_pr;
                let sum_mu: f64 = mu_g.iter().sum();
                mu_pop = (o00 * sum_mu) / pr_m + (1.0 / pr_m).sqrt() * rng.rnorm();
                let rss_mu: f64 = mu_g.iter().map(|&v| (v - mu_pop).powi(2)).sum();
                let (sh, sc) = if prior_mode == 1 {
                    ((0.5 * g_cnt as f64 - 1.0).max(0.5), 0.5 * rss_mu.max(1e-8))
                } else {
                    (0.01 + 0.5 * g_cnt as f64, 0.01 + 0.5 * rss_mu)
                };
                omega_b[0] = rng.rinv_gamma(sh, sc);
            }
        } else {
            // N=1 ARMA(1,1): exact collapsed updates for mu_pop and s_pop
            let tg = offsets[1];
            if tg >= 2 {
                let n = tg - 1;
                let mut pr_mu = p_pr;
                let mut inf_mu = 0.0;
                let c_const = 1.0 - s_pop;
                let mut ur_p = 0.0;
                let mut uc_p = 0.0;
                for i in 0..n {
                    let ri = y_work[i + 1] - s_pop * y_work[i];
                    let (ur, uc) = if i == 0 {
                        (ri / chol_l[0], c_const / chol_l[0])
                    } else {
                        (
                            (ri - chol_m[i] * ur_p) / chol_l[i],
                            (c_const - chol_m[i] * uc_p) / chol_l[i],
                        )
                    };
                    pr_mu += uc * uc;
                    inf_mu += uc * ur;
                    ur_p = ur;
                    uc_p = uc;
                }
                mu_pop = inf_mu / pr_mu.max(1e-10) + (1.0 / pr_mu.max(1e-10)).sqrt() * rng.rnorm();
                mu_g[0] = mu_pop;

                let mut pr_s = p_pr;
                let mut inf_s = 0.0;
                let mut ur_p2 = 0.0;
                let mut uc_p2 = 0.0;
                for i in 0..n {
                    let ri = y_work[i + 1] - mu_pop;
                    let ci = y_work[i] - mu_pop;
                    let (ur, uc) = if i == 0 {
                        (ri / chol_l[0], ci / chol_l[0])
                    } else {
                        (
                            (ri - chol_m[i] * ur_p2) / chol_l[i],
                            (ci - chol_m[i] * uc_p2) / chol_l[i],
                        )
                    };
                    pr_s += uc * uc;
                    inf_s += uc * ur;
                    ur_p2 = ur;
                    uc_p2 = uc;
                }
                s_pop = rng.rtruncnorm(
                    inf_s / pr_s.max(1e-10),
                    (1.0 / pr_s.max(1e-10)).sqrt(),
                    -ar_bound,
                    ar_bound,
                );
                s_g[0] = s_pop;
            }
        }

        // 4. Exact Collapsed Marginal MH for (theta_ma, log(sigma2_e)) (integrating out ALL e_{gt}!)
        {
            let mut l_tmp = vec![0.0; max_t];
            let mut m_tmp = vec![0.0; max_t];
            let eval_lp = |th_v: f64, lse: f64, l_b: &mut [f64], m_b: &mut [f64]| -> f64 {
                let se2 = lse.exp();
                let a = (1.0 + th_v * th_v) * se2 + theta_y;
                let b = th_v * se2;
                let max_n = max_t.saturating_sub(1);
                factor_toeplitz_tridiag(a, b, max_n, l_b, m_b);
                let mut logdet_prefix = vec![0.0; max_n + 1];
                for i in 0..max_n {
                    logdet_prefix[i + 1] = logdet_prefix[i] + 2.0 * l_b[i].ln();
                }
                let mut ll = 0.0;
                let log2pi = (2.0 * PI).ln();
                for g in 0..g_cnt {
                    let st = offsets[g];
                    let tg = offsets[g + 1] - st;
                    if tg < 2 {
                        continue;
                    }
                    let n = tg - 1;
                    let mug = mu_g[g];
                    let sg = if is_twolevel && random_ar {
                        s_g[g]
                    } else {
                        s_pop
                    };
                    let mut quad = 0.0;
                    let mut u_prev = 0.0;
                    for i in 0..n {
                        let zi = (y_work[st + i + 1] - mug) - sg * (y_work[st + i] - mug);
                        let ui = if i == 0 {
                            zi / l_b[0]
                        } else {
                            (zi - m_b[i] * u_prev) / l_b[i]
                        };
                        quad += ui * ui;
                        u_prev = ui;
                    }
                    ll += -0.5 * (n as f64 * log2pi + logdet_prefix[n] + quad);
                }
                // Priors + Jacobian (+lse for log-variance transform)
                // Diffuse IG(-1, 0): p(se2) = const -> log p + log(se2) = lse
                let lp_prior = -0.5 * p_pr * th_v * th_v
                    + if prior_mode == 1 {
                        lse
                    } else {
                        -0.01 * lse - 0.01 / se2
                    };
                ll + lp_prior
            };
            let mut cur_th = theta_ma;
            let mut cur_lse = sigma2_e.max(1e-6).ln();
            let mut cur_lp = eval_lp(cur_th, cur_lse, &mut l_tmp, &mut m_tmp);
            for mv in 0..4 {
                let (prop_th, prop_lse) = match mv {
                    0 => (cur_th + 0.04 * rng.rnorm(), cur_lse),
                    1 => (cur_th, cur_lse + 0.03 * rng.rnorm()),
                    2 => (cur_th + 0.025 * rng.rnorm(), cur_lse + 0.02 * rng.rnorm()),
                    _ => (cur_th + 0.012 * rng.rnorm(), cur_lse + 0.01 * rng.rnorm()),
                };
                if prop_th.abs() < ma_bound && prop_lse > -8.0 && prop_lse < 8.0 {
                    let prop_lp = eval_lp(prop_th, prop_lse, &mut l_tmp, &mut m_tmp);
                    let acc = prop_lp - cur_lp;
                    if acc.is_finite() && (acc >= 0.0 || rng.runif().ln() < acc) {
                        cur_th = prop_th;
                        cur_lse = prop_lse;
                        cur_lp = prop_lp;
                    }
                }
            }
            theta_ma = cur_th;
            sigma2_e = cur_lse.exp();

            // 4b. Joint Collapsed ARMA(1,1) Ridge Move:
            // Shifts (s_pop -> s_pop + d, s_g -> s_g + d, theta_ma -> theta_ma - d, log(sigma2_e) -> log(sigma2_e) + eps).
            // Since (s_g' - s_pop') == (s_g - s_pop), the between-level hierarchical prior N(mu_pop, s_pop, Omega_B)
            // is 100% invariant and the translation Jacobian is 1!
            let eval_joint_lp = |th_v: f64,
                                 lse: f64,
                                 sp: f64,
                                 sg_slice: &[f64],
                                 l_b: &mut [f64],
                                 m_b: &mut [f64]|
             -> f64 {
                let se2 = lse.exp();
                let a = (1.0 + th_v * th_v) * se2 + theta_y;
                let b = th_v * se2;
                let max_n = max_t.saturating_sub(1);
                factor_toeplitz_tridiag(a, b, max_n, l_b, m_b);
                let mut logdet_prefix = vec![0.0; max_n + 1];
                for i in 0..max_n {
                    logdet_prefix[i + 1] = logdet_prefix[i] + 2.0 * l_b[i].ln();
                }
                let mut ll = 0.0;
                let log2pi = (2.0 * PI).ln();
                for g in 0..g_cnt {
                    let st = offsets[g];
                    let tg = offsets[g + 1] - st;
                    if tg < 2 {
                        continue;
                    }
                    let n = tg - 1;
                    let mug = mu_g[g];
                    let sg = if is_twolevel && random_ar {
                        sg_slice[g]
                    } else {
                        sp
                    };
                    let mut quad = 0.0;
                    let mut u_prev = 0.0;
                    for i in 0..n {
                        let zi = (y_work[st + i + 1] - mug) - sg * (y_work[st + i] - mug);
                        let ui = if i == 0 {
                            zi / l_b[0]
                        } else {
                            (zi - m_b[i] * u_prev) / l_b[i]
                        };
                        quad += ui * ui;
                        u_prev = ui;
                    }
                    ll += -0.5 * (n as f64 * log2pi + logdet_prefix[n] + quad);
                }
                let lp_prior = -0.5 * p_pr * (th_v * th_v + sp * sp)
                    + if prior_mode == 1 {
                        lse
                    } else {
                        -0.01 * lse - 0.01 / se2
                    };
                ll + lp_prior
            };
            let mut cur_sp = s_pop;
            let mut cur_th_j = theta_ma;
            let mut cur_lse_j = sigma2_e.max(1e-6).ln();
            let mut cur_jlp =
                eval_joint_lp(cur_th_j, cur_lse_j, cur_sp, &s_g, &mut l_tmp, &mut m_tmp);
            let mut prop_sg = s_g.clone();
            for mv in 0..4 {
                let sd_d = if mv < 2 { 0.035 } else { 0.015 };
                let delta = sd_d * rng.rnorm();
                let eps = 0.008 * rng.rnorm();
                let prop_th = cur_th_j - delta;
                let prop_sp = cur_sp + delta;
                let prop_lse = cur_lse_j + eps;
                if prop_th.abs() >= ma_bound
                    || prop_sp.abs() >= ar_bound
                    || prop_lse <= -8.0
                    || prop_lse >= 8.0
                {
                    continue;
                }
                let mut valid = true;
                if is_twolevel && random_ar {
                    for g in 0..g_cnt {
                        let sg_p = s_g[g] + delta;
                        if sg_p.abs() >= ar_bound {
                            valid = false;
                            break;
                        }
                        prop_sg[g] = sg_p;
                    }
                }
                if !valid {
                    continue;
                }
                let prop_jlp =
                    eval_joint_lp(prop_th, prop_lse, prop_sp, &prop_sg, &mut l_tmp, &mut m_tmp);
                let acc = prop_jlp - cur_jlp;
                if acc.is_finite() && (acc >= 0.0 || rng.runif().ln() < acc) {
                    cur_th_j = prop_th;
                    cur_sp = prop_sp;
                    cur_lse_j = prop_lse;
                    cur_jlp = prop_jlp;
                    if is_twolevel && random_ar {
                        s_g.copy_from_slice(&prop_sg);
                    } else {
                        s_g.fill(cur_sp);
                    }
                }
            }
            theta_ma = cur_th_j;
            s_pop = cur_sp;
            sigma2_e = cur_lse_j.exp();
        }

        // 5. Retain posterior draw + conditional latent e_{gt} & DIC
        if keep_pos < n_keep && step == keep_indices[keep_pos] {
            draws[idx(keep_pos, 0, n_keep)] = theta_ma;
            draws[idx(keep_pos, 1, n_keep)] = sigma2_e;
            if !is_twolevel {
                draws[idx(keep_pos, 2, n_keep)] = s_pop;
                draws[idx(keep_pos, 3, n_keep)] = mu_pop;
            } else if random_ar {
                draws[idx(keep_pos, 2, n_keep)] = mu_pop;
                draws[idx(keep_pos, 3, n_keep)] = s_pop;
                draws[idx(keep_pos, 4, n_keep)] = omega_b[idx(0, 0, 2)];
                draws[idx(keep_pos, 5, n_keep)] = omega_b[idx(1, 0, 2)];
                draws[idx(keep_pos, 6, n_keep)] = omega_b[idx(1, 1, 2)];
            } else {
                draws[idx(keep_pos, 2, n_keep)] = s_pop;
                draws[idx(keep_pos, 3, n_keep)] = mu_pop;
                draws[idx(keep_pos, 4, n_keep)] = omega_b[0];
            }

            if is_twolevel {
                for g in 0..g_cnt {
                    re_draws[keep_pos + n_keep * g] = mu_g[g];
                    if random_ar {
                        re_draws[keep_pos + n_keep * (g + g_cnt)] = s_g[g];
                    }
                }
            }

            // Sample exact conditional e_{gt} | y_g for latent_draws and fitted_mean,
            // and evaluate exact collapsed marginal log-likelihood for well-posed DIC/pD
            let mut l_dic = vec![0.0; max_t];
            let mut m_dic = vec![0.0; max_t];
            let max_n_dic = max_t.saturating_sub(1);
            factor_toeplitz_tridiag(
                (1.0 + theta_ma * theta_ma) * sigma2_e + theta_y,
                theta_ma * sigma2_e,
                max_n_dic,
                &mut l_dic,
                &mut m_dic,
            );
            let mut logdet_dic = vec![0.0; max_n_dic + 1];
            for i in 0..max_n_dic {
                logdet_dic[i + 1] = logdet_dic[i] + 2.0 * l_dic[i].ln();
            }
            let log2pi_dic = (2.0 * PI).ln();
            let mut ll_marg = 0.0;
            theta_ma_accum += theta_ma;
            sigma2_e_accum += sigma2_e;
            for g in 0..g_cnt {
                let st = offsets[g];
                let en = offsets[g + 1];
                let tg = en - st;
                let mug = mu_g[g];
                let sg = if is_twolevel && random_ar {
                    s_g[g]
                } else {
                    s_pop
                };
                mu_g_accum[g] += mug;
                s_g_accum[g] += sg;
                if tg >= 2 {
                    let n = tg - 1;
                    let mut quad = 0.0;
                    let mut u_prev = 0.0;
                    for i in 0..n {
                        let zi = (y_work[st + i + 1] - mug) - sg * (y_work[st + i] - mug);
                        let ui = if i == 0 {
                            zi / l_dic[0]
                        } else {
                            (zi - m_dic[i] * u_prev) / l_dic[i]
                        };
                        quad += ui * ui;
                        u_prev = ui;
                    }
                    ll_marg += -0.5 * (n as f64 * log2pi_dic + logdet_dic[n] + quad);
                    sample_e_given_y(
                        &mut rng,
                        &y_work[st..en],
                        mug,
                        sg,
                        theta_ma,
                        sigma2_e,
                        theta_y,
                        &mut e_buf[..tg],
                    );
                } else if tg == 1 {
                    e_buf[0] = sigma2_e.sqrt() * rng.rnorm();
                }
                for t_loc in 0..tg {
                    let r = st + t_loc;
                    if store_latent {
                        latent_draws[keep_pos + n_keep * r] = e_buf[t_loc];
                    }
                    let fv = if t_loc == 0 {
                        mug + e_buf[0]
                    } else {
                        mug + sg * (y_work[r - 1] - mug)
                            + e_buf[t_loc]
                            + theta_ma * e_buf[t_loc - 1]
                    };
                    fitted_accum[r] += fv;
                }
            }
            deviance_sum += -2.0 * ll_marg;
            keep_pos += 1;
        }
    }

    let inv_keep = 1.0 / (n_keep.max(1) as f64);
    let fitted_mean: Vec<f64> = fitted_accum.into_iter().map(|v| v * inv_keep).collect();
    let d_bar = deviance_sum * inv_keep;
    let th_bar = theta_ma_accum * inv_keep;
    let se_bar = (sigma2_e_accum * inv_keep).max(1e-6);
    let max_n_dic = max_t.saturating_sub(1);
    let mut l_bar = vec![0.0; max_t];
    let mut m_bar = vec![0.0; max_t];
    factor_toeplitz_tridiag(
        (1.0 + th_bar * th_bar) * se_bar + theta_y,
        th_bar * se_bar,
        max_n_dic,
        &mut l_bar,
        &mut m_bar,
    );
    let mut logdet_bar = vec![0.0; max_n_dic + 1];
    for i in 0..max_n_dic {
        logdet_bar[i + 1] = logdet_bar[i] + 2.0 * l_bar[i].ln();
    }
    let log2pi_dic = (2.0 * PI).ln();
    let mut ll_hat = 0.0;
    for g in 0..g_cnt {
        let st = offsets[g];
        let en = offsets[g + 1];
        let tg = en - st;
        if tg < 2 {
            continue;
        }
        let n = tg - 1;
        let mug = mu_g_accum[g] * inv_keep;
        let sg = s_g_accum[g] * inv_keep;
        let mut quad = 0.0;
        let mut u_prev = 0.0;
        for i in 0..n {
            let zi = (y_work[st + i + 1] - mug) - sg * (y_work[st + i] - mug);
            let ui = if i == 0 {
                zi / l_bar[0]
            } else {
                (zi - m_bar[i] * u_prev) / l_bar[i]
            };
            quad += ui * ui;
            u_prev = ui;
        }
        ll_hat += -0.5 * (n as f64 * log2pi_dic + logdet_bar[n] + quad);
    }
    let pd = (d_bar - (-2.0 * ll_hat)).max(0.0);
    let dic = d_bar + pd;

    ArmaOutput {
        draws,
        param_count: n_params,
        latent_draws,
        re_draws,
        fitted_mean,
        dic,
        pd,
    }
}

/// Cholesky factor L of an n x n symmetric tridiagonal Toeplitz matrix with main diagonal `a`
/// and sub/super-diagonal `b`. Stores diagonal in `l[0..n]` and subdiagonal in `m[1..n]`.
fn factor_toeplitz_tridiag(a: f64, b: f64, n: usize, l: &mut [f64], m: &mut [f64]) {
    if n == 0 {
        return;
    }
    l[0] = a.max(1e-12).sqrt();
    m[0] = 0.0;
    for i in 1..n {
        let mi = b / l[i - 1];
        m[i] = mi;
        l[i] = (a - mi * mi).max(1e-12).sqrt();
    }
}

/// Exact O(T) Gaussian Gibbs draw of latent innovations e[0..T] conditional on y[0..T]
/// via Cholesky of the tridiagonal precision matrix Q_e.
#[allow(clippy::too_many_arguments)]
fn sample_e_given_y(
    rng: &mut Rng,
    y_g: &[f64],
    mu_g: f64,
    s_g: f64,
    theta: f64,
    sigma2_e: f64,
    theta_y: f64,
    e_out: &mut [f64],
) {
    let tg = y_g.len();
    if tg < 2 {
        return;
    }
    let inv_se = 1.0 / sigma2_e.max(1e-8);
    let inv_ty = 1.0 / theta_y.max(1e-8);
    let off = theta * inv_ty;

    // Tridiagonal precision Q_e = L L^T and info vector h_e
    let mut l_d = vec![0.0; tg];
    let mut l_sub = vec![0.0; tg];
    let mut h = vec![0.0; tg];

    for t in 1..tg {
        let zt = (y_g[t] - mu_g) - s_g * (y_g[t - 1] - mu_g);
        h[t] += zt * inv_ty;
        h[t - 1] += theta * zt * inv_ty;
    }

    for t in 0..tg {
        let q_tt = inv_se
            + if t == 0 {
                theta * theta * inv_ty
            } else if t == tg - 1 {
                inv_ty
            } else {
                (1.0 + theta * theta) * inv_ty
            };
        if t == 0 {
            l_d[0] = q_tt.max(1e-12).sqrt();
        } else {
            let sub = off / l_d[t - 1];
            l_sub[t] = sub;
            l_d[t] = (q_tt - sub * sub).max(1e-12).sqrt();
        }
    }

    // Forward solve L w = h
    let mut w = vec![0.0; tg];
    w[0] = h[0] / l_d[0];
    for t in 1..tg {
        w[t] = (h[t] - l_sub[t] * w[t - 1]) / l_d[t];
    }

    // Backward solve L^T e = w + z, z ~ N(0, I)
    e_out[tg - 1] = (w[tg - 1] + rng.rnorm()) / l_d[tg - 1];
    for t in (0..(tg - 1)).rev() {
        e_out[t] = (w[t] + rng.rnorm() - l_sub[t + 1] * e_out[t + 1]) / l_d[t];
    }
}
