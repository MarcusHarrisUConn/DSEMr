use crate::kalman::ffbs_dafs;
use crate::linalg::idx;
use crate::rng::Rng;
use std::f64::consts::PI;

pub struct CcDynamicCfaOutput {
    pub draws: Vec<f64>,
    pub param_count: usize,
    pub latent_draws: Vec<f64>, // n_keep x (n_id * n_time) (within factor f_w)
    pub id_draws: Vec<f64>,     // n_keep x n_id x q_id_out
    pub time_draws: Vec<f64>,   // n_keep x n_time x q_time_out
    pub fitted_mean: Vec<f64>,  // (n_id * n_time) x n_items
    pub dic: f64,
    pub pd: f64,
}

/// Cross-Classified & Two-Level Dynamic CFA with Measurement Invariance (Ex 9.40a)
/// and/or Random Factor Loadings (Ex 9.40b):
///   y_{gtj} = nu_j + u_{gj} + v_{tj} + lam_{gj} * (eta^w_{gt} + f_subj_g + f_time_t) + eps_{gtj}
///   eta^w_{gt} = phi_g * eta^w_{g,t-1} + zeta_{gt},  zeta_{gt} ~ N(0, psi_w)
/// Supports fixed loadings (equal across within/subject/time), random subject-specific loadings
/// lam_{gj} ~ N(lam_{0j}, tau^2_{lam,j}), fixed or random AR(1) slope, and optional time-level effects.
#[allow(clippy::too_many_arguments)]
pub fn mcmc_cc_dynamic_cfa(
    y_grid: &[f64],        // (n_id * n_time) x n_items, ordered by subject g, time t
    n_id: usize,
    n_time: usize,
    n_items: usize,
    random_loadings: bool, // true = Ex 9.40b (s1-sp | f BY y1-yp)
    has_f_subj: bool,      // true = subject-level factor fsubj / f
    has_f_time: bool,      // true = time-level factor ftime / f
    has_time_int: bool,    // true = time-level indicator random intercepts v_{tj} (cross-classified)
    random_ar: bool,       // true = subject-varying AR(1) slope s_g
    psi_fixed_one: bool,   // true = f@1 (unit within factor innovation variance)
    prior_mode: i32,       // 0 = weakly_informative, 1 = diffuse
    ar_bound: f64,
    store_latent: bool,
    iter: usize,
    warmup: usize,
    thin: usize,
    seed: u64,
) -> CcDynamicCfaOutput {
    let mut rng = Rng::from_seed(seed);
    let n_obs = n_id * n_time;
    let p_pr = if prior_mode == 1 { 1e-8 } else { 0.01 };

    // Grand indicator means for initialization
    let mut nu0 = vec![0.0; n_items];
    for j in 0..n_items {
        let mut s = 0.0;
        let mut c = 0usize;
        for r in 0..n_obs {
            let v = y_grid[idx(r, j, n_obs)];
            if v.is_finite() {
                s += v;
                c += 1;
            }
        }
        nu0[j] = if c > 0 { s / c as f64 } else { 0.0 };
    }

    // Measurement & structural states
    let mut lam0 = vec![0.8_f64; n_items];
    if !psi_fixed_one && !random_loadings {
        lam0[0] = 1.0;
    }
    let mut lam_g = vec![0.8_f64; n_id * n_items];
    for g in 0..n_id {
        for j in 0..n_items {
            lam_g[idx(g, j, n_id)] = lam0[j];
        }
    }
    let mut tau2_lam = vec![0.1_f64; n_items];
    let mut theta_w = vec![0.5_f64; n_items];
    let mut theta_subj = vec![0.3_f64; n_items];
    let mut theta_time = vec![0.1_f64; n_items];

    let mut u_subj = vec![0.0_f64; n_id * n_items];
    let mut v_time = vec![0.0_f64; n_time * n_items];
    let mut f_subj = vec![0.0_f64; n_id];
    let mut f_time = vec![0.0_f64; n_time];
    let mut psi_subj = 0.5_f64;
    let mut psi_time = 0.2_f64;

    let mut eta_w = vec![0.0_f64; n_obs];
    let mut phi_pop = 0.3_f64;
    let mut phi_g = vec![0.3_f64; n_id];
    let mut var_phi = 0.02_f64;
    let mut psi_w = 1.0_f64;

    // Parameter output order:
    //   1. Loading means: lam0[0..n_items] (if !psi_fixed_one && !random_loadings, skip j=0 marker)
    //   2. AR(1) slope (phi_pop)
    //   3. Within factor variance psi_w (if !psi_fixed_one)
    //   4. Indicator intercepts nu0[0..n_items]
    //   5. Within residual variances theta_w[0..n_items]
    //   6. Subject factor variance psi_subj (if has_f_subj)
    //   7. Subject indicator residual variances theta_subj[0..n_items]
    //   8. Subject random loading variances tau2_lam[0..n_items] (if random_loadings)
    //   9. Subject random AR slope variance var_phi (if random_ar)
    //  10. Time factor variance psi_time (if has_f_time)
    //  11. Time indicator residual variances theta_time[0..n_items] (if has_time_int)
    let free_lam_idx: Vec<usize> = (0..n_items)
        .filter(|&j| psi_fixed_one || random_loadings || j > 0)
        .collect();
    let n_params = free_lam_idx.len()
        + 1
        + usize::from(!psi_fixed_one)
        + n_items
        + n_items
        + usize::from(has_f_subj)
        + n_items
        + if random_loadings { n_items } else { 0 }
        + usize::from(random_ar)
        + usize::from(has_f_time)
        + if has_time_int { n_items } else { 0 };

    // Subject random effects stored in id_draws: [f_subj (if has_f_subj), lam_g (if random_loadings), phi_g (if random_ar)]
    let q_id_out = usize::from(has_f_subj)
        + if random_loadings { n_items } else { 0 }
        + usize::from(random_ar);
    let q_time_out = usize::from(has_f_time);

    let keep_indices: Vec<usize> = ((warmup + 1)..=iter)
        .filter(|i| (i - warmup - 1) % thin == 0)
        .collect();
    let n_keep = keep_indices.len();
    let mut draws = vec![0.0; n_keep * n_params];
    let mut latent_draws = vec![0.0; if store_latent { n_keep * n_obs } else { 0 }];
    let mut id_draws = vec![0.0; n_keep * n_id * q_id_out.max(1)];
    let mut time_draws = vec![0.0; n_keep * n_time * q_time_out.max(1)];
    let mut fitted_accum = vec![0.0; n_obs * n_items];
    let mut theta_w_accum = vec![0.0; n_items];
    let mut deviance_sum = 0.0;
    let mut keep_pos = 0usize;

    let alpha_zero = [0.0_f64];
    let mut y_g_buf = vec![0.0; n_time * n_items];
    let mut nu_g_buf = vec![0.0; n_items];
    let mut lam_g_buf = vec![0.0; n_items];

    for step in 1..=iter {
        // 1. Sample within-level AR(1) factor trajectory eta_w[g, 0..n_time] via exact 1D FFBS
        for g in 0..n_id {
            let st = g * n_time;
            let fs_g = if has_f_subj { f_subj[g] } else { 0.0 };
            for j in 0..n_items {
                let lgj = if random_loadings {
                    lam_g[idx(g, j, n_id)]
                } else {
                    lam0[j]
                };
                lam_g_buf[j] = lgj;
                nu_g_buf[j] = 0.0;
                for t in 0..n_time {
                    let yv = y_grid[idx(st + t, j, n_obs)];
                    if yv.is_finite() {
                        let vt = if has_time_int {
                            v_time[idx(t, j, n_time)]
                        } else {
                            0.0
                        };
                        let ft = if has_f_time { f_time[t] } else { 0.0 };
                        y_g_buf[idx(t, j, n_time)] =
                            yv - nu0[j] - u_subj[idx(g, j, n_id)] - vt - lgj * (fs_g + ft);
                    } else {
                        y_g_buf[idx(t, j, n_time)] = f64::NAN;
                    }
                }
            }
            let ph = if random_ar { phi_g[g] } else { phi_pop };
            let (st_g, _) = ffbs_dafs(
                &mut rng,
                &y_g_buf,
                n_time,
                n_items,
                1,
                &nu_g_buf,
                &lam_g_buf,
                &theta_w,
                &alpha_zero,
                &[ph],
                &[psi_w],
                ar_bound,
            );
            eta_w[st..(st + n_time)].copy_from_slice(&st_g[..n_time]);
        }

        // 2. Sample subject-level factor f_subj[g] (exact 1D Gaussian Gibbs) + variance psi_subj
        if has_f_subj {
            let inv_ps = 1.0 / psi_subj.max(1e-6);
            for g in 0..n_id {
                let st = g * n_time;
                let mut pr = inv_ps;
                let mut inf = 0.0;
                for j in 0..n_items {
                    let lgj = if random_loadings {
                        lam_g[idx(g, j, n_id)]
                    } else {
                        lam0[j]
                    };
                    let inv_tw = 1.0 / theta_w[j].max(1e-6);
                    let u_gj = u_subj[idx(g, j, n_id)];
                    for t in 0..n_time {
                        let yv = y_grid[idx(st + t, j, n_obs)];
                        if yv.is_finite() {
                            let vt = if has_time_int {
                                v_time[idx(t, j, n_time)]
                            } else {
                                0.0
                            };
                            let ft = if has_f_time { f_time[t] } else { 0.0 };
                            let r = yv - nu0[j] - u_gj - vt - lgj * (eta_w[st + t] + ft);
                            pr += lgj * lgj * inv_tw;
                            inf += lgj * r * inv_tw;
                        }
                    }
                }
                f_subj[g] = inf / pr + (1.0 / pr).sqrt() * rng.rnorm();
            }
            let m_fs: f64 = f_subj.iter().sum::<f64>() / n_id as f64;
            for v in &mut f_subj {
                *v -= m_fs;
            }
            let ss_fs: f64 = f_subj.iter().map(|v| v * v).sum();
            let (sh, sc) = if prior_mode == 1 {
                ((0.5 * n_id as f64 - 1.0).max(0.5), 0.5 * ss_fs.max(1e-8))
            } else {
                (0.01 + 0.5 * n_id as f64, 0.01 + 0.5 * ss_fs)
            };
            psi_subj = rng.rinv_gamma(sh, sc);
        }

        // 3. Sample time-level factor f_time[t] (exact 1D Gaussian Gibbs) + variance psi_time
        if has_f_time {
            let inv_pt = 1.0 / psi_time.max(1e-6);
            for t in 0..n_time {
                let mut pr = inv_pt;
                let mut inf = 0.0;
                for g in 0..n_id {
                    let st = g * n_time;
                    let fs_g = if has_f_subj { f_subj[g] } else { 0.0 };
                    for j in 0..n_items {
                        let yv = y_grid[idx(st + t, j, n_obs)];
                        if yv.is_finite() {
                            let lgj = if random_loadings {
                                lam_g[idx(g, j, n_id)]
                            } else {
                                lam0[j]
                            };
                            let inv_tw = 1.0 / theta_w[j].max(1e-6);
                            let vt = if has_time_int {
                                v_time[idx(t, j, n_time)]
                            } else {
                                0.0
                            };
                            let r = yv
                                - nu0[j]
                                - u_subj[idx(g, j, n_id)]
                                - vt
                                - lgj * (eta_w[st + t] + fs_g);
                            pr += lgj * lgj * inv_tw;
                            inf += lgj * r * inv_tw;
                        }
                    }
                }
                f_time[t] = inf / pr + (1.0 / pr).sqrt() * rng.rnorm();
            }
            let m_ft: f64 = f_time.iter().sum::<f64>() / n_time as f64;
            for v in &mut f_time {
                *v -= m_ft;
            }
            let ss_ft: f64 = f_time.iter().map(|v| v * v).sum();
            let (sh, sc) = if prior_mode == 1 {
                ((0.5 * n_time as f64 - 1.0).max(0.5), 0.5 * ss_ft.max(1e-8))
            } else {
                (0.01 + 0.5 * n_time as f64, 0.01 + 0.5 * ss_ft)
            };
            psi_time = rng.rinv_gamma(sh, sc);
        }

        // 4. Sample subject indicator random intercepts u_subj[g, j], time indicator intercepts v_time[t, j], nu0[j]
        for j in 0..n_items {
            let inv_tw = 1.0 / theta_w[j].max(1e-6);
            let inv_ts = 1.0 / theta_subj[j].max(1e-6);

            // 4a. u_subj[g, j]
            for g in 0..n_id {
                let st = g * n_time;
                let lgj = if random_loadings {
                    lam_g[idx(g, j, n_id)]
                } else {
                    lam0[j]
                };
                let fs_g = if has_f_subj { f_subj[g] } else { 0.0 };
                let mut sum_r = 0.0;
                let mut cnt = 0usize;
                for t in 0..n_time {
                    let yv = y_grid[idx(st + t, j, n_obs)];
                    if yv.is_finite() {
                        let vt = if has_time_int {
                            v_time[idx(t, j, n_time)]
                        } else {
                            0.0
                        };
                        let ft = if has_f_time { f_time[t] } else { 0.0 };
                        sum_r += yv - nu0[j] - vt - lgj * (eta_w[st + t] + fs_g + ft);
                        cnt += 1;
                    }
                }
                let pr = inv_ts + cnt as f64 * inv_tw;
                let inf = sum_r * inv_tw;
                u_subj[idx(g, j, n_id)] = inf / pr + (1.0 / pr).sqrt() * rng.rnorm();
            }

            // 4b. v_time[t, j] if cross-classified
            if has_time_int {
                let inv_tt = 1.0 / theta_time[j].max(1e-6);
                for t in 0..n_time {
                    let ft = if has_f_time { f_time[t] } else { 0.0 };
                    let mut sum_r = 0.0;
                    let mut cnt = 0usize;
                    for g in 0..n_id {
                        let st = g * n_time;
                        let yv = y_grid[idx(st + t, j, n_obs)];
                        if yv.is_finite() {
                            let lgj = if random_loadings {
                                lam_g[idx(g, j, n_id)]
                            } else {
                                lam0[j]
                            };
                            let fs_g = if has_f_subj { f_subj[g] } else { 0.0 };
                            sum_r += yv
                                - nu0[j]
                                - u_subj[idx(g, j, n_id)]
                                - lgj * (eta_w[st + t] + fs_g + ft);
                            cnt += 1;
                        }
                    }
                    let pr = inv_tt + cnt as f64 * inv_tw;
                    let inf = sum_r * inv_tw;
                    v_time[idx(t, j, n_time)] = inf / pr + (1.0 / pr).sqrt() * rng.rnorm();
                }
                // Center v_time into nu0
                let m_vt: f64 =
                    (0..n_time).map(|t| v_time[idx(t, j, n_time)]).sum::<f64>() / n_time as f64;
                for t in 0..n_time {
                    v_time[idx(t, j, n_time)] -= m_vt;
                }
                nu0[j] += m_vt;
                let ss_vt: f64 = (0..n_time)
                    .map(|t| v_time[idx(t, j, n_time)].powi(2))
                    .sum();
                let (sh, sc) = if prior_mode == 1 {
                    ((0.5 * n_time as f64 - 1.0).max(0.5), 0.5 * ss_vt.max(1e-8))
                } else {
                    (0.01 + 0.5 * n_time as f64, 0.01 + 0.5 * ss_vt)
                };
                theta_time[j] = rng.rinv_gamma(sh, sc);
            }

            // 4c. Update grand intercept nu0[j] via hierarchical centering (u_subj + nu0)
            let m_u: f64 =
                (0..n_id).map(|g| u_subj[idx(g, j, n_id)]).sum::<f64>() / n_id as f64;
            let pr_nu = n_id as f64 * inv_ts + p_pr;
            let old_nu = nu0[j];
            let new_nu = (n_id as f64 * inv_ts * (old_nu + m_u)) / pr_nu
                + (1.0 / pr_nu).sqrt() * rng.rnorm();
            nu0[j] = new_nu;
            for g in 0..n_id {
                u_subj[idx(g, j, n_id)] += old_nu - new_nu;
            }
            let ss_u: f64 = (0..n_id)
                .map(|g| u_subj[idx(g, j, n_id)].powi(2))
                .sum();
            let (sh, sc) = if prior_mode == 1 {
                ((0.5 * n_id as f64 - 1.0).max(0.5), 0.5 * ss_u.max(1e-8))
            } else {
                (0.01 + 0.5 * n_id as f64, 0.01 + 0.5 * ss_u)
            };
            theta_subj[j] = rng.rinv_gamma(sh, sc);
        }

        // 5. Update factor loadings (random lam_g[g, j] + lam0[j] + tau2_lam[j], OR fixed lam0[j])
        if random_loadings {
            for j in 0..n_items {
                let inv_tw = 1.0 / theta_w[j].max(1e-6);
                let inv_tl = 1.0 / tau2_lam[j].max(1e-6);
                for g in 0..n_id {
                    let st = g * n_time;
                    let fs_g = if has_f_subj { f_subj[g] } else { 0.0 };
                    let u_gj = u_subj[idx(g, j, n_id)];
                    let mut pr = inv_tl;
                    let mut inf = lam0[j] * inv_tl;
                    for t in 0..n_time {
                        let yv = y_grid[idx(st + t, j, n_obs)];
                        if yv.is_finite() {
                            let vt = if has_time_int {
                                v_time[idx(t, j, n_time)]
                            } else {
                                0.0
                            };
                            let ft = if has_f_time { f_time[t] } else { 0.0 };
                            let f_tot = eta_w[st + t] + fs_g + ft;
                            let r = yv - nu0[j] - u_gj - vt;
                            pr += f_tot * f_tot * inv_tw;
                            inf += f_tot * r * inv_tw;
                        }
                    }
                    lam_g[idx(g, j, n_id)] = inf / pr + (1.0 / pr).sqrt() * rng.rnorm();
                }
                let mean_lg: f64 =
                    (0..n_id).map(|g| lam_g[idx(g, j, n_id)]).sum::<f64>() / n_id as f64;
                let pr_l0 = n_id as f64 * inv_tl + p_pr;
                lam0[j] = (n_id as f64 * inv_tl * mean_lg) / pr_l0
                    + (1.0 / pr_l0).sqrt() * rng.rnorm();
                let ss_l: f64 = (0..n_id)
                    .map(|g| (lam_g[idx(g, j, n_id)] - lam0[j]).powi(2))
                    .sum();
                let (sh, sc) = if prior_mode == 1 {
                    ((0.5 * n_id as f64 - 1.0).max(0.5), 0.5 * ss_l.max(1e-8))
                } else {
                    (0.01 + 0.5 * n_id as f64, 0.01 + 0.5 * ss_l)
                };
                tau2_lam[j] = rng.rinv_gamma(sh, sc);
            }
        } else {
            for &j in &free_lam_idx {
                let inv_tw = 1.0 / theta_w[j].max(1e-6);
                let mut pr = p_pr;
                let mut inf = 0.0;
                for g in 0..n_id {
                    let st = g * n_time;
                    let fs_g = if has_f_subj { f_subj[g] } else { 0.0 };
                    let u_gj = u_subj[idx(g, j, n_id)];
                    for t in 0..n_time {
                        let yv = y_grid[idx(st + t, j, n_obs)];
                        if yv.is_finite() {
                            let vt = if has_time_int {
                                v_time[idx(t, j, n_time)]
                            } else {
                                0.0
                            };
                            let ft = if has_f_time { f_time[t] } else { 0.0 };
                            let f_tot = eta_w[st + t] + fs_g + ft;
                            let r = yv - nu0[j] - u_gj - vt;
                            pr += f_tot * f_tot * inv_tw;
                            inf += f_tot * r * inv_tw;
                        }
                    }
                }
                lam0[j] = inf / pr + (1.0 / pr).sqrt() * rng.rnorm();
            }
        }

        // Polarity anchoring when psi_fixed_one (all loadings free) so factor sign stays positive
        if psi_fixed_one {
            let sum_l: f64 = lam0.iter().sum();
            if sum_l < 0.0 {
                for l in &mut lam0 {
                    *l = -*l;
                }
                if random_loadings {
                    for l in &mut lam_g {
                        *l = -*l;
                    }
                }
                for e in &mut eta_w {
                    *e = -*e;
                }
                for fs in &mut f_subj {
                    *fs = -*fs;
                }
                for ft in &mut f_time {
                    *ft = -*ft;
                }
            }
        }

        // 6. Update within indicator residual variances theta_w[j]
        for j in 0..n_items {
            let mut rss = 0.0;
            let mut cnt = 0usize;
            for g in 0..n_id {
                let st = g * n_time;
                let lgj = if random_loadings {
                    lam_g[idx(g, j, n_id)]
                } else {
                    lam0[j]
                };
                let fs_g = if has_f_subj { f_subj[g] } else { 0.0 };
                let u_gj = u_subj[idx(g, j, n_id)];
                for t in 0..n_time {
                    let yv = y_grid[idx(st + t, j, n_obs)];
                    if yv.is_finite() {
                        let vt = if has_time_int {
                            v_time[idx(t, j, n_time)]
                        } else {
                            0.0
                        };
                        let ft = if has_f_time { f_time[t] } else { 0.0 };
                        let fv = nu0[j] + u_gj + vt + lgj * (eta_w[st + t] + fs_g + ft);
                        let e = yv - fv;
                        rss += e * e;
                        cnt += 1;
                    }
                }
            }
            let (sh, sc) = if prior_mode == 1 {
                ((0.5 * cnt as f64 - 1.0).max(0.5), 0.5 * rss.max(1e-8))
            } else {
                (0.01 + 0.5 * cnt as f64, 0.01 + 0.5 * rss)
            };
            theta_w[j] = rng.rinv_gamma(sh, sc);
        }

        // 7. Update AR(1) slope(s) and optional psi_w
        let inv_pw = 1.0 / psi_w.max(1e-6);
        if random_ar {
            let inv_vp = 1.0 / var_phi.max(1e-6);
            for (g, phi_slot) in phi_g.iter_mut().enumerate().take(n_id) {
                let st = g * n_time;
                let mut pr = inv_vp;
                let mut inf = phi_pop * inv_vp;
                for t in 1..n_time {
                    let ep = eta_w[st + t - 1];
                    pr += ep * ep * inv_pw;
                    inf += ep * eta_w[st + t] * inv_pw;
                }
                *phi_slot = rng.rtruncnorm(inf / pr, (1.0 / pr).sqrt(), -ar_bound, ar_bound);
            }
            let m_p: f64 = phi_g.iter().sum::<f64>() / n_id as f64;
            let pr_mp = n_id as f64 * inv_vp + p_pr;
            phi_pop = rng.rtruncnorm(
                (n_id as f64 * inv_vp * m_p) / pr_mp,
                (1.0 / pr_mp).sqrt(),
                -ar_bound,
                ar_bound,
            );
            let ss_p: f64 = phi_g.iter().map(|v| (v - phi_pop).powi(2)).sum();
            let (sh, sc) = if prior_mode == 1 {
                ((0.5 * n_id as f64 - 1.0).max(0.5), 0.5 * ss_p.max(1e-8))
            } else {
                (0.01 + 0.5 * n_id as f64, 0.01 + 0.5 * ss_p)
            };
            var_phi = rng.rinv_gamma(sh, sc);
        } else {
            let mut pr = p_pr;
            let mut inf = 0.0;
            for g in 0..n_id {
                let st = g * n_time;
                for t in 1..n_time {
                    let ep = eta_w[st + t - 1];
                    pr += ep * ep * inv_pw;
                    inf += ep * eta_w[st + t] * inv_pw;
                }
            }
            phi_pop = rng.rtruncnorm(inf / pr, (1.0 / pr).sqrt(), -ar_bound, ar_bound);
            phi_g.fill(phi_pop);
        }
        if !psi_fixed_one {
            let mut rss_f = 0.0;
            let mut n_tr = 0usize;
            for (g, &ph_val) in phi_g.iter().enumerate().take(n_id) {
                let st = g * n_time;
                let ph = if random_ar { ph_val } else { phi_pop };
                for t in 1..n_time {
                    let ef = eta_w[st + t] - ph * eta_w[st + t - 1];
                    rss_f += ef * ef;
                    n_tr += 1;
                }
            }
            let (sh, sc) = if prior_mode == 1 {
                ((0.5 * n_tr as f64 - 1.0).max(0.5), 0.5 * rss_f.max(1e-8))
            } else {
                (0.01 + 0.5 * n_tr as f64, 0.01 + 0.5 * rss_f)
            };
            psi_w = rng.rinv_gamma(sh, sc);
        }

        // 8. Record retained draw
        if keep_pos < n_keep && step == keep_indices[keep_pos] {
            let mut cur = 0usize;
            for &j in &free_lam_idx {
                draws[idx(keep_pos, cur, n_keep)] = lam0[j];
                cur += 1;
            }
            draws[idx(keep_pos, cur, n_keep)] = phi_pop;
            cur += 1;
            if !psi_fixed_one {
                draws[idx(keep_pos, cur, n_keep)] = psi_w;
                cur += 1;
            }
            for &v in &nu0 {
                draws[idx(keep_pos, cur, n_keep)] = v;
                cur += 1;
            }
            for &v in &theta_w {
                draws[idx(keep_pos, cur, n_keep)] = v;
                cur += 1;
            }
            if has_f_subj {
                draws[idx(keep_pos, cur, n_keep)] = psi_subj;
                cur += 1;
            }
            for &v in &theta_subj {
                draws[idx(keep_pos, cur, n_keep)] = v;
                cur += 1;
            }
            if random_loadings {
                for &v in &tau2_lam {
                    draws[idx(keep_pos, cur, n_keep)] = v;
                    cur += 1;
                }
            }
            if random_ar {
                draws[idx(keep_pos, cur, n_keep)] = var_phi;
                cur += 1;
            }
            if has_f_time {
                draws[idx(keep_pos, cur, n_keep)] = psi_time;
                cur += 1;
            }
            if has_time_int {
                for &v in &theta_time {
                    draws[idx(keep_pos, cur, n_keep)] = v;
                    cur += 1;
                }
            }

            // Store subject & time factor/random-loading draws
            if q_id_out > 0 {
                for g in 0..n_id {
                    let mut q_c = 0usize;
                    if has_f_subj {
                        id_draws[keep_pos + n_keep * (g + n_id * q_c)] = f_subj[g];
                        q_c += 1;
                    }
                    if random_loadings {
                        for j in 0..n_items {
                            id_draws[keep_pos + n_keep * (g + n_id * q_c)] = lam_g[idx(g, j, n_id)];
                            q_c += 1;
                        }
                    }
                    if random_ar {
                        id_draws[keep_pos + n_keep * (g + n_id * q_c)] = phi_g[g];
                    }
                }
            }
            if has_f_time {
                for t in 0..n_time {
                    time_draws[keep_pos + n_keep * t] = f_time[t];
                }
            }
            if store_latent {
                for r in 0..n_obs {
                    latent_draws[keep_pos + n_keep * r] = eta_w[r];
                }
            }

            let mut ll_cond = 0.0;
            for j in 0..n_items {
                theta_w_accum[j] += theta_w[j];
            }
            for g in 0..n_id {
                let st = g * n_time;
                let fs_g = if has_f_subj { f_subj[g] } else { 0.0 };
                for j in 0..n_items {
                    let lgj = if random_loadings {
                        lam_g[idx(g, j, n_id)]
                    } else {
                        lam0[j]
                    };
                    let u_gj = u_subj[idx(g, j, n_id)];
                    let tw_j = theta_w[j].max(1e-6);
                    let log_c = (2.0 * PI * tw_j).ln();
                    for t in 0..n_time {
                        let r_abs = st + t;
                        let vt = if has_time_int {
                            v_time[idx(t, j, n_time)]
                        } else {
                            0.0
                        };
                        let ft = if has_f_time { f_time[t] } else { 0.0 };
                        let fv = nu0[j] + u_gj + vt + lgj * (eta_w[r_abs] + fs_g + ft);
                        fitted_accum[idx(r_abs, j, n_obs)] += fv;
                        let yv = y_grid[idx(r_abs, j, n_obs)];
                        if yv.is_finite() {
                            let e = yv - fv;
                            ll_cond += -0.5 * (log_c + e * e / tw_j);
                        }
                    }
                }
            }
            deviance_sum += -2.0 * ll_cond;
            keep_pos += 1;
        }
    }

    let inv_keep = 1.0 / (n_keep.max(1) as f64);
    let fitted_mean: Vec<f64> = fitted_accum.into_iter().map(|v| v * inv_keep).collect();
    let d_bar = deviance_sum * inv_keep;
    let mut ll_hat = 0.0;
    for j in 0..n_items {
        let tw_bar = (theta_w_accum[j] * inv_keep).max(1e-6);
        let log_c = (2.0 * PI * tw_bar).ln();
        for r in 0..n_obs {
            let yv = y_grid[idx(r, j, n_obs)];
            if yv.is_finite() {
                let e = yv - fitted_mean[idx(r, j, n_obs)];
                ll_hat += -0.5 * (log_c + e * e / tw_bar);
            }
        }
    }
    let pd = (d_bar - (-2.0 * ll_hat)).max(0.0);
    let dic = d_bar + pd;

    CcDynamicCfaOutput {
        draws,
        param_count: n_params,
        latent_draws,
        id_draws,
        time_draws,
        fitted_mean,
        dic,
        pd,
    }
}
