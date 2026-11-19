use crate::kalman::{ffbs_dafs, ffbs_lagged_measurement};
use crate::linalg::{
    eye, idx, is_stable_transition, mat_t_mul, quad_form, rinv_wishart, rmvnorm_precision,
    spd_invert, symmetrize,
};
use crate::rng::Rng;
use std::f64::consts::PI;

pub struct DynamicSemOutput {
    pub draws: Vec<f64>,
    pub param_count: usize,
    pub latent_draws: Vec<f64>, // n_keep x n_obs x m (if store_latent)
    pub re_draws: Vec<f64>,     // n_keep x g_cnt x q_b (if g_cnt > 1)
    pub fitted_mean: Vec<f64>,  // n_obs x n_items
    pub dic: f64,
    pub pd: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn mcmc_dynamic_sem(
    y_in: &[f64],            // n_obs x n_items (contiguous by cluster)
    cluster_sizes: &[usize], // length g_cnt (1 for N=1, >=2 for two-level)
    n_items: usize,
    m: usize,             // number of within factors
    m_b: usize,           // number of between factors (0 if N=1 or direct random intercepts)
    lambda0_mode: &[i32], // n_items x m: 0 = zero, 1 = free, 2 = fixed to lambda0_init
    lambda0_init: &[f64], // n_items x m
    use_wnfs: bool,
    lambda1_mode: &[i32],     // n_items x m: 0 = zero, 1 = free
    lambdab_mode: &[i32],     // n_items x m_b: 0 = zero, 1 = free, 2 = fixed 1.0
    equal_wb_loadings: bool,  // true in Mplus 9.35: f BY u1-u4*(1-4); fb BY u1-u4*(1-4);
    phi_mode: &[i32],         // m x m: 0 = zero, 1 = fixed free, 2 = random across clusters
    psi_fixed_one: bool,      // true in Mplus 6.27 (f@1)
    psi_correlated: bool,     // true in Mplus 6.28 / 9.36 (f1 WITH f2)
    has_random_logv: &[i32],  // length m: 1 if factor k has random log innovation variance
    fix_logv_mean_zero: bool, // true in Mplus 9.35 ([logvf@0])
    item_types: &[i32],       // 0 = continuous, >=2 = K categories
    theta_fixed: &[f64], // length n_items: <=0 means free, >0 means fixed residual variance (e.g. 9.33b y@.01)
    prior_mode: i32,
    ar_bound: f64,
    store_latent: bool,
    iter: usize,
    warmup: usize,
    thin: usize,
    seed: u64,
) -> DynamicSemOutput {
    let mut rng = Rng::from_seed(seed);
    let g_cnt = cluster_sizes.len();
    let is_twolevel = g_cnt > 1;
    let n_obs: usize = cluster_sizes.iter().sum();
    let is_cat: Vec<bool> = item_types.iter().map(|&k| k >= 2).collect();

    let mut offsets = Vec::with_capacity(g_cnt + 1);
    offsets.push(0usize);
    for &sz in cluster_sizes {
        offsets.push(offsets.last().unwrap() + sz);
    }

    // Initialize continuous working y_work for probit items
    let mut y_work = y_in.to_vec();
    for j in 0..n_items {
        if is_cat[j] {
            let kc = item_types[j].max(2) as f64;
            for r in 0..n_obs {
                let v = y_in[idx(r, j, n_obs)];
                if v.is_finite() {
                    y_work[idx(r, j, n_obs)] = ((v + 0.5) / kc - 0.5) * 2.0;
                }
            }
        }
    }

    // Measurement parameters
    let mut nu0 = vec![0.0; n_items];
    for j in 0..n_items {
        let mut s = 0.0;
        let mut c = 0usize;
        for r in 0..n_obs {
            let v = y_work[idx(r, j, n_obs)];
            if v.is_finite() {
                s += v;
                c += 1;
            }
        }
        nu0[j] = if c > 0 { s / c as f64 } else { 0.0 };
    }
    let mut lambda0 = lambda0_init.to_vec();
    let mut lambda1 = vec![0.0; n_items * m];
    if use_wnfs {
        for i in 0..(n_items * m) {
            if lambda1_mode[i] == 1 {
                lambda1[i] = 0.25;
            }
        }
    }
    let mut theta_w = vec![0.8; n_items];
    for j in 0..n_items {
        if is_cat[j] {
            theta_w[j] = 1.0;
        } else if theta_fixed[j] > 0.0 {
            theta_w[j] = theta_fixed[j];
        }
    }

    // Dynamic transition matrix Phi (fixed baseline) and Psi
    let mut phi_pop: Vec<f64> = vec![0.0_f64; m * m];
    let mut rand_phi_slots: Vec<(usize, usize)> = Vec::new();
    let mut fixed_phi_slots: Vec<(usize, usize)> = Vec::new();
    for col in 0..m {
        for row in 0..m {
            let md = phi_mode[idx(row, col, m)];
            if md == 1 {
                fixed_phi_slots.push((row, col));
                if row == col {
                    phi_pop[idx(row, col, m)] = 0.25;
                }
            } else if md == 2 {
                rand_phi_slots.push((row, col));
                if row == col {
                    phi_pop[idx(row, col, m)] = 0.25;
                }
            }
        }
    }
    let mut psi_pop = eye(m, 1.0);

    // Two-level between structure:
    // Between random-effect vector per cluster g:
    //   if m_b > 0: [fb_{g,0..m_b}, s_{g,0..n_rs}, logv_{g,0..n_lv}]
    //   if is_twolevel && m_b == 0: [nu_dev_{g,0..n_items}, s_{g,0..n_rs}, logv_{g,0..n_lv}]
    let n_b_loc = if !is_twolevel {
        0
    } else if m_b > 0 {
        m_b
    } else {
        n_items
    };
    let n_rs = if is_twolevel { rand_phi_slots.len() } else { 0 };
    let rand_lv_factors: Vec<usize> = if is_twolevel {
        (0..m).filter(|&k| has_random_logv[k] != 0).collect()
    } else {
        Vec::new()
    };
    let n_lv = rand_lv_factors.len();
    let q_b = n_b_loc + n_rs + n_lv;

    let mut lambdab = vec![0.0; n_items * m_b.max(1)];
    if is_twolevel && m_b > 0 {
        for col in 0..m_b {
            for j in 0..n_items {
                if lambdab_mode[idx(j, col, n_items)] != 0 {
                    lambdab[idx(j, col, n_items)] = if equal_wb_loadings {
                        lambda0[idx(j, col.min(m - 1), n_items)]
                    } else {
                        0.6
                    };
                }
            }
        }
    }
    let mut theta_b: Vec<f64> = vec![0.25_f64; if is_twolevel && m_b > 0 { n_items } else { 0 }];
    let mut nu_cluster: Vec<f64> = vec![0.0_f64; g_cnt * n_items]; // person-specific indicator intercepts
    for g in 0..g_cnt {
        for j in 0..n_items {
            nu_cluster[idx(g, j, g_cnt)] = nu0[j];
        }
    }
    let mut re_b: Vec<f64> = vec![0.0_f64; g_cnt * q_b.max(1)]; // [loc, s_dev, logv_dev] -> we store full s and logv centered around pop mean
    let mut logv_pop: Vec<f64> = vec![0.0_f64; n_lv];
    let mut omega_b = eye(q_b, 0.15);
    if is_twolevel && m_b > 0 && !equal_wb_loadings {
        // When fb variance is fixed at 1 (e.g. fb@1 in 9.34 / 9.36), rescale fb block diagonal to 1
        for k in 0..m_b {
            omega_b[idx(k, k, q_b)] = 1.0;
        }
    }
    let fb_unit_var = is_twolevel && m_b > 0 && !equal_wb_loadings;

    // Latent factor trajectories eta (n_obs x m) and lag0 states (g_cnt x m)
    let mut eta_all = vec![0.0; n_obs * m];
    let mut eta0_all = vec![0.0; g_cnt * m];

    // Count output parameters in deterministic canonical order:
    // 1. Free within loadings (lambda0_mode == 1)
    // 2. Free lagged loadings if use_wnfs (lambda1_mode == 1)
    // 3. Fixed dynamic slopes (fixed_phi_slots) + population means of random slopes (rand_phi_slots)
    // 4. Population means of random logv (n_lv)
    // 5. Indicator intercepts / thresholds (n_items)
    // 6. Within indicator residual variances (continuous free items)
    // 7. Within factor variances / covariances (if !psi_fixed_one && n_lv == 0)
    // 8. If two-level & m_b > 0 && !equal_wb_loadings: free between loadings + theta_b (n_items)
    //    If two-level & m_b > 0 && equal_wb_loadings: theta_b (n_items)
    // 9. If two-level: lower triangle of omega_b (q_b*(q_b+1)/2)
    let free_l0: Vec<(usize, usize)> = (0..m)
        .flat_map(|c| (0..n_items).map(move |r| (r, c)))
        .filter(|&(r, c)| lambda0_mode[idx(r, c, n_items)] == 1)
        .collect();
    let free_l1: Vec<(usize, usize)> = if use_wnfs {
        (0..m)
            .flat_map(|c| (0..n_items).map(move |r| (r, c)))
            .filter(|&(r, c)| lambda1_mode[idx(r, c, n_items)] == 1)
            .collect()
    } else {
        Vec::new()
    };
    let free_theta_w: Vec<usize> = (0..n_items)
        .filter(|&j| !is_cat[j] && theta_fixed[j] <= 0.0)
        .collect();
    let n_psi_params = if psi_fixed_one || n_lv == m {
        0
    } else if psi_correlated {
        m * (m + 1) / 2
    } else {
        m - n_lv
    };
    let free_lb: Vec<(usize, usize)> = if is_twolevel && m_b > 0 && !equal_wb_loadings {
        (0..m_b)
            .flat_map(|c| (0..n_items).map(move |r| (r, c)))
            .filter(|&(r, c)| lambdab_mode[idx(r, c, n_items)] == 1)
            .collect()
    } else {
        Vec::new()
    };
    let n_theta_b = if is_twolevel && m_b > 0 { n_items } else { 0 };
    let n_om_b = if is_twolevel { q_b * (q_b + 1) / 2 } else { 0 };

    let n_params = free_l0.len()
        + free_l1.len()
        + fixed_phi_slots.len()
        + rand_phi_slots.len()
        + n_lv
        + n_items
        + free_theta_w.len()
        + n_psi_params
        + free_lb.len()
        + n_theta_b
        + n_om_b;

    let keep_indices: Vec<usize> = ((warmup + 1)..=iter)
        .filter(|i| (i - warmup - 1) % thin == 0)
        .collect();
    let n_keep = keep_indices.len();
    let mut draws = vec![0.0; n_keep * n_params];
    let mut latent_draws = vec![0.0; if store_latent { n_keep * n_obs * m } else { 0 }];
    let mut re_draws = vec![0.0; if is_twolevel { n_keep * g_cnt * q_b } else { 0 }];
    let mut fitted_accum = vec![0.0; n_obs * n_items];
    let mut theta_w_accum = vec![0.0; n_items];
    let mut deviance_sum = 0.0;
    let mut keep_pos = 0usize;

    let alpha_zero = vec![0.0; m];
    let p_pr = if prior_mode == 1 { 1e-8 } else { 0.01 };

    for step in 1..=iter {
        let om_b_inv = if is_twolevel {
            spd_invert(&omega_b, q_b)
        } else {
            Vec::new()
        };

        // 1. Sample latent factor trajectories per cluster via FFBS & probit y* augmentation
        for g in 0..g_cnt {
            let st = offsets[g];
            let en = offsets[g + 1];
            let tg = en - st;

            let nu_g: Vec<f64> = (0..n_items)
                .map(|j| {
                    if is_twolevel {
                        nu_cluster[idx(g, j, g_cnt)]
                    } else {
                        nu0[j]
                    }
                })
                .collect();

            let mut phi_g = phi_pop.clone();
            if is_twolevel {
                for (si, &(r, c)) in rand_phi_slots.iter().enumerate() {
                    phi_g[idx(r, c, m)] = (phi_pop[idx(r, c, m)]
                        + re_b[idx(g, n_b_loc + si, g_cnt)])
                    .clamp(-ar_bound, ar_bound);
                }
            }
            let mut psi_g = psi_pop.clone();
            if is_twolevel && n_lv > 0 {
                for (li, &fk) in rand_lv_factors.iter().enumerate() {
                    let lv = logv_pop[li] + re_b[idx(g, n_b_loc + n_rs + li, g_cnt)];
                    psi_g[idx(fk, fk, m)] = lv.clamp(-10.0, 10.0).exp();
                }
            }

            // Probit data augmentation for binary/ordinal indicators
            for j in 0..n_items {
                if is_cat[j] {
                    for t_loc in 0..tg {
                        let r_abs = st + t_loc;
                        let obs = y_in[idx(r_abs, j, n_obs)];
                        if obs.is_finite() {
                            let mut mean_ij = nu_g[j];
                            for k in 0..m {
                                mean_ij +=
                                    lambda0[idx(j, k, n_items)] * eta_all[idx(r_abs, k, n_obs)];
                            }
                            let (lo, hi) = if obs.round() as i32 <= 0 {
                                (f64::NEG_INFINITY, 0.0)
                            } else {
                                (0.0, f64::INFINITY)
                            };
                            y_work[idx(r_abs, j, n_obs)] = rng.rtruncnorm(mean_ij, 1.0, lo, hi);
                        }
                    }
                }
            }

            // Copy cluster indicator slice tg x n_items
            let mut y_g = vec![0.0; tg * n_items];
            for j in 0..n_items {
                for t_loc in 0..tg {
                    y_g[idx(t_loc, j, tg)] = y_work[idx(st + t_loc, j, n_obs)];
                }
            }

            if use_wnfs {
                let (st_g, e0_g, _) = ffbs_lagged_measurement(
                    &mut rng,
                    &y_g,
                    tg,
                    n_items,
                    m,
                    &nu_g,
                    &lambda0,
                    &lambda1,
                    &theta_w,
                    &alpha_zero,
                    &phi_g,
                    &psi_g,
                    ar_bound,
                );
                for k in 0..m {
                    eta0_all[idx(g, k, g_cnt)] = e0_g[k];
                    for t_loc in 0..tg {
                        eta_all[idx(st + t_loc, k, n_obs)] = st_g[idx(t_loc, k, tg)];
                    }
                }
            } else {
                let (st_g, _) = ffbs_dafs(
                    &mut rng,
                    &y_g,
                    tg,
                    n_items,
                    m,
                    &nu_g,
                    &lambda0,
                    &theta_w,
                    &alpha_zero,
                    &phi_g,
                    &psi_g,
                    ar_bound,
                );
                for k in 0..m {
                    for t_loc in 0..tg {
                        eta_all[idx(st + t_loc, k, n_obs)] = st_g[idx(t_loc, k, tg)];
                    }
                }
            }
        }

        // 2. Update measurement intercepts, within loadings (Lambda0, Lambda1), and within residual variances theta_w
        for j in 0..n_items {
            let th_j = theta_w[j].max(1e-6);
            // Active loading predictors on item j: [if !is_twolevel { intercept }, free l0 cols, free l1 cols]
            let l0_cols: Vec<usize> = (0..m)
                .filter(|&k| lambda0_mode[idx(j, k, n_items)] == 1)
                .collect();
            let l1_cols: Vec<usize> = if use_wnfs {
                (0..m)
                    .filter(|&k| lambda1_mode[idx(j, k, n_items)] == 1)
                    .collect()
            } else {
                Vec::new()
            };
            let include_int = !is_twolevel;
            let n_p = (if include_int { 1 } else { 0 }) + l0_cols.len() + l1_cols.len();

            if n_p > 0 {
                let mut ztz = eye(n_p, p_pr);
                let mut zty = vec![0.0; n_p];
                for g in 0..g_cnt {
                    let st = offsets[g];
                    let en = offsets[g + 1];
                    let tg = en - st;
                    for t_loc in 0..tg {
                        let r_abs = st + t_loc;
                        let yv = y_work[idx(r_abs, j, n_obs)];
                        if !yv.is_finite() {
                            continue;
                        }
                        let mut rem = yv
                            - if is_twolevel {
                                nu_cluster[idx(g, j, g_cnt)]
                            } else {
                                0.0
                            };
                        // Subtract fixed marker loadings (lambda0_mode == 2)
                        for k in 0..m {
                            if lambda0_mode[idx(j, k, n_items)] == 2 {
                                rem -= lambda0[idx(j, k, n_items)] * eta_all[idx(r_abs, k, n_obs)];
                            }
                        }
                        let mut z_row = Vec::with_capacity(n_p);
                        if include_int {
                            z_row.push(1.0);
                        }
                        for &k in &l0_cols {
                            z_row.push(eta_all[idx(r_abs, k, n_obs)]);
                        }
                        for &k in &l1_cols {
                            let e_prev = if t_loc == 0 {
                                eta0_all[idx(g, k, g_cnt)]
                            } else {
                                eta_all[idx(r_abs - 1, k, n_obs)]
                            };
                            z_row.push(e_prev);
                        }
                        for a in 0..n_p {
                            zty[a] += z_row[a] * rem / th_j;
                            for b in 0..n_p {
                                ztz[idx(a, b, n_p)] += z_row[a] * z_row[b] / th_j;
                            }
                        }
                    }
                }
                // Cross-level invariance contribution from between model (9.35: f BY u1-u4*(1-4); fb BY u1-u4*(1-4))
                if is_twolevel && m_b > 0 && equal_wb_loadings && !l0_cols.is_empty() {
                    let th_bj = theta_b[j].max(1e-6);
                    for g in 0..g_cnt {
                        let rem_b = nu_cluster[idx(g, j, g_cnt)] - nu0[j];
                        for (ci, &k) in l0_cols.iter().enumerate() {
                            if k < m_b && lambdab_mode[idx(j, k, n_items)] != 0 {
                                let fb_gk = re_b[idx(g, k, g_cnt)];
                                zty[ci] += fb_gk * rem_b / th_bj;
                                ztz[idx(ci, ci, n_p)] += fb_gk * fb_gk / th_bj;
                            }
                        }
                    }
                }
                let draw_b = rmvnorm_precision(&mut rng, &ztz, &zty, n_p);
                let mut pos = 0usize;
                if include_int {
                    nu0[j] = draw_b[0];
                    pos += 1;
                }
                for &k in &l0_cols {
                    // Enforce positive sign when psi_fixed_one && k == 0 && j == 0 for sign stability, or natural Gibbs
                    lambda0[idx(j, k, n_items)] = draw_b[pos];
                    if equal_wb_loadings && k < m_b {
                        lambdab[idx(j, k, n_items)] = draw_b[pos];
                    }
                    pos += 1;
                }
                for &k in &l1_cols {
                    lambda1[idx(j, k, n_items)] = draw_b[pos];
                    pos += 1;
                }
            }

            // Sign-anchoring when all loadings are free with fixed factor variance (e.g. Mplus 6.27 IRT):
            // Mplus starts loadings positive and identifies factor polarity positively on primary indicators.
            if (psi_fixed_one || fix_logv_mean_zero) && j == n_items - 1 {
                for k in 0..m {
                    let sum_l: f64 = (0..n_items)
                        .filter(|&jj| lambda0_mode[idx(jj, k, n_items)] != 0)
                        .map(|jj| lambda0[idx(jj, k, n_items)])
                        .sum();
                    if sum_l < 0.0 {
                        for jj in 0..n_items {
                            lambda0[idx(jj, k, n_items)] = -lambda0[idx(jj, k, n_items)];
                            if equal_wb_loadings && k < m_b {
                                lambdab[idx(jj, k, n_items)] = -lambdab[idx(jj, k, n_items)];
                            }
                        }
                        for r in 0..n_obs {
                            eta_all[idx(r, k, n_obs)] = -eta_all[idx(r, k, n_obs)];
                        }
                        if is_twolevel && k < m_b && equal_wb_loadings {
                            for g in 0..g_cnt {
                                re_b[idx(g, k, g_cnt)] = -re_b[idx(g, k, g_cnt)];
                            }
                        }
                    }
                }
            }

            // Update continuous residual variance theta_w[j] if free
            if !is_cat[j] && theta_fixed[j] <= 0.0 {
                let mut rss = 0.0;
                let mut n_obs_j = 0usize;
                for g in 0..g_cnt {
                    let st = offsets[g];
                    let en = offsets[g + 1];
                    let tg = en - st;
                    let nug = if is_twolevel {
                        nu_cluster[idx(g, j, g_cnt)]
                    } else {
                        nu0[j]
                    };
                    for t_loc in 0..tg {
                        let r_abs = st + t_loc;
                        let yv = y_work[idx(r_abs, j, n_obs)];
                        if yv.is_finite() {
                            let mut fv = nug;
                            for k in 0..m {
                                fv += lambda0[idx(j, k, n_items)] * eta_all[idx(r_abs, k, n_obs)];
                                if use_wnfs {
                                    let ep = if t_loc == 0 {
                                        eta0_all[idx(g, k, g_cnt)]
                                    } else {
                                        eta_all[idx(r_abs - 1, k, n_obs)]
                                    };
                                    fv += lambda1[idx(j, k, n_items)] * ep;
                                }
                            }
                            let e = yv - fv;
                            rss += e * e;
                            n_obs_j += 1;
                        }
                    }
                }
                let (sh, sc) = if prior_mode == 1 {
                    ((0.5 * n_obs_j as f64 - 1.0).max(0.5), 0.5 * rss.max(1e-8))
                } else {
                    (0.01 + 0.5 * n_obs_j as f64, 0.01 + 0.5 * rss)
                };
                theta_w[j] = rng.rinv_gamma(sh, sc);
            }
        }

        // 3. Update Two-Level Between Measurement & Random Effects (9.33, 9.34, 9.35, 9.36)
        if is_twolevel {
            // 3a. Update person-specific indicator intercepts nu_cluster[g, j]
            for g in 0..g_cnt {
                let st = offsets[g];
                let en = offsets[g + 1];
                let tg = en - st;
                for j in 0..n_items {
                    let th_j = theta_w[j].max(1e-6);
                    let mut sum_r = 0.0;
                    let mut cnt = 0usize;
                    for t_loc in 0..tg {
                        let r_abs = st + t_loc;
                        let yv = y_work[idx(r_abs, j, n_obs)];
                        if yv.is_finite() {
                            let mut fit_w = 0.0;
                            for k in 0..m {
                                fit_w +=
                                    lambda0[idx(j, k, n_items)] * eta_all[idx(r_abs, k, n_obs)];
                                if use_wnfs {
                                    let ep = if t_loc == 0 {
                                        eta0_all[idx(g, k, g_cnt)]
                                    } else {
                                        eta_all[idx(r_abs - 1, k, n_obs)]
                                    };
                                    fit_w += lambda1[idx(j, k, n_items)] * ep;
                                }
                            }
                            sum_r += yv - fit_w;
                            cnt += 1;
                        }
                    }
                    if m_b > 0 {
                        let th_bj = theta_b[j].max(1e-6);
                        let mut prior_m = nu0[j];
                        for kb in 0..m_b {
                            prior_m += lambdab[idx(j, kb, n_items)] * re_b[idx(g, kb, g_cnt)];
                        }
                        let pr = cnt as f64 / th_j + 1.0 / th_bj;
                        let inf = sum_r / th_j + prior_m / th_bj;
                        nu_cluster[idx(g, j, g_cnt)] = inf / pr + (1.0 / pr).sqrt() * rng.rnorm();
                    } else {
                        // Direct random intercept in re_b[g, j] around nu0[j]
                        let o_jj = om_b_inv[idx(j, j, q_b)].max(1e-6);
                        let mut inf = sum_r / th_j + o_jj * nu0[j];
                        for q2 in 0..q_b {
                            if q2 != j {
                                inf -= om_b_inv[idx(j, q2, q_b)] * re_b[idx(g, q2, g_cnt)];
                            }
                        }
                        let pr = cnt as f64 / th_j + o_jj;
                        let draw_nu = inf / pr + (1.0 / pr).sqrt() * rng.rnorm();
                        nu_cluster[idx(g, j, g_cnt)] = draw_nu;
                        re_b[idx(g, j, g_cnt)] = draw_nu - nu0[j];
                    }
                }
            }

            // Update grand indicator intercepts nu0[j], between loadings lambdab, and theta_b
            for j in 0..n_items {
                if m_b > 0 {
                    let th_bj = theta_b[j].max(1e-6);
                    let lb_cols: Vec<usize> = if !equal_wb_loadings {
                        (0..m_b)
                            .filter(|&kb| lambdab_mode[idx(j, kb, n_items)] == 1)
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let pa = 1 + lb_cols.len();
                    let mut wd = vec![0.0; g_cnt * pa];
                    let mut yv = vec![0.0; g_cnt];
                    for g in 0..g_cnt {
                        wd[idx(g, 0, g_cnt)] = 1.0;
                        let mut rem = nu_cluster[idx(g, j, g_cnt)];
                        for kb in 0..m_b {
                            if equal_wb_loadings || lambdab_mode[idx(j, kb, n_items)] == 2 {
                                rem -= lambdab[idx(j, kb, n_items)] * re_b[idx(g, kb, g_cnt)];
                            }
                        }
                        for (ci, &kb) in lb_cols.iter().enumerate() {
                            wd[idx(g, 1 + ci, g_cnt)] = re_b[idx(g, kb, g_cnt)];
                        }
                        yv[g] = rem;
                    }
                    let wtw = mat_t_mul(&wd, &wd, pa, g_cnt, pa);
                    let wty = mat_t_mul(&wd, &yv, pa, g_cnt, 1);
                    let mut pr = vec![0.0; pa * pa];
                    for i in 0..(pa * pa) {
                        pr[i] = wtw[i] / th_bj;
                    }
                    for i in 0..pa {
                        pr[idx(i, i, pa)] += p_pr;
                    }
                    let inf: Vec<f64> = wty.iter().map(|v| v / th_bj).collect();
                    let b = rmvnorm_precision(&mut rng, &pr, &inf, pa);
                    nu0[j] = b[0];
                    for (ci, &kb) in lb_cols.iter().enumerate() {
                        lambdab[idx(j, kb, n_items)] = b[1 + ci];
                    }
                    // Anchor positive sign on free between loadings when fb_unit_var
                    if fb_unit_var && j == n_items - 1 {
                        for kb in 0..m_b {
                            let s_lb: f64 =
                                (0..n_items).map(|jj| lambdab[idx(jj, kb, n_items)]).sum();
                            if s_lb < 0.0 {
                                for jj in 0..n_items {
                                    lambdab[idx(jj, kb, n_items)] = -lambdab[idx(jj, kb, n_items)];
                                }
                                for g in 0..g_cnt {
                                    re_b[idx(g, kb, g_cnt)] = -re_b[idx(g, kb, g_cnt)];
                                }
                            }
                        }
                    }
                    let mut rss_b = 0.0;
                    for g in 0..g_cnt {
                        let mut pr_b = nu0[j];
                        for kb in 0..m_b {
                            pr_b += lambdab[idx(j, kb, n_items)] * re_b[idx(g, kb, g_cnt)];
                        }
                        let eb = nu_cluster[idx(g, j, g_cnt)] - pr_b;
                        rss_b += eb * eb;
                    }
                    theta_b[j] = rng.rinv_gamma(0.01 + 0.5 * g_cnt as f64, 0.01 + 0.5 * rss_b);
                } else {
                    let v_j = omega_b[idx(j, j, q_b)].max(1e-6);
                    let m_nu: f64 = (0..g_cnt)
                        .map(|g| nu_cluster[idx(g, j, g_cnt)])
                        .sum::<f64>()
                        / g_cnt as f64;
                    let old_nu = nu0[j];
                    nu0[j] = m_nu + (v_j / g_cnt as f64).sqrt() * rng.rnorm();
                    for g in 0..g_cnt {
                        re_b[idx(g, j, g_cnt)] += old_nu - nu0[j];
                    }
                }
            }

            // 3b. Update between factors fb_{g, 0..m_b}, random dynamic slopes s_g, and random logv_g!
            for g in 0..g_cnt {
                let st = offsets[g];
                let en = offsets[g + 1];
                let tg = en - st;

                // Update fb_{g, 0..m_b} conditional on nu_cluster[g, .] and (s_g, logv_g)
                if m_b > 0 {
                    let mut pr = vec![0.0; m_b * m_b];
                    let mut inf = vec![0.0; m_b];
                    for ca in 0..m_b {
                        for ra in 0..m_b {
                            pr[idx(ra, ca, m_b)] = om_b_inv[idx(ra, ca, q_b)];
                        }
                        for q_o in m_b..q_b {
                            inf[ca] -= om_b_inv[idx(ca, q_o, q_b)] * re_b[idx(g, q_o, g_cnt)];
                        }
                    }
                    for j in 0..n_items {
                        let th_bj = theta_b[j].max(1e-6);
                        let y_b = nu_cluster[idx(g, j, g_cnt)] - nu0[j];
                        for ra in 0..m_b {
                            let lr = lambdab[idx(j, ra, n_items)];
                            inf[ra] += lr * y_b / th_bj;
                            for ca in 0..m_b {
                                pr[idx(ra, ca, m_b)] += lr * lambdab[idx(j, ca, n_items)] / th_bj;
                            }
                        }
                    }
                    let fb_draw = rmvnorm_precision(&mut rng, &pr, &inf, m_b);
                    for kb in 0..m_b {
                        re_b[idx(g, kb, g_cnt)] = fb_draw[kb];
                    }
                }

                // Update random dynamic slopes re_b[g, n_b_loc..n_b_loc+n_rs]
                if n_rs > 0 && tg >= 2 {
                    let mut pr = vec![0.0; n_rs * n_rs];
                    let mut inf = vec![0.0; n_rs];
                    for ca in 0..n_rs {
                        for ra in 0..n_rs {
                            pr[idx(ra, ca, n_rs)] = om_b_inv[idx(n_b_loc + ra, n_b_loc + ca, q_b)];
                        }
                        for q_o in 0..q_b {
                            if q_o < n_b_loc || q_o >= n_b_loc + n_rs {
                                inf[ca] -= om_b_inv[idx(n_b_loc + ca, q_o, q_b)]
                                    * re_b[idx(g, q_o, g_cnt)];
                            }
                        }
                    }
                    let mut psi_g = psi_pop.clone();
                    for (li, &fk) in rand_lv_factors.iter().enumerate() {
                        psi_g[idx(fk, fk, m)] = (logv_pop[li]
                            + re_b[idx(g, n_b_loc + n_rs + li, g_cnt)])
                        .clamp(-10.0, 10.0)
                        .exp();
                    }
                    let psi_g_inv = spd_invert(&psi_g, m);
                    for t_loc in 1..tg {
                        let r_abs = st + t_loc;
                        // Residual around phi_pop * eta_{t-1}
                        let mut r_base = vec![0.0; m];
                        for r_f in 0..m {
                            let mut fit0 = 0.0;
                            for c_f in 0..m {
                                fit0 +=
                                    phi_pop[idx(r_f, c_f, m)] * eta_all[idx(r_abs - 1, c_f, n_obs)];
                            }
                            r_base[r_f] = eta_all[idx(r_abs, r_f, n_obs)] - fit0;
                        }
                        for (a, &(ra, ca)) in rand_phi_slots.iter().enumerate() {
                            let xa = eta_all[idx(r_abs - 1, ca, n_obs)];
                            for rf in 0..m {
                                inf[a] += xa * psi_g_inv[idx(ra, rf, m)] * r_base[rf];
                            }
                            for (b, &(rb, cb)) in rand_phi_slots.iter().enumerate() {
                                let xb = eta_all[idx(r_abs - 1, cb, n_obs)];
                                pr[idx(a, b, n_rs)] += xa * xb * psi_g_inv[idx(ra, rb, m)];
                            }
                        }
                    }
                    let s_draw = rmvnorm_precision(&mut rng, &pr, &inf, n_rs);
                    // Check stationarity of proposed Phi_g
                    let mut cand_phi = phi_pop.clone();
                    for (si, &(r, c)) in rand_phi_slots.iter().enumerate() {
                        cand_phi[idx(r, c, m)] += s_draw[si];
                    }
                    if is_stable_transition(&cand_phi, m, ar_bound) {
                        for si in 0..n_rs {
                            re_b[idx(g, n_b_loc + si, g_cnt)] = s_draw[si];
                        }
                    }
                }

                // Update random log factor innovation variances via MH
                if n_lv > 0 && tg >= 2 {
                    let mut phi_g = phi_pop.clone();
                    for (si, &(r, c)) in rand_phi_slots.iter().enumerate() {
                        phi_g[idx(r, c, m)] = (phi_pop[idx(r, c, m)]
                            + re_b[idx(g, n_b_loc + si, g_cnt)])
                        .clamp(-ar_bound, ar_bound);
                    }
                    for (li, &fk) in rand_lv_factors.iter().enumerate() {
                        let q_lv = n_b_loc + n_rs + li;
                        let cur_u = re_b[idx(g, q_lv, g_cnt)];
                        let prop_u = (cur_u + 0.22 * rng.rnorm()).clamp(-8.0, 8.0);
                        let mut rss_f = 0.0;
                        for t_loc in 1..tg {
                            let r_abs = st + t_loc;
                            let mut pr_f = 0.0;
                            for cf in 0..m {
                                pr_f += phi_g[idx(fk, cf, m)] * eta_all[idx(r_abs - 1, cf, n_obs)];
                            }
                            let ef = eta_all[idx(r_abs, fk, n_obs)] - pr_f;
                            rss_f += ef * ef;
                        }
                        let ne = (tg - 1) as f64;
                        let lp = |uv: f64| -> f64 {
                            let lv = logv_pop[li] + uv;
                            let mut vec_b: Vec<f64> =
                                (0..q_b).map(|q| re_b[idx(g, q, g_cnt)]).collect();
                            vec_b[q_lv] = uv;
                            -0.5 * ne * lv
                                - 0.5 * (-lv).exp() * rss_f
                                - 0.5 * quad_form(&vec_b, &om_b_inv, q_b)
                        };
                        if rng.runif().ln() < lp(prop_u) - lp(cur_u) {
                            re_b[idx(g, q_lv, g_cnt)] = prop_u;
                        }
                    }
                }
            }

            // Update population means of random slopes (phi_pop) and random logv (logv_pop) + Omega_B!
            for (si, &(r, c)) in rand_phi_slots.iter().enumerate() {
                let q_s = n_b_loc + si;
                let vs = omega_b[idx(q_s, q_s, q_b)].max(1e-6);
                let mean_u: f64 =
                    (0..g_cnt).map(|g| re_b[idx(g, q_s, g_cnt)]).sum::<f64>() / g_cnt as f64;
                let old_m = phi_pop[idx(r, c, m)];
                let new_m = (old_m + mean_u + (vs / g_cnt as f64).sqrt() * rng.rnorm())
                    .clamp(-ar_bound, ar_bound);
                phi_pop[idx(r, c, m)] = new_m;
                for g in 0..g_cnt {
                    re_b[idx(g, q_s, g_cnt)] += old_m - new_m;
                }
            }
            for (li, lv_slot) in logv_pop.iter_mut().enumerate().take(n_lv) {
                let q_lv = n_b_loc + n_rs + li;
                if fix_logv_mean_zero {
                    *lv_slot = 0.0;
                } else {
                    let vl = omega_b[idx(q_lv, q_lv, q_b)].max(1e-6);
                    let mean_u: f64 =
                        (0..g_cnt).map(|g| re_b[idx(g, q_lv, g_cnt)]).sum::<f64>() / g_cnt as f64;
                    let old_m = *lv_slot;
                    let new_m = old_m + mean_u + (vl / g_cnt as f64).sqrt() * rng.rnorm();
                    *lv_slot = new_m;
                    for g in 0..g_cnt {
                        re_b[idx(g, q_lv, g_cnt)] += old_m - new_m;
                    }
                }
            }
            // Center fb if m_b > 0
            for kb in 0..m_b {
                let mb_mean: f64 =
                    (0..g_cnt).map(|g| re_b[idx(g, kb, g_cnt)]).sum::<f64>() / g_cnt as f64;
                for g in 0..g_cnt {
                    re_b[idx(g, kb, g_cnt)] -= mb_mean;
                }
            }
            let mut sb = eye(q_b, if prior_mode == 1 { 1e-4 } else { 0.05 });
            for g in 0..g_cnt {
                for ca in 0..q_b {
                    for ra in 0..q_b {
                        sb[idx(ra, ca, q_b)] += re_b[idx(g, ra, g_cnt)] * re_b[idx(g, ca, g_cnt)];
                    }
                }
            }
            omega_b = rinv_wishart(&mut rng, (q_b as f64 + 2.0) + g_cnt as f64, &sb, q_b);
            if fb_unit_var {
                // Parameter-expanded rescaling: enforce diag(Omega_B)[0..m_b] = 1 (matching Mplus fb@1!)
                for kb in 0..m_b {
                    let sd_k = omega_b[idx(kb, kb, q_b)].max(1e-8).sqrt();
                    for q in 0..q_b {
                        omega_b[idx(kb, q, q_b)] /= sd_k;
                        omega_b[idx(q, kb, q_b)] /= sd_k;
                    }
                    omega_b[idx(kb, kb, q_b)] = 1.0;
                    for g in 0..g_cnt {
                        re_b[idx(g, kb, g_cnt)] /= sd_k;
                    }
                    for j in 0..n_items {
                        lambdab[idx(j, kb, n_items)] *= sd_k;
                    }
                }
                symmetrize(&mut omega_b, q_b);
            }
        }

        // 4. Update fixed dynamic slopes (fixed_phi_slots) & factor innovation covariance psi_pop!
        if !fixed_phi_slots.is_empty() {
            let nf = fixed_phi_slots.len();
            let psi_inv = spd_invert(&psi_pop, m);
            let mut pr = eye(nf, p_pr);
            let mut inf = vec![0.0; nf];
            for g in 0..g_cnt {
                let st = offsets[g];
                let en = offsets[g + 1];
                let tg = en - st;
                for t_loc in 1..tg {
                    let r_abs = st + t_loc;
                    let mut y_f = vec![0.0; m];
                    for rf in 0..m {
                        y_f[rf] = eta_all[idx(r_abs, rf, n_obs)];
                        if is_twolevel {
                            for (si, &(rr, cc)) in rand_phi_slots.iter().enumerate() {
                                if rr == rf {
                                    let ph =
                                        phi_pop[idx(rr, cc, m)] + re_b[idx(g, n_b_loc + si, g_cnt)];
                                    y_f[rf] -= ph * eta_all[idx(r_abs - 1, cc, n_obs)];
                                }
                            }
                        }
                    }
                    for (a, &(ra, ca)) in fixed_phi_slots.iter().enumerate() {
                        let xa = eta_all[idx(r_abs - 1, ca, n_obs)];
                        for rf in 0..m {
                            inf[a] += xa * psi_inv[idx(ra, rf, m)] * y_f[rf];
                        }
                        for (b, &(rb, cb)) in fixed_phi_slots.iter().enumerate() {
                            let xb = eta_all[idx(r_abs - 1, cb, n_obs)];
                            pr[idx(a, b, nf)] += xa * xb * psi_inv[idx(ra, rb, m)];
                        }
                    }
                }
            }
            let cand_vec = rmvnorm_precision(&mut rng, &pr, &inf, nf);
            let mut cand_phi = phi_pop.clone();
            for (a, &(r, c)) in fixed_phi_slots.iter().enumerate() {
                cand_phi[idx(r, c, m)] = cand_vec[a];
            }
            if is_stable_transition(&cand_phi, m, ar_bound) {
                phi_pop = cand_phi;
            }
        }

        if !psi_fixed_one && n_lv < m {
            // Collect factor innovation residuals across clusters
            let mut e_f = Vec::new();
            let mut n_trans = 0usize;
            for g in 0..g_cnt {
                let st = offsets[g];
                let en = offsets[g + 1];
                let tg = en - st;
                let mut phi_g = phi_pop.clone();
                if is_twolevel {
                    for (si, &(r, c)) in rand_phi_slots.iter().enumerate() {
                        phi_g[idx(r, c, m)] += re_b[idx(g, n_b_loc + si, g_cnt)];
                    }
                }
                let start_t = if use_wnfs && phi_mode.iter().all(|&v| v == 0) {
                    0
                } else {
                    1
                };
                for t_loc in start_t..tg {
                    let r_abs = st + t_loc;
                    for rf in 0..m {
                        let mut pr_f = 0.0;
                        if t_loc >= 1 {
                            for cf in 0..m {
                                pr_f += phi_g[idx(rf, cf, m)] * eta_all[idx(r_abs - 1, cf, n_obs)];
                            }
                        }
                        e_f.push(eta_all[idx(r_abs, rf, n_obs)] - pr_f);
                    }
                    n_trans += 1;
                }
            }
            if n_trans > m {
                if psi_correlated {
                    let mut sp = eye(m, if prior_mode == 1 { 1e-4 } else { 0.02 });
                    for r_t in 0..n_trans {
                        for ca in 0..m {
                            for ra in 0..m {
                                sp[idx(ra, ca, m)] += e_f[r_t * m + ra] * e_f[r_t * m + ca];
                            }
                        }
                    }
                    let df_p = if prior_mode == 1 {
                        (n_trans as f64 - m as f64 - 1.0).max(m as f64 + 1.0)
                    } else {
                        (m as f64 + 2.0) + n_trans as f64
                    };
                    psi_pop = rinv_wishart(&mut rng, df_p, &sp, m);
                } else {
                    for fk in 0..m {
                        if !rand_lv_factors.contains(&fk) {
                            let mut rss_k = 0.0;
                            for r_t in 0..n_trans {
                                let ek = e_f[r_t * m + fk];
                                rss_k += ek * ek;
                            }
                            if !is_twolevel && m == 1 && !use_wnfs {
                                let phi0 = phi_pop[0].clamp(-ar_bound, ar_bound);
                                let e0 = eta_all[0];
                                rss_k += (1.0 - phi0 * phi0).max(0.01) * e0 * e0;
                            }
                            let n_eff_k = if !is_twolevel && m == 1 && !use_wnfs {
                                n_trans + 1
                            } else {
                                n_trans
                            };
                            let (sh, sc) = if prior_mode == 1 {
                                ((0.5 * n_eff_k as f64 - 1.0).max(0.5), 0.5 * rss_k.max(1e-8))
                            } else {
                                (0.01 + 0.5 * n_eff_k as f64, 0.01 + 0.5 * rss_k)
                            };
                            psi_pop[idx(fk, fk, m)] = rng.rinv_gamma(sh, sc);
                        }
                    }
                }
            }
        }

        // Store retained posterior draw
        if keep_pos < n_keep && step == keep_indices[keep_pos] {
            let mut cur = 0usize;
            for &(r, c) in &free_l0 {
                draws[idx(keep_pos, cur, n_keep)] = lambda0[idx(r, c, n_items)];
                cur += 1;
            }
            for &(r, c) in &free_l1 {
                draws[idx(keep_pos, cur, n_keep)] = lambda1[idx(r, c, n_items)];
                cur += 1;
            }
            for &(r, c) in &fixed_phi_slots {
                draws[idx(keep_pos, cur, n_keep)] = phi_pop[idx(r, c, m)];
                cur += 1;
            }
            for &(r, c) in &rand_phi_slots {
                draws[idx(keep_pos, cur, n_keep)] = phi_pop[idx(r, c, m)];
                cur += 1;
            }
            for &lv_m in &logv_pop {
                draws[idx(keep_pos, cur, n_keep)] = lv_m;
                cur += 1;
            }
            for j in 0..n_items {
                let v = if is_cat[j] { -nu0[j] } else { nu0[j] };
                draws[idx(keep_pos, cur, n_keep)] = v;
                cur += 1;
            }
            for &j in &free_theta_w {
                draws[idx(keep_pos, cur, n_keep)] = theta_w[j];
                cur += 1;
            }
            if !psi_fixed_one && n_lv < m {
                if psi_correlated {
                    for ca in 0..m {
                        for ra in ca..m {
                            draws[idx(keep_pos, cur, n_keep)] = psi_pop[idx(ra, ca, m)];
                            cur += 1;
                        }
                    }
                } else {
                    for fk in 0..m {
                        if !rand_lv_factors.contains(&fk) {
                            draws[idx(keep_pos, cur, n_keep)] = psi_pop[idx(fk, fk, m)];
                            cur += 1;
                        }
                    }
                }
            }
            for &(r, c) in &free_lb {
                draws[idx(keep_pos, cur, n_keep)] = lambdab[idx(r, c, n_items)];
                cur += 1;
            }
            for &tb in &theta_b {
                draws[idx(keep_pos, cur, n_keep)] = tb;
                cur += 1;
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
                        let base = if q >= n_b_loc && q < n_b_loc + n_rs {
                            let (rr, cc) = rand_phi_slots[q - n_b_loc];
                            phi_pop[idx(rr, cc, m)]
                        } else if q >= n_b_loc + n_rs {
                            logv_pop[q - n_b_loc - n_rs]
                        } else if m_b == 0 {
                            nu0[q]
                        } else {
                            0.0
                        };
                        re_draws[keep_pos + n_keep * (g + g_cnt * q)] =
                            base + re_b[idx(g, q, g_cnt)];
                    }
                }
            }
            if store_latent {
                for k in 0..m {
                    for r in 0..n_obs {
                        latent_draws[keep_pos + n_keep * (r + n_obs * k)] =
                            eta_all[idx(r, k, n_obs)];
                    }
                }
            }

            // Conditional measurement deviance D(theta, eta) for Mplus-compatible DIC/pD!
            let mut ll_cond = 0.0;
            for j in 0..n_items {
                theta_w_accum[j] += theta_w[j];
            }
            for g in 0..g_cnt {
                let st = offsets[g];
                let en = offsets[g + 1];
                let tg = en - st;
                for j in 0..n_items {
                    let nug = if is_twolevel {
                        nu_cluster[idx(g, j, g_cnt)]
                    } else {
                        nu0[j]
                    };
                    let th_j = theta_w[j].max(1e-6);
                    for t_loc in 0..tg {
                        let r_abs = st + t_loc;
                        let mut fv = nug;
                        for k in 0..m {
                            fv += lambda0[idx(j, k, n_items)] * eta_all[idx(r_abs, k, n_obs)];
                            if use_wnfs {
                                let ep = if t_loc == 0 {
                                    eta0_all[idx(g, k, g_cnt)]
                                } else {
                                    eta_all[idx(r_abs - 1, k, n_obs)]
                                };
                                fv += lambda1[idx(j, k, n_items)] * ep;
                            }
                        }
                        fitted_accum[idx(r_abs, j, n_obs)] += fv;
                        let y_obs = y_in[idx(r_abs, j, n_obs)];
                        if y_obs.is_finite() && !is_cat[j] {
                            let e = y_obs - fv;
                            ll_cond += -0.5 * ((2.0 * PI * th_j).ln() + e * e / th_j);
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
        if !is_cat[j] {
            let th_bar = (theta_w_accum[j] * inv_keep).max(1e-6);
            for r in 0..n_obs {
                let y_obs = y_in[idx(r, j, n_obs)];
                if y_obs.is_finite() {
                    let e = y_obs - fitted_mean[idx(r, j, n_obs)];
                    ll_hat += -0.5 * ((2.0 * PI * th_bar).ln() + e * e / th_bar);
                }
            }
        }
    }
    let pd = (d_bar - (-2.0 * ll_hat)).max(0.0);
    let dic = d_bar + pd;

    DynamicSemOutput {
        draws,
        param_count: n_params,
        latent_draws,
        re_draws,
        fitted_mean,
        dic,
        pd,
    }
}
