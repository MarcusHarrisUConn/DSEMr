use crate::linalg::{
    eye, idx, mat_mul, mat_mul_t, mat_t_mul, mat_vec, rinv_wishart, rmvnorm_precision, spd_invert,
    symmetrize,
};
use crate::rng::Rng;
use std::f64::consts::PI;

pub struct ThreeLevelOutput {
    pub draws: Vec<f64>,
    pub param_count: usize,
    pub l2_draws: Vec<f64>,
    pub l3_draws: Vec<f64>,
    pub fitted_mean: Vec<f64>,
    pub dic: f64,
    pub pd: f64,
}

/// Three-Level DSEM (DSEM3, Asparouhov & Muthen 2026) exact simultaneous two-level block Gibbs,
/// non-random Level-2 AR, random residual variances & covariances, short-T stationary MH,
/// PSEM shrinkage, and Newton-Laplace Independence MH sampler.
#[allow(clippy::too_many_arguments)]
pub fn mcmc_threelevel(
    y_in: &[f64],
    x_in: &[f64],
    l2_sizes: &[usize],
    l3_of_l2: &[usize],
    g3_cnt: usize,
    w2_in: &[f64],
    w3_in: &[f64],
    d: usize,
    kx: usize,
    kw2: usize,
    kw3: usize,
    lags: &[usize],
    has_l2_re: &[i32],
    has_l3_re: &[i32],
    latent_center_x: &[i32],
    l2_ar_mask: &[i32],      // length d: 1 if outcome j has non-random AR(1) at Level 2
    re_l3_on_w3_mask: &[i32], // kw3 x q_l3: 1 if L3 random effect p regresses on w3[k]
    re_l2_on_w2_mask: &[i32], // kw2 x q_l2: 1 if L2 random effect p regresses on w2[k]
    shared_cov_loadings: &[f64], // length d: shared within residual covariance factor loadings (logvf)
    residual_dynamics: bool,
    prior_mode: i32,
    beta_sd: f64,
    iter: usize,
    warmup: usize,
    thin: usize,
    seed: u64,
) -> ThreeLevelOutput {
    let mut rng = Rng::from_seed(seed);
    let g2_cnt = l2_sizes.len();
    let n_obs: usize = l2_sizes.iter().sum();
    let max_lag = lags.iter().copied().max().unwrap_or(1).max(1);
    let n_lags = lags.len();
    let n_slopes = n_lags * d + kx;
    let lin_sl_end = d + d * n_slopes;
    let has_random_logvf = d > 1
        && shared_cov_loadings.len() >= d
        && shared_cov_loadings.iter().take(d).any(|&v| v.abs() > 1e-12);
    let cov_lam: Vec<f64> = if has_random_logvf {
        shared_cov_loadings[..d].to_vec()
    } else {
        vec![0.0; d]
    };
    let q_tot = lin_sl_end + d + usize::from(has_random_logvf);
    let q_vf_slot = if has_random_logvf {
        Some(lin_sl_end + d)
    } else {
        None
    };

    let mut offsets = Vec::with_capacity(g2_cnt + 1);
    offsets.push(0usize);
    for &sz in l2_sizes {
        offsets.push(offsets.last().unwrap() + sz);
    }

    let l2_slots: Vec<usize> = (0..q_tot).filter(|&q| has_l2_re[q] != 0).collect();
    let l3_slots: Vec<usize> = (0..q_tot).filter(|&q| has_l3_re[q] != 0).collect();
    let q_l2 = l2_slots.len();
    let q_l3 = l3_slots.len();
    let mut slot_to_l2 = vec![-1i32; q_tot];
    let mut slot_to_l3 = vec![-1i32; q_tot];
    for (i, &q) in l2_slots.iter().enumerate() {
        slot_to_l2[q] = i as i32;
    }
    for (i, &q) in l3_slots.iter().enumerate() {
        slot_to_l3[q] = i as i32;
    }

    let any_rand_logv = (0..d)
        .any(|j| slot_to_l2[lin_sl_end + j] >= 0 || slot_to_l3[lin_sl_end + j] >= 0);
    let use_full_sigma_w = d > 1 && !any_rand_logv && !has_random_logvf;
    let l2_ar_active: Vec<usize> = (0..d)
        .filter(|&j| l2_ar_mask.get(j).copied().unwrap_or(0) != 0 && slot_to_l2[j] >= 0)
        .collect();
    let any_l2_ar = !l2_ar_active.is_empty();

    // Impute missing y
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
                y_work[idx(r, j, n_obs)] = m;
            }
        }
    }

    // Three-level latent covariate decomposition
    let mut xb3: Vec<f64> = vec![0.0_f64; g3_cnt * kx];
    let mut xb2: Vec<f64> = vec![0.0_f64; g2_cnt * kx];
    let mut xv1: Vec<f64> = vec![1.0_f64; kx];
    let mut xv2: Vec<f64> = vec![0.5_f64; kx];
    let mut xv3: Vec<f64> = vec![0.5_f64; kx];
    let mut x_gm: Vec<f64> = vec![0.0_f64; kx];
    for k in 0..kx {
        if latent_center_x[k] != 0 {
            let mut l3_sum = vec![0.0; g3_cnt];
            let mut l3_n = vec![0usize; g3_cnt];
            for c in 0..g2_cnt {
                let st = offsets[c];
                let en = offsets[c + 1];
                let m_c = (st..en)
                    .map(|r| x_in[idx(r, k, n_obs)])
                    .filter(|v| v.is_finite())
                    .sum::<f64>()
                    / ((en - st).max(1) as f64);
                xb2[idx(c, k, g2_cnt)] = m_c;
                l3_sum[l3_of_l2[c]] += m_c;
                l3_n[l3_of_l2[c]] += 1;
            }
            for g in 0..g3_cnt {
                xb3[idx(g, k, g3_cnt)] = if l3_n[g] > 0 {
                    l3_sum[g] / l3_n[g] as f64
                } else {
                    0.0
                };
            }
            for c in 0..g2_cnt {
                xb2[idx(c, k, g2_cnt)] -= xb3[idx(l3_of_l2[c], k, g3_cnt)];
            }
            x_gm[k] = (0..g3_cnt).map(|g| xb3[idx(g, k, g3_cnt)]).sum::<f64>() / g3_cnt as f64;
        }
    }

    let mut gamma0: Vec<f64> = vec![0.0_f64; q_tot];
    for j in 0..d {
        gamma0[j] = (0..n_obs).map(|r| y_work[idx(r, j, n_obs)]).sum::<f64>() / n_obs as f64;
        if n_lags >= 1 {
            gamma0[d + j * n_slopes + j] = 0.25;
        }
    }
    if let Some(qvf) = q_vf_slot {
        gamma0[qvf] = -0.7;
    }
    let mut phi_l2: Vec<f64> = vec![0.0_f64; d];
    for &j in &l2_ar_active {
        phi_l2[j] = 0.2;
    }
    let mut gamma_w3: Vec<f64> = vec![0.0_f64; kw3 * q_tot];
    let mut gamma_w2: Vec<f64> = vec![0.0_f64; kw2 * q_tot];
    let mut u3: Vec<f64> = vec![0.0_f64; g3_cnt * q_l3.max(1)];
    let mut u2: Vec<f64> = vec![0.0_f64; g2_cnt * q_l2.max(1)];
    let mut omega3 = eye(q_l3, 0.1);
    let mut omega2 = eye(q_l2, 0.1);
    let mut sigma_w = eye(d, 1.0);
    let mut f_shock: Vec<f64> = if has_random_logvf {
        vec![0.0; n_obs]
    } else {
        Vec::new()
    };

    // PSEM Bayesian Lasso scale-mixture states (prior_mode == 2)
    let mut psem_tau2_g0 = vec![1.0_f64; q_tot];
    let mut psem_tau2_w3 = vec![1.0_f64; kw3 * q_tot];
    let mut psem_tau2_w2 = vec![1.0_f64; kw2 * q_tot];
    let mut psem_lambda2 = 1.0_f64;

    let n_w3_active: usize = (0..q_l3)
        .flat_map(|p| (0..kw3).map(move |k| (k, p)))
        .filter(|&(k, p)| re_l3_on_w3_mask.get(idx(k, p, kw3)).copied().unwrap_or(1) != 0)
        .count();
    let n_w2_active: usize = (0..q_l2)
        .flat_map(|p| (0..kw2).map(move |k| (k, p)))
        .filter(|&(k, p)| re_l2_on_w2_mask.get(idx(k, p, kw2)).copied().unwrap_or(1) != 0)
        .count();
    let n_om3 = q_l3 * (q_l3 + 1) / 2;
    let n_om2 = q_l2 * (q_l2 + 1) / 2;
    let n_sw_offdiag = if use_full_sigma_w { d * (d - 1) / 2 } else { 0 };
    let n_params = q_tot + l2_ar_active.len() + n_w3_active + n_w2_active + n_om3 + n_om2 + n_sw_offdiag;

    let keep_indices: Vec<usize> = ((warmup + 1)..=iter)
        .filter(|i| (i - warmup - 1) % thin == 0)
        .collect();
    let n_keep = keep_indices.len();
    let mut draws = vec![0.0; n_keep * n_params];
    let mut l2_draws = vec![0.0; n_keep * g2_cnt * q_l2.max(1)];
    let mut l3_draws = vec![0.0; n_keep * g3_cnt * q_l3.max(1)];
    let mut fitted_accum = vec![0.0; n_obs * d];
    let mut sig_accum = vec![0.0; g2_cnt * d];
    let mut sig_vf_accum = vec![0.0; if has_random_logvf { g2_cnt } else { 0 }];
    let mut sigma_w_accum = vec![0.0; if use_full_sigma_w { d * d } else { 0 }];
    let mut deviance_sum = 0.0;
    let mut keep_pos = 0usize;

    let mut l2_by_l3: Vec<Vec<usize>> = vec![Vec::new(); g3_cnt];
    for (c, &g) in l3_of_l2.iter().enumerate() {
        l2_by_l3[g].push(c);
    }

    let l2_int_pos: Vec<usize> = (0..q_l2).filter(|&i| l2_slots[i] < d).collect();
    let l2_sl_pos: Vec<usize> = (0..q_l2)
        .filter(|&i| l2_slots[i] >= d && l2_slots[i] < lin_sl_end)
        .collect();
    let l3_int_pos: Vec<usize> = (0..q_l3).filter(|&i| l3_slots[i] < d).collect();
    let l3_sl_pos: Vec<usize> = (0..q_l3)
        .filter(|&i| l3_slots[i] >= d && l3_slots[i] < lin_sl_end)
        .collect();

    let use_stat_t0 = d == 1 && n_lags == 1 && (kx == 0 || residual_dynamics);

    for step in 1..=iter {
        let om3_inv = if q_l3 > 0 {
            spd_invert(&omega3, q_l3)
        } else {
            Vec::new()
        };
        let om2_inv = if q_l2 > 0 {
            spd_invert(&omega2, q_l2)
        } else {
            Vec::new()
        };
        let sigma_w_inv = if use_full_sigma_w {
            spd_invert(&sigma_w, d)
        } else {
            Vec::new()
        };

        // Three-level latent covariate decomposition update
        for k in 0..kx {
            if latent_center_x[k] != 0 {
                let mut ss1 = 0.0;
                let mut n1 = 0usize;
                for c in 0..g2_cnt {
                    let g = l3_of_l2[c];
                    let st = offsets[c];
                    let en = offsets[c + 1];
                    let mut sum_d = 0.0;
                    let mut cnt = 0usize;
                    for r in st..en {
                        let v = x_in[idx(r, k, n_obs)];
                        if v.is_finite() {
                            sum_d += v - xb3[idx(g, k, g3_cnt)];
                            cnt += 1;
                        }
                    }
                    let pr = cnt as f64 / xv1[k].max(1e-6) + 1.0 / xv2[k].max(1e-6);
                    xb2[idx(c, k, g2_cnt)] =
                        (sum_d / xv1[k].max(1e-6)) / pr + (1.0 / pr).sqrt() * rng.rnorm();
                    for r in st..en {
                        let v = x_in[idx(r, k, n_obs)];
                        if v.is_finite() {
                            let e1 = v - xb3[idx(g, k, g3_cnt)] - xb2[idx(c, k, g2_cnt)];
                            ss1 += e1 * e1;
                            n1 += 1;
                        }
                    }
                }
                for g in 0..g3_cnt {
                    let mut sum_g = 0.0;
                    let mut cnt_g = 0usize;
                    for &c in &l2_by_l3[g] {
                        let st = offsets[c];
                        let en = offsets[c + 1];
                        for r in st..en {
                            let v = x_in[idx(r, k, n_obs)];
                            if v.is_finite() {
                                sum_g += v - xb2[idx(c, k, g2_cnt)];
                                cnt_g += 1;
                            }
                        }
                    }
                    let pr = cnt_g as f64 / xv1[k].max(1e-6) + 1.0 / xv3[k].max(1e-6);
                    let inf = sum_g / xv1[k].max(1e-6) + x_gm[k] / xv3[k].max(1e-6);
                    xb3[idx(g, k, g3_cnt)] = inf / pr + (1.0 / pr).sqrt() * rng.rnorm();
                }
                xv1[k] = rng.rinv_gamma(0.5 * n1 as f64 + 0.01, 0.5 * ss1 + 0.01);
                let ss2: f64 = (0..g2_cnt).map(|c| xb2[idx(c, k, g2_cnt)].powi(2)).sum();
                xv2[k] = rng.rinv_gamma(0.5 * g2_cnt as f64 + 0.01, 0.5 * ss2 + 0.01);
                x_gm[k] = (0..g3_cnt).map(|g| xb3[idx(g, k, g3_cnt)]).sum::<f64>() / g3_cnt as f64;
                let ss3: f64 = (0..g3_cnt)
                    .map(|g| (xb3[idx(g, k, g3_cnt)] - x_gm[k]).powi(2))
                    .sum();
                xv3[k] = rng.rinv_gamma(0.5 * g3_cnt as f64 + 0.01, 0.5 * ss3 + 0.01);
            }
        }

        let build_theta_gc = |c: usize,
                              g0: &[f64],
                              gw3: &[f64],
                              gw2: &[f64],
                              u3_m: &[f64],
                              u2_m: &[f64]|
         -> Vec<f64> {
            let g = l3_of_l2[c];
            let mut th = g0.to_vec();
            for q in 0..q_tot {
                if slot_to_l3[q] >= 0 {
                    let p3 = slot_to_l3[q] as usize;
                    for k in 0..kw3 {
                        if re_l3_on_w3_mask.get(idx(k, p3, kw3)).copied().unwrap_or(1) != 0 {
                            th[q] += gw3[idx(k, q, kw3)] * w3_in[idx(g, k, g3_cnt)];
                        }
                    }
                    th[q] += u3_m[idx(g, p3, g3_cnt)];
                }
                if slot_to_l2[q] >= 0 {
                    let p2 = slot_to_l2[q] as usize;
                    for k in 0..kw2 {
                        if re_l2_on_w2_mask.get(idx(k, p2, kw2)).copied().unwrap_or(1) != 0 {
                            th[q] += gw2[idx(k, q, kw2)] * w2_in[idx(c, k, g2_cnt)];
                        }
                    }
                    th[q] += u2_m[idx(c, p2, g2_cnt)];
                }
            }
            th
        };

        // 1. Impute missing y and sample shared residual covariance shocks f_shock (if has_random_logvf)
        for c in 0..g2_cnt {
            let st = offsets[c];
            let en = offsets[c + 1];
            let tc = en - st;
            if tc <= max_lag {
                continue;
            }
            let g = l3_of_l2[c];
            let th = build_theta_gc(c, &gamma0, &gamma_w3, &gamma_w2, &u3, &u2);
            let sig2_c: Vec<f64> = (0..d)
                .map(|j| th[lin_sl_end + j].clamp(-10.0, 10.0).exp().max(1e-6))
                .collect();
            for t_loc in 0..tc {
                let r_abs = st + t_loc;
                if has_random_logvf && t_loc >= max_lag {
                    let qvf = q_vf_slot.unwrap();
                    let vf_c = th[qvf].clamp(-10.0, 10.0).exp().max(1e-6);
                    let mut p_f = 1.0 / vf_c;
                    let mut i_f = 0.0;
                    for j in 0..d {
                        if !miss_y[idx(r_abs, j, n_obs)] {
                            let pred = cell_mean_3l(
                                &y_work,
                                x_in,
                                &xb3,
                                &xb2,
                                latent_center_x,
                                g,
                                c,
                                g3_cnt,
                                g2_cnt,
                                st,
                                t_loc,
                                j,
                                d,
                                kx,
                                lags,
                                max_lag,
                                &th,
                                residual_dynamics,
                                n_obs,
                            );
                            let res = y_work[idx(r_abs, j, n_obs)] - pred;
                            let lj = cov_lam[j];
                            p_f += (lj * lj) / sig2_c[j];
                            i_f += lj * res / sig2_c[j];
                        }
                    }
                    f_shock[r_abs] = i_f / p_f + (1.0 / p_f).sqrt() * rng.rnorm();
                }
                for j in 0..d {
                    if miss_y[idx(r_abs, j, n_obs)] {
                        let pred = cell_mean_3l(
                            &y_work,
                            x_in,
                            &xb3,
                            &xb2,
                            latent_center_x,
                            g,
                            c,
                            g3_cnt,
                            g2_cnt,
                            st,
                            t_loc,
                            j,
                            d,
                            kx,
                            lags,
                            max_lag,
                            &th,
                            residual_dynamics,
                            n_obs,
                        ) + if has_random_logvf && t_loc >= max_lag {
                            cov_lam[j] * f_shock[r_abs]
                        } else {
                            0.0
                        };
                        let s2 = if use_full_sigma_w {
                            sigma_w[idx(j, j, d)].max(1e-6)
                        } else {
                            sig2_c[j]
                        };
                        y_work[idx(r_abs, j, n_obs)] = pred + s2.sqrt() * rng.rnorm();
                    }
                }
            }
        }

        // 2. SIMULTANEOUS TWO-LEVEL BLOCK GIBBS (Asparouhov & Muthen, 2026b):
        //    For each Level-3 cluster g, jointly sample (u3[g], {u2[c]}_{c in C_g}) in O(C_g)
        //    via exact block-arrow Schur complement marginalization (eliminates L2-L3 autocorrelation!).
        //    When non-random Level-2 AR(1) is active on intercepts, sample sequentially along L2 time
        //    with exact Markov prior precision + exact joint L3-L2 shift move to maintain fast mixing.
        for is_slope_block in [false, true] {
            let l3_pos: &[usize] = if is_slope_block {
                &l3_sl_pos
            } else {
                &l3_int_pos
            };
            let l2_pos: &[usize] = if is_slope_block {
                &l2_sl_pos
            } else {
                &l2_int_pos
            };
            let n3 = l3_pos.len();
            let n2 = l2_pos.len();
            if n3 == 0 && n2 == 0 {
                continue;
            }

            if !is_slope_block && any_l2_ar {
                // Level-2 AR(1) intercept block: exact conditional Gibbs over L2 sequence + exact L3 update + joint L2-L3 shift move
                // Helper: map L2 vector u2[c, .] to innovation zeta[c, .]
                let l2_slot_phi = |p: usize| -> f64 {
                    let q = l2_slots[p];
                    if q < d && l2_ar_mask.get(q).copied().unwrap_or(0) != 0 {
                        phi_l2[q]
                    } else {
                        0.0
                    }
                };
                for g in 0..g3_cnt {
                    let children = &l2_by_l3[g];
                    let m_len = children.len();
                    // 2a-i. Update u2[c_m, l2_pos] for m = 0..m_len with exact AR(1) Markov prior from m-1 and m+1
                    if n2 > 0 {
                        for (m_idx, &c) in children.iter().enumerate() {
                            let st = offsets[c];
                            let en = offsets[c + 1];
                            let tc = en - st;
                            let mut pcc = vec![0.0; n2 * n2];
                            let mut hc = vec![0.0; n2];

                            // Term from zeta_m: for m == 0, D0 * u2_0 ~ N(0, Omega2); for m >= 1, u2_m - Phi2 u2_{m-1} ~ N(0, Omega2)
                            let mut d_cur = vec![1.0; q_l2];
                            let mut shift_prev = vec![0.0; q_l2];
                            if m_idx == 0 {
                                for (p, dc) in d_cur.iter_mut().enumerate().take(q_l2) {
                                    let ph = l2_slot_phi(p);
                                    if ph.abs() > 1e-12 {
                                        *dc = (1.0 - ph * ph).max(0.02).sqrt();
                                    }
                                }
                            } else {
                                let c_prev = children[m_idx - 1];
                                for (p, sp) in shift_prev.iter_mut().enumerate().take(q_l2) {
                                    let ph = l2_slot_phi(p);
                                    if ph.abs() > 1e-12 {
                                        *sp = ph * u2[idx(c_prev, p, g2_cnt)];
                                    }
                                }
                            }
                            for (a, &pa) in l2_pos.iter().enumerate() {
                                for (b, &pb) in l2_pos.iter().enumerate() {
                                    pcc[idx(a, b, n2)] +=
                                        d_cur[pa] * om2_inv[idx(pa, pb, q_l2)] * d_cur[pb];
                                }
                                let mut rhs = 0.0;
                                for p_all in 0..q_l2 {
                                    let o_ap = d_cur[pa] * om2_inv[idx(pa, p_all, q_l2)];
                                    if l2_pos.contains(&p_all) {
                                        rhs += o_ap * shift_prev[p_all];
                                    } else {
                                        rhs -= o_ap
                                            * (d_cur[p_all] * u2[idx(c, p_all, g2_cnt)]
                                                - shift_prev[p_all]);
                                    }
                                }
                                hc[a] += rhs;
                            }

                            // Term from zeta_{m+1} = u2_{m+1} - Phi2 u2_m ~ N(0, Omega2)
                            if m_idx + 1 < m_len {
                                let c_next = children[m_idx + 1];
                                let mut phi_vec = vec![0.0; q_l2];
                                for (p, pv) in phi_vec.iter_mut().enumerate().take(q_l2) {
                                    *pv = l2_slot_phi(p);
                                }
                                for (a, &pa) in l2_pos.iter().enumerate() {
                                    if phi_vec[pa].abs() < 1e-12 {
                                        continue;
                                    }
                                    for (b, &pb) in l2_pos.iter().enumerate() {
                                        if phi_vec[pb].abs() >= 1e-12 {
                                            pcc[idx(a, b, n2)] += phi_vec[pa]
                                                * om2_inv[idx(pa, pb, q_l2)]
                                                * phi_vec[pb];
                                        }
                                    }
                                    let mut rhs = 0.0;
                                    for p_all in 0..q_l2 {
                                        let o_ap = phi_vec[pa] * om2_inv[idx(pa, p_all, q_l2)];
                                        if l2_pos.contains(&p_all) {
                                            rhs += o_ap * u2[idx(c_next, p_all, g2_cnt)];
                                        } else {
                                            rhs += o_ap
                                                * (u2[idx(c_next, p_all, g2_cnt)]
                                                    - phi_vec[p_all] * u2[idx(c, p_all, g2_cnt)]);
                                        }
                                    }
                                    hc[a] += rhs;
                                }
                            }

                            // Level-1 likelihood contribution for child c
                            if tc > max_lag {
                                let mut u2_z = u2.clone();
                                for &pa in l2_pos {
                                    u2_z[idx(c, pa, g2_cnt)] = 0.0;
                                }
                                let th_base = build_theta_gc(
                                    c, &gamma0, &gamma_w3, &gamma_w2, &u3, &u2_z,
                                );
                                if use_stat_t0 {
                                    let s2 = th_base[lin_sl_end].clamp(-10.0, 10.0).exp().max(1e-6);
                                    let phi = th_base[d].clamp(-0.98, 0.98);
                                    let w0 = (1.0 - phi * phi).max(0.02) / s2;
                                    let mut r0 = y_work[idx(st, 0, n_obs)] - th_base[0];
                                    if residual_dynamics {
                                        for k in 0..kx {
                                            let x0 = if latent_center_x[k] != 0 {
                                                x_in[idx(st, k, n_obs)]
                                                    - xb3[idx(g, k, g3_cnt)]
                                                    - xb2[idx(c, k, g2_cnt)]
                                            } else {
                                                x_in[idx(st, k, n_obs)]
                                            };
                                            r0 -= th_base[d + n_lags * d + k] * x0;
                                        }
                                    }
                                    pcc[0] += w0;
                                    hc[0] += w0 * r0;
                                }
                                let mut gr2 = vec![0.0; d * n2];
                                let mut y_rem_vec = vec![0.0; d];
                                for t_loc in max_lag..tc {
                                    for j in 0..d {
                                        let bp = cell_mean_3l(
                                            &y_work,
                                            x_in,
                                            &xb3,
                                            &xb2,
                                            latent_center_x,
                                            g,
                                            c,
                                            g3_cnt,
                                            g2_cnt,
                                            st,
                                            t_loc,
                                            j,
                                            d,
                                            kx,
                                            lags,
                                            max_lag,
                                            &th_base,
                                            residual_dynamics,
                                            n_obs,
                                        ) + if has_random_logvf {
                                            cov_lam[j] * f_shock[st + t_loc]
                                        } else {
                                            0.0
                                        };
                                        y_rem_vec[j] = y_work[idx(st + t_loc, j, n_obs)] - bp;
                                        for (a, &pa) in l2_pos.iter().enumerate() {
                                            gr2[idx(j, a, d)] = sensitivity_3l(
                                                &y_work,
                                                x_in,
                                                &xb3,
                                                &xb2,
                                                latent_center_x,
                                                g,
                                                c,
                                                g3_cnt,
                                                g2_cnt,
                                                st,
                                                t_loc,
                                                j,
                                                d,
                                                kx,
                                                lags,
                                                &th_base,
                                                l2_slots[pa],
                                                residual_dynamics,
                                                n_obs,
                                            );
                                        }
                                    }
                                    if use_full_sigma_w {
                                        for j in 0..d {
                                            for k in 0..d {
                                                let w_jk = sigma_w_inv[idx(j, k, d)];
                                                if w_jk.abs() < 1e-14 {
                                                    continue;
                                                }
                                                let rk = y_rem_vec[k];
                                                for a in 0..n2 {
                                                    let gja = gr2[idx(j, a, d)];
                                                    if gja.abs() < 1e-14 {
                                                        continue;
                                                    }
                                                    hc[a] += w_jk * gja * rk;
                                                    for b in 0..n2 {
                                                        pcc[idx(a, b, n2)] +=
                                                            w_jk * gja * gr2[idx(k, b, d)];
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        for j in 0..d {
                                            let inv_s2 =
                                                1.0 / th_base[lin_sl_end + j].clamp(-10.0, 10.0).exp().max(1e-6);
                                            let rj = y_rem_vec[j];
                                            for a in 0..n2 {
                                                let gja = gr2[idx(j, a, d)];
                                                if gja.abs() < 1e-14 {
                                                    continue;
                                                }
                                                hc[a] += gja * rj * inv_s2;
                                                for b in 0..n2 {
                                                    pcc[idx(a, b, n2)] +=
                                                        gja * gr2[idx(j, b, d)] * inv_s2;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            symmetrize(&mut pcc, n2);
                            let dr2 = rmvnorm_precision(&mut rng, &pcc, &hc, n2);
                            for (a, &pa) in l2_pos.iter().enumerate() {
                                u2[idx(c, pa, g2_cnt)] = dr2[a];
                            }
                        }
                    }

                    // 2a-ii. Exact L3 intercept update given updated u2
                    if n3 > 0 {
                        let mut p00 = vec![0.0; n3 * n3];
                        let mut h0 = vec![0.0; n3];
                        for (a, &pa) in l3_pos.iter().enumerate() {
                            for (b, &pb) in l3_pos.iter().enumerate() {
                                p00[idx(a, b, n3)] = om3_inv[idx(pa, pb, q_l3)];
                            }
                            for p_o in 0..q_l3 {
                                if !l3_pos.contains(&p_o) {
                                    h0[a] -=
                                        om3_inv[idx(pa, p_o, q_l3)] * u3[idx(g, p_o, g3_cnt)];
                                }
                            }
                        }
                        let mut u3_z = u3.clone();
                        for &pa in l3_pos {
                            u3_z[idx(g, pa, g3_cnt)] = 0.0;
                        }
                        let mut gr3 = vec![0.0; d * n3];
                        let mut y_rem_vec = vec![0.0; d];
                        for &c in children {
                            let st = offsets[c];
                            let en = offsets[c + 1];
                            let tc = en - st;
                            if tc <= max_lag {
                                continue;
                            }
                            let th_base =
                                build_theta_gc(c, &gamma0, &gamma_w3, &gamma_w2, &u3_z, &u2);
                            if use_stat_t0 {
                                let s2 = th_base[lin_sl_end].clamp(-10.0, 10.0).exp().max(1e-6);
                                let phi = th_base[d].clamp(-0.98, 0.98);
                                let w0 = (1.0 - phi * phi).max(0.02) / s2;
                                let mut r0 = y_work[idx(st, 0, n_obs)] - th_base[0];
                                if residual_dynamics {
                                    for k in 0..kx {
                                        let x0 = if latent_center_x[k] != 0 {
                                            x_in[idx(st, k, n_obs)]
                                                - xb3[idx(g, k, g3_cnt)]
                                                - xb2[idx(c, k, g2_cnt)]
                                        } else {
                                            x_in[idx(st, k, n_obs)]
                                        };
                                        r0 -= th_base[d + n_lags * d + k] * x0;
                                    }
                                }
                                p00[0] += w0;
                                h0[0] += w0 * r0;
                            }
                            for t_loc in max_lag..tc {
                                for j in 0..d {
                                    let bp = cell_mean_3l(
                                        &y_work,
                                        x_in,
                                        &xb3,
                                        &xb2,
                                        latent_center_x,
                                        g,
                                        c,
                                        g3_cnt,
                                        g2_cnt,
                                        st,
                                        t_loc,
                                        j,
                                        d,
                                        kx,
                                        lags,
                                        max_lag,
                                        &th_base,
                                        residual_dynamics,
                                        n_obs,
                                    ) + if has_random_logvf {
                                        cov_lam[j] * f_shock[st + t_loc]
                                    } else {
                                        0.0
                                    };
                                    y_rem_vec[j] = y_work[idx(st + t_loc, j, n_obs)] - bp;
                                    for (a, &pa) in l3_pos.iter().enumerate() {
                                        gr3[idx(j, a, d)] = sensitivity_3l(
                                            &y_work,
                                            x_in,
                                            &xb3,
                                            &xb2,
                                            latent_center_x,
                                            g,
                                            c,
                                            g3_cnt,
                                            g2_cnt,
                                            st,
                                            t_loc,
                                            j,
                                            d,
                                            kx,
                                            lags,
                                            &th_base,
                                            l3_slots[pa],
                                            residual_dynamics,
                                            n_obs,
                                        );
                                    }
                                }
                                if use_full_sigma_w {
                                    for j in 0..d {
                                        for k in 0..d {
                                            let w_jk = sigma_w_inv[idx(j, k, d)];
                                            if w_jk.abs() < 1e-14 {
                                                continue;
                                            }
                                            let rk = y_rem_vec[k];
                                            for a in 0..n3 {
                                                let gja = gr3[idx(j, a, d)];
                                                if gja.abs() < 1e-14 {
                                                    continue;
                                                }
                                                h0[a] += w_jk * gja * rk;
                                                for b in 0..n3 {
                                                    p00[idx(a, b, n3)] +=
                                                        w_jk * gja * gr3[idx(k, b, d)];
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    for j in 0..d {
                                        let inv_s2 =
                                            1.0 / th_base[lin_sl_end + j].clamp(-10.0, 10.0).exp().max(1e-6);
                                        let rj = y_rem_vec[j];
                                        for a in 0..n3 {
                                            let gja = gr3[idx(j, a, d)];
                                            if gja.abs() < 1e-14 {
                                                continue;
                                            }
                                            h0[a] += gja * rj * inv_s2;
                                            for b in 0..n3 {
                                                p00[idx(a, b, n3)] +=
                                                    gja * gr3[idx(j, b, d)] * inv_s2;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        symmetrize(&mut p00, n3);
                        let u3_draw = rmvnorm_precision(&mut rng, &p00, &h0, n3);
                        for (a, &pa) in l3_pos.iter().enumerate() {
                            u3[idx(g, pa, g3_cnt)] = u3_draw[a];
                        }

                        // 2a-iii. Exact L2-L3 likelihood-invariant shift move: u3[g, p3] += delta, u2[c, p2] -= delta
                        // Directly samples the L3/L2 mean partition under the AR(1) prior!
                        for j in 0..d {
                            if slot_to_l3[j] >= 0 && slot_to_l2[j] >= 0 && !children.is_empty() {
                                let p3 = slot_to_l3[j] as usize;
                                let p2 = slot_to_l2[j] as usize;
                                let ph = l2_slot_phi(p2);
                                // Prior precision & info on shift delta from u3[g] + e_{p3}*delta
                                let mut p_d = om3_inv[idx(p3, p3, q_l3)];
                                let mut i_d = 0.0;
                                for p_o in 0..q_l3 {
                                    i_d -= om3_inv[idx(p3, p_o, q_l3)] * u3[idx(g, p_o, g3_cnt)];
                                }
                                // Prior precision & info on shift delta from zeta_m(delta) = zeta_m - c_m * e_{p2} * delta
                                for (m_idx, &c) in children.iter().enumerate() {
                                    let c_m = if m_idx == 0 {
                                        (1.0 - ph * ph).max(0.02).sqrt()
                                    } else {
                                        1.0 - ph
                                    };
                                    p_d += c_m * c_m * om2_inv[idx(p2, p2, q_l2)];
                                    for p_o in 0..q_l2 {
                                        let z_mo = if m_idx == 0 {
                                            let ph_o = l2_slot_phi(p_o);
                                            let d_o = if ph_o.abs() > 1e-12 {
                                                (1.0 - ph_o * ph_o).max(0.02).sqrt()
                                            } else {
                                                1.0
                                            };
                                            d_o * u2[idx(c, p_o, g2_cnt)]
                                        } else {
                                            let ph_o = l2_slot_phi(p_o);
                                            u2[idx(c, p_o, g2_cnt)]
                                                - ph_o * u2[idx(children[m_idx - 1], p_o, g2_cnt)]
                                        };
                                        i_d += c_m * om2_inv[idx(p2, p_o, q_l2)] * z_mo;
                                    }
                                }
                                let delta = i_d / p_d.max(1e-8)
                                    + (1.0 / p_d.max(1e-8)).sqrt() * rng.rnorm();
                                u3[idx(g, p3, g3_cnt)] += delta;
                                for &c in children {
                                    u2[idx(c, p2, g2_cnt)] -= delta;
                                }
                            }
                        }
                    }
                }
                continue;
            }

            // Standard exact O(C_g) block-arrow Schur complement simultaneous draw of (u3[g], {u2[c]})
            for g in 0..g3_cnt {
                let mut p00 = vec![0.0; n3 * n3];
                let mut h0 = vec![0.0; n3];
                for (a, &pa) in l3_pos.iter().enumerate() {
                    for (b, &pb) in l3_pos.iter().enumerate() {
                        p00[idx(a, b, n3)] = om3_inv[idx(pa, pb, q_l3)];
                    }
                    for p_o in 0..q_l3 {
                        if !l3_pos.contains(&p_o) {
                            h0[a] -= om3_inv[idx(pa, p_o, q_l3)] * u3[idx(g, p_o, g3_cnt)];
                        }
                    }
                }

                type ChildBlock = (usize, Vec<f64>, Vec<f64>, Vec<f64>);
                let mut child_blocks: Vec<ChildBlock> = Vec::with_capacity(l2_by_l3[g].len());

                let mut u3_z = u3.clone();
                for &pa in l3_pos {
                    u3_z[idx(g, pa, g3_cnt)] = 0.0;
                }

                for &c in &l2_by_l3[g] {
                    let st = offsets[c];
                    let en = offsets[c + 1];
                    let tc = en - st;
                    let mut pcc = vec![0.0; n2 * n2];
                    let mut hc = vec![0.0; n2];
                    let mut p0c = vec![0.0; n3 * n2];
                    for (a, &pa) in l2_pos.iter().enumerate() {
                        for (b, &pb) in l2_pos.iter().enumerate() {
                            pcc[idx(a, b, n2)] = om2_inv[idx(pa, pb, q_l2)];
                        }
                        for p_o in 0..q_l2 {
                            if !l2_pos.contains(&p_o) {
                                hc[a] -= om2_inv[idx(pa, p_o, q_l2)] * u2[idx(c, p_o, g2_cnt)];
                            }
                        }
                    }

                    if tc > max_lag {
                        let mut u2_z = u2.clone();
                        for &pa in l2_pos {
                            u2_z[idx(c, pa, g2_cnt)] = 0.0;
                        }
                        let th_base =
                            build_theta_gc(c, &gamma0, &gamma_w3, &gamma_w2, &u3_z, &u2_z);

                        // Stationary initial-state contribution for short-T univariate AR(1) DSEM3 & RDSEM3 (Asparouhov 2026)
                        if use_stat_t0 {
                            let s2 = th_base[lin_sl_end].clamp(-10.0, 10.0).exp().max(1e-6);
                            let phi = th_base[d].clamp(-0.98, 0.98);
                            let w0 = (1.0 - phi * phi).max(0.02) / s2;
                            if !is_slope_block {
                                let mut r0 = y_work[idx(st, 0, n_obs)] - th_base[0];
                                if residual_dynamics {
                                    for k in 0..kx {
                                        let x0 = if latent_center_x[k] != 0 {
                                            x_in[idx(st, k, n_obs)]
                                                - xb3[idx(g, k, g3_cnt)]
                                                - xb2[idx(c, k, g2_cnt)]
                                        } else {
                                            x_in[idx(st, k, n_obs)]
                                        };
                                        r0 -= th_base[d + n_lags * d + k] * x0;
                                    }
                                }
                                if n3 > 0 {
                                    p00[0] += w0;
                                    h0[0] += w0 * r0;
                                }
                                if n2 > 0 {
                                    pcc[0] += w0;
                                    hc[0] += w0 * r0;
                                }
                                if n3 > 0 && n2 > 0 {
                                    p0c[0] += w0;
                                }
                            } else if residual_dynamics && kx > 0 {
                                // Exogenous slope contribution from t_loc = 0 in RDSEM3
                                let mut r0 = y_work[idx(st, 0, n_obs)] - th_base[0];
                                let mut gr3_0 = vec![0.0; n3];
                                let mut gr2_0 = vec![0.0; n2];
                                for k in 0..kx {
                                    let x0 = if latent_center_x[k] != 0 {
                                        x_in[idx(st, k, n_obs)]
                                            - xb3[idx(g, k, g3_cnt)]
                                            - xb2[idx(c, k, g2_cnt)]
                                    } else {
                                        x_in[idx(st, k, n_obs)]
                                    };
                                    let q_k = d + n_lags * d + k;
                                    r0 -= th_base[q_k] * x0;
                                    for (a, &pa) in l3_pos.iter().enumerate() {
                                        if l3_slots[pa] == q_k {
                                            gr3_0[a] = x0;
                                        }
                                    }
                                    for (a, &pa) in l2_pos.iter().enumerate() {
                                        if l2_slots[pa] == q_k {
                                            gr2_0[a] = x0;
                                        }
                                    }
                                }
                                for a in 0..n3 {
                                    h0[a] += w0 * gr3_0[a] * r0;
                                    for b in 0..n3 {
                                        p00[idx(a, b, n3)] += w0 * gr3_0[a] * gr3_0[b];
                                    }
                                    for b in 0..n2 {
                                        p0c[idx(a, b, n3)] += w0 * gr3_0[a] * gr2_0[b];
                                    }
                                }
                                for a in 0..n2 {
                                    hc[a] += w0 * gr2_0[a] * r0;
                                    for b in 0..n2 {
                                        pcc[idx(a, b, n2)] += w0 * gr2_0[a] * gr2_0[b];
                                    }
                                }
                            }
                        }

                        let mut gr3 = vec![0.0; d * n3];
                        let mut gr2 = vec![0.0; d * n2];
                        let mut y_rem_vec = vec![0.0; d];
                        for t_loc in max_lag..tc {
                            for j in 0..d {
                                let bp = cell_mean_3l(
                                    &y_work,
                                    x_in,
                                    &xb3,
                                    &xb2,
                                    latent_center_x,
                                    g,
                                    c,
                                    g3_cnt,
                                    g2_cnt,
                                    st,
                                    t_loc,
                                    j,
                                    d,
                                    kx,
                                    lags,
                                    max_lag,
                                    &th_base,
                                    residual_dynamics,
                                    n_obs,
                                ) + if has_random_logvf {
                                    cov_lam[j] * f_shock[st + t_loc]
                                } else {
                                    0.0
                                };
                                y_rem_vec[j] = y_work[idx(st + t_loc, j, n_obs)] - bp;
                                for (a, &pa) in l3_pos.iter().enumerate() {
                                    gr3[idx(j, a, d)] = sensitivity_3l(
                                        &y_work,
                                        x_in,
                                        &xb3,
                                        &xb2,
                                        latent_center_x,
                                        g,
                                        c,
                                        g3_cnt,
                                        g2_cnt,
                                        st,
                                        t_loc,
                                        j,
                                        d,
                                        kx,
                                        lags,
                                        &th_base,
                                        l3_slots[pa],
                                        residual_dynamics,
                                        n_obs,
                                    );
                                }
                                for (a, &pa) in l2_pos.iter().enumerate() {
                                    gr2[idx(j, a, d)] = sensitivity_3l(
                                        &y_work,
                                        x_in,
                                        &xb3,
                                        &xb2,
                                        latent_center_x,
                                        g,
                                        c,
                                        g3_cnt,
                                        g2_cnt,
                                        st,
                                        t_loc,
                                        j,
                                        d,
                                        kx,
                                        lags,
                                        &th_base,
                                        l2_slots[pa],
                                        residual_dynamics,
                                        n_obs,
                                    );
                                }
                            }
                            if use_full_sigma_w {
                                for j in 0..d {
                                    for k in 0..d {
                                        let w_jk = sigma_w_inv[idx(j, k, d)];
                                        if w_jk.abs() < 1e-14 {
                                            continue;
                                        }
                                        let rk = y_rem_vec[k];
                                        for a in 0..n3 {
                                            let gja = gr3[idx(j, a, d)];
                                            if gja.abs() < 1e-14 {
                                                continue;
                                            }
                                            h0[a] += w_jk * gja * rk;
                                            for b in 0..n3 {
                                                p00[idx(a, b, n3)] +=
                                                    w_jk * gja * gr3[idx(k, b, d)];
                                            }
                                            for b in 0..n2 {
                                                p0c[idx(a, b, n3)] +=
                                                    w_jk * gja * gr2[idx(k, b, d)];
                                            }
                                        }
                                        for a in 0..n2 {
                                            let gja = gr2[idx(j, a, d)];
                                            if gja.abs() < 1e-14 {
                                                continue;
                                            }
                                            hc[a] += w_jk * gja * rk;
                                            for b in 0..n2 {
                                                pcc[idx(a, b, n2)] +=
                                                    w_jk * gja * gr2[idx(k, b, d)];
                                            }
                                        }
                                    }
                                }
                            } else {
                                for j in 0..d {
                                    let inv_s2 = 1.0
                                        / th_base[lin_sl_end + j].clamp(-10.0, 10.0).exp().max(1e-6);
                                    let rj = y_rem_vec[j];
                                    for a in 0..n3 {
                                        let gja = gr3[idx(j, a, d)];
                                        if gja.abs() < 1e-14 {
                                            continue;
                                        }
                                        h0[a] += gja * rj * inv_s2;
                                        for b in 0..n3 {
                                            p00[idx(a, b, n3)] +=
                                                gja * gr3[idx(j, b, d)] * inv_s2;
                                        }
                                        for b in 0..n2 {
                                            p0c[idx(a, b, n3)] +=
                                                gja * gr2[idx(j, b, d)] * inv_s2;
                                        }
                                    }
                                    for a in 0..n2 {
                                        let gja = gr2[idx(j, a, d)];
                                        if gja.abs() < 1e-14 {
                                            continue;
                                        }
                                        hc[a] += gja * rj * inv_s2;
                                        for b in 0..n2 {
                                            pcc[idx(a, b, n2)] +=
                                                gja * gr2[idx(j, b, d)] * inv_s2;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Fold child c out of (p00, h0) via exact Schur complement
                    if n3 > 0 && n2 > 0 {
                        let pcc_inv = spd_invert(&pcc, n2);
                        let p0c_inv = mat_mul(&p0c, &pcc_inv, n3, n2, n2);
                        let schur_p = mat_mul_t(&p0c_inv, &p0c, n3, n2, n3);
                        let schur_h = mat_vec(&p0c_inv, &hc, n3, n2);
                        for i in 0..(n3 * n3) {
                            p00[i] -= schur_p[i];
                        }
                        for a in 0..n3 {
                            h0[a] -= schur_h[a];
                        }
                    }
                    child_blocks.push((c, pcc, hc, p0c));
                }

                if n3 > 0 {
                    symmetrize(&mut p00, n3);
                }
                let u3_draw = if n3 > 0 {
                    rmvnorm_precision(&mut rng, &p00, &h0, n3)
                } else {
                    Vec::new()
                };
                let mut u2_draws_g: Vec<(usize, Vec<f64>)> = Vec::with_capacity(child_blocks.len());
                for (c, pcc, hc, p0c) in child_blocks {
                    if n2 > 0 {
                        let mut hc_cond = hc;
                        for b in 0..n2 {
                            let mut sub = 0.0;
                            for a in 0..n3 {
                                sub += p0c[idx(a, b, n3)] * u3_draw[a];
                            }
                            hc_cond[b] -= sub;
                        }
                        let dr2 = rmvnorm_precision(&mut rng, &pcc, &hc_cond, n2);
                        u2_draws_g.push((c, dr2));
                    }
                }

                // Stationarity check + exact short-T stationary initial-state MH ratio for AR slope block
                let mut accept_block = true;
                let mut log_mh_t0 = 0.0;
                if is_slope_block {
                    for &(c, ref dr2) in &u2_draws_g {
                        for (a, &pa) in l2_pos.iter().enumerate() {
                            let q = l2_slots[pa];
                            let is_ar = (q - d) % n_slopes < n_lags * d;
                            if is_ar {
                                let mut tot = gamma0[q] + dr2[a];
                                if slot_to_l3[q] >= 0 {
                                    let p3 = slot_to_l3[q] as usize;
                                    for k in 0..kw3 {
                                        if re_l3_on_w3_mask.get(idx(k, p3, kw3)).copied().unwrap_or(1)
                                            != 0
                                        {
                                            tot +=
                                                gamma_w3[idx(k, q, kw3)] * w3_in[idx(g, k, g3_cnt)];
                                        }
                                    }
                                    if let Some(pos3) =
                                        l3_pos.iter().position(|&p| l3_slots[p] == q)
                                    {
                                        tot += u3_draw[pos3];
                                    } else {
                                        tot += u3[idx(g, p3, g3_cnt)];
                                    }
                                }
                                for k in 0..kw2 {
                                    if re_l2_on_w2_mask.get(idx(k, pa, kw2)).copied().unwrap_or(1)
                                        != 0
                                    {
                                        tot +=
                                            gamma_w2[idx(k, q, kw2)] * w2_in[idx(c, k, g2_cnt)];
                                    }
                                }
                                if !tot.is_finite() || tot.abs() >= 0.98 {
                                    accept_block = false;
                                    break;
                                }
                            }
                        }
                        if !accept_block {
                            break;
                        }
                    }
                    if accept_block && n3 > 0 {
                        for (a, &pa) in l3_pos.iter().enumerate() {
                            let q = l3_slots[pa];
                            let is_ar = (q - d) % n_slopes < n_lags * d;
                            if is_ar && slot_to_l2[q] < 0 {
                                let mut tot = gamma0[q] + u3_draw[a];
                                for k in 0..kw3 {
                                    if re_l3_on_w3_mask.get(idx(k, pa, kw3)).copied().unwrap_or(1)
                                        != 0
                                    {
                                        tot += gamma_w3[idx(k, q, kw3)] * w3_in[idx(g, k, g3_cnt)];
                                    }
                                }
                                if !tot.is_finite() || tot.abs() >= 0.98 {
                                    accept_block = false;
                                    break;
                                }
                            }
                        }
                    }
                    // Exact stationary initial-state MH ratio for short-T univariate AR(1) DSEM3 / RDSEM3
                    if accept_block && use_stat_t0 && (slot_to_l3[d] >= 0 || slot_to_l2[d] >= 0) {
                        let mut u3_cand = u3.clone();
                        for (a, &pa) in l3_pos.iter().enumerate() {
                            u3_cand[idx(g, pa, g3_cnt)] = u3_draw[a];
                        }
                        let mut u2_cand = u2.clone();
                        for &(c, ref dr2) in &u2_draws_g {
                            for (a, &pa) in l2_pos.iter().enumerate() {
                                u2_cand[idx(c, pa, g2_cnt)] = dr2[a];
                            }
                        }
                        for &c in &l2_by_l3[g] {
                            let st = offsets[c];
                            let en = offsets[c + 1];
                            if en - st <= max_lag {
                                continue;
                            }
                            let th_old = build_theta_gc(c, &gamma0, &gamma_w3, &gamma_w2, &u3, &u2);
                            let th_new =
                                build_theta_gc(c, &gamma0, &gamma_w3, &gamma_w2, &u3_cand, &u2_cand);
                            let phi_old = th_old[d].clamp(-0.98, 0.98);
                            let phi_new = th_new[d].clamp(-0.98, 0.98);
                            let s2 = th_old[lin_sl_end].clamp(-10.0, 10.0).exp().max(1e-6);
                            let mut e0_new = y_work[idx(st, 0, n_obs)] - th_new[0];
                            if residual_dynamics {
                                for k in 0..kx {
                                    let x0 = if latent_center_x[k] != 0 {
                                        x_in[idx(st, k, n_obs)]
                                            - xb3[idx(g, k, g3_cnt)]
                                            - xb2[idx(c, k, g2_cnt)]
                                    } else {
                                        x_in[idx(st, k, n_obs)]
                                    };
                                    e0_new -= th_new[d + n_lags * d + k] * x0;
                                }
                            }
                            let v_old = (1.0 - phi_old * phi_old).max(0.02);
                            let v_new = (1.0 - phi_new * phi_new).max(0.02);
                            // Note: exogenous slopes already included t0 in Gaussian proposal; only AR(1) phi factor differs
                            log_mh_t0 += 0.5 * (v_new.ln() - v_old.ln())
                                - 0.5 * (v_new - v_old) * e0_new * e0_new / s2;
                        }
                        if !log_mh_t0.is_finite()
                            || (log_mh_t0 < 0.0 && rng.runif().ln() >= log_mh_t0)
                        {
                            accept_block = false;
                        }
                    }
                }

                if accept_block {
                    for (a, &pa) in l3_pos.iter().enumerate() {
                        u3[idx(g, pa, g3_cnt)] = u3_draw[a];
                    }
                    for (c, dr2) in u2_draws_g {
                        for (a, &pa) in l2_pos.iter().enumerate() {
                            u2[idx(c, pa, g2_cnt)] = dr2[a];
                        }
                    }
                }
            }
        }

        // 2c. L2 random log-variances & logvf (strictly-concave Newton-Laplace Independence MH)
        if q_l2 > 0 {
            let mut l2_lv_targets: Vec<(usize, Option<usize>)> = Vec::new();
            for j in 0..d {
                if slot_to_l2[lin_sl_end + j] >= 0 {
                    l2_lv_targets.push((lin_sl_end + j, Some(j)));
                }
            }
            if let Some(qvf) = q_vf_slot {
                if slot_to_l2[qvf] >= 0 {
                    l2_lv_targets.push((qvf, None));
                }
            }
            for c in 0..g2_cnt {
                let st = offsets[c];
                let en = offsets[c + 1];
                let tc = en - st;
                if tc <= max_lag {
                    continue;
                }
                let g = l3_of_l2[c];
                for &(q_lv, j_opt) in &l2_lv_targets {
                    let p_lv = slot_to_l2[q_lv] as usize;
                    let mut cur_v = u2[idx(c, p_lv, g2_cnt)];
                    let th_cur = build_theta_gc(c, &gamma0, &gamma_w3, &gamma_w2, &u3, &u2);
                    let mut rss = 0.0;
                    let mut n_eff = (tc - max_lag) as f64;
                    if let Some(j) = j_opt {
                        for t_loc in max_lag..tc {
                            let pr = cell_mean_3l(
                                &y_work,
                                x_in,
                                &xb3,
                                &xb2,
                                latent_center_x,
                                g,
                                c,
                                g3_cnt,
                                g2_cnt,
                                st,
                                t_loc,
                                j,
                                d,
                                kx,
                                lags,
                                max_lag,
                                &th_cur,
                                residual_dynamics,
                                n_obs,
                            ) + if has_random_logvf {
                                cov_lam[j] * f_shock[st + t_loc]
                            } else {
                                0.0
                            };
                            let e = y_work[idx(st + t_loc, j, n_obs)] - pr;
                            rss += e * e;
                        }
                        if use_stat_t0 {
                            let phi = th_cur[d].clamp(-0.98, 0.98);
                            let mut e0 = y_work[idx(st, 0, n_obs)] - th_cur[0];
                            if residual_dynamics {
                                for k in 0..kx {
                                    let x0 = if latent_center_x[k] != 0 {
                                        x_in[idx(st, k, n_obs)]
                                            - xb3[idx(g, k, g3_cnt)]
                                            - xb2[idx(c, k, g2_cnt)]
                                    } else {
                                        x_in[idx(st, k, n_obs)]
                                    };
                                    e0 -= th_cur[d + n_lags * d + k] * x0;
                                }
                            }
                            rss += (1.0 - phi * phi).max(0.02) * e0 * e0;
                            n_eff += 1.0;
                        }
                    } else {
                        for t_loc in max_lag..tc {
                            let fv = f_shock[st + t_loc];
                            rss += fv * fv;
                        }
                    }
                    let base_lv = th_cur[q_lv] - cur_v;
                    let p_tot = om2_inv[idx(p_lv, p_lv, q_l2)].max(1e-8);
                    let mut info_tot = 0.0;
                    for p in 0..q_l2 {
                        if p != p_lv {
                            info_tot -= om2_inv[idx(p_lv, p, q_l2)] * u2[idx(c, p, g2_cnt)];
                        }
                    }
                    let m_prior = (info_tot / p_tot).clamp(-8.0, 8.0);
                    let mut u_mode = m_prior;
                    for _ in 0..6 {
                        let exp_m = (-(base_lv + u_mode)).exp().min(1e12);
                        let g1 = -0.5 * n_eff + 0.5 * exp_m * rss - p_tot * (u_mode - m_prior);
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
                        -0.5 * n_eff * lv
                            - 0.5 * (-lv).exp().min(1e12) * rss
                            - 0.5 * p_tot * (uv - m_prior) * (uv - m_prior)
                    };
                    let logq = |uv: f64| -> f64 {
                        let z = (uv - u_mode) / prop_sd;
                        -0.5 * z * z
                    };
                    for _ in 0..2 {
                        let prop_v = (u_mode + prop_sd * rng.rnorm()).clamp(-8.0, 8.0);
                        let log_acc =
                            (lp_fn(prop_v) - logq(prop_v)) - (lp_fn(cur_v) - logq(cur_v));
                        if rng.runif().ln() < log_acc {
                            cur_v = prop_v;
                        }
                    }
                    u2[idx(c, p_lv, g2_cnt)] = cur_v;
                }
            }
        }

        // 3c. L3 random log-variances & logvf (strictly-concave Newton-Laplace Independence MH)
        if q_l3 > 0 {
            let mut l3_lv_targets: Vec<(usize, Option<usize>)> = Vec::new();
            for j in 0..d {
                if slot_to_l3[lin_sl_end + j] >= 0 {
                    l3_lv_targets.push((lin_sl_end + j, Some(j)));
                }
            }
            if let Some(qvf) = q_vf_slot {
                if slot_to_l3[qvf] >= 0 {
                    l3_lv_targets.push((qvf, None));
                }
            }
            for g in 0..g3_cnt {
                for &(q_lv, j_opt) in &l3_lv_targets {
                    let p_lv = slot_to_l3[q_lv] as usize;
                    let mut cur_v = u3[idx(g, p_lv, g3_cnt)];
                    let mut c_rss: Vec<(f64, f64, f64)> = Vec::new();
                    for &c in &l2_by_l3[g] {
                        let st = offsets[c];
                        let en = offsets[c + 1];
                        let tc = en - st;
                        if tc <= max_lag {
                            continue;
                        }
                        let th_c = build_theta_gc(c, &gamma0, &gamma_w3, &gamma_w2, &u3, &u2);
                        let mut rss = 0.0;
                        let mut n_eff = (tc - max_lag) as f64;
                        if let Some(j) = j_opt {
                            for t_loc in max_lag..tc {
                                let pr = cell_mean_3l(
                                    &y_work,
                                    x_in,
                                    &xb3,
                                    &xb2,
                                    latent_center_x,
                                    g,
                                    c,
                                    g3_cnt,
                                    g2_cnt,
                                    st,
                                    t_loc,
                                    j,
                                    d,
                                    kx,
                                    lags,
                                    max_lag,
                                    &th_c,
                                    residual_dynamics,
                                    n_obs,
                                ) + if has_random_logvf {
                                    cov_lam[j] * f_shock[st + t_loc]
                                } else {
                                    0.0
                                };
                                let e = y_work[idx(st + t_loc, j, n_obs)] - pr;
                                rss += e * e;
                            }
                            if use_stat_t0 {
                                let phi = th_c[d].clamp(-0.98, 0.98);
                                let mut e0 = y_work[idx(st, 0, n_obs)] - th_c[0];
                                if residual_dynamics {
                                    for k in 0..kx {
                                        let x0 = if latent_center_x[k] != 0 {
                                            x_in[idx(st, k, n_obs)]
                                                - xb3[idx(g, k, g3_cnt)]
                                                - xb2[idx(c, k, g2_cnt)]
                                        } else {
                                            x_in[idx(st, k, n_obs)]
                                        };
                                        e0 -= th_c[d + n_lags * d + k] * x0;
                                    }
                                }
                                rss += (1.0 - phi * phi).max(0.02) * e0 * e0;
                                n_eff += 1.0;
                            }
                        } else {
                            for t_loc in max_lag..tc {
                                let fv = f_shock[st + t_loc];
                                rss += fv * fv;
                            }
                        }
                        c_rss.push((n_eff, th_c[q_lv] - cur_v, rss));
                    }
                    let p_tot = om3_inv[idx(p_lv, p_lv, q_l3)].max(1e-8);
                    let mut info_tot = 0.0;
                    for p in 0..q_l3 {
                        if p != p_lv {
                            info_tot -= om3_inv[idx(p_lv, p, q_l3)] * u3[idx(g, p, g3_cnt)];
                        }
                    }
                    let m_prior = (info_tot / p_tot).clamp(-8.0, 8.0);
                    let mut u_mode = m_prior;
                    for _ in 0..6 {
                        let mut g1 = -p_tot * (u_mode - m_prior);
                        let mut neg_g2 = p_tot;
                        for &(ne, blv, rss) in &c_rss {
                            let exp_m = (-(blv + u_mode)).exp().min(1e12);
                            g1 += -0.5 * ne + 0.5 * exp_m * rss;
                            neg_g2 += 0.5 * exp_m * rss;
                        }
                        let delta = (g1 / neg_g2.max(1e-6)).clamp(-2.0, 2.0);
                        u_mode = (u_mode + delta).clamp(-8.0, 8.0);
                        if delta.abs() < 1e-6 {
                            break;
                        }
                    }
                    let mut hess = p_tot;
                    for &(_, blv, rss) in &c_rss {
                        hess += 0.5 * (-(blv + u_mode)).exp().min(1e12) * rss;
                    }
                    let prop_sd = (1.25 / hess.max(1e-6).sqrt()).clamp(0.01, 2.5);
                    let lp_fn = |uv: f64| -> f64 {
                        let mut lp = -0.5 * p_tot * (uv - m_prior) * (uv - m_prior);
                        for &(ne, blv, rss) in &c_rss {
                            let lv = blv + uv;
                            lp += -0.5 * ne * lv - 0.5 * (-lv).exp().min(1e12) * rss;
                        }
                        lp
                    };
                    let logq = |uv: f64| -> f64 {
                        let z = (uv - u_mode) / prop_sd;
                        -0.5 * z * z
                    };
                    for _ in 0..2 {
                        let prop_v = (u_mode + prop_sd * rng.rnorm()).clamp(-8.0, 8.0);
                        let log_acc =
                            (lp_fn(prop_v) - logq(prop_v)) - (lp_fn(cur_v) - logq(cur_v));
                        if rng.runif().ln() < log_acc {
                            cur_v = prop_v;
                        }
                    }
                    u3[idx(g, p_lv, g3_cnt)] = cur_v;
                }
            }
        }

        // 4. Update gamma0, gamma_w3, gamma_w2, non-random Level-2 AR(1) phi_l2, and fixed slots!
        let p_pr_base = if prior_mode == 1 {
            1e-10
        } else {
            1.0 / (beta_sd * beta_sd)
        };
        for q in 0..q_tot {
            if slot_to_l3[q] >= 0 {
                let p_l3 = slot_to_l3[q] as usize;
                let o_qq = om3_inv[idx(p_l3, p_l3, q_l3)].max(1e-8);
                let mut active_k: Vec<usize> = Vec::new();
                for k in 0..kw3 {
                    if re_l3_on_w3_mask.get(idx(k, p_l3, kw3)).copied().unwrap_or(1) != 0 {
                        active_k.push(k);
                    }
                }
                let pw = 1 + active_k.len();
                let mut wd = vec![0.0; g3_cnt * pw];
                let mut yv = vec![0.0; g3_cnt];
                for g in 0..g3_cnt {
                    wd[idx(g, 0, g3_cnt)] = 1.0;
                    let mut tot = gamma0[q] + u3[idx(g, p_l3, g3_cnt)];
                    for (ai, &k) in active_k.iter().enumerate() {
                        let wv = w3_in[idx(g, k, g3_cnt)];
                        wd[idx(g, 1 + ai, g3_cnt)] = wv;
                        tot += gamma_w3[idx(k, q, kw3)] * wv;
                    }
                    for p2 in 0..q_l3 {
                        if p2 != p_l3 {
                            let o_q2 = om3_inv[idx(p_l3, p2, q_l3)];
                            if o_q2.abs() > 1e-14 {
                                tot += (o_q2 / o_qq) * u3[idx(g, p2, g3_cnt)];
                            }
                        }
                    }
                    yv[g] = tot;
                }
                let wtw = mat_t_mul(&wd, &wd, pw, g3_cnt, pw);
                let wty = mat_t_mul(&wd, &yv, pw, g3_cnt, 1);
                let mut pr = vec![0.0; pw * pw];
                for i in 0..(pw * pw) {
                    pr[i] = wtw[i] * o_qq;
                }
                pr[idx(0, 0, pw)] += p_pr_base;
                for (ai, &k) in active_k.iter().enumerate() {
                    let pk = if prior_mode == 2 {
                        1.0 / psem_tau2_w3[idx(k, q, kw3)].max(1e-6)
                    } else {
                        p_pr_base
                    };
                    pr[idx(1 + ai, 1 + ai, pw)] += pk;
                }
                let inf: Vec<f64> = wty.iter().map(|v| v * o_qq).collect();
                let b = rmvnorm_precision(&mut rng, &pr, &inf, pw);
                let old_g0 = gamma0[q];
                let old_gw: Vec<f64> = (0..kw3).map(|k| gamma_w3[idx(k, q, kw3)]).collect();
                gamma0[q] = b[0];
                for k in 0..kw3 {
                    gamma_w3[idx(k, q, kw3)] = 0.0;
                }
                for (ai, &k) in active_k.iter().enumerate() {
                    gamma_w3[idx(k, q, kw3)] = b[1 + ai];
                }
                for g in 0..g3_cnt {
                    let mut delta = old_g0 - gamma0[q];
                    for &k in &active_k {
                        delta += (old_gw[k] - gamma_w3[idx(k, q, kw3)]) * w3_in[idx(g, k, g3_cnt)];
                    }
                    u3[idx(g, p_l3, g3_cnt)] += delta;
                }
            } else if slot_to_l2[q] >= 0 {
                let p_l2 = slot_to_l2[q] as usize;
                let o_qq = om2_inv[idx(p_l2, p_l2, q_l2)].max(1e-8);
                let mut active_k: Vec<usize> = Vec::new();
                for k in 0..kw2 {
                    if re_l2_on_w2_mask.get(idx(k, p_l2, kw2)).copied().unwrap_or(1) != 0 {
                        active_k.push(k);
                    }
                }
                let pw = 1 + active_k.len();
                let mut wd = vec![0.0; g2_cnt * pw];
                let mut yv = vec![0.0; g2_cnt];
                for c in 0..g2_cnt {
                    wd[idx(c, 0, g2_cnt)] = 1.0;
                    let mut tot = gamma0[q] + u2[idx(c, p_l2, g2_cnt)];
                    for (ai, &k) in active_k.iter().enumerate() {
                        let wv = w2_in[idx(c, k, g2_cnt)];
                        wd[idx(c, 1 + ai, g2_cnt)] = wv;
                        tot += gamma_w2[idx(k, q, kw2)] * wv;
                    }
                    for p2 in 0..q_l2 {
                        if p2 != p_l2 {
                            let o_q2 = om2_inv[idx(p_l2, p2, q_l2)];
                            if o_q2.abs() > 1e-14 {
                                tot += (o_q2 / o_qq) * u2[idx(c, p2, g2_cnt)];
                            }
                        }
                    }
                    yv[c] = tot;
                }
                let wtw = mat_t_mul(&wd, &wd, pw, g2_cnt, pw);
                let wty = mat_t_mul(&wd, &yv, pw, g2_cnt, 1);
                let mut pr = vec![0.0; pw * pw];
                for i in 0..(pw * pw) {
                    pr[i] = wtw[i] * o_qq;
                }
                pr[idx(0, 0, pw)] += p_pr_base;
                for (ai, &k) in active_k.iter().enumerate() {
                    let pk = if prior_mode == 2 {
                        1.0 / psem_tau2_w2[idx(k, q, kw2)].max(1e-6)
                    } else {
                        p_pr_base
                    };
                    pr[idx(1 + ai, 1 + ai, pw)] += pk;
                }
                let inf: Vec<f64> = wty.iter().map(|v| v * o_qq).collect();
                let b = rmvnorm_precision(&mut rng, &pr, &inf, pw);
                let old_g0 = gamma0[q];
                let old_gw: Vec<f64> = (0..kw2).map(|k| gamma_w2[idx(k, q, kw2)]).collect();
                gamma0[q] = b[0];
                for k in 0..kw2 {
                    gamma_w2[idx(k, q, kw2)] = 0.0;
                }
                for (ai, &k) in active_k.iter().enumerate() {
                    gamma_w2[idx(k, q, kw2)] = b[1 + ai];
                }
                for c in 0..g2_cnt {
                    let mut delta = old_g0 - gamma0[q];
                    for &k in &active_k {
                        delta += (old_gw[k] - gamma_w2[idx(k, q, kw2)]) * w2_in[idx(c, k, g2_cnt)];
                    }
                    u2[idx(c, p_l2, g2_cnt)] += delta;
                }
            } else if q < lin_sl_end {
                let p_q = if prior_mode == 2 && q >= d {
                    1.0 / psem_tau2_g0[q].max(1e-6)
                } else {
                    p_pr_base
                };
                let mut pr = p_q;
                let mut inf = 0.0;
                let mut g0_zero = gamma0.clone();
                g0_zero[q] = 0.0;
                for c in 0..g2_cnt {
                    let st = offsets[c];
                    let en = offsets[c + 1];
                    let tc = en - st;
                    if tc <= max_lag {
                        continue;
                    }
                    let g = l3_of_l2[c];
                    let th_b = build_theta_gc(c, &g0_zero, &gamma_w3, &gamma_w2, &u3, &u2);
                    // RDSEM3 stationary t0 contribution for fixed exogenous slopes
                    if use_stat_t0 && residual_dynamics && q >= d + n_lags * d {
                        let k = q - (d + n_lags * d);
                        let s2 = th_b[lin_sl_end].clamp(-10.0, 10.0).exp().max(1e-6);
                        let phi = th_b[d].clamp(-0.98, 0.98);
                        let w0 = (1.0 - phi * phi).max(0.02) / s2;
                        let x0 = if latent_center_x[k] != 0 {
                            x_in[idx(st, k, n_obs)]
                                - xb3[idx(g, k, g3_cnt)]
                                - xb2[idx(c, k, g2_cnt)]
                        } else {
                            x_in[idx(st, k, n_obs)]
                        };
                        let mut r0 = y_work[idx(st, 0, n_obs)] - th_b[0];
                        for kk in 0..kx {
                            if kk != k {
                                let xk0 = if latent_center_x[kk] != 0 {
                                    x_in[idx(st, kk, n_obs)]
                                        - xb3[idx(g, kk, g3_cnt)]
                                        - xb2[idx(c, kk, g2_cnt)]
                                } else {
                                    x_in[idx(st, kk, n_obs)]
                                };
                                r0 -= th_b[d + n_lags * d + kk] * xk0;
                            }
                        }
                        pr += w0 * x0 * x0;
                        inf += w0 * x0 * r0;
                    }
                    let mut gr_vec = vec![0.0; d];
                    let mut yr_vec = vec![0.0; d];
                    for t_loc in max_lag..tc {
                        let mut any_nz = false;
                        for (j, gr_j) in gr_vec.iter_mut().enumerate().take(d) {
                            let gr = sensitivity_3l(
                                &y_work,
                                x_in,
                                &xb3,
                                &xb2,
                                latent_center_x,
                                g,
                                c,
                                g3_cnt,
                                g2_cnt,
                                st,
                                t_loc,
                                j,
                                d,
                                kx,
                                lags,
                                &th_b,
                                q,
                                residual_dynamics,
                                n_obs,
                            );
                            *gr_j = gr;
                            if gr.abs() > 1e-14 {
                                any_nz = true;
                            }
                        }
                        if !any_nz {
                            continue;
                        }
                        for j in 0..d {
                            if use_full_sigma_w || gr_vec[j].abs() > 1e-14 {
                                let bp = cell_mean_3l(
                                    &y_work,
                                    x_in,
                                    &xb3,
                                    &xb2,
                                    latent_center_x,
                                    g,
                                    c,
                                    g3_cnt,
                                    g2_cnt,
                                    st,
                                    t_loc,
                                    j,
                                    d,
                                    kx,
                                    lags,
                                    max_lag,
                                    &th_b,
                                    residual_dynamics,
                                    n_obs,
                                ) + if has_random_logvf {
                                    cov_lam[j] * f_shock[st + t_loc]
                                } else {
                                    0.0
                                };
                                yr_vec[j] = y_work[idx(st + t_loc, j, n_obs)] - bp;
                            }
                        }
                        if use_full_sigma_w {
                            for j in 0..d {
                                if gr_vec[j].abs() < 1e-14 {
                                    continue;
                                }
                                for k in 0..d {
                                    let w_jk = sigma_w_inv[idx(j, k, d)];
                                    pr += gr_vec[j] * w_jk * gr_vec[k];
                                    inf += gr_vec[j] * w_jk * yr_vec[k];
                                }
                            }
                        } else {
                            for j in 0..d {
                                if gr_vec[j].abs() > 1e-14 {
                                    let s2 =
                                        th_b[lin_sl_end + j].clamp(-10.0, 10.0).exp().max(1e-6);
                                    pr += gr_vec[j] * gr_vec[j] / s2;
                                    inf += gr_vec[j] * yr_vec[j] / s2;
                                }
                            }
                        }
                    }
                }
                let cand = inf / pr + (1.0 / pr).sqrt() * rng.rnorm();
                if use_stat_t0 && q == d {
                    // Exact stationary t0 MH step for fixed univariate AR(1) slope
                    if cand.abs() < 0.98 {
                        let phi_old = gamma0[d].clamp(-0.98, 0.98);
                        let v_old = (1.0 - phi_old * phi_old).max(0.02);
                        let v_new = (1.0 - cand * cand).max(0.02);
                        let mut log_mh = 0.0;
                        for c in 0..g2_cnt {
                            let st = offsets[c];
                            let en = offsets[c + 1];
                            if en - st <= max_lag {
                                continue;
                            }
                            let g = l3_of_l2[c];
                            let th_c = build_theta_gc(c, &gamma0, &gamma_w3, &gamma_w2, &u3, &u2);
                            let s2 = th_c[lin_sl_end].clamp(-10.0, 10.0).exp().max(1e-6);
                            let mut e0 = y_work[idx(st, 0, n_obs)] - th_c[0];
                            if residual_dynamics {
                                for k in 0..kx {
                                    let x0 = if latent_center_x[k] != 0 {
                                        x_in[idx(st, k, n_obs)]
                                            - xb3[idx(g, k, g3_cnt)]
                                            - xb2[idx(c, k, g2_cnt)]
                                    } else {
                                        x_in[idx(st, k, n_obs)]
                                    };
                                    e0 -= th_c[d + n_lags * d + k] * x0;
                                }
                            }
                            log_mh += 0.5 * (v_new.ln() - v_old.ln())
                                - 0.5 * (v_new - v_old) * e0 * e0 / s2;
                        }
                        if log_mh.is_finite() && (log_mh >= 0.0 || rng.runif().ln() < log_mh) {
                            gamma0[d] = cand;
                        }
                    }
                } else {
                    gamma0[q] = cand;
                }
            } else if q < lin_sl_end + d && !use_full_sigma_w {
                let j = q - lin_sl_end;
                let mut rss = 0.0;
                let mut cnt = 0usize;
                for c in 0..g2_cnt {
                    let st = offsets[c];
                    let en = offsets[c + 1];
                    let tc = en - st;
                    if tc <= max_lag {
                        continue;
                    }
                    let g = l3_of_l2[c];
                    let th_c = build_theta_gc(c, &gamma0, &gamma_w3, &gamma_w2, &u3, &u2);
                    for t_loc in max_lag..tc {
                        let pr = cell_mean_3l(
                            &y_work,
                            x_in,
                            &xb3,
                            &xb2,
                            latent_center_x,
                            g,
                            c,
                            g3_cnt,
                            g2_cnt,
                            st,
                            t_loc,
                            j,
                            d,
                            kx,
                            lags,
                            max_lag,
                            &th_c,
                            residual_dynamics,
                            n_obs,
                        ) + if has_random_logvf {
                            cov_lam[j] * f_shock[st + t_loc]
                        } else {
                            0.0
                        };
                        let e = y_work[idx(st + t_loc, j, n_obs)] - pr;
                        rss += e * e;
                        cnt += 1;
                    }
                    if use_stat_t0 {
                        let phi = th_c[d].clamp(-0.98, 0.98);
                        let mut e0 = y_work[idx(st, 0, n_obs)] - th_c[0];
                        if residual_dynamics {
                            for k in 0..kx {
                                let x0 = if latent_center_x[k] != 0 {
                                    x_in[idx(st, k, n_obs)]
                                        - xb3[idx(g, k, g3_cnt)]
                                        - xb2[idx(c, k, g2_cnt)]
                                } else {
                                    x_in[idx(st, k, n_obs)]
                                };
                                e0 -= th_c[d + n_lags * d + k] * x0;
                            }
                        }
                        rss += (1.0 - phi * phi).max(0.02) * e0 * e0;
                        cnt += 1;
                    }
                }
                let (sh, sc) = if prior_mode == 1 {
                    ((0.5 * cnt as f64 - 1.0).max(0.5), 0.5 * rss.max(1e-8))
                } else {
                    (0.001 + 0.5 * cnt as f64, 0.001 + 0.5 * rss)
                };
                let s2 = rng.rinv_gamma(sh, sc);
                gamma0[q] = s2.ln();
            } else if Some(q) == q_vf_slot {
                // Non-random shared residual covariance variance
                let mut rss_f = 0.0;
                let mut cnt = 0usize;
                for c in 0..g2_cnt {
                    let st = offsets[c];
                    let en = offsets[c + 1];
                    if en - st <= max_lag {
                        continue;
                    }
                    for t_loc in max_lag..(en - st) {
                        let fv = f_shock[st + t_loc];
                        rss_f += fv * fv;
                        cnt += 1;
                    }
                }
                let (sh, sc) = if prior_mode == 1 {
                    ((0.5 * cnt as f64 - 1.0).max(0.5), 0.5 * rss_f.max(1e-8))
                } else {
                    (0.001 + 0.5 * cnt as f64, 0.001 + 0.5 * rss_f)
                };
                gamma0[q] = rng.rinv_gamma(sh, sc).ln();
            }

            if slot_to_l3[q] >= 0 && slot_to_l2[q] >= 0 && kw2 > 0 {
                let p_l2 = slot_to_l2[q] as usize;
                let mut active_k: Vec<usize> = Vec::new();
                for k in 0..kw2 {
                    if re_l2_on_w2_mask.get(idx(k, p_l2, kw2)).copied().unwrap_or(1) != 0 {
                        active_k.push(k);
                    }
                }
                if !active_k.is_empty() {
                    let pa = active_k.len();
                    let o_qq = om2_inv[idx(p_l2, p_l2, q_l2)].max(1e-8);
                    let mut wd = vec![0.0; g2_cnt * pa];
                    let mut yv = vec![0.0; g2_cnt];
                    for c in 0..g2_cnt {
                        let mut tot = u2[idx(c, p_l2, g2_cnt)];
                        for (ai, &k) in active_k.iter().enumerate() {
                            let wv = w2_in[idx(c, k, g2_cnt)];
                            wd[idx(c, ai, g2_cnt)] = wv;
                            tot += gamma_w2[idx(k, q, kw2)] * wv;
                        }
                        for p2 in 0..q_l2 {
                            if p2 != p_l2 {
                                let o_q2 = om2_inv[idx(p_l2, p2, q_l2)];
                                if o_q2.abs() > 1e-14 {
                                    tot += (o_q2 / o_qq) * u2[idx(c, p2, g2_cnt)];
                                }
                            }
                        }
                        yv[c] = tot;
                    }
                    let wtw = mat_t_mul(&wd, &wd, pa, g2_cnt, pa);
                    let wty = mat_t_mul(&wd, &yv, pa, g2_cnt, 1);
                    let mut pr = vec![0.0; pa * pa];
                    for i in 0..(pa * pa) {
                        pr[i] = wtw[i] * o_qq;
                    }
                    for (ai, &k) in active_k.iter().enumerate() {
                        let pk = if prior_mode == 2 {
                            1.0 / psem_tau2_w2[idx(k, q, kw2)].max(1e-6)
                        } else {
                            p_pr_base
                        };
                        pr[idx(ai, ai, pa)] += pk;
                    }
                    let inf: Vec<f64> = wty.iter().map(|v| v * o_qq).collect();
                    let b = rmvnorm_precision(&mut rng, &pr, &inf, pa);
                    let old_gw: Vec<f64> = (0..kw2).map(|k| gamma_w2[idx(k, q, kw2)]).collect();
                    for k in 0..kw2 {
                        gamma_w2[idx(k, q, kw2)] = 0.0;
                    }
                    for (ai, &k) in active_k.iter().enumerate() {
                        gamma_w2[idx(k, q, kw2)] = b[ai];
                    }
                    for c in 0..g2_cnt {
                        let mut delta = 0.0;
                        for &k in &active_k {
                            delta +=
                                (old_gw[k] - gamma_w2[idx(k, q, kw2)]) * w2_in[idx(c, k, g2_cnt)];
                        }
                        u2[idx(c, p_l2, g2_cnt)] += delta;
                    }
                }
            }
        }

        // Update full unstructured fixed Sigma_W if multivariate without random logv / logvf
        if use_full_sigma_w {
            let mut s_w = eye(d, if prior_mode == 1 { 1e-5 } else { 0.01 });
            let mut n_tot_w = 0usize;
            let mut e_row = vec![0.0; d];
            for c in 0..g2_cnt {
                let st = offsets[c];
                let en = offsets[c + 1];
                let tc = en - st;
                if tc <= max_lag {
                    continue;
                }
                let g = l3_of_l2[c];
                let th_c = build_theta_gc(c, &gamma0, &gamma_w3, &gamma_w2, &u3, &u2);
                for t_loc in max_lag..tc {
                    for j in 0..d {
                        let pr = cell_mean_3l(
                            &y_work,
                            x_in,
                            &xb3,
                            &xb2,
                            latent_center_x,
                            g,
                            c,
                            g3_cnt,
                            g2_cnt,
                            st,
                            t_loc,
                            j,
                            d,
                            kx,
                            lags,
                            max_lag,
                            &th_c,
                            residual_dynamics,
                            n_obs,
                        );
                        e_row[j] = y_work[idx(st + t_loc, j, n_obs)] - pr;
                    }
                    for ca in 0..d {
                        for ra in 0..d {
                            s_w[idx(ra, ca, d)] += e_row[ra] * e_row[ca];
                        }
                    }
                    n_tot_w += 1;
                }
            }
            let df_w = if prior_mode == 1 {
                (n_tot_w as f64 - d as f64 - 1.0).max(d as f64 + 1.0)
            } else {
                (d as f64 + 2.0) + n_tot_w as f64
            };
            sigma_w = rinv_wishart(&mut rng, df_w, &s_w, d);
            for j in 0..d {
                gamma0[lin_sl_end + j] = sigma_w[idx(j, j, d)].max(1e-6).ln();
            }
        }

        // Update non-random Level-2 AR(1) slopes phi_l2[j] with exact SUR conditioning + stationary m=0 MH
        if any_l2_ar {
            let l2_slot_phi = |p: usize, ph_v: &[f64]| -> f64 {
                let q = l2_slots[p];
                if q < d && l2_ar_mask.get(q).copied().unwrap_or(0) != 0 {
                    ph_v[q]
                } else {
                    0.0
                }
            };
            for &j in &l2_ar_active {
                let p_j = slot_to_l2[j] as usize;
                let o_jj = om2_inv[idx(p_j, p_j, q_l2)].max(1e-8);
                let mut pr = p_pr_base;
                let mut inf = 0.0;
                let mut u0_ss = 0.0;
                let mut g_with_l2 = 0usize;
                for children in l2_by_l3.iter().take(g3_cnt) {
                    if children.is_empty() {
                        continue;
                    }
                    g_with_l2 += 1;
                    u0_ss += u2[idx(children[0], p_j, g2_cnt)].powi(2);
                    for m_idx in 1..children.len() {
                        let c_cur = children[m_idx];
                        let c_prv = children[m_idx - 1];
                        let x_lag = u2[idx(c_prv, p_j, g2_cnt)];
                        let mut y_adj = u2[idx(c_cur, p_j, g2_cnt)];
                        for p2 in 0..q_l2 {
                            if p2 != p_j {
                                let o_j2 = om2_inv[idx(p_j, p2, q_l2)];
                                if o_j2.abs() > 1e-14 {
                                    let z2 = u2[idx(c_cur, p2, g2_cnt)]
                                        - l2_slot_phi(p2, &phi_l2) * u2[idx(c_prv, p2, g2_cnt)];
                                    y_adj += (o_j2 / o_jj) * z2;
                                }
                            }
                        }
                        pr += x_lag * x_lag * o_jj;
                        inf += x_lag * y_adj * o_jj;
                    }
                }
                let cand = inf / pr + (1.0 / pr).sqrt() * rng.rnorm();
                if cand.is_finite() && cand.abs() < 0.98 {
                    let ph_old = phi_l2[j].clamp(-0.98, 0.98);
                    let v_old = (1.0 - ph_old * ph_old).max(0.02);
                    let v_new = (1.0 - cand * cand).max(0.02);
                    let log_mh = 0.5 * (g_with_l2 as f64) * (v_new.ln() - v_old.ln())
                        - 0.5 * o_jj * (v_new - v_old) * u0_ss;
                    if log_mh.is_finite() && (log_mh >= 0.0 || rng.runif().ln() < log_mh) {
                        phi_l2[j] = cand;
                    }
                }
            }
        }

        // Update PSEM Bayesian Lasso scale-mixture hyperparameters (prior_mode == 2)
        if prior_mode == 2 {
            let mut sum_tau2 = 0.0;
            let mut n_pen = 0usize;
            for q in d..lin_sl_end {
                if slot_to_l3[q] < 0 && slot_to_l2[q] < 0 {
                    let b = gamma0[q].abs().max(1e-6);
                    let inv_t2 = rng.rinv_gaussian(psem_lambda2.sqrt() / b, psem_lambda2);
                    psem_tau2_g0[q] = (1.0 / inv_t2.max(1e-8)).clamp(1e-6, 1e6);
                    sum_tau2 += psem_tau2_g0[q];
                    n_pen += 1;
                }
            }
            for &q in &l3_slots {
                let p3 = slot_to_l3[q] as usize;
                for k in 0..kw3 {
                    if re_l3_on_w3_mask.get(idx(k, p3, kw3)).copied().unwrap_or(1) != 0 {
                        let b = gamma_w3[idx(k, q, kw3)].abs().max(1e-6);
                        let inv_t2 = rng.rinv_gaussian(psem_lambda2.sqrt() / b, psem_lambda2);
                        psem_tau2_w3[idx(k, q, kw3)] = (1.0 / inv_t2.max(1e-8)).clamp(1e-6, 1e6);
                        sum_tau2 += psem_tau2_w3[idx(k, q, kw3)];
                        n_pen += 1;
                    }
                }
            }
            for &q in &l2_slots {
                let p2 = slot_to_l2[q] as usize;
                for k in 0..kw2 {
                    if re_l2_on_w2_mask.get(idx(k, p2, kw2)).copied().unwrap_or(1) != 0 {
                        let b = gamma_w2[idx(k, q, kw2)].abs().max(1e-6);
                        let inv_t2 = rng.rinv_gaussian(psem_lambda2.sqrt() / b, psem_lambda2);
                        psem_tau2_w2[idx(k, q, kw2)] = (1.0 / inv_t2.max(1e-8)).clamp(1e-6, 1e6);
                        sum_tau2 += psem_tau2_w2[idx(k, q, kw2)];
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

        // Re-center non-AR L2 deviations within L3 clusters for clean L2/L3 hierarchical mixing!
        for p2 in 0..q_l2 {
            let q = l2_slots[p2];
            if slot_to_l3[q] >= 0 && !(q < d && l2_ar_mask.get(q).copied().unwrap_or(0) != 0) {
                let p3 = slot_to_l3[q] as usize;
                for g in 0..g3_cnt {
                    if !l2_by_l3[g].is_empty() {
                        let m_g = l2_by_l3[g]
                            .iter()
                            .map(|&c| u2[idx(c, p2, g2_cnt)])
                            .sum::<f64>()
                            / l2_by_l3[g].len() as f64;
                        for &c in &l2_by_l3[g] {
                            u2[idx(c, p2, g2_cnt)] -= m_g;
                        }
                        u3[idx(g, p3, g3_cnt)] += m_g;
                    }
                }
            }
        }

        // 5. Update Omega^{(3)} and Omega^{(2)} (+ Exact PX Scale Move on random logv / logvf)
        let psi0_diag = if prior_mode == 1 { 1e-5 } else { 0.05 };
        let nu0_iw3 = if prior_mode == 1 {
            -(q_l3 as f64 + 1.0)
        } else {
            q_l3 as f64 + 2.0
        };
        let nu0_iw2 = if prior_mode == 1 {
            -(q_l2 as f64 + 1.0)
        } else {
            q_l2 as f64 + 2.0
        };
        if q_l3 > 0 {
            let mut s3 = eye(q_l3, psi0_diag);
            for g in 0..g3_cnt {
                for ca in 0..q_l3 {
                    for ra in 0..q_l3 {
                        s3[idx(ra, ca, q_l3)] += u3[idx(g, ra, g3_cnt)] * u3[idx(g, ca, g3_cnt)];
                    }
                }
            }
            let df3 = (g3_cnt as f64 + nu0_iw3).max(q_l3 as f64 + 1.0);
            omega3 = rinv_wishart(&mut rng, df3, &s3, q_l3);
        }
        if q_l2 > 0 {
            let mut s2 = eye(q_l2, psi0_diag);
            if any_l2_ar {
                let mut z_vec = vec![0.0; q_l2];
                for children in l2_by_l3.iter().take(g3_cnt) {
                    for (m_idx, &c) in children.iter().enumerate() {
                        for p in 0..q_l2 {
                            let q = l2_slots[p];
                            let ph = if q < d && l2_ar_mask.get(q).copied().unwrap_or(0) != 0 {
                                phi_l2[q]
                            } else {
                                0.0
                            };
                            z_vec[p] = if ph.abs() < 1e-12 {
                                u2[idx(c, p, g2_cnt)]
                            } else if m_idx == 0 {
                                (1.0 - ph * ph).max(0.02).sqrt() * u2[idx(c, p, g2_cnt)]
                            } else {
                                u2[idx(c, p, g2_cnt)] - ph * u2[idx(children[m_idx - 1], p, g2_cnt)]
                            };
                        }
                        for ca in 0..q_l2 {
                            for ra in 0..q_l2 {
                                s2[idx(ra, ca, q_l2)] += z_vec[ra] * z_vec[ca];
                            }
                        }
                    }
                }
            } else {
                for c in 0..g2_cnt {
                    for ca in 0..q_l2 {
                        for ra in 0..q_l2 {
                            s2[idx(ra, ca, q_l2)] +=
                                u2[idx(c, ra, g2_cnt)] * u2[idx(c, ca, g2_cnt)];
                        }
                    }
                }
            }
            let df2 = (g2_cnt as f64 + nu0_iw2).max(q_l2 as f64 + 1.0);
            omega2 = rinv_wishart(&mut rng, df2, &s2, q_l2);
        }

        // 5b. Exact PX Scale Group Move for L2 & L3 random log-variances and logvf
        if q_l2 > 0 && (any_rand_logv || has_random_logvf) {
            let mut om2_inv_px = spd_invert(&omega2, q_l2);
            let mut px_l2: Vec<(usize, usize, Option<usize>)> = Vec::new();
            for j in 0..d {
                let q_lv = lin_sl_end + j;
                if slot_to_l2[q_lv] >= 0 {
                    px_l2.push((q_lv, slot_to_l2[q_lv] as usize, Some(j)));
                }
            }
            if let Some(qvf) = q_vf_slot {
                if slot_to_l2[qvf] >= 0 {
                    px_l2.push((qvf, slot_to_l2[qvf] as usize, None));
                }
            }
            for (q_lv, p_lv, j_opt) in px_l2 {
                let mut base_vec = vec![0.0; g2_cnt];
                let mut u_vec = vec![0.0; g2_cnt];
                let mut n_vec = vec![0.0; g2_cnt];
                let mut rss_vec = vec![0.0; g2_cnt];
                for c in 0..g2_cnt {
                    let st = offsets[c];
                    let en = offsets[c + 1];
                    let tc = en - st;
                    let g = l3_of_l2[c];
                    let th_c = build_theta_gc(c, &gamma0, &gamma_w3, &gamma_w2, &u3, &u2);
                    u_vec[c] = u2[idx(c, p_lv, g2_cnt)];
                    base_vec[c] = th_c[q_lv] - u_vec[c];
                    if tc <= max_lag {
                        continue;
                    }
                    let mut n_eff = (tc - max_lag) as f64;
                    let mut rss = 0.0;
                    if let Some(j) = j_opt {
                        for t_loc in max_lag..tc {
                            let pr = cell_mean_3l(
                                &y_work,
                                x_in,
                                &xb3,
                                &xb2,
                                latent_center_x,
                                g,
                                c,
                                g3_cnt,
                                g2_cnt,
                                st,
                                t_loc,
                                j,
                                d,
                                kx,
                                lags,
                                max_lag,
                                &th_c,
                                residual_dynamics,
                                n_obs,
                            ) + if has_random_logvf {
                                cov_lam[j] * f_shock[st + t_loc]
                            } else {
                                0.0
                            };
                            let e = y_work[idx(st + t_loc, j, n_obs)] - pr;
                            rss += e * e;
                        }
                        if use_stat_t0 {
                            let phi = th_c[d].clamp(-0.98, 0.98);
                            let mut e0 = y_work[idx(st, 0, n_obs)] - th_c[0];
                            if residual_dynamics {
                                for k in 0..kx {
                                    let x0 = if latent_center_x[k] != 0 {
                                        x_in[idx(st, k, n_obs)]
                                            - xb3[idx(g, k, g3_cnt)]
                                            - xb2[idx(c, k, g2_cnt)]
                                    } else {
                                        x_in[idx(st, k, n_obs)]
                                    };
                                    e0 -= th_c[d + n_lags * d + k] * x0;
                                }
                            }
                            rss += (1.0 - phi * phi).max(0.02) * e0 * e0;
                            n_eff += 1.0;
                        }
                    } else {
                        for t_loc in max_lag..tc {
                            let fv = f_shock[st + t_loc];
                            rss += fv * fv;
                        }
                    }
                    n_vec[c] = n_eff;
                    rss_vec[c] = rss;
                }
                for _ in 0..3 {
                    let delta = 0.18 * rng.rnorm();
                    let a = delta.exp();
                    let inv_a2 = (-2.0 * delta).exp();
                    let mut valid = true;
                    let mut ll_diff = 0.0;
                    for c in 0..g2_cnt {
                        let cur_v = base_vec[c] + u_vec[c];
                        let prop_v = base_vec[c] + a * u_vec[c];
                        if prop_v.abs() > 10.0 {
                            valid = false;
                            break;
                        }
                        if n_vec[c] > 0.0 {
                            ll_diff += -0.5 * n_vec[c] * (prop_v - cur_v)
                                - 0.5
                                    * ((-prop_v).exp().min(1e12) - (-cur_v).exp().min(1e12))
                                    * rss_vec[c];
                        }
                    }
                    if !valid {
                        continue;
                    }
                    let om_pp = om2_inv_px[idx(p_lv, p_lv, q_l2)];
                    let log_acc =
                        ll_diff - nu0_iw2 * delta - 0.5 * psi0_diag * (inv_a2 - 1.0) * om_pp;
                    if log_acc.is_finite() && (log_acc >= 0.0 || rng.runif().ln() < log_acc) {
                        for c in 0..g2_cnt {
                            u_vec[c] *= a;
                            u2[idx(c, p_lv, g2_cnt)] = u_vec[c];
                        }
                        let inv_a = 1.0 / a;
                        for r in 0..q_l2 {
                            omega2[idx(r, p_lv, q_l2)] *= a;
                            omega2[idx(p_lv, r, q_l2)] *= a;
                            om2_inv_px[idx(r, p_lv, q_l2)] *= inv_a;
                            om2_inv_px[idx(p_lv, r, q_l2)] *= inv_a;
                        }
                    }
                }
            }
        }
        if q_l3 > 0 && (any_rand_logv || has_random_logvf) {
            let mut om3_inv_px = spd_invert(&omega3, q_l3);
            let mut px_l3: Vec<(usize, usize, Option<usize>)> = Vec::new();
            for j in 0..d {
                let q_lv = lin_sl_end + j;
                if slot_to_l3[q_lv] >= 0 {
                    px_l3.push((q_lv, slot_to_l3[q_lv] as usize, Some(j)));
                }
            }
            if let Some(qvf) = q_vf_slot {
                if slot_to_l3[qvf] >= 0 {
                    px_l3.push((qvf, slot_to_l3[qvf] as usize, None));
                }
            }
            for (q_lv, p_lv, j_opt) in px_l3 {
                let mut u3_vec = vec![0.0; g3_cnt];
                let mut g_items: Vec<Vec<(f64, f64, f64)>> = vec![Vec::new(); g3_cnt];
                for g in 0..g3_cnt {
                    u3_vec[g] = u3[idx(g, p_lv, g3_cnt)];
                    for &c in &l2_by_l3[g] {
                        let st = offsets[c];
                        let en = offsets[c + 1];
                        let tc = en - st;
                        if tc <= max_lag {
                            continue;
                        }
                        let th_c = build_theta_gc(c, &gamma0, &gamma_w3, &gamma_w2, &u3, &u2);
                        let blv = th_c[q_lv] - u3_vec[g];
                        let mut n_eff = (tc - max_lag) as f64;
                        let mut rss = 0.0;
                        if let Some(j) = j_opt {
                            for t_loc in max_lag..tc {
                                let pr = cell_mean_3l(
                                    &y_work,
                                    x_in,
                                    &xb3,
                                    &xb2,
                                    latent_center_x,
                                    g,
                                    c,
                                    g3_cnt,
                                    g2_cnt,
                                    st,
                                    t_loc,
                                    j,
                                    d,
                                    kx,
                                    lags,
                                    max_lag,
                                    &th_c,
                                    residual_dynamics,
                                    n_obs,
                                ) + if has_random_logvf {
                                    cov_lam[j] * f_shock[st + t_loc]
                                } else {
                                    0.0
                                };
                                let e = y_work[idx(st + t_loc, j, n_obs)] - pr;
                                rss += e * e;
                            }
                            if use_stat_t0 {
                                let phi = th_c[d].clamp(-0.98, 0.98);
                                let mut e0 = y_work[idx(st, 0, n_obs)] - th_c[0];
                                if residual_dynamics {
                                    for k in 0..kx {
                                        let x0 = if latent_center_x[k] != 0 {
                                            x_in[idx(st, k, n_obs)]
                                                - xb3[idx(g, k, g3_cnt)]
                                                - xb2[idx(c, k, g2_cnt)]
                                        } else {
                                            x_in[idx(st, k, n_obs)]
                                        };
                                        e0 -= th_c[d + n_lags * d + k] * x0;
                                    }
                                }
                                rss += (1.0 - phi * phi).max(0.02) * e0 * e0;
                                n_eff += 1.0;
                            }
                        } else {
                            for t_loc in max_lag..tc {
                                let fv = f_shock[st + t_loc];
                                rss += fv * fv;
                            }
                        }
                        g_items[g].push((n_eff, blv, rss));
                    }
                }
                for _ in 0..3 {
                    let delta = 0.18 * rng.rnorm();
                    let a = delta.exp();
                    let inv_a2 = (-2.0 * delta).exp();
                    let mut valid = true;
                    let mut ll_diff = 0.0;
                    for g in 0..g3_cnt {
                        for &(ne, blv, rss) in &g_items[g] {
                            let cur_v = blv + u3_vec[g];
                            let prop_v = blv + a * u3_vec[g];
                            if prop_v.abs() > 10.0 {
                                valid = false;
                                break;
                            }
                            ll_diff += -0.5 * ne * (prop_v - cur_v)
                                - 0.5
                                    * ((-prop_v).exp().min(1e12) - (-cur_v).exp().min(1e12))
                                    * rss;
                        }
                        if !valid {
                            break;
                        }
                    }
                    if !valid {
                        continue;
                    }
                    let om_pp = om3_inv_px[idx(p_lv, p_lv, q_l3)];
                    let log_acc =
                        ll_diff - nu0_iw3 * delta - 0.5 * psi0_diag * (inv_a2 - 1.0) * om_pp;
                    if log_acc.is_finite() && (log_acc >= 0.0 || rng.runif().ln() < log_acc) {
                        for g in 0..g3_cnt {
                            u3_vec[g] *= a;
                            u3[idx(g, p_lv, g3_cnt)] = u3_vec[g];
                        }
                        let inv_a = 1.0 / a;
                        for r in 0..q_l3 {
                            omega3[idx(r, p_lv, q_l3)] *= a;
                            omega3[idx(p_lv, r, q_l3)] *= a;
                            om3_inv_px[idx(r, p_lv, q_l3)] *= inv_a;
                            om3_inv_px[idx(p_lv, r, q_l3)] *= inv_a;
                        }
                    }
                }
            }
        }

        if keep_pos < n_keep && step == keep_indices[keep_pos] {
            let mut cur = 0usize;
            for &g0 in &gamma0 {
                draws[idx(keep_pos, cur, n_keep)] = g0;
                cur += 1;
            }
            for &j in &l2_ar_active {
                draws[idx(keep_pos, cur, n_keep)] = phi_l2[j];
                cur += 1;
            }
            for (p3, &q) in l3_slots.iter().enumerate() {
                for k in 0..kw3 {
                    if re_l3_on_w3_mask.get(idx(k, p3, kw3)).copied().unwrap_or(1) != 0 {
                        draws[idx(keep_pos, cur, n_keep)] = gamma_w3[idx(k, q, kw3)];
                        cur += 1;
                    }
                }
            }
            for (p2, &q) in l2_slots.iter().enumerate() {
                for k in 0..kw2 {
                    if re_l2_on_w2_mask.get(idx(k, p2, kw2)).copied().unwrap_or(1) != 0 {
                        draws[idx(keep_pos, cur, n_keep)] = gamma_w2[idx(k, q, kw2)];
                        cur += 1;
                    }
                }
            }
            for ca in 0..q_l3 {
                for ra in ca..q_l3 {
                    draws[idx(keep_pos, cur, n_keep)] = omega3[idx(ra, ca, q_l3)];
                    cur += 1;
                }
            }
            for ca in 0..q_l2 {
                for ra in ca..q_l2 {
                    draws[idx(keep_pos, cur, n_keep)] = omega2[idx(ra, ca, q_l2)];
                    cur += 1;
                }
            }
            if use_full_sigma_w {
                for ca in 0..d {
                    for ra in (ca + 1)..d {
                        draws[idx(keep_pos, cur, n_keep)] = sigma_w[idx(ra, ca, d)];
                        cur += 1;
                    }
                }
                for i in 0..(d * d) {
                    sigma_w_accum[i] += sigma_w[i];
                }
            }
            for p in 0..q_l2 {
                for c in 0..g2_cnt {
                    l2_draws[keep_pos + n_keep * (c + g2_cnt * p)] = u2[idx(c, p, g2_cnt)];
                }
            }
            for p in 0..q_l3 {
                for g in 0..g3_cnt {
                    l3_draws[keep_pos + n_keep * (g + g3_cnt * p)] = u3[idx(g, p, g3_cnt)];
                }
            }

            let mut ll = 0.0;
            let sw_inv_ll = if use_full_sigma_w {
                spd_invert(&sigma_w, d)
            } else {
                Vec::new()
            };
            let sw_logdet = if use_full_sigma_w {
                log_det_spd(&sigma_w, d)
            } else {
                0.0
            };
            let mut e_vec = vec![0.0; d];
            for c in 0..g2_cnt {
                let st = offsets[c];
                let en = offsets[c + 1];
                let tc = en - st;
                let g = l3_of_l2[c];
                let th_c = build_theta_gc(c, &gamma0, &gamma_w3, &gamma_w2, &u3, &u2);
                let mut sig2_c = vec![1.0; d];
                for j in 0..d {
                    let s2 = th_c[lin_sl_end + j].clamp(-10.0, 10.0).exp().max(1e-6);
                    sig2_c[j] = s2;
                    sig_accum[idx(c, j, g2_cnt)] += s2;
                }
                let vf_c = if has_random_logvf {
                    let v = th_c[q_vf_slot.unwrap()].clamp(-10.0, 10.0).exp().max(1e-6);
                    sig_vf_accum[c] += v;
                    v
                } else {
                    0.0
                };
                // Rank-1 Woodbury quantities for exact marginal multivariate Gaussian LL under random residual covariance
                let mut wb_denom = 1.0;
                let mut wb_logdet = 0.0;
                if has_random_logvf {
                    let mut quad_lam = 0.0;
                    for j in 0..d {
                        quad_lam += cov_lam[j] * cov_lam[j] / sig2_c[j];
                        wb_logdet += sig2_c[j].ln();
                    }
                    wb_denom = 1.0 + vf_c * quad_lam;
                    wb_logdet += wb_denom.ln();
                }
                for t_loc in 0..tc {
                    let mut all_obs = true;
                    for j in 0..d {
                        let fv = if t_loc >= max_lag {
                            cell_mean_3l(
                                &y_work,
                                x_in,
                                &xb3,
                                &xb2,
                                latent_center_x,
                                g,
                                c,
                                g3_cnt,
                                g2_cnt,
                                st,
                                t_loc,
                                j,
                                d,
                                kx,
                                lags,
                                max_lag,
                                &th_c,
                                residual_dynamics,
                                n_obs,
                            )
                        } else {
                            th_c[j]
                        };
                        fitted_accum[idx(st + t_loc, j, n_obs)] += fv;
                        e_vec[j] = y_work[idx(st + t_loc, j, n_obs)] - fv;
                        if miss_y[idx(st + t_loc, j, n_obs)] {
                            all_obs = false;
                        }
                    }
                    if t_loc >= max_lag {
                        if use_full_sigma_w && all_obs {
                            let mut quad = 0.0;
                            for ca in 0..d {
                                for ra in 0..d {
                                    quad += e_vec[ra] * sw_inv_ll[idx(ra, ca, d)] * e_vec[ca];
                                }
                            }
                            ll += -0.5 * (d as f64 * (2.0 * PI).ln() + sw_logdet + quad);
                        } else if has_random_logvf && all_obs {
                            let mut q_diag = 0.0;
                            let mut q_dot = 0.0;
                            for j in 0..d {
                                let ej = e_vec[j];
                                q_diag += ej * ej / sig2_c[j];
                                q_dot += cov_lam[j] * ej / sig2_c[j];
                            }
                            let quad = q_diag - (vf_c / wb_denom) * q_dot * q_dot;
                            ll += -0.5 * (d as f64 * (2.0 * PI).ln() + wb_logdet + quad);
                        } else {
                            for j in 0..d {
                                if !miss_y[idx(st + t_loc, j, n_obs)] {
                                    let s2 = sig2_c[j];
                                    let e = e_vec[j];
                                    ll += -0.5 * ((2.0 * PI * s2).ln() + e * e / s2);
                                }
                            }
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
    let sw_bar: Vec<f64> = if use_full_sigma_w {
        sigma_w_accum.iter().map(|&v| v * inv_keep).collect()
    } else {
        Vec::new()
    };
    let sw_bar_inv = if use_full_sigma_w {
        spd_invert(&sw_bar, d)
    } else {
        Vec::new()
    };
    let sw_bar_logdet = if use_full_sigma_w {
        log_det_spd(&sw_bar, d)
    } else {
        0.0
    };
    let mut e_vec = vec![0.0; d];
    for c in 0..g2_cnt {
        let st = offsets[c];
        let en = offsets[c + 1];
        let tc = en - st;
        let sig2_bar: Vec<f64> = (0..d)
            .map(|j| (sig_accum[idx(c, j, g2_cnt)] * inv_keep).max(1e-6))
            .collect();
        let vf_bar = if has_random_logvf {
            (sig_vf_accum[c] * inv_keep).max(1e-6)
        } else {
            0.0
        };
        let mut wb_denom = 1.0;
        let mut wb_logdet = 0.0;
        if has_random_logvf {
            let mut quad_lam = 0.0;
            for j in 0..d {
                quad_lam += cov_lam[j] * cov_lam[j] / sig2_bar[j];
                wb_logdet += sig2_bar[j].ln();
            }
            wb_denom = 1.0 + vf_bar * quad_lam;
            wb_logdet += wb_denom.ln();
        }
        for t_loc in max_lag..tc {
            let mut all_obs = true;
            for j in 0..d {
                e_vec[j] =
                    y_work[idx(st + t_loc, j, n_obs)] - fitted_mean[idx(st + t_loc, j, n_obs)];
                if miss_y[idx(st + t_loc, j, n_obs)] {
                    all_obs = false;
                }
            }
            if use_full_sigma_w && all_obs {
                let mut quad = 0.0;
                for ca in 0..d {
                    for ra in 0..d {
                        quad += e_vec[ra] * sw_bar_inv[idx(ra, ca, d)] * e_vec[ca];
                    }
                }
                ll_hat += -0.5 * (d as f64 * (2.0 * PI).ln() + sw_bar_logdet + quad);
            } else if has_random_logvf && all_obs {
                let mut q_diag = 0.0;
                let mut q_dot = 0.0;
                for j in 0..d {
                    let ej = e_vec[j];
                    q_diag += ej * ej / sig2_bar[j];
                    q_dot += cov_lam[j] * ej / sig2_bar[j];
                }
                let quad = q_diag - (vf_bar / wb_denom) * q_dot * q_dot;
                ll_hat += -0.5 * (d as f64 * (2.0 * PI).ln() + wb_logdet + quad);
            } else {
                for j in 0..d {
                    if !miss_y[idx(st + t_loc, j, n_obs)] {
                        let s2 = sig2_bar[j];
                        let e = e_vec[j];
                        ll_hat += -0.5 * ((2.0 * PI * s2).ln() + e * e / s2);
                    }
                }
            }
        }
    }
    let pd = (d_bar - (-2.0 * ll_hat)).max(0.0);
    let dic = d_bar + pd;

    ThreeLevelOutput {
        draws,
        param_count: n_params,
        l2_draws,
        l3_draws,
        fitted_mean,
        dic,
        pd,
    }
}

fn log_det_spd(a: &[f64], n: usize) -> f64 {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut s = a[idx(j, j, n)];
        for k in 0..j {
            s -= l[idx(j, k, n)] * l[idx(j, k, n)];
        }
        let d_val = s.max(1e-12).sqrt();
        l[idx(j, j, n)] = d_val;
        for i in (j + 1)..n {
            let mut si = a[idx(i, j, n)];
            for k in 0..j {
                si -= l[idx(i, k, n)] * l[idx(j, k, n)];
            }
            l[idx(i, j, n)] = si / d_val;
        }
    }
    2.0 * (0..n).map(|j| l[idx(j, j, n)].ln()).sum::<f64>()
}

#[allow(clippy::too_many_arguments)]
fn cell_mean_3l(
    y: &[f64],
    x: &[f64],
    xb3: &[f64],
    xb2: &[f64],
    lc_x: &[i32],
    g: usize,
    c: usize,
    g3_cnt: usize,
    g2_cnt: usize,
    st: usize,
    t_loc: usize,
    j: usize,
    d: usize,
    kx: usize,
    lags: &[usize],
    max_lag: usize,
    th: &[f64],
    residual_dynamics: bool,
    n_obs: usize,
) -> f64 {
    let n_lags = lags.len();
    let n_sl = n_lags * d + kx;
    let mut pr = th[j];
    for k in 0..kx {
        let xv = if lc_x[k] != 0 {
            x[idx(st + t_loc, k, n_obs)] - xb3[idx(g, k, g3_cnt)] - xb2[idx(c, k, g2_cnt)]
        } else {
            x[idx(st + t_loc, k, n_obs)]
        };
        pr += th[d + j * n_sl + n_lags * d + k] * xv;
    }
    if t_loc < max_lag {
        return pr;
    }
    for (li, &lv) in lags.iter().enumerate() {
        for mc in 0..d {
            let phi = th[d + j * n_sl + li * d + mc];
            let mut dev = y[idx(st + t_loc - lv, mc, n_obs)] - th[mc];
            if residual_dynamics {
                for k in 0..kx {
                    let xl = if lc_x[k] != 0 {
                        x[idx(st + t_loc - lv, k, n_obs)]
                            - xb3[idx(g, k, g3_cnt)]
                            - xb2[idx(c, k, g2_cnt)]
                    } else {
                        x[idx(st + t_loc - lv, k, n_obs)]
                    };
                    dev -= th[d + mc * n_sl + n_lags * d + k] * xl;
                }
            }
            pr += phi * dev;
        }
    }
    pr
}

#[allow(clippy::too_many_arguments)]
fn sensitivity_3l(
    y: &[f64],
    x: &[f64],
    xb3: &[f64],
    xb2: &[f64],
    lc_x: &[i32],
    g: usize,
    c: usize,
    g3_cnt: usize,
    g2_cnt: usize,
    st: usize,
    t_loc: usize,
    j: usize,
    d: usize,
    kx: usize,
    lags: &[usize],
    th: &[f64],
    q: usize,
    residual_dynamics: bool,
    n_obs: usize,
) -> f64 {
    let n_lags = lags.len();
    let n_sl = n_lags * d + kx;
    if q < d {
        let mut drv = if q == j { 1.0 } else { 0.0 };
        for (li, _) in lags.iter().enumerate() {
            drv -= th[d + j * n_sl + li * d + q];
        }
        return drv;
    }
    let rel = q - d;
    let out_j = rel / n_sl;
    let sl_i = rel % n_sl;
    if out_j != j {
        return 0.0;
    }
    if sl_i < n_lags * d {
        let li = sl_i / d;
        let mc = sl_i % d;
        let lv = lags[li];
        let mut dev = y[idx(st + t_loc - lv, mc, n_obs)] - th[mc];
        if residual_dynamics {
            for k in 0..kx {
                let xl = if lc_x[k] != 0 {
                    x[idx(st + t_loc - lv, k, n_obs)]
                        - xb3[idx(g, k, g3_cnt)]
                        - xb2[idx(c, k, g2_cnt)]
                } else {
                    x[idx(st + t_loc - lv, k, n_obs)]
                };
                dev -= th[d + mc * n_sl + n_lags * d + k] * xl;
            }
        }
        dev
    } else {
        let k = sl_i - n_lags * d;
        if lc_x[k] != 0 {
            x[idx(st + t_loc, k, n_obs)] - xb3[idx(g, k, g3_cnt)] - xb2[idx(c, k, g2_cnt)]
        } else {
            x[idx(st + t_loc, k, n_obs)]
        }
    }
}
