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

/// Three-Level DSEM (DSEM3, Asparouhov & Muthen 2026) exact simultaneous two-level block Gibbs + Newton-Laplace MH sampler.
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
    let q_tot = lin_sl_end + d;

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
    let mut gamma_w3: Vec<f64> = vec![0.0_f64; kw3 * q_tot];
    let mut gamma_w2: Vec<f64> = vec![0.0_f64; kw2 * q_tot];
    let mut u3: Vec<f64> = vec![0.0_f64; g3_cnt * q_l3.max(1)];
    let mut u2: Vec<f64> = vec![0.0_f64; g2_cnt * q_l2.max(1)];
    let mut omega3 = eye(q_l3, 0.1);
    let mut omega2 = eye(q_l2, 0.1);

    let n_om3 = q_l3 * (q_l3 + 1) / 2;
    let n_om2 = q_l2 * (q_l2 + 1) / 2;
    let n_params = q_tot + q_l3 * kw3 + q_l2 * kw2 + n_om3 + n_om2;

    let keep_indices: Vec<usize> = ((warmup + 1)..=iter)
        .filter(|i| (i - warmup - 1) % thin == 0)
        .collect();
    let n_keep = keep_indices.len();
    let mut draws = vec![0.0; n_keep * n_params];
    let mut l2_draws = vec![0.0; n_keep * g2_cnt * q_l2.max(1)];
    let mut l3_draws = vec![0.0; n_keep * g3_cnt * q_l3.max(1)];
    let mut fitted_accum = vec![0.0; n_obs * d];
    let mut sig_accum = vec![0.0; g2_cnt * d];
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
                    for k in 0..kw3 {
                        th[q] += gw3[idx(k, q, kw3)] * w3_in[idx(g, k, g3_cnt)];
                    }
                    th[q] += u3_m[idx(g, slot_to_l3[q] as usize, g3_cnt)];
                }
                if slot_to_l2[q] >= 0 {
                    for k in 0..kw2 {
                        th[q] += gw2[idx(k, q, kw2)] * w2_in[idx(c, k, g2_cnt)];
                    }
                    th[q] += u2_m[idx(c, slot_to_l2[q] as usize, g2_cnt)];
                }
            }
            th
        };

        // 1. Impute missing y
        for c in 0..g2_cnt {
            let st = offsets[c];
            let en = offsets[c + 1];
            let tc = en - st;
            if tc <= max_lag {
                continue;
            }
            let th = build_theta_gc(c, &gamma0, &gamma_w3, &gamma_w2, &u3, &u2);
            for t_loc in 0..tc {
                let r_abs = st + t_loc;
                for j in 0..d {
                    if miss_y[idx(r_abs, j, n_obs)] {
                        let s2 = th[lin_sl_end + j].clamp(-10.0, 10.0).exp();
                        let pred = cell_mean_3l(
                            &y_work,
                            x_in,
                            &xb3,
                            &xb2,
                            latent_center_x,
                            l3_of_l2[c],
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
                        y_work[idx(r_abs, j, n_obs)] = pred + s2.sqrt() * rng.rnorm();
                    }
                }
            }
        }

        // 2. SIMULTANEOUS TWO-LEVEL BLOCK GIBBS (Asparouhov & Muthen, 2026b):
        //    For each Level-3 cluster g, jointly sample (u3[g], {u2[c]}_{c in C_g}) in O(C_g)
        //    via exact block-arrow Schur complement marginalization (eliminates L2-L3 autocorrelation!).
        //    2a. Joint (L3, L2) intercepts
        //    2b. Joint (L3, L2) slopes (conditional on updated intercepts)
        //    2c. L2 & L3 random log-variances via strictly-concave Newton-Laplace Independence MH
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
            for g in 0..g3_cnt {
                // Prior precision & info for u3[g, l3_pos] conditional on other u3 slots
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

                // Store per-child (c, P_cc, h_c, P_0c) for exact backward conditional draw
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
                        for j in 0..d {
                            let s2 = th_base[lin_sl_end + j].clamp(-10.0, 10.0).exp();
                            // Stationary initial-state contribution for pure univariate AR(1) intercepts
                            if !is_slope_block
                                && d == 1
                                && n_lags == 1
                                && kx == 0
                                && !residual_dynamics
                            {
                                let phi = th_base[d].clamp(-0.95, 0.95);
                                let w0 = (1.0 - phi * phi).max(0.05) / s2;
                                let r0 = y_work[idx(st, 0, n_obs)] - th_base[0];
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
                            }
                            let mut gr3 = vec![0.0; n3];
                            let mut gr2 = vec![0.0; n2];
                            for t_loc in max_lag..tc {
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
                                );
                                let y_rem = y_work[idx(st + t_loc, j, n_obs)] - bp;
                                for (a, &pa) in l3_pos.iter().enumerate() {
                                    gr3[a] = sensitivity_3l(
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
                                    gr2[a] = sensitivity_3l(
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
                                let inv_s2 = 1.0 / s2;
                                for a in 0..n3 {
                                    h0[a] += gr3[a] * y_rem * inv_s2;
                                    for b in 0..n3 {
                                        p00[idx(a, b, n3)] += gr3[a] * gr3[b] * inv_s2;
                                    }
                                    for b in 0..n2 {
                                        p0c[idx(a, b, n3)] += gr3[a] * gr2[b] * inv_s2;
                                    }
                                }
                                for a in 0..n2 {
                                    hc[a] += gr2[a] * y_rem * inv_s2;
                                    for b in 0..n2 {
                                        pcc[idx(a, b, n2)] += gr2[a] * gr2[b] * inv_s2;
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

                // Draw u3_sub[g] from exact marginal conditional posterior, then u2_sub[c] | u3_sub[g]
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

                // Stationarity check for autoregressive slope block before committing joint draw
                let mut accept_block = true;
                if is_slope_block {
                    for &(c, ref dr2) in &u2_draws_g {
                        for (a, &pa) in l2_pos.iter().enumerate() {
                            let q = l2_slots[pa];
                            let is_ar = (q - d) % n_slopes < n_lags * d;
                            if is_ar {
                                let mut tot = gamma0[q] + dr2[a];
                                if slot_to_l3[q] >= 0 {
                                    for k in 0..kw3 {
                                        tot += gamma_w3[idx(k, q, kw3)] * w3_in[idx(g, k, g3_cnt)];
                                    }
                                    if let Some(pos3) =
                                        l3_pos.iter().position(|&p| l3_slots[p] == q)
                                    {
                                        tot += u3_draw[pos3];
                                    } else {
                                        tot += u3[idx(g, slot_to_l3[q] as usize, g3_cnt)];
                                    }
                                }
                                for k in 0..kw2 {
                                    tot += gamma_w2[idx(k, q, kw2)] * w2_in[idx(c, k, g2_cnt)];
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
                                    tot += gamma_w3[idx(k, q, kw3)] * w3_in[idx(g, k, g3_cnt)];
                                }
                                if !tot.is_finite() || tot.abs() >= 0.98 {
                                    accept_block = false;
                                    break;
                                }
                            }
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

        // 2c. L2 random log-variances (strictly-concave Newton-Laplace Independence MH)
        if q_l2 > 0 {
            for c in 0..g2_cnt {
                let st = offsets[c];
                let en = offsets[c + 1];
                let tc = en - st;
                if tc <= max_lag {
                    continue;
                }
                let g = l3_of_l2[c];
                for j in 0..d {
                    let q_lv = lin_sl_end + j;
                    if slot_to_l2[q_lv] >= 0 {
                        let p_lv = slot_to_l2[q_lv] as usize;
                        let mut cur_v = u2[idx(c, p_lv, g2_cnt)];
                        let th_cur = build_theta_gc(c, &gamma0, &gamma_w3, &gamma_w2, &u3, &u2);
                        let mut rss = 0.0;
                        let n_eff = (tc - max_lag) as f64;
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
                            );
                            let e = y_work[idx(st + t_loc, j, n_obs)] - pr;
                            rss += e * e;
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
        }

        // 3c. L3 random log-variances (strictly-concave Newton-Laplace Independence MH)
        if q_l3 > 0 {
            for g in 0..g3_cnt {
                for j in 0..d {
                    let q_lv = lin_sl_end + j;
                    if slot_to_l3[q_lv] >= 0 {
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
                                );
                                let e = y_work[idx(st + t_loc, j, n_obs)] - pr;
                                rss += e * e;
                            }
                            c_rss.push(((tc - max_lag) as f64, th_c[q_lv] - cur_v, rss));
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
        }

        // 4. Update gamma0, gamma_w3, gamma_w2, and fixed non-random slots!
        let p_pr = if prior_mode == 1 {
            1e-10
        } else {
            1.0 / (beta_sd * beta_sd)
        };
        for q in 0..q_tot {
            if slot_to_l3[q] >= 0 {
                let p_l3 = slot_to_l3[q] as usize;
                let o_qq = om3_inv[idx(p_l3, p_l3, q_l3)].max(1e-8);
                let pw = 1 + kw3;
                let mut wd = vec![0.0; g3_cnt * pw];
                let mut yv = vec![0.0; g3_cnt];
                for g in 0..g3_cnt {
                    wd[idx(g, 0, g3_cnt)] = 1.0;
                    let mut tot = gamma0[q] + u3[idx(g, p_l3, g3_cnt)];
                    for k in 0..kw3 {
                        wd[idx(g, 1 + k, g3_cnt)] = w3_in[idx(g, k, g3_cnt)];
                        tot += gamma_w3[idx(k, q, kw3)] * w3_in[idx(g, k, g3_cnt)];
                    }
                    // Exact SUR adjustment for correlated L3 random effects
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
                for i in 0..pw {
                    pr[idx(i, i, pw)] += p_pr;
                }
                let inf: Vec<f64> = wty.iter().map(|v| v * o_qq).collect();
                let b = rmvnorm_precision(&mut rng, &pr, &inf, pw);
                let old_g0 = gamma0[q];
                let old_gw: Vec<f64> = (0..kw3).map(|k| gamma_w3[idx(k, q, kw3)]).collect();
                gamma0[q] = b[0];
                for k in 0..kw3 {
                    gamma_w3[idx(k, q, kw3)] = b[1 + k];
                }
                for g in 0..g3_cnt {
                    let mut delta = old_g0 - gamma0[q];
                    for k in 0..kw3 {
                        delta += (old_gw[k] - gamma_w3[idx(k, q, kw3)]) * w3_in[idx(g, k, g3_cnt)];
                    }
                    u3[idx(g, p_l3, g3_cnt)] += delta;
                }
            } else if slot_to_l2[q] >= 0 {
                let p_l2 = slot_to_l2[q] as usize;
                let o_qq = om2_inv[idx(p_l2, p_l2, q_l2)].max(1e-8);
                let pw = 1 + kw2;
                let mut wd = vec![0.0; g2_cnt * pw];
                let mut yv = vec![0.0; g2_cnt];
                for c in 0..g2_cnt {
                    wd[idx(c, 0, g2_cnt)] = 1.0;
                    let mut tot = gamma0[q] + u2[idx(c, p_l2, g2_cnt)];
                    for k in 0..kw2 {
                        wd[idx(c, 1 + k, g2_cnt)] = w2_in[idx(c, k, g2_cnt)];
                        tot += gamma_w2[idx(k, q, kw2)] * w2_in[idx(c, k, g2_cnt)];
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
                for i in 0..pw {
                    pr[idx(i, i, pw)] += p_pr;
                }
                let inf: Vec<f64> = wty.iter().map(|v| v * o_qq).collect();
                let b = rmvnorm_precision(&mut rng, &pr, &inf, pw);
                let old_g0 = gamma0[q];
                let old_gw: Vec<f64> = (0..kw2).map(|k| gamma_w2[idx(k, q, kw2)]).collect();
                gamma0[q] = b[0];
                for k in 0..kw2 {
                    gamma_w2[idx(k, q, kw2)] = b[1 + k];
                }
                for c in 0..g2_cnt {
                    let mut delta = old_g0 - gamma0[q];
                    for k in 0..kw2 {
                        delta += (old_gw[k] - gamma_w2[idx(k, q, kw2)]) * w2_in[idx(c, k, g2_cnt)];
                    }
                    u2[idx(c, p_l2, g2_cnt)] += delta;
                }
            } else if q < lin_sl_end {
                let mut pr = p_pr;
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
                    let th_b = build_theta_gc(c, &g0_zero, &gamma_w3, &gamma_w2, &u3, &u2);
                    for j in 0..d {
                        let s2 = th_b[lin_sl_end + j].clamp(-10.0, 10.0).exp();
                        for t_loc in max_lag..tc {
                            let gr = sensitivity_3l(
                                &y_work,
                                x_in,
                                &xb3,
                                &xb2,
                                latent_center_x,
                                l3_of_l2[c],
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
                            if gr.abs() > 1e-14 {
                                let bp = cell_mean_3l(
                                    &y_work,
                                    x_in,
                                    &xb3,
                                    &xb2,
                                    latent_center_x,
                                    l3_of_l2[c],
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
                                );
                                let yr = y_work[idx(st + t_loc, j, n_obs)] - bp;
                                pr += gr * gr / s2;
                                inf += gr * yr / s2;
                            }
                        }
                    }
                }
                gamma0[q] = inf / pr + (1.0 / pr).sqrt() * rng.rnorm();
            } else {
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
                    let th_c = build_theta_gc(c, &gamma0, &gamma_w3, &gamma_w2, &u3, &u2);
                    for t_loc in max_lag..tc {
                        let pr = cell_mean_3l(
                            &y_work,
                            x_in,
                            &xb3,
                            &xb2,
                            latent_center_x,
                            l3_of_l2[c],
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
                        let e = y_work[idx(st + t_loc, j, n_obs)] - pr;
                        rss += e * e;
                        cnt += 1;
                    }
                }
                let s2 = rng.rinv_gamma(0.001 + 0.5 * cnt as f64, 0.001 + 0.5 * rss);
                gamma0[q] = s2.ln();
            }

            if slot_to_l3[q] >= 0 && slot_to_l2[q] >= 0 && kw2 > 0 {
                let p_l2 = slot_to_l2[q] as usize;
                let var_q2 = omega2[idx(p_l2, p_l2, q_l2)].max(1e-6);
                let mut wd = vec![0.0; g2_cnt * kw2];
                let mut yv = vec![0.0; g2_cnt];
                for c in 0..g2_cnt {
                    let mut tot = u2[idx(c, p_l2, g2_cnt)];
                    for k in 0..kw2 {
                        wd[idx(c, k, g2_cnt)] = w2_in[idx(c, k, g2_cnt)];
                        tot += gamma_w2[idx(k, q, kw2)] * w2_in[idx(c, k, g2_cnt)];
                    }
                    yv[c] = tot;
                }
                let wtw = mat_t_mul(&wd, &wd, kw2, g2_cnt, kw2);
                let wty = mat_t_mul(&wd, &yv, kw2, g2_cnt, 1);
                let mut pr = vec![0.0; kw2 * kw2];
                for i in 0..(kw2 * kw2) {
                    pr[i] = wtw[i] / var_q2;
                }
                for i in 0..kw2 {
                    pr[idx(i, i, kw2)] += p_pr;
                }
                let inf: Vec<f64> = wty.iter().map(|v| v / var_q2).collect();
                let b = rmvnorm_precision(&mut rng, &pr, &inf, kw2);
                let old_gw: Vec<f64> = (0..kw2).map(|k| gamma_w2[idx(k, q, kw2)]).collect();
                for k in 0..kw2 {
                    gamma_w2[idx(k, q, kw2)] = b[k];
                }
                for c in 0..g2_cnt {
                    let mut delta = 0.0;
                    for k in 0..kw2 {
                        delta += (old_gw[k] - gamma_w2[idx(k, q, kw2)]) * w2_in[idx(c, k, g2_cnt)];
                    }
                    u2[idx(c, p_l2, g2_cnt)] += delta;
                }
            }
        }

        // Re-center L2 deviations within L3 clusters for clean L2/L3 hierarchical mixing!
        for p2 in 0..q_l2 {
            let q = l2_slots[p2];
            if slot_to_l3[q] >= 0 {
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

        // 5. Update Omega^{(3)} and Omega^{(2)}
        if q_l3 > 0 {
            let mut s3 = eye(q_l3, if prior_mode == 1 { 1e-4 } else { 0.05 });
            for g in 0..g3_cnt {
                for ca in 0..q_l3 {
                    for ra in 0..q_l3 {
                        s3[idx(ra, ca, q_l3)] += u3[idx(g, ra, g3_cnt)] * u3[idx(g, ca, g3_cnt)];
                    }
                }
            }
            omega3 = rinv_wishart(&mut rng, (q_l3 as f64 + 2.0) + g3_cnt as f64, &s3, q_l3);
        }
        if q_l2 > 0 {
            let mut s2 = eye(q_l2, if prior_mode == 1 { 1e-4 } else { 0.05 });
            for c in 0..g2_cnt {
                for ca in 0..q_l2 {
                    for ra in 0..q_l2 {
                        s2[idx(ra, ca, q_l2)] += u2[idx(c, ra, g2_cnt)] * u2[idx(c, ca, g2_cnt)];
                    }
                }
            }
            omega2 = rinv_wishart(&mut rng, (q_l2 as f64 + 2.0) + g2_cnt as f64, &s2, q_l2);
        }

        if keep_pos < n_keep && step == keep_indices[keep_pos] {
            let mut cur = 0usize;
            for &g0 in &gamma0 {
                draws[idx(keep_pos, cur, n_keep)] = g0;
                cur += 1;
            }
            for &q in &l3_slots {
                for k in 0..kw3 {
                    draws[idx(keep_pos, cur, n_keep)] = gamma_w3[idx(k, q, kw3)];
                    cur += 1;
                }
            }
            for &q in &l2_slots {
                for k in 0..kw2 {
                    draws[idx(keep_pos, cur, n_keep)] = gamma_w2[idx(k, q, kw2)];
                    cur += 1;
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
            for c in 0..g2_cnt {
                let st = offsets[c];
                let en = offsets[c + 1];
                let tc = en - st;
                let th_c = build_theta_gc(c, &gamma0, &gamma_w3, &gamma_w2, &u3, &u2);
                for j in 0..d {
                    let s2 = th_c[lin_sl_end + j].clamp(-10.0, 10.0).exp();
                    sig_accum[idx(c, j, g2_cnt)] += s2;
                    for t_loc in 0..tc {
                        let fv = if t_loc >= max_lag {
                            cell_mean_3l(
                                &y_work,
                                x_in,
                                &xb3,
                                &xb2,
                                latent_center_x,
                                l3_of_l2[c],
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
                        if t_loc >= max_lag && !miss_y[idx(st + t_loc, j, n_obs)] {
                            let e = y_work[idx(st + t_loc, j, n_obs)] - fv;
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
    for c in 0..g2_cnt {
        let st = offsets[c];
        let en = offsets[c + 1];
        let tc = en - st;
        for j in 0..d {
            let s2 = (sig_accum[idx(c, j, g2_cnt)] * inv_keep).max(1e-6);
            for t_loc in max_lag..tc {
                if !miss_y[idx(st + t_loc, j, n_obs)] {
                    let e =
                        y_work[idx(st + t_loc, j, n_obs)] - fitted_mean[idx(st + t_loc, j, n_obs)];
                    ll_hat += -0.5 * ((2.0 * PI * s2).ln() + e * e / s2);
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
