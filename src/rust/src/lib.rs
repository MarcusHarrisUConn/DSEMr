mod arma;
mod cc_dynamic_cfa;
mod crossclassified;
mod ct_dsem;
mod dynamic_sem;
mod kalman;
mod linalg;
mod rng;
mod threelevel;
mod twolevel;
mod var_n1;

use extendr_api::prelude::*;

#[extendr]
fn dsem_ar_sufficient(
    x: Vec<f64>,
    y: Vec<f64>,
    n: i32,
    p: i32,
) -> std::result::Result<List, Error> {
    if n <= 0 || p <= 0 {
        return Err(Error::Other("n and p must be positive".into()));
    }
    let n = n as usize;
    let p = p as usize;
    let (xtx, xty, yty) =
        sufficient_kernel(&x, &y, n, p).map_err(|message| Error::Other(message.into()))?;

    Ok(list!(
        xtx = xtx,
        xty = xty,
        yty = yty,
        n = n as i32,
        p = p as i32
    ))
}

#[extendr]
fn dsem_grouped_sufficient(
    x: Vec<f64>,
    y: Vec<f64>,
    group_sizes: Vec<i32>,
    n: i32,
    p: i32,
) -> std::result::Result<List, Error> {
    if n <= 0 || p <= 0 || group_sizes.is_empty() {
        return Err(Error::Other(
            "n, p, and group_sizes must be positive".into(),
        ));
    }
    let n_usize = n as usize;
    let p_usize = p as usize;
    let sizes: std::result::Result<Vec<usize>, _> = group_sizes
        .iter()
        .map(|size| {
            if *size <= 0 {
                Err("every group size must be positive")
            } else {
                Ok(*size as usize)
            }
        })
        .collect();
    let sizes = sizes.map_err(|message| Error::Other(message.into()))?;
    let (xtx, xty, yty) = grouped_sufficient_kernel(&x, &y, &sizes, n_usize, p_usize)
        .map_err(|message| Error::Other(message.into()))?;
    Ok(list!(
        xtx = xtx,
        xty = xty,
        yty = yty,
        group_sizes = group_sizes,
        groups = sizes.len() as i32,
        p = p
    ))
}

#[extendr]
fn dsem_factor_sufficient(
    y: Vec<f64>,
    eta: Vec<f64>,
    n: i32,
    items: i32,
    factors: i32,
) -> std::result::Result<List, Error> {
    if n <= 0 || items <= 0 || factors <= 0 {
        return Err(Error::Other(
            "n, items, and factors must be positive".into(),
        ));
    }
    let n_usize = n as usize;
    let items_usize = items as usize;
    let factors_usize = factors as usize;
    let (ztz, zty, yty, n_obs) =
        factor_sufficient_kernel(&y, &eta, n_usize, items_usize, factors_usize)
            .map_err(|message| Error::Other(message.into()))?;
    let n_obs: Vec<i32> = n_obs.into_iter().map(|value| value as i32).collect();

    Ok(list!(
        ztz = ztz,
        zty = zty,
        yty = yty,
        n_obs = n_obs,
        n = n,
        items = items,
        factors = factors,
        design_cols = factors + 1
    ))
}

/// Native Rust MCMC kernel for N=1 VAR(p), Covariate, RDSEM, Probit, and Tobit Censored DSEM.
/// dims = [n_time, d, kx, residual_dynamics, prior_mode, iter, warmup, thin, seed]
/// hyper = [beta_sd, sigma_shape, sigma_scale, optional censor_lo[0..d], optional censor_hi[0..d]]
#[extendr]
fn dsem_rust_var_n1(
    y: Vec<f64>,
    x: Vec<f64>,
    lags: Vec<i32>,
    outcome_types: Vec<i32>,
    dims: Vec<i32>,
    hyper: Vec<f64>,
) -> std::result::Result<List, Error> {
    if dims.len() < 9 || hyper.len() < 3 {
        return Err(Error::Other("invalid dims or hyper vector length".into()));
    }
    let d = dims[1] as usize;
    let (censor_lo, censor_hi) = if hyper.len() >= 3 + 2 * d {
        (
            hyper[3..(3 + d)].to_vec(),
            hyper[(3 + d)..(3 + 2 * d)].to_vec(),
        )
    } else {
        (vec![f64::NEG_INFINITY; d], vec![f64::INFINITY; d])
    };
    let lags_u: Vec<usize> = lags.into_iter().map(|v| v.max(1) as usize).collect();
    let out = var_n1::mcmc_var_n1(
        &y,
        &x,
        dims[0] as usize,
        d,
        dims[2] as usize,
        &lags_u,
        &outcome_types,
        &censor_lo,
        &censor_hi,
        dims[3] != 0,
        dims[4],
        hyper[0],
        hyper[1],
        hyper[2],
        dims[5] as usize,
        dims[6] as usize,
        dims[7] as usize,
        dims[8] as u64,
    );
    Ok(list!(
        draws = out.draws,
        param_count = out.param_count as i32,
        fitted_mean = out.fitted_mean,
        dic = out.dic,
        pd = out.pd,
        d_bar = out.d_bar
    ))
}

/// Native Rust MCMC kernel for Two-Level Location-Scale, Multivariate VAR, Censored, and RDSEM.
/// dims = [d, kx, kw, kz, latent_center_y, residual_dynamics, prior_mode, iter, warmup, thin, seed]
/// hyper = [beta_sd, omega_df_add, omega_scale_diag, optional censor_lo[0..d], optional censor_hi[0..d]]
/// masks = concatenated [z_on_re_mask, z_on_w_mask, re_on_w_mask, slope_is_random, has_random_logv, latent_center_x, outcome_types]
#[extendr]
#[allow(clippy::too_many_arguments)]
fn dsem_rust_twolevel(
    y: Vec<f64>,
    x: Vec<f64>,
    cluster_sizes: Vec<i32>,
    w: Vec<f64>,
    z: Vec<f64>,
    lags: Vec<i32>,
    masks: Vec<i32>,
    dims: Vec<i32>,
    hyper: Vec<f64>,
) -> std::result::Result<List, Error> {
    if dims.len() < 11 || hyper.len() < 3 {
        return Err(Error::Other("invalid twolevel dims/hyper length".into()));
    }
    let d = dims[0] as usize;
    let kx = dims[1] as usize;
    let kw = dims[2] as usize;
    let kz = dims[3] as usize;
    let lags_u: Vec<usize> = lags.into_iter().map(|v| v.max(1) as usize).collect();
    let n_sl = lags_u.len() * d + kx;
    let n_slope_entries = d * n_sl;

    // Unpack masks deterministically from tail-known sizes
    let tail_len = n_slope_entries + d + kx + d;
    if masks.len() < tail_len {
        return Err(Error::Other("masks vector too short".into()));
    }
    let head_len = masks.len() - tail_len;
    let slope_is_random = &masks[head_len..(head_len + n_slope_entries)];
    let has_random_logv = &masks[(head_len + n_slope_entries)..(head_len + n_slope_entries + d)];
    let latent_center_x =
        &masks[(head_len + n_slope_entries + d)..(head_len + n_slope_entries + d + kx)];
    let outcome_types = &masks[(head_len + n_slope_entries + d + kx)..];

    let (censor_lo, censor_hi) = if hyper.len() >= 3 + 2 * d {
        (
            hyper[3..(3 + d)].to_vec(),
            hyper[(3 + d)..(3 + 2 * d)].to_vec(),
        )
    } else {
        (vec![f64::NEG_INFINITY; d], vec![f64::INFINITY; d])
    };
    let shared_cov_loadings: Vec<f64> = if hyper.len() >= 3 + 3 * d {
        hyper[(3 + 2 * d)..(3 + 3 * d)].to_vec()
    } else {
        Vec::new()
    };
    let has_random_logvf = d > 1
        && shared_cov_loadings.len() >= d
        && shared_cov_loadings.iter().take(d).any(|&v| v.abs() > 1e-12);

    let n_rs = slope_is_random.iter().filter(|&&v| v != 0).count();
    let n_lv = (0..d)
        .filter(|&j| has_random_logv[j] != 0 && outcome_types[j] < 2)
        .count();
    let q_re = d + n_rs + n_lv + usize::from(has_random_logvf);
    let expected_head = q_re * kz + kw * kz + kw * q_re;
    if head_len != expected_head {
        return Err(Error::Other("between mask dimensions mismatch".into()));
    }
    let z_on_re_mask = &masks[0..(q_re * kz)];
    let z_on_w_mask = &masks[(q_re * kz)..(q_re * kz + kw * kz)];
    let re_on_w_mask = &masks[(q_re * kz + kw * kz)..head_len];

    let sizes_u: Vec<usize> = cluster_sizes.into_iter().map(|s| s as usize).collect();
    let out = twolevel::mcmc_twolevel(
        &y,
        &x,
        &sizes_u,
        &w,
        &z,
        z_on_re_mask,
        z_on_w_mask,
        re_on_w_mask,
        d,
        kx,
        kw,
        kz,
        &lags_u,
        slope_is_random,
        has_random_logv,
        dims[4] != 0,
        latent_center_x,
        dims[5] != 0,
        outcome_types,
        &censor_lo,
        &censor_hi,
        &shared_cov_loadings,
        dims[6],
        hyper[0],
        hyper[1],
        hyper[2],
        dims[7] as usize,
        dims[8] as usize,
        dims[9] as usize,
        dims[10] as u64,
    );
    Ok(list!(
        draws = out.draws,
        param_count = out.param_count as i32,
        re_draws = out.re_draws,
        q_re = out.q_re as i32,
        fitted_mean = out.fitted_mean,
        dic = out.dic,
        pd = out.pd
    ))
}

/// Native Rust MCMC kernel for Three-Level DSEM (DSEM3, simultaneous two-level block Gibbs,
/// non-random Level-2 AR, random residual covariance, short-T stationary MH, and PSEM shrinkage).
/// dims = [g3_cnt, d, kx, kw2, kw3, residual_dynamics, prior_mode, iter, warmup, thin, seed]
/// hyper = [beta_sd, optional shared_cov_loadings[0..d]]
/// masks = [has_l2_re (q_tot), has_l3_re (q_tot), latent_center_x (kx),
///          optional l2_ar_mask (d), optional re_l3_on_w3_mask (kw3*q_l3), optional re_l2_on_w2_mask (kw2*q_l2)]
#[extendr]
#[allow(clippy::too_many_arguments)]
fn dsem_rust_threelevel(
    y: Vec<f64>,
    x: Vec<f64>,
    l2_sizes: Vec<i32>,
    l3_of_l2: Vec<i32>,
    w2: Vec<f64>,
    w3: Vec<f64>,
    lags: Vec<i32>,
    masks: Vec<i32>,
    dims: Vec<i32>,
    hyper: Vec<f64>,
) -> std::result::Result<List, Error> {
    if dims.len() < 11 || hyper.is_empty() {
        return Err(Error::Other("invalid threelevel dims or hyper length".into()));
    }
    let g3_cnt = dims[0] as usize;
    let d = dims[1] as usize;
    let kx = dims[2] as usize;
    let kw2 = dims[3] as usize;
    let kw3 = dims[4] as usize;
    let beta_sd = hyper[0];
    let shared_cov_loadings: Vec<f64> = if hyper.len() > d {
        hyper[1..(1 + d)].to_vec()
    } else {
        Vec::new()
    };
    let has_random_logvf = d > 1
        && shared_cov_loadings.len() >= d
        && shared_cov_loadings.iter().take(d).any(|&v| v.abs() > 1e-12);
    let lags_u: Vec<usize> = lags.into_iter().map(|v| v.max(1) as usize).collect();
    let q_tot = d + d * (lags_u.len() * d + kx) + d + usize::from(has_random_logvf);
    let base_len = 2 * q_tot + kx;
    if masks.len() < base_len {
        return Err(Error::Other("invalid threelevel masks length".into()));
    }
    let has_l2_re = &masks[0..q_tot];
    let has_l3_re = &masks[q_tot..(2 * q_tot)];
    let latent_center_x = &masks[(2 * q_tot)..base_len];
    let q_l2 = has_l2_re.iter().filter(|&&v| v != 0).count();
    let q_l3 = has_l3_re.iter().filter(|&&v| v != 0).count();
    let extra = &masks[base_len..];
    let l2_ar_mask: Vec<i32> = if extra.len() >= d {
        extra[0..d].to_vec()
    } else {
        vec![0; d]
    };
    let re_l3_on_w3_mask: Vec<i32> = if extra.len() >= d + kw3 * q_l3 {
        extra[d..(d + kw3 * q_l3)].to_vec()
    } else {
        vec![1; kw3 * q_l3]
    };
    let re_l2_on_w2_mask: Vec<i32> = if extra.len() >= d + kw3 * q_l3 + kw2 * q_l2 {
        extra[(d + kw3 * q_l3)..(d + kw3 * q_l3 + kw2 * q_l2)].to_vec()
    } else {
        vec![1; kw2 * q_l2]
    };

    let sizes_u: Vec<usize> = l2_sizes.into_iter().map(|v| v as usize).collect();
    let l3_u: Vec<usize> = l3_of_l2.into_iter().map(|v| v as usize).collect();
    let out = threelevel::mcmc_threelevel(
        &y,
        &x,
        &sizes_u,
        &l3_u,
        g3_cnt,
        &w2,
        &w3,
        d,
        kx,
        kw2,
        kw3,
        &lags_u,
        has_l2_re,
        has_l3_re,
        latent_center_x,
        &l2_ar_mask,
        &re_l3_on_w3_mask,
        &re_l2_on_w2_mask,
        &shared_cov_loadings,
        dims[5] != 0,
        dims[6],
        beta_sd,
        dims[7] as usize,
        dims[8] as usize,
        dims[9] as usize,
        dims[10] as u64,
    );
    Ok(list!(
        draws = out.draws,
        param_count = out.param_count as i32,
        l2_draws = out.l2_draws,
        l3_draws = out.l3_draws,
        fitted_mean = out.fitted_mean,
        dic = out.dic,
        pd = out.pd
    ))
}

/// Native Rust MCMC kernel for Cross-Classified DSEM with selective subject/time random slopes and predictors.
/// dims = [n_id, n_time, d, kx, kw_id, kw_time, prior_mode, iter, warmup, thin, seed]
/// masks = [slope_rand_id (d*n_sl), slope_rand_time (d*n_sl), has_random_logv (d),
///          optional re_id_on_w_mask (kw_id*q_id), optional re_time_on_w_mask (kw_time*q_time)]
#[extendr]
#[allow(clippy::too_many_arguments)]
fn dsem_rust_crossclassified(
    y_grid: Vec<f64>,
    x_grid: Vec<f64>,
    w_id: Vec<f64>,
    w_time: Vec<f64>,
    lags: Vec<i32>,
    masks: Vec<i32>,
    dims: Vec<i32>,
    beta_sd: f64,
) -> std::result::Result<List, Error> {
    if dims.len() < 11 {
        return Err(Error::Other("invalid crossclassified dims length".into()));
    }
    let d = dims[2] as usize;
    let kx = dims[3] as usize;
    let kw_id = dims[4] as usize;
    let kw_time = dims[5] as usize;
    let lags_u: Vec<usize> = lags.into_iter().map(|v| v.max(1) as usize).collect();
    let n_sl_tot = d * (lags_u.len() * d + kx);
    let base_len = 2 * n_sl_tot + d;
    if masks.len() < base_len {
        return Err(Error::Other("invalid crossclassified masks length".into()));
    }
    let sl_id = &masks[0..n_sl_tot];
    let sl_tm = &masks[n_sl_tot..(2 * n_sl_tot)];
    let lv_m = &masks[(2 * n_sl_tot)..base_len];
    let q_id = d
        + sl_id.iter().filter(|&&v| v != 0).count()
        + lv_m.iter().filter(|&&v| v != 0).count();
    let q_time = d + sl_tm.iter().filter(|&&v| v != 0).count();
    let extra = &masks[base_len..];
    let re_id_on_w_mask: Vec<i32> = if extra.len() >= kw_id * q_id {
        extra[0..(kw_id * q_id)].to_vec()
    } else {
        vec![1; kw_id * q_id]
    };
    let re_time_on_w_mask: Vec<i32> = if extra.len() >= kw_id * q_id + kw_time * q_time {
        extra[(kw_id * q_id)..(kw_id * q_id + kw_time * q_time)].to_vec()
    } else {
        vec![1; kw_time * q_time]
    };

    let out = crossclassified::mcmc_crossclassified(
        &y_grid,
        &x_grid,
        &w_id,
        &w_time,
        sl_id,
        sl_tm,
        lv_m,
        &re_id_on_w_mask,
        &re_time_on_w_mask,
        dims[0] as usize,
        dims[1] as usize,
        d,
        kx,
        kw_id,
        kw_time,
        &lags_u,
        dims[6],
        beta_sd,
        dims[7] as usize,
        dims[8] as usize,
        dims[9] as usize,
        dims[10] as u64,
    );
    Ok(list!(
        draws = out.draws,
        param_count = out.param_count as i32,
        id_draws = out.id_draws,
        time_draws = out.time_draws,
        q_id = out.q_id as i32,
        q_time = out.q_time as i32,
        fitted_mean = out.fitted_mean,
        dic = out.dic,
        pd = out.pd
    ))
}

/// Native Rust collapsed MCMC kernel for N=1 and Two-Level ARMA(1,1) DSEM.
/// dims = [random_ar, prior_mode, store_latent, iter, warmup, thin, seed]
/// hyper = [theta_y_fixed, ar_bound]
#[extendr]
fn dsem_rust_arma11(
    y: Vec<f64>,
    cluster_sizes: Vec<i32>,
    dims: Vec<i32>,
    hyper: Vec<f64>,
) -> std::result::Result<List, Error> {
    if dims.len() < 7 || hyper.len() < 2 {
        return Err(Error::Other("invalid arma11 dims/hyper length".into()));
    }
    let sizes_u: Vec<usize> = cluster_sizes.into_iter().map(|v| v as usize).collect();
    let out = arma::mcmc_arma11(
        &y,
        &sizes_u,
        dims[0] != 0,
        hyper[0],
        dims[1],
        hyper[1],
        dims[2] != 0,
        dims[3] as usize,
        dims[4] as usize,
        dims[5] as usize,
        dims[6] as u64,
    );
    Ok(list!(
        draws = out.draws,
        param_count = out.param_count as i32,
        latent_draws = out.latent_draws,
        re_draws = out.re_draws,
        fitted_mean = out.fitted_mean,
        dic = out.dic,
        pd = out.pd
    ))
}

/// Native Rust MCMC kernel for Cross-Classified / Two-Level Dynamic CFA with Measurement Invariance
/// or Random Factor Loadings.
/// dims = [n_id, n_time, n_items, random_loadings, has_f_subj, has_f_time, has_time_int,
///         random_ar, psi_fixed_one, prior_mode, store_latent, iter, warmup, thin, seed]
#[extendr]
fn dsem_rust_cc_dynamic_cfa(
    y_grid: Vec<f64>,
    dims: Vec<i32>,
    ar_bound: f64,
) -> std::result::Result<List, Error> {
    if dims.len() < 15 {
        return Err(Error::Other("invalid cc_dynamic_cfa dims length".into()));
    }
    let out = cc_dynamic_cfa::mcmc_cc_dynamic_cfa(
        &y_grid,
        dims[0] as usize,
        dims[1] as usize,
        dims[2] as usize,
        dims[3] != 0,
        dims[4] != 0,
        dims[5] != 0,
        dims[6] != 0,
        dims[7] != 0,
        dims[8] != 0,
        dims[9],
        ar_bound,
        dims[10] != 0,
        dims[11] as usize,
        dims[12] as usize,
        dims[13] as usize,
        dims[14] as u64,
    );
    Ok(list!(
        draws = out.draws,
        param_count = out.param_count as i32,
        latent_draws = out.latent_draws,
        id_draws = out.id_draws,
        time_draws = out.time_draws,
        fitted_mean = out.fitted_mean,
        dic = out.dic,
        pd = out.pd
    ))
}

/// Native Rust MCMC kernel for Continuous-Time DSEM / CT-RDSEM (Ornstein-Uhlenbeck exact exp(A dt)).
/// dims = [d, kx, random_drift, random_logv, prior_mode, iter, warmup, thin, seed]
#[extendr]
fn dsem_rust_ct_dsem(
    y: Vec<f64>,
    x: Vec<f64>,
    time: Vec<f64>,
    cluster_sizes: Vec<i32>,
    dims: Vec<i32>,
    beta_sd: f64,
) -> std::result::Result<List, Error> {
    if dims.len() < 9 {
        return Err(Error::Other("invalid ct_dsem dims length".into()));
    }
    let sizes_u: Vec<usize> = cluster_sizes.into_iter().map(|v| v as usize).collect();
    let out = ct_dsem::mcmc_ct_dsem(
        &y,
        &x,
        &time,
        &sizes_u,
        dims[0] as usize,
        dims[1] as usize,
        dims[2] != 0,
        dims[3] != 0,
        dims[4],
        beta_sd,
        dims[5] as usize,
        dims[6] as usize,
        dims[7] as usize,
        dims[8] as u64,
    );
    Ok(list!(
        draws = out.draws,
        param_count = out.param_count as i32,
        re_draws = out.re_draws,
        fitted_mean = out.fitted_mean,
        dic = out.dic,
        pd = out.pd
    ))
}

/// Native Rust MCMC kernel for Single-Level & Two-Level Dynamic CFA, Dynamic SEM, WNFS, and Dynamic IRT.
/// dims = [n_items, m, m_b, use_wnfs, equal_wb_loadings, psi_fixed_one, psi_correlated, fix_logv_mean_zero, prior_mode, store_latent, iter, warmup, thin, seed]
/// modes = concatenated [lambda0_mode (J*m), lambda1_mode (J*m), lambdab_mode (J*m_b), phi_mode (m*m), has_random_logv (m), item_types (J)]
#[extendr]
fn dsem_rust_dynamic_sem(
    y: Vec<f64>,
    cluster_sizes: Vec<i32>,
    lambda0_init: Vec<f64>,
    theta_fixed: Vec<f64>,
    modes: Vec<i32>,
    dims: Vec<i32>,
    ar_bound: f64,
) -> std::result::Result<List, Error> {
    if dims.len() < 14 {
        return Err(Error::Other("invalid dynamic_sem dims length".into()));
    }
    let j = dims[0] as usize;
    let m = dims[1] as usize;
    let mb = dims[2] as usize;
    let need = j * m + j * m + j * mb + m * m + m + j;
    if modes.len() != need {
        return Err(Error::Other("invalid dynamic_sem modes length".into()));
    }
    let c1 = j * m;
    let c2 = c1 + j * m;
    let c3 = c2 + j * mb;
    let c4 = c3 + m * m;
    let c5 = c4 + m;

    let sizes_u: Vec<usize> = cluster_sizes.into_iter().map(|v| v as usize).collect();
    let out = dynamic_sem::mcmc_dynamic_sem(
        &y,
        &sizes_u,
        j,
        m,
        mb,
        &modes[0..c1],
        &lambda0_init,
        dims[3] != 0,
        &modes[c1..c2],
        &modes[c2..c3],
        dims[4] != 0,
        &modes[c3..c4],
        dims[5] != 0,
        dims[6] != 0,
        &modes[c4..c5],
        dims[7] != 0,
        &modes[c5..],
        &theta_fixed,
        dims[8],
        ar_bound,
        dims[9] != 0,
        dims[10] as usize,
        dims[11] as usize,
        dims[12] as usize,
        dims[13] as u64,
    );
    Ok(list!(
        draws = out.draws,
        param_count = out.param_count as i32,
        latent_draws = out.latent_draws,
        re_draws = out.re_draws,
        fitted_mean = out.fitted_mean,
        dic = out.dic,
        pd = out.pd
    ))
}

fn sufficient_kernel(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
) -> std::result::Result<(Vec<f64>, Vec<f64>, f64), &'static str> {
    if x.len() != n * p {
        return Err("x length does not match n * p");
    }
    if y.len() != n {
        return Err("y length does not match n");
    }
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    let mut yty = 0.0;
    for row in 0..n {
        let response = y[row];
        yty += response * response;
        for col in 0..p {
            let x_col = x[row + n * col];
            xty[col] += x_col * response;
            for other in 0..p {
                xtx[col + p * other] += x_col * x[row + n * other];
            }
        }
    }
    Ok((xtx, xty, yty))
}

type GroupedStatistics = (Vec<f64>, Vec<f64>, Vec<f64>);
type FactorStatistics = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>);

#[derive(Clone, Copy, Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let adjusted = value - self.correction;
        let updated = self.sum + adjusted;
        self.correction = (updated - self.sum) - adjusted;
        self.sum = updated;
    }
}

fn grouped_sufficient_kernel(
    x: &[f64],
    y: &[f64],
    group_sizes: &[usize],
    n: usize,
    p: usize,
) -> std::result::Result<GroupedStatistics, &'static str> {
    if x.len() != n * p || y.len() != n {
        return Err("x or y dimensions do not match n and p");
    }
    if group_sizes.iter().sum::<usize>() != n {
        return Err("group sizes do not sum to n");
    }
    let groups = group_sizes.len();
    let mut xtx = vec![0.0; groups * p * p];
    let mut xty = vec![0.0; groups * p];
    let mut yty = vec![0.0; groups];
    let mut start = 0usize;
    for (group, size) in group_sizes.iter().enumerate() {
        for row in start..(start + size) {
            let response = y[row];
            yty[group] += response * response;
            for col in 0..p {
                let x_col = x[row + n * col];
                xty[group * p + col] += x_col * response;
                for other in 0..p {
                    xtx[group * p * p + col + p * other] += x_col * x[row + n * other];
                }
            }
        }
        start += size;
    }
    Ok((xtx, xty, yty))
}

fn factor_sufficient_kernel(
    y: &[f64],
    eta: &[f64],
    n: usize,
    items: usize,
    factors: usize,
) -> std::result::Result<FactorStatistics, &'static str> {
    if y.len() != n * items || eta.len() != n * factors {
        return Err("y or eta dimensions do not match n, items, and factors");
    }
    if eta.iter().any(|value| !value.is_finite()) {
        return Err("eta must contain only finite latent scores");
    }
    if y.iter().any(|value| value.is_infinite()) {
        return Err("y may contain missing values but not infinite values");
    }

    let design_cols = factors + 1;
    let mut ztz = vec![CompensatedSum::default(); items * design_cols * design_cols];
    let mut zty = vec![CompensatedSum::default(); items * design_cols];
    let mut yty = vec![CompensatedSum::default(); items];
    let mut n_obs = vec![0usize; items];

    for item in 0..items {
        for row in 0..n {
            let response = y[row + n * item];
            if response.is_nan() {
                continue;
            }
            n_obs[item] += 1;
            yty[item].add(response * response);
            for col in 0..design_cols {
                let z_col = if col == 0 {
                    1.0
                } else {
                    eta[row + n * (col - 1)]
                };
                zty[item * design_cols + col].add(z_col * response);
                for other in 0..design_cols {
                    let z_other = if other == 0 {
                        1.0
                    } else {
                        eta[row + n * (other - 1)]
                    };
                    let offset = item * design_cols * design_cols + col + design_cols * other;
                    ztz[offset].add(z_col * z_other);
                }
            }
        }
    }

    Ok((
        ztz.into_iter().map(|value| value.sum).collect(),
        zty.into_iter().map(|value| value.sum).collect(),
        yty.into_iter().map(|value| value.sum).collect(),
        n_obs,
    ))
}

extendr_module! {
    mod dsemr;
    fn dsem_ar_sufficient;
    fn dsem_grouped_sufficient;
    fn dsem_factor_sufficient;
    fn dsem_rust_var_n1;
    fn dsem_rust_twolevel;
    fn dsem_rust_threelevel;
    fn dsem_rust_crossclassified;
    fn dsem_rust_dynamic_sem;
    fn dsem_rust_arma11;
    fn dsem_rust_cc_dynamic_cfa;
    fn dsem_rust_ct_dsem;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_cross_products() {
        let (xtx, xty, yty) =
            sufficient_kernel(&[1.0, 1.0, 1.0, 1.0, 2.0, 3.0], &[2.0, 4.0, 6.0], 3, 2).unwrap();
        assert_eq!(xtx, vec![3.0, 6.0, 6.0, 14.0]);
        assert_eq!(xty, vec![12.0, 28.0]);
        assert_eq!(yty, 56.0);
    }

    #[test]
    fn truncnorm_and_linalg_are_stable() {
        let mut r = rng::Rng::from_seed(42);
        for _ in 0..200 {
            let v = r.rtruncnorm(0.0, 1.0, 4.5, f64::INFINITY);
            assert!(v >= 4.5 && v.is_finite());
            let v2 = r.rtruncnorm(0.0, 1.0, f64::NEG_INFINITY, -4.5);
            assert!(v2 <= -4.5 && v2.is_finite());
        }
        let a = vec![4.0, 1.0, 1.0, 3.0];
        let inv = linalg::spd_invert(&a, 2);
        let prod = linalg::mat_mul(&a, &inv, 2, 2, 2);
        assert!((prod[0] - 1.0).abs() < 1e-12);
        assert!((prod[3] - 1.0).abs() < 1e-12);
        assert!(prod[1].abs() < 1e-12);
    }

    #[test]
    fn threelevel_and_twolevel_samplers_recover_ar() {
        let mut r = rng::Rng::from_seed(123);
        let mut y = Vec::new();
        for g in 0..8 {
            let mu_g = (g as f64 - 3.5) * 0.2;
            let mut prev = 0.0;
            for _ in 0..25 {
                let cur = 0.5 * prev + 0.5 * r.rnorm();
                y.push(mu_g + cur);
                prev = cur;
            }
        }
        let out2 = twolevel::mcmc_twolevel(
            &y,
            &[],
            &[25; 8],
            &[],
            &[],
            &[],
            &[],
            &[],
            1,
            0,
            0,
            0,
            &[1],
            &[1],
            &[1],
            true,
            &[],
            false,
            &[0],
            &[f64::NEG_INFINITY],
            &[f64::INFINITY],
            &[],
            0,
            10.0,
            2.0,
            0.05,
            200,
            100,
            1,
            99,
        );
        assert_eq!(out2.draws.len(), 100 * out2.param_count);
        assert!(out2.dic.is_finite());

        let out3 = threelevel::mcmc_threelevel(
            &y,
            &[],
            &[25; 8],
            &[0, 0, 1, 1, 2, 2, 3, 3],
            4,
            &[],
            &[],
            1,
            0,
            0,
            0,
            &[1],
            &[1, 0, 0],
            &[1, 1, 1],
            &[],
            &[1],
            &[],
            &[],
            &[],
            false,
            0,
            10.0,
            120,
            60,
            1,
            77,
        );
        assert_eq!(out3.draws.len(), 60 * out3.param_count);
        assert!(out3.dic.is_finite());
    }

    #[test]
    fn tobit_censored_and_multivariate_twolevel_and_crossclassified_work() {
        let mut r = rng::Rng::from_seed(2026);
        // 1. Left-censored N=1 AR(1) at floor 0.0
        let mut y_cens = vec![0.0; 120];
        let mut prev = 0.2;
        for val in &mut y_cens {
            let y_star = 0.25 + 0.55 * prev + 0.6 * r.rnorm();
            *val = y_star.max(0.0);
            prev = y_star;
        }
        let out_c = var_n1::mcmc_var_n1(
            &y_cens,
            &[],
            120,
            1,
            0,
            &[1],
            &[-1],
            &[0.0],
            &[f64::INFINITY],
            false,
            0,
            10.0,
            0.001,
            0.001,
            200,
            100,
            1,
            55,
        );
        assert_eq!(out_c.draws.len(), 100 * out_c.param_count);
        assert!(out_c.dic.is_finite());

        // 2. Bivariate two-level VAR(1) with random residual covariance (Ex 9.32b)
        let mut y2 = vec![0.0; 8 * 20 * 2];
        for g in 0..8 {
            let mut p1 = 0.0;
            let mut p2 = 0.0;
            for t in 0..20 {
                let f_sh = 0.5 * r.rnorm();
                let e1 = f_sh + 0.5 * r.rnorm();
                let e2 = f_sh + 0.5 * r.rnorm();
                let c1 = 0.4 * p1 + 0.2 * p2 + e1;
                let c2 = 0.3 * p2 + 0.15 * p1 + e2;
                let row = g * 20 + t;
                y2[row] = c1;
                y2[160 + row] = c2;
                p1 = c1;
                p2 = c2;
            }
        }
        let out_mv = twolevel::mcmc_twolevel(
            &y2,
            &[],
            &[20; 8],
            &[],
            &[],
            &[],
            &[],
            &[],
            2,
            0,
            0,
            0,
            &[1],
            &[1, 1, 1, 1],
            &[1, 1],
            true,
            &[],
            false,
            &[0, 0],
            &[f64::NEG_INFINITY; 2],
            &[f64::INFINITY; 2],
            &[1.0, 1.0],
            2, // PSEM mode
            10.0,
            2.0,
            0.05,
            160,
            80,
            1,
            88,
        );
        assert_eq!(out_mv.q_re, 9);
        assert_eq!(out_mv.draws.len(), 80 * out_mv.param_count);
        assert!(
            out_mv.dic.is_finite(),
            "dic={} pd={} first_row={:?}",
            out_mv.dic,
            out_mv.pd,
            (0..out_mv.param_count.min(15))
                .map(|j| out_mv.draws[j * 80])
                .collect::<Vec<_>>()
        );

        // 3. Cross-classified with selective random covariate slope & selective time predictor mask
        let n_id = 6;
        let n_t = 15;
        let mut yc = vec![0.0; n_id * n_t];
        let mut xc = vec![0.0; n_id * n_t];
        let wt: Vec<f64> = (0..n_t).map(|t| (t as f64 - 7.0) / 5.0).collect();
        for g in 0..n_id {
            let mut prev_y = 0.0;
            for t in 0..n_t {
                let x_gt = r.rnorm();
                xc[g * n_t + t] = x_gt;
                let cur_y = 0.3 * prev_y + (0.5 + 0.2 * wt[t]) * x_gt + 0.5 * r.rnorm();
                yc[g * n_t + t] = cur_y;
                prev_y = cur_y;
            }
        }
        let out_cc = crossclassified::mcmc_crossclassified(
            &yc,
            &xc,
            &[],
            &wt,
            &[0, 1], // AR fixed, x slope random across subject
            &[0, 1], // AR fixed, x slope random across time
            &[0],
            &[],
            &[0, 1], // only sx on w_time (not y)
            n_id,
            n_t,
            1,
            1,
            0,
            1,
            &[1],
            0,
            10.0,
            160,
            80,
            1,
            66,
        );
        assert_eq!(out_cc.q_id, 2);
        assert_eq!(out_cc.q_time, 2);
        assert_eq!(out_cc.draws.len(), 80 * out_cc.param_count);
        assert!(out_cc.dic.is_finite());
    }

    #[test]
    fn arma11_cc_dynamic_cfa_and_ct_dsem_kernels_work() {
        let mut r = rng::Rng::from_seed(31415);

        // 1. Two-level ARMA(1,1)
        let mut y_arma = Vec::new();
        for g in 0..10 {
            let mu = (g as f64 - 4.5) * 0.2;
            let s = 0.35;
            let mut e_prev = r.rnorm();
            let mut y_prev = e_prev;
            y_arma.push(mu + y_prev);
            for _ in 1..40 {
                let e_cur = r.rnorm();
                let y_cur = s * y_prev + e_cur - 0.25 * e_prev + 0.1 * r.rnorm();
                y_arma.push(mu + y_cur);
                e_prev = e_cur;
                y_prev = y_cur;
            }
        }
        let out_a = arma::mcmc_arma11(&y_arma, &[40; 10], true, 0.01, 1, 0.98, false, 160, 80, 1, 11);
        assert_eq!(out_a.param_count, 7);
        assert_eq!(out_a.draws.len(), 80 * 7);
        assert!(out_a.dic.is_finite());

        // 2. Cross-classified Dynamic CFA with random factor loadings
        let n_id = 8;
        let n_t = 12;
        let p = 3;
        let mut y_cfa = vec![0.0; n_id * n_t * p];
        for g in 0..n_id {
            let mut eta = 0.0;
            for t in 0..n_t {
                eta = 0.4 * eta + r.rnorm();
                for j in 0..p {
                    let row = g * n_t + t;
                    y_cfa[j * (n_id * n_t) + row] = (0.8 + 0.1 * j as f64) * eta + 0.5 * r.rnorm();
                }
            }
        }
        let out_cfa = cc_dynamic_cfa::mcmc_cc_dynamic_cfa(
            &y_cfa, n_id, n_t, p, true, true, true, true, false, true, 0, 0.98, false, 120, 60, 1, 22,
        );
        assert_eq!(out_cfa.draws.len(), 60 * out_cfa.param_count);
        assert!(out_cfa.dic.is_finite());

        // 3. Continuous-Time Bivariate OU (N=1) with irregular time gaps
        let n_ct = 120;
        let mut t_ct = vec![0.0; n_ct];
        let mut y_ct = vec![0.0; n_ct * 2];
        let a_true = vec![-0.7, 0.15, 0.2, -0.5]; // col-major
        let q_true = vec![0.8, 0.2, 0.2, 0.6];
        let sig_inf = ct_dsem::solve_lyapunov_ct(&a_true, &q_true, 2).unwrap();
        assert!(sig_inf[0] > 0.0 && sig_inf[3] > 0.0);
        let mut prev = [0.0, 0.0];
        for i in 1..n_ct {
            let dt = 0.5 + 1.5 * r.runif();
            t_ct[i] = t_ct[i - 1] + dt;
            let phi = ct_dsem::mat_exp_scaled(&a_true, dt, 2);
            let c0 = phi[0] * prev[0] + phi[2] * prev[1] + 0.5 * r.rnorm();
            let c1 = phi[1] * prev[0] + phi[3] * prev[1] + 0.5 * r.rnorm();
            y_ct[i] = c0;
            y_ct[n_ct + i] = c1;
            prev = [c0, c1];
        }
        let out_ct = ct_dsem::mcmc_ct_dsem(&y_ct, &[], &t_ct, &[n_ct], 2, 0, false, false, 0, 10.0, 140, 70, 1, 33);
        assert_eq!(out_ct.draws.len(), 70 * out_ct.param_count);
        assert!(out_ct.dic.is_finite());
    }
}
