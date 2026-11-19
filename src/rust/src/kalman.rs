use crate::linalg::{
    chol_lower_jitter, chol_solve_vec, eye, idx, is_stable_transition, mat_mul, mat_mul_t, mat_vec,
    rmvnorm_cov, symmetrize,
};
use crate::rng::Rng;
use std::f64::consts::PI;

/// Solve discrete-time Lyapunov equation P = F P F' + Q via Smith doubling.
pub fn stationary_lyapunov_cov(f: &[f64], q: &[f64], m: usize, bound: f64) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        let phi = f[0];
        let psi = q[0].max(1e-8);
        if phi.is_finite() && phi.abs() < bound {
            return vec![(psi / (1.0 - phi * phi)).clamp(1e-8, 1e4)];
        }
        return vec![10.0];
    }
    if !is_stable_transition(f, m, bound) {
        return eye(m, 10.0);
    }
    let mut p = q.to_vec();
    let mut fk = f.to_vec();
    for _ in 0..12 {
        let fp = mat_mul(&fk, &p, m, m, m);
        let fpf = mat_mul_t(&fp, &fk, m, m, m);
        for i in 0..(m * m) {
            p[i] += fpf[i];
        }
        symmetrize(&mut p, m);
        if p.iter().any(|v| !v.is_finite() || v.abs() > 1e6) {
            return eye(m, 10.0);
        }
        fk = mat_mul(&fk, &fk, m, m, m);
    }
    p
}

/// Multivariate DAFS FFBS smoother:
///   eta_t = alpha + Phi * eta_{t-1} + zeta_t,  zeta_t ~ N(0, Psi)
///   y_t   = nu_t  + Lambda_0 * eta_t + eps_t,  eps_t  ~ N(0, diag(theta))
/// Returns (eta_draws [n_time x m col-major], marginal_loglik).
#[allow(clippy::too_many_arguments)]
pub fn ffbs_dafs(
    rng: &mut Rng,
    y: &[f64],
    n_time: usize,
    n_items: usize,
    m: usize,
    nu: &[f64], // length n_items (or n_time * n_items if time-varying; here length n_items)
    lambda0: &[f64], // n_items x m (col-major)
    theta: &[f64], // length n_items
    alpha: &[f64], // length m
    phi: &[f64], // m x m (col-major)
    psi: &[f64], // m x m (col-major)
    ar_bound: f64,
) -> (Vec<f64>, f64) {
    let mut pred_mean = vec![0.0; m * n_time];
    let mut filt_mean = vec![0.0; m * n_time];
    let mut pred_cov = vec![0.0; m * m * n_time];
    let mut filt_cov = vec![0.0; m * m * n_time];

    let p0 = stationary_lyapunov_cov(phi, psi, m, ar_bound);
    let m0 = if is_stable_transition(phi, m, ar_bound) {
        // (I - Phi)^{-1} alpha
        let mut i_minus_phi = eye(m, 1.0);
        for i in 0..(m * m) {
            i_minus_phi[i] -= phi[i];
        }
        // General small linear solve via normal equations if nonsymmetric
        let ata = crate::linalg::mat_t_mul(&i_minus_phi, &i_minus_phi, m, m, m);
        let atb = crate::linalg::mat_t_mul(&i_minus_phi, alpha, m, m, 1);
        let l = chol_lower_jitter(&ata, m);
        chol_solve_vec(&l, &atb, m)
    } else {
        vec![0.0; m]
    };

    let mut loglik = 0.0;

    for t in 0..n_time {
        let mut cur_m = if t == 0 {
            m0.clone()
        } else {
            let prev_m = &filt_mean[(t - 1) * m..t * m];
            let mut pm = mat_vec(phi, prev_m, m, m);
            for k in 0..m {
                pm[k] += alpha[k];
            }
            pm
        };
        let mut cur_p = if t == 0 {
            p0.clone()
        } else {
            let prev_p = &filt_cov[(t - 1) * m * m..t * m * m];
            let fp = mat_mul(phi, prev_p, m, m, m);
            let mut fpf = mat_mul_t(&fp, phi, m, m, m);
            for i in 0..(m * m) {
                fpf[i] += psi[i];
            }
            symmetrize(&mut fpf, m);
            fpf
        };

        pred_mean[t * m..(t + 1) * m].copy_from_slice(&cur_m);
        pred_cov[t * m * m..(t + 1) * m * m].copy_from_slice(&cur_p);

        // Sequential scalar measurement updates across observed indicators j
        for j in 0..n_items {
            let y_tj = y[t + n_time * j];
            if y_tj.is_nan() {
                continue;
            }
            let r_j = theta[j].max(1e-8);
            // h_j is row j of Lambda_0 (length m)
            let mut h_j = vec![0.0; m];
            let mut yhat = nu[j];
            for k in 0..m {
                let l_jk = lambda0[idx(j, k, n_items)];
                h_j[k] = l_jk;
                yhat += l_jk * cur_m[k];
            }
            let ph = mat_vec(&cur_p, &h_j, m, m);
            let mut f_var = r_j;
            for k in 0..m {
                f_var += h_j[k] * ph[k];
            }
            f_var = f_var.max(1e-10);
            let v_err = y_tj - yhat;
            loglik += -0.5 * ((2.0 * PI * f_var).ln() + (v_err * v_err) / f_var);

            let inv_f = 1.0 / f_var;
            for k in 0..m {
                cur_m[k] += ph[k] * inv_f * v_err;
            }
            // Joseph-stabilized rank-1 covariance update: P <- P - (P h)(P h)' / f_var
            for c in 0..m {
                for r in 0..m {
                    cur_p[idx(r, c, m)] -= ph[r] * ph[c] * inv_f;
                }
            }
            symmetrize(&mut cur_p, m);
            for k in 0..m {
                if cur_p[idx(k, k, m)] < 1e-10 {
                    cur_p[idx(k, k, m)] = 1e-10;
                }
            }
        }

        filt_mean[t * m..(t + 1) * m].copy_from_slice(&cur_m);
        filt_cov[t * m * m..(t + 1) * m * m].copy_from_slice(&cur_p);
    }

    // Backward simulation smoother
    let mut states = vec![0.0; n_time * m]; // col-major n_time x m
    let last_m = &filt_mean[(n_time - 1) * m..n_time * m];
    let last_p = &filt_cov[(n_time - 1) * m * m..n_time * m * m];
    let mut next_state = rmvnorm_cov(rng, last_m, last_p, m);
    for k in 0..m {
        states[(n_time - 1) + n_time * k] = next_state[k];
    }

    for t in (0..(n_time - 1)).rev() {
        let fm = &filt_mean[t * m..(t + 1) * m];
        let fp = &filt_cov[t * m * m..(t + 1) * m * m];
        let pm_next = &pred_mean[(t + 1) * m..(t + 2) * m];
        let pp_next = &pred_cov[(t + 1) * m * m..(t + 2) * m * m];

        // J_t = P_{t|t} * Phi' * P_{t+1|t}^{-1}
        let fp_phit = mat_mul_t(fp, phi, m, m, m);
        let l_pp = chol_lower_jitter(pp_next, m);
        // Solve P_{t+1|t} * X = (Phi * P_{t|t}) = fp_phit' -> J_t = X'
        let mut j_mat = vec![0.0; m * m];
        let mut rhs = vec![0.0; m];
        for r in 0..m {
            for c in 0..m {
                rhs[c] = fp_phit[idx(r, c, m)];
            }
            let sol = chol_solve_vec(&l_pp, &rhs, m);
            for c in 0..m {
                j_mat[idx(r, c, m)] = sol[c];
            }
        }

        let diff: Vec<f64> = next_state.iter().zip(pm_next).map(|(s, p)| s - p).collect();
        let j_diff = mat_vec(&j_mat, &diff, m, m);
        let sm_mean: Vec<f64> = fm.iter().zip(j_diff).map(|(a, b)| a + b).collect();

        // P_smooth = P_{t|t} - J_t * P_{t+1|t} * J_t'
        let j_pp = mat_mul(&j_mat, pp_next, m, m, m);
        let j_pp_jt = mat_mul_t(&j_pp, &j_mat, m, m, m);
        let mut sm_cov = vec![0.0; m * m];
        for i in 0..(m * m) {
            sm_cov[i] = fp[i] - j_pp_jt[i];
        }
        symmetrize(&mut sm_cov, m);
        for k in 0..m {
            if sm_cov[idx(k, k, m)] < 1e-10 {
                sm_cov[idx(k, k, m)] = 1e-10;
            }
        }

        let cur_state = rmvnorm_cov(rng, &sm_mean, &sm_cov, m);
        for k in 0..m {
            states[t + n_time * k] = cur_state[k];
        }
        next_state = cur_state;
    }

    (states, loglik)
}

/// Companion-form FFBS for WNFS / lagged-loading state-space models:
///   eta_t = alpha + Phi * eta_{t-1} + zeta_t,  zeta_t ~ N(0, Psi)
///   y_t   = nu    + Lambda_0 * eta_t + Lambda_1 * eta_{t-1} + eps_t
/// Returns (eta_1..T [n_time x m col-major], eta_0 [length m], marginal_loglik).
#[allow(clippy::too_many_arguments)]
pub fn ffbs_lagged_measurement(
    rng: &mut Rng,
    y: &[f64],
    n_time: usize,
    n_items: usize,
    m: usize,
    nu: &[f64],
    lambda0: &[f64], // n_items x m
    lambda1: &[f64], // n_items x m
    theta: &[f64],
    alpha: &[f64],
    phi: &[f64],
    psi: &[f64],
    ar_bound: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let d = 2 * m;
    let mut filt_mean = vec![0.0; d * n_time];
    let mut filt_cov = vec![0.0; d * d * n_time];

    // Stationary joint distribution of s_1 = (eta_1, eta_0):
    // Cov(eta_0) = P0, Cov(eta_1, eta_0) = Phi P0, Cov(eta_1) = Phi P0 Phi' + Psi = P0
    let p0 = stationary_lyapunov_cov(phi, psi, m, ar_bound);
    let phi_p0 = mat_mul(phi, &p0, m, m, m);
    let mut cur_p = vec![0.0; d * d];
    for c in 0..m {
        for r in 0..m {
            cur_p[idx(r, c, d)] = p0[idx(r, c, m)];
            cur_p[idx(r + m, c + m, d)] = p0[idx(r, c, m)];
            cur_p[idx(r, c + m, d)] = phi_p0[idx(r, c, m)];
            cur_p[idx(c + m, r, d)] = phi_p0[idx(r, c, m)];
        }
    }
    let mut cur_m = vec![0.0; d];
    let mut loglik = 0.0;

    // Companion transition F = [[Phi, 0], [I, 0]], Q_comp = [[Psi, 0], [0, 0]]
    let mut f_comp = vec![0.0; d * d];
    let mut q_comp = vec![0.0; d * d];
    for c in 0..m {
        for r in 0..m {
            f_comp[idx(r, c, d)] = phi[idx(r, c, m)];
            q_comp[idx(r, c, d)] = psi[idx(r, c, m)];
        }
        f_comp[idx(c + m, c, d)] = 1.0;
    }

    for t in 0..n_time {
        if t > 0 {
            let prev_m = &filt_mean[(t - 1) * d..t * d];
            let prev_p = &filt_cov[(t - 1) * d * d..t * d * d];
            cur_m = mat_vec(&f_comp, prev_m, d, d);
            for k in 0..m {
                cur_m[k] += alpha[k];
            }
            let fp = mat_mul(&f_comp, prev_p, d, d, d);
            cur_p = mat_mul_t(&fp, &f_comp, d, d, d);
            for i in 0..(d * d) {
                cur_p[i] += q_comp[i];
            }
            symmetrize(&mut cur_p, d);
        }

        for j in 0..n_items {
            let y_tj = y[t + n_time * j];
            if y_tj.is_nan() {
                continue;
            }
            let r_j = theta[j].max(1e-8);
            let mut h_j = vec![0.0; d];
            let mut yhat = nu[j];
            for k in 0..m {
                h_j[k] = lambda0[idx(j, k, n_items)];
                h_j[k + m] = lambda1[idx(j, k, n_items)];
            }
            for k in 0..d {
                yhat += h_j[k] * cur_m[k];
            }
            let ph = mat_vec(&cur_p, &h_j, d, d);
            let mut f_var = r_j;
            for k in 0..d {
                f_var += h_j[k] * ph[k];
            }
            f_var = f_var.max(1e-10);
            let v_err = y_tj - yhat;
            loglik += -0.5 * ((2.0 * PI * f_var).ln() + (v_err * v_err) / f_var);
            let inv_f = 1.0 / f_var;
            for k in 0..d {
                cur_m[k] += ph[k] * inv_f * v_err;
            }
            for c in 0..d {
                for r in 0..d {
                    cur_p[idx(r, c, d)] -= ph[r] * ph[c] * inv_f;
                }
            }
            symmetrize(&mut cur_p, d);
        }

        filt_mean[t * d..(t + 1) * d].copy_from_slice(&cur_m);
        filt_cov[t * d * d..(t + 1) * d * d].copy_from_slice(&cur_p);
    }

    // Exact non-degenerate block-conditional backward sampling:
    // Draw s_T = (eta_T, eta_{T-1}) ~ N(m_{T|T}, P_{T|T}), then for t = T-1..1
    // draw eta_{t-1} | eta_t, y_{1:t} from the Gaussian partition of s_t = (eta_t, eta_{t-1}).
    let mut states = vec![0.0; n_time * m];
    let last_m = &filt_mean[(n_time - 1) * d..n_time * d];
    let last_p = &filt_cov[(n_time - 1) * d * d..n_time * d * d];
    let s_last = rmvnorm_cov(rng, last_m, last_p, d);
    let eta_curr = s_last[0..m].to_vec();
    let eta_prev = s_last[m..d].to_vec();
    for k in 0..m {
        states[(n_time - 1) + n_time * k] = eta_curr[k];
    }
    if n_time >= 2 {
        for k in 0..m {
            states[(n_time - 2) + n_time * k] = eta_prev[k];
        }
    }

    // Now sample eta_{t-1} for t = n_time - 2 down to 0 (t=0 gives eta_0!)
    let mut eta0 = eta_prev.clone();
    let start_t = n_time.saturating_sub(2);
    for t in (0..=start_t).rev() {
        let eta_known: Vec<f64> = (0..m).map(|k| states[t + n_time * k]).collect();
        let fm = &filt_mean[t * d..(t + 1) * d];
        let fp = &filt_cov[t * d * d..(t + 1) * d * d];
        let m1 = &fm[0..m];
        let m2 = &fm[m..d];
        let mut p11 = vec![0.0; m * m];
        let mut p21 = vec![0.0; m * m];
        let mut p22 = vec![0.0; m * m];
        for c in 0..m {
            for r in 0..m {
                p11[idx(r, c, m)] = fp[idx(r, c, d)];
                p21[idx(r, c, m)] = fp[idx(r + m, c, d)];
                p22[idx(r, c, m)] = fp[idx(r + m, c + m, d)];
            }
        }
        let l11 = chol_lower_jitter(&p11, m);
        let diff1: Vec<f64> = eta_known.iter().zip(m1).map(|(a, b)| a - b).collect();
        let p11_inv_diff = chol_solve_vec(&l11, &diff1, m);
        let p21_delta = mat_vec(&p21, &p11_inv_diff, m, m);
        let cond_mean: Vec<f64> = m2.iter().zip(p21_delta).map(|(a, b)| a + b).collect();

        // P22 - P21 P11^{-1} P12
        let mut p11_inv_p12 = vec![0.0; m * m];
        let mut col_vec = vec![0.0; m];
        for c in 0..m {
            for r in 0..m {
                col_vec[r] = p21[idx(c, r, m)]; // P12 = P21'
            }
            let sol = chol_solve_vec(&l11, &col_vec, m);
            for r in 0..m {
                p11_inv_p12[idx(r, c, m)] = sol[r];
            }
        }
        let sub = mat_mul(&p21, &p11_inv_p12, m, m, m);
        let mut cond_cov = vec![0.0; m * m];
        for i in 0..(m * m) {
            cond_cov[i] = p22[i] - sub[i];
        }
        symmetrize(&mut cond_cov, m);
        let sampled_prev = rmvnorm_cov(rng, &cond_mean, &cond_cov, m);
        if t == 0 {
            eta0 = sampled_prev;
        } else if t >= 1 {
            // For t >= 1, sampled_prev is eta_{t-1}
            for k in 0..m {
                states[(t - 1) + n_time * k] = sampled_prev[k];
            }
        }
    }

    (states, eta0, loglik)
}
