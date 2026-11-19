use crate::rng::Rng;
use std::f64::consts::PI;

#[inline]
pub fn idx(row: usize, col: usize, nrows: usize) -> usize {
    row + nrows * col
}

pub fn eye(p: usize, scale: f64) -> Vec<f64> {
    let mut out = vec![0.0; p * p];
    for i in 0..p {
        out[idx(i, i, p)] = scale;
    }
    out
}

pub fn symmetrize(a: &mut [f64], p: usize) {
    for c in 0..p {
        for r in (c + 1)..p {
            let avg = 0.5 * (a[idx(r, c, p)] + a[idx(c, r, p)]);
            a[idx(r, c, p)] = avg;
            a[idx(c, r, p)] = avg;
        }
    }
}

pub fn chol_lower_strict(a: &[f64], p: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; p * p];
    for j in 0..p {
        let mut sum_diag = a[idx(j, j, p)];
        for k in 0..j {
            let v = l[idx(j, k, p)];
            sum_diag -= v * v;
        }
        if !sum_diag.is_finite() || sum_diag <= 1e-14 {
            return None;
        }
        let diag = sum_diag.sqrt();
        l[idx(j, j, p)] = diag;
        for i in (j + 1)..p {
            let mut sum = a[idx(i, j, p)];
            for k in 0..j {
                sum -= l[idx(i, k, p)] * l[idx(j, k, p)];
            }
            l[idx(i, j, p)] = sum / diag;
        }
    }
    Some(l)
}

pub fn chol_lower_jitter(a: &[f64], p: usize) -> Vec<f64> {
    if p == 0 {
        return Vec::new();
    }
    if let Some(l) = chol_lower_strict(a, p) {
        return l;
    }
    let jitters = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0];
    let mut work = a.to_vec();
    symmetrize(&mut work, p);
    for jit in jitters {
        let mut cand = work.clone();
        for i in 0..p {
            if !cand[idx(i, i, p)].is_finite() || cand[idx(i, i, p)] <= 0.0 {
                cand[idx(i, i, p)] = jit;
            } else {
                cand[idx(i, i, p)] += jit;
            }
        }
        if let Some(l) = chol_lower_strict(&cand, p) {
            return l;
        }
    }
    eye(p, 1.0)
}

pub fn forward_solve(l: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    let mut y = vec![0.0; p];
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= l[idx(i, k, p)] * y[k];
        }
        y[i] = s / l[idx(i, i, p)].max(1e-12);
    }
    y
}

pub fn backward_solve_lt(l: &[f64], y: &[f64], p: usize) -> Vec<f64> {
    let mut x = vec![0.0; p];
    for i in (0..p).rev() {
        let mut s = y[i];
        for k in (i + 1)..p {
            s -= l[idx(k, i, p)] * x[k];
        }
        x[i] = s / l[idx(i, i, p)].max(1e-12);
    }
    x
}

pub fn chol_solve_vec(l: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    let y = forward_solve(l, b, p);
    backward_solve_lt(l, &y, p)
}

pub fn spd_invert(a: &[f64], p: usize) -> Vec<f64> {
    let l = chol_lower_jitter(a, p);
    let mut inv = vec![0.0; p * p];
    let mut e = vec![0.0; p];
    for c in 0..p {
        e.fill(0.0);
        e[c] = 1.0;
        let col = chol_solve_vec(&l, &e, p);
        for r in 0..p {
            inv[idx(r, c, p)] = col[r];
        }
    }
    symmetrize(&mut inv, p);
    inv
}

pub fn spd_logdet(l: &[f64], p: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..p {
        s += l[idx(i, i, p)].max(1e-15).ln();
    }
    2.0 * s
}

pub fn mat_mul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for col in 0..n {
        for inner in 0..k {
            let b_val = b[idx(inner, col, k)];
            if b_val == 0.0 {
                continue;
            }
            for row in 0..m {
                out[idx(row, col, m)] += a[idx(row, inner, m)] * b_val;
            }
        }
    }
    out
}

pub fn mat_t_mul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    // A' * B where A is k x m (so A' is m x k) and B is k x n -> m x n
    let mut out = vec![0.0; m * n];
    for col in 0..n {
        for row in 0..m {
            let mut s = 0.0;
            for inner in 0..k {
                s += a[idx(inner, row, k)] * b[idx(inner, col, k)];
            }
            out[idx(row, col, m)] = s;
        }
    }
    out
}

pub fn mat_mul_t(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    // A * B' where A is m x k and B is n x k -> m x n
    let mut out = vec![0.0; m * n];
    for col in 0..n {
        for inner in 0..k {
            let b_val = b[idx(col, inner, n)];
            for row in 0..m {
                out[idx(row, col, m)] += a[idx(row, inner, m)] * b_val;
            }
        }
    }
    out
}

pub fn mat_vec(a: &[f64], x: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m];
    for col in 0..n {
        let xc = x[col];
        for row in 0..m {
            out[row] += a[idx(row, col, m)] * xc;
        }
    }
    out
}

pub fn quad_form(x: &[f64], a: &[f64], p: usize) -> f64 {
    let mut s = 0.0;
    for c in 0..p {
        let mut col_sum = 0.0;
        for r in 0..p {
            col_sum += a[idx(r, c, p)] * x[r];
        }
        s += col_sum * x[c];
    }
    s
}

/// Sample theta ~ N(P^{-1} info, P^{-1}) directly using Cholesky of precision P.
pub fn rmvnorm_precision(rng: &mut Rng, precision: &[f64], info: &[f64], p: usize) -> Vec<f64> {
    if p == 0 {
        return Vec::new();
    }
    let l_p = chol_lower_jitter(precision, p);
    let mean = chol_solve_vec(&l_p, info, p);
    let mut z = vec![0.0; p];
    for v in &mut z {
        *v = rng.rnorm();
    }
    let delta = backward_solve_lt(&l_p, &z, p);
    mean.into_iter().zip(delta).map(|(m, d)| m + d).collect()
}

/// Sample theta ~ N(mean, cov) using Cholesky of covariance.
pub fn rmvnorm_cov(rng: &mut Rng, mean: &[f64], cov: &[f64], p: usize) -> Vec<f64> {
    if p == 0 {
        return Vec::new();
    }
    let l = chol_lower_jitter(cov, p);
    let mut z = vec![0.0; p];
    for v in &mut z {
        *v = rng.rnorm();
    }
    let lz = mat_vec(&l, &z, p, p);
    mean.iter().zip(lz).map(|(m, d)| m + d).collect()
}

/// Sample W ~ Wishart(df, S) via Bartlett decomposition.
pub fn rwishart(rng: &mut Rng, df: f64, scale_mat: &[f64], p: usize) -> Vec<f64> {
    if p == 0 {
        return Vec::new();
    }
    let eff_df = df.max(p as f64 + 0.5);
    let l_s = chol_lower_jitter(scale_mat, p);
    let mut a = vec![0.0; p * p];
    for j in 0..p {
        let chi_df = (eff_df - j as f64).max(0.5);
        a[idx(j, j, p)] = (2.0 * rng.rgamma(0.5 * chi_df, 1.0)).sqrt();
        for i in (j + 1)..p {
            a[idx(i, j, p)] = rng.rnorm();
        }
    }
    let la = mat_mul(&l_s, &a, p, p, p);
    let mut w = mat_mul_t(&la, &la, p, p, p);
    symmetrize(&mut w, p);
    w
}

/// Sample Omega ~ Inverse-Wishart(df, Psi) where density proportional to
/// |Omega|^{-(df + p + 1)/2} exp(-0.5 tr(Psi Omega^{-1})).
pub fn rinv_wishart(rng: &mut Rng, df: f64, scale_psi: &[f64], p: usize) -> Vec<f64> {
    if p == 0 {
        return Vec::new();
    }
    if p == 1 {
        let a = 0.5 * df.max(1.0);
        let b = 0.5 * scale_psi[0].max(1e-8);
        return vec![rng.rinv_gamma(a, b)];
    }
    let psi_inv = spd_invert(scale_psi, p);
    let w = rwishart(rng, df, &psi_inv, p);
    spd_invert(&w, p)
}

#[allow(dead_code)]
pub fn mvnorm_logpdf_prec(
    x: &[f64],
    mean: &[f64],
    precision: &[f64],
    logdet_prec: f64,
    p: usize,
) -> f64 {
    let diff: Vec<f64> = x.iter().zip(mean).map(|(xi, mi)| xi - mi).collect();
    let qf = quad_form(&diff, precision, p);
    0.5 * (logdet_prec - (p as f64) * (2.0 * PI).ln() - qf)
}

/// Spectral radius check via power/QR bound or Frobenius/companion check for stationarity.
pub fn max_row_abs_sum(a: &[f64], p: usize) -> f64 {
    let mut m = 0.0_f64;
    for r in 0..p {
        let mut s = 0.0;
        for c in 0..p {
            s += a[idx(r, c, p)].abs();
        }
        m = m.max(s);
    }
    m
}

/// Check discrete-time stability rho(F) < bound by repeated squaring of F:
/// rho(F) < 1 iff ||F^{2^k}||_inf < 1 for small k.
pub fn is_stable_transition(f: &[f64], d: usize, bound: f64) -> bool {
    if d == 0 {
        return true;
    }
    if d == 1 {
        return f[0].is_finite() && f[0].abs() < bound;
    }
    if f.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let scale = 1.0 / bound.max(1e-6);
    let mut power: Vec<f64> = f.iter().map(|v| v * scale).collect();
    for _ in 0..7 {
        let norm = max_row_abs_sum(&power, d);
        if norm < 0.999 {
            return true;
        }
        if norm > 1e8 || !norm.is_finite() {
            return false;
        }
        power = mat_mul(&power, &power, d, d, d);
    }
    max_row_abs_sum(&power, d) < 1.0
}
