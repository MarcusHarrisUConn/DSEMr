use std::f64::consts::{PI, SQRT_2};

#[derive(Clone, Debug)]
pub struct Rng {
    state: [u64; 4],
    spare_normal: Option<f64>,
}

impl Rng {
    pub fn from_seed(seed: u64) -> Self {
        let mut sm = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next_sm = || {
            sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self {
            state: [next_sm(), next_sm(), next_sm(), next_sm()],
            spare_normal: None,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.state[0].wrapping_add(self.state[3]))
            .rotate_left(23)
            .wrapping_add(self.state[0]);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    #[inline]
    pub fn runif(&mut self) -> f64 {
        // Strictly in (0, 1)
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / 9007199254740992.0)
    }

    pub fn rnorm(&mut self) -> f64 {
        if let Some(val) = self.spare_normal.take() {
            return val;
        }
        let u1 = self.runif();
        let u2 = self.runif();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        self.spare_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    pub fn rgamma(&mut self, shape: f64, rate: f64) -> f64 {
        let alpha = shape.max(1e-8);
        let beta = rate.max(1e-12);
        if alpha < 1.0 {
            let u = self.runif();
            return self.rgamma(alpha + 1.0, beta) * u.powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.rnorm();
            let v_root = 1.0 + c * x;
            if v_root <= 0.0 {
                continue;
            }
            let v = v_root * v_root * v_root;
            let u = self.runif();
            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return (d * v / beta).max(1e-12);
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return (d * v / beta).max(1e-12);
            }
        }
    }

    pub fn rinv_gamma(&mut self, shape: f64, scale: f64) -> f64 {
        let a = shape.max(0.5);
        let b = scale.max(1e-10);
        let g = self.rgamma(a, 1.0);
        (b / g).clamp(1e-10, 1e8)
    }

    /// Exact Michael-Schucany-Haas inverse-Gaussian sampler IG(mu, lambda)
    /// for Bayesian Lasso / PSEM scale-mixture updates.
    pub fn rinv_gaussian(&mut self, mu: f64, lambda: f64) -> f64 {
        let m = mu.clamp(1e-8, 1e8);
        let lam = lambda.clamp(1e-8, 1e8);
        let z = self.rnorm();
        let y = z * z;
        let my = m * y;
        let disc = (4.0 * m * lam * y + my * my).max(0.0).sqrt();
        let x = m + (m * my - m * disc) / (2.0 * lam);
        let u = self.runif();
        if u <= m / (m + x) {
            x.clamp(1e-10, 1e8)
        } else {
            (m * m / x.max(1e-12)).clamp(1e-10, 1e8)
        }
    }

    pub fn rtruncnorm(&mut self, mean: f64, sd: f64, lower: f64, upper: f64) -> f64 {
        let s = sd.max(1e-8);
        let a = if lower.is_finite() {
            (lower - mean) / s
        } else {
            f64::NEG_INFINITY
        };
        let b = if upper.is_finite() {
            (upper - mean) / s
        } else {
            f64::INFINITY
        };
        if a >= b {
            return mean + s * a;
        }
        let z = self.rtruncnorm_std(a, b);
        mean + s * z
    }

    fn rtruncnorm_std(&mut self, a: f64, b: f64) -> f64 {
        if !a.is_finite() && !b.is_finite() {
            return self.rnorm();
        }
        // Extreme positive tail: Robert (1995) translated exponential rejection
        if a >= 3.2 {
            let alpha = 0.5 * (a + (a * a + 4.0).sqrt());
            for _ in 0..200 {
                let z = a - self.runif().ln() / alpha;
                if z <= b {
                    let rho = (-0.5 * (z - alpha) * (z - alpha)).exp();
                    if self.runif() <= rho {
                        return z;
                    }
                }
            }
            return a + 1.0 / a.max(1.0);
        }
        // Extreme negative tail: symmetry
        if b <= -3.2 {
            return -self.rtruncnorm_std(-b, -a);
        }
        let pa = std_norm_cdf(a);
        let pb = std_norm_cdf(b);
        let span = pb - pa;
        if span > 1e-12 {
            let u = pa + self.runif() * span;
            return std_norm_inv_cdf(u.clamp(1e-15, 1.0 - 1e-15)).clamp(a, b);
        }
        0.5 * (a.max(-10.0) + b.min(10.0))
    }
}

#[inline]
pub fn erf_approx(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 (max error 1.5e-7) + Halley refinement
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    if ax > 8.0 {
        return sign;
    }
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-ax * ax).exp())
}

#[inline]
pub fn std_norm_cdf(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_negative() { 0.0 } else { 1.0 };
    }
    0.5 * (1.0 + erf_approx(x / SQRT_2))
}

#[inline]
pub fn log_std_norm_cdf(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_negative() { -1e12 } else { 0.0 };
    }
    if x > -5.5 {
        std_norm_cdf(x).max(1e-300).ln()
    } else {
        // Asymptotic Mills-ratio expansion for extreme left tail: Phi(x) ~ phi(x) / (-x)
        let ax = -x;
        let inv2 = 1.0 / (ax * ax);
        let corr = (1.0 - inv2 + 3.0 * inv2 * inv2).max(0.5);
        -0.5 * x * x - 0.5 * (2.0 * PI).ln() - ax.ln() + corr.ln()
    }
}

pub fn std_norm_inv_cdf(p: f64) -> f64 {
    // Acklam's rational approximation to standard normal quantile
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let u = p.clamp(1e-15, 1.0 - 1e-15);
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if u < p_low {
        let q = (-2.0 * u.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if u <= p_high {
        let q = u - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - u).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}
