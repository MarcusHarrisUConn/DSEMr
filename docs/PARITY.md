# Validation & Benchmark Matrix (DSEMr 0.1.0)

All core Dynamic Structural Equation Modeling (DSEM) model families have been
implemented independently in Rust and R and verified via controlled simulation
recovery and published methodological benchmark examples from the DSEM
literature (Asparouhov, Hamaker, & Muthén, 2018; Asparouhov & Muthén, 2020,
2024, 2025, 2026; Hamaker et al., 2018; McNeish & Hamaker, 2020; Oud & Jansen,
2000).

| Benchmark / Model Family | Statistical Specification | Compiler | Estimator | Evidence | Status |
|---|---|---|---|---|---|
| Internal AR(1) | N=1 Gaussian AR(1) | Yes | R + Rust kernel | Exact R/Rust numerical agreement (`1e-12`) and quadrature audit | validated |
| Ch. 6 (6.23–6.25) | N=1 continuous VAR(p), sparse lags (`y&1@0 y&2`), lagged covariates (`x&1`) | Yes | Rust + R reference | Parameter recovery, DIC/pD, PSEM Bayesian Lasso shrinkage | validated |
| [Benchmark 6.26](../inst/parity/mplus-6.26.yml) | N=1 continuous dynamic CFA (DAFS) | Yes | R + Rust collapsed Kalman FFBS | Published benchmark agreement (`ar1=0.307 vs 0.310`, `f_var=0.946`, `DIC=2438.7 vs 2435.9`, `ESS > 144`, `max Rhat = 1.019`) | validated |
| [Benchmark 6.26b](../inst/parity/mplus-6.26b.yml) | N=1 continuous dynamic CFA (WNFS) | Yes | Rust companion FFBS | Published benchmark agreement (`loading_y2..4=1.129, 1.206, 1.116`, `DIC=2446.8 vs 2444.5`, `max Rhat = 1.028`) | validated |
| Benchmark 6.27 | N=1 dynamic IRT with binary indicators | Yes | Rust probit FFBS | Published benchmark agreement (`ar1=0.241 vs 0.249`, all 4 thresholds & loadings match) | validated |
| Benchmark 6.28 | N=1 bivariate cross-lagged 2-factor SEM | Yes | Rust multivariate FFBS | Published benchmark agreement across all 29 parameters (`ar1=0.448/0.415`, `f2_with_f1=0.515`, `DIC=2408.9 vs 2403.2`) | validated |
| [Benchmark 9.30](../inst/parity/mplus-9.30.yml) | Two-level location-scale AR(1), `TINTERVAL`, `logv`, `w` predictors, distal `z` | Yes | Rust Gibbs/MH + PX scale move | Published benchmark agreement across all 17 parameters (`y_on_w=0.290 vs 0.290`, `var_y=0.107 vs 0.107`, `var_logv=0.005 vs 0.005`, `var_z=0.050 vs 0.050`) | validated |
| Benchmarks 9.31–9.32b | Two-level continuous VAR with correlated $\Sigma_W$, random residual variances, and random residual covariance (`logvf`) | Yes | Rust Gibbs/MH + shared shock factor | Simulation recovery of cross-lagged slopes, correlated innovations, random `logvf`, and cluster-specific `STDYX` | validated |
| Benchmark 9.33 | Two-level single-indicator measurement error | Yes | Rust 3-way collapsed Kalman FFBS | Published benchmark agreement (`mean_s=0.300 vs 0.302`, `var_s=0.025 vs 0.025`, `residual_var_y=0.533 vs 0.549`, `f_var=0.976 vs 0.960`, `max Rhat=1.030`) | validated |
| Benchmark 9.33b | Two-level ARMA(1,1) | Yes | Rust exact collapsed Toeplitz Cholesky + ridge move | Published benchmark agreement (`y_on_lag1_e=-0.121 vs -0.100`, `var_e=1.539 vs 1.539`, `mean_s=0.319 vs 0.301`, `max Rhat=1.005`, `min ESS=184`) | validated |
| Benchmark 9.34 | Two-level continuous dynamic CFA + between factor + random `logv` | Yes | Rust dynamic SEM + Newton–Laplace + PX | Published benchmark agreement across all 27 parameters (`mean_s=0.300`, `between_var_logv=0.012 vs 0.012`, `max Rhat=1.025`, `min ESS=110`) | validated |
| Benchmark 9.35 | Two-level binary dynamic IRT + cross-level invariance + `[logvf@0]` | Yes | Rust probit dynamic SEM + Newton–Laplace + PX | Published benchmark agreement (`between_var_logvf=0.111 vs 0.111`, `between_var_fb=0.168 vs 0.165`, all loadings, thresholds, and between residual variances match) | validated |
| Benchmark 9.36 | Two-level bivariate 2-factor cross-lagged SEM | Yes | Rust multivariate dynamic SEM | Published benchmark agreement across all 64 parameters (`mean_s11..22` within `0.001`, `factor_cov=0.259 vs 0.259`, `DIC=380549 vs 380426`) | validated |
| Benchmark 9.37 | Two-level AR(1), groupmean-centered covariate, linear trend, 4 random slopes + `logv` | Yes | Rust Joint Linear Block A + PX | Published benchmark agreement across all 31 parameters (`var_y=0.644`, `cov_s_y=-0.001 vs -0.001`, `mean_sy=0.323`, `sx_on_w=0.234`) | validated |
| Benchmarks 9.38–9.40b | Cross-classified DSEM, `DEFINE` transforms, selective time-level predictors, Cross-Classified Dynamic CFA, & Random Factor Loadings | Yes | Rust Cross-Classified & Dynamic CFA kernels | Simulation recovery of subject/occasion random intercepts, slopes, factor variances, and random factor loadings (`s1-s3 | f BY y1-y3`) | validated |
| Three-Level DSEM3 (Asparouhov & Muthén, 2026) | 3-level latent centering, simultaneous two-level block Gibbs, non-random Level-2 AR(1), 3L random residual covariance, short-$T$ RDSEM3, PSEM Lasso | Yes | Rust block-arrow Schur Gibbs/MH + PX | Simulation recovery across L1/L2/L3 random intercepts, slopes, `l2_ar1_y`, `logv`, `logvf`, and selective L2/L3 covariates | validated |
| Continuous-Time DSEM (`dsem_ct()`) | Irregular-interval Ornstein–Uhlenbeck N=1 & 2L CT-DSEM with Padé `[6/6]` $\exp(A \Delta t)$ & continuous-time Lyapunov covariance | Yes | Rust CT-DSEM kernel | Simulation recovery of continuous-time drift, diffusion, random intercepts, and continuous-time IRFs | validated |
| Residual DSEM (Asparouhov & Muthén, 2020) | Single-level, two-level, and three-level RDSEM | Yes | Rust + R reference | Exact separation of structural covariate slope and residual autocorrelation | validated |
| Categorical & Censored DSEM | N=1 & 2L binary/ordinal ($K \ge 3$) probit and left/right/interval Tobit censored DSEM | Yes | Rust probit & Tobit Gibbs | Simulation recovery of ordered thresholds, uncensored AR slopes, and innovation variances | validated |
