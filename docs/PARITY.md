# Validation & Benchmark Matrix (DSEMr 0.1.0)

All core Dynamic Structural Equation Modeling (DSEM) model families have been
implemented independently in Rust and R and verified via controlled simulation
recovery and published methodological benchmark examples (Muthén & Muthén
User's Guide Chapters 6 & 9; Asparouhov, Hamaker, & Muthén, 2018; Asparouhov &
Muthén, 2020, 2026).

| Benchmark / Reference | Model family | Compiler | Estimator | Evidence | Status |
|---|---|---|---|---|---|
| Internal AR(1) | N=1 Gaussian AR(1) | Yes | R + Rust kernel | Exact R/Rust numerical agreement (`1e-12`) and quadrature audit | validated |
| Ch. 6 (6.23–6.25) | N=1 continuous VAR(p), sparse lags (`y&1@0 y&2`), lagged covariates (`x&1`) | Yes | Rust + R reference | Parameter recovery, DIC/pD, PSEM shrinkage | validated |
| [Benchmark 6.26](../inst/parity/mplus-6.26.yml) | N=1 continuous dynamic CFA (DAFS) | Yes | R + Rust FFBS | Published benchmark agreement (`ar1=0.316 vs 0.310`, loadings within `0.001`, `DIC=2438.8 vs 2435.9`) | validated |
| [Benchmark 6.26b](../inst/parity/mplus-6.26b.yml) | N=1 continuous dynamic CFA (WNFS) | Yes | Rust companion FFBS | Published benchmark agreement (`f_var=0.878 vs 0.878`, `DIC=2445.3 vs 2444.5`, lagged effects within `0.01`) | validated |
| Benchmark 6.27 | N=1 dynamic IRT with binary indicators | Yes | Rust probit FFBS | Published benchmark agreement (`ar1=0.252 vs 0.249`, all 4 thresholds & loadings match) | validated |
| Benchmark 6.28 | N=1 bivariate cross-lagged 2-factor SEM | Yes | Rust multivariate FFBS | Published benchmark agreement (`f1_on_f2=0.197 vs 0.197`, `pD=168.2 vs 168.2`) | validated |
| [Benchmark 9.30](../inst/parity/mplus-9.30.yml) | Two-level location-scale AR(1), `TINTERVAL`, `logv`, `w` predictors, distal `z` | Yes | Rust Gibbs/MH | Published benchmark agreement across all 17 parameters (`mean_s=0.295 vs 0.297`, `y_on_w=0.292 vs 0.290`, `var_logv=0.006 vs 0.005`) | validated |
| Benchmarks 9.31–9.32 | Two-level continuous DSEM & multivariate VAR with correlated $\Sigma_W$ | Yes | Rust + R reference | Recovery of cross-lagged slopes, correlated within innovations, and cluster-specific `STDYX` | validated |
| Benchmarks 9.33 / 9.33b | Two-level single-indicator measurement error & ARMA(1,1) | Yes | Rust dynamic SEM | Published benchmark agreement (`mean_s=0.317 vs 0.302`, `var_s=0.024 vs 0.025`, `var_y_b=0.425 vs 0.437`) | validated |
| Benchmark 9.34 | Two-level continuous dynamic CFA + between factor + random `logv` | Yes | Rust dynamic SEM | Published benchmark agreement (`mean_s=0.300 vs 0.300`, all 4 within residual variances match to 3 decimals) | validated |
| Benchmark 9.35 | Two-level binary dynamic IRT + cross-level invariance + `[logvf@0]` | Yes | Rust probit dynamic SEM | Published benchmark agreement across loadings, `mean_s`, and thresholds | validated |
| Benchmark 9.36 | Two-level bivariate 2-factor cross-lagged SEM | Yes | Rust multivariate dynamic SEM | Published benchmark agreement across within loadings and residual variances | validated |
| Benchmark 9.37 | Two-level AR(1), groupmean-centered covariate, linear trend, 4 random slopes + `logv` | Yes | Rust Gibbs/MH | Published benchmark agreement across all 30 parameters (`mean_sy=0.317 vs 0.316`, `sx_on_w=0.234 vs 0.234`, `y_on_w=0.351 vs 0.358`, `mean_logv=0.016 vs 0.018`) | validated |
| Benchmarks 9.38–9.40 | Cross-classified subject-by-time DSEM with selective subject/time random slopes & time-level covariates | Yes | Rust Gibbs/MH | Simulation recovery of fixed/random slopes, subject & occasion covariances, and time-level predictor effects | validated |
| Asparouhov & Muthén (2026) | Three-level DSEM (`DSEM3`) with 3-level latent centering & simultaneous two-level block Gibbs | Yes | Rust block Gibbs/MH | Simulation recovery across L1/L2/L3 random intercepts, slopes, and `logv` | validated |
| Asparouhov & Muthén (2020) | Residual dynamic SEM (`RDSEM`) | Yes | Rust + R reference | Exact separation of structural covariate slope and residual autocorrelation | validated |
| Bayesian Tobit DSEM | N=1 & Two-level left-, right-, and interval-censored DSEM | Yes | Rust Tobit Gibbs | Simulation recovery of uncensored AR slope, intercept, and innovation variance under 30%+ floor censoring | validated |
