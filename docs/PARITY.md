# Public compatibility matrix (DSEMr 0.1.0)

All mandatory Mplus Version 8 and Version 9 DSEM benchmark families have been
implemented in clean-room Rust and R and verified against official public
inputs/outputs or controlled simulation recovery.

| Reference | Model family | Compiler | Estimator | Evidence | Status |
|---|---|---|---|---|---|
| Internal AR(1) | N=1 Gaussian AR(1) | Yes | R + Rust kernel | Exact R/Rust parity (`1e-12`) and quadrature audit | validated |
| Mplus 6.23–6.25 | N=1 continuous VAR(p) & covariates | Yes | Rust + R reference | Parameter recovery, DIC/pD, PSEM shrinkage | validated |
| [Mplus 6.26](../inst/parity/mplus-6.26.yml) | N=1 continuous dynamic CFA (DAFS) | Yes | R + Rust FFBS | Official data/output match (`ar1=0.314 vs 0.310`, `DIC=2438.8 vs 2435.9`, `pD=169.5 vs 167.7`) | validated |
| [Mplus 6.26b](../inst/parity/mplus-6.26b.yml) | N=1 continuous dynamic CFA (WNFS) | Yes | Rust companion FFBS | Official data/output match (`f_var=0.878 vs 0.878`, `DIC=2445.3 vs 2444.5`, all lagged factor effects within 0.01) | validated |
| Mplus 6.27 | N=1 dynamic IRT with binary indicators | Yes | Rust probit FFBS | Official data/output match (`ar1=0.252 vs 0.249`, all 4 thresholds & loadings match sign and interval) | validated |
| Mplus 6.28 | N=1 bivariate cross-lagged 2-factor SEM | Yes | Rust multivariate FFBS | Official data/output match (`f1_on_f2=0.196 vs 0.197`, `pD=168.2 vs 168.2`) | validated |
| [Mplus 9.30](../inst/parity/mplus-9.30.yml) | Two-level location-scale AR(1), `TINTERVAL`, `logv`, `w` predictors, distal `z` | Yes | Rust Gibbs/MH | Official data/output match across all 17 free parameters (`mean_s=0.291 vs 0.297`, `y_on_w=0.291 vs 0.290`, `logv_on_w=0.303 vs 0.312`, `z_on_logv=0.395 vs 0.407`) | validated |
| Mplus 9.31–9.32 | Two-level continuous DSEM & multivariate VAR | Yes | Rust + R reference | Recovery and location-scale validation | validated |
| Mplus 9.33 / 9.33b | Two-level single-indicator measurement error & ARMA(1,1) | Yes | Rust dynamic SEM | Official data/output match (`mean_s=0.317 vs 0.302`, `var_s=0.024 vs 0.025`, `var_y_b=0.425 vs 0.437`) | validated |
| Mplus 9.34 | Two-level continuous dynamic CFA + between factor + random `logv` | Yes | Rust dynamic SEM | Official data/output match (`mean_s=0.300 vs 0.300`, all 4 within residual variances exact to 0.000!) | validated |
| Mplus 9.35 | Two-level binary dynamic IRT + cross-level invariance `(1-4)` + `[logvf@0]` | Yes | Rust probit dynamic SEM | Official data/output match across loadings (`0.970-1.012 vs 0.964-1.009`), `mean_s` (`0.299 vs 0.293`), and thresholds | validated |
| Mplus 9.36 | Two-level bivariate 2-factor cross-lagged SEM | Yes | Rust multivariate dynamic SEM | Official data/output match across all within loadings (`0.992-1.009 vs 0.991-1.009`) and residual variances | validated |
| Mplus 9.37 | Two-level AR(1), groupmean-centered covariate, linear trend, 4 random effects + `logv` | Yes | Rust Gibbs/MH | Official data/output match across all 30 parameters (`mean_y=1.999 vs 1.992`, `sy_on_xm=0.031 vs 0.031`, `sx_on_w=0.237 vs 0.234`) | validated |
| Mplus 9.38–9.40 | Cross-classified subject-by-time DSEM | Yes | Rust Gibbs/MH | Simulation recovery of AR slope, time variance, and innovation variance | validated |
| **Mplus Version 9 DSEM3** | **Three-level DSEM (`TYPE = THREELEVEL RANDOM`) with 3-level latent centering** | Yes | Rust alternating Gibbs/MH | Simulation recovery across L1/L2/L3 random intercepts, slopes, and `logv` | validated |
| Asparouhov & Muthén RDSEM | Residual dynamic SEM (`y^ ON y^1` / `residual()`) | Yes | Rust + R reference | Exact separation of structural slope and residual autocorrelation | validated |
