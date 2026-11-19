# DSEMr

`DSEMr` is an open-source R package for Bayesian **Dynamic Structural Equation Modeling (DSEM)** built around a unified canonical model representation, dual `lavaan`-extended and clean-room **Mplus Version 9** compatibility frontends, and a compiled offline-vendored **Rust** numerical backend (`extendr`).

## Capabilities & Mplus Version 9 Parity

| Model family | Mplus reference | Compiler | Rust + R Estimator | Status |
|---|---|---:|---:|---|
| N=1 Gaussian AR(p) / VAR(p) & covariates | 6.23–6.25 | Yes | Yes | Validated |
| N=1 Dynamic CFA — DAFS & WNFS | 6.26, 6.26b | Yes | Yes | Validated |
| N=1 Binary & Ordinal Probit DSEM / Dynamic IRT | 6.27 | Yes | Yes | Validated |
| N=1 Multivariate Cross-Lagged Dynamic SEM | 6.28 | Yes | Yes | Validated |
| Two-level Location-Scale DSEM (latent centering, random slopes & `logv`, `%BETWEEN%` predictors & distal outcomes) | 9.30–9.32, 9.37 | Yes | Yes | Validated |
| Two-level Single-Indicator Measurement Error & ARMA(1,1) | 9.33, 9.33b | Yes | Yes | Validated |
| Two-level Dynamic CFA, Dynamic IRT (cross-level invariance), & Multi-Factor Cross-Lagged SEM | 9.34–9.36 | Yes | Yes | Validated |
| Residual DSEM (`RDSEM`: `y^ ON y^1` / `residual()`) | Asparouhov & Muthén (2020) | Yes | Yes | Validated |
| Cross-Classified DSEM (`TYPE = CROSSCLASSIFIED RANDOM`) | 9.38–9.40 | Yes | Yes | Validated |
| **Three-Level DSEM (`TYPE = THREELEVEL RANDOM` / DSEM3)** with 3-level latent decomposition | **Mplus Version 9** | Yes | Yes | Validated |
| **Diagnostics & Post-Estimation:** `TINTERVAL` grid alignment, `DIC`/`pD`, `STDYX`/`STDY`/`STD` (cluster-wise within-person), `VIF`, `PSEM` shrinkage priors, split-Rhat, PSR (`TECH8`), bulk/tail ESS, MCSE, one-tailed p-values | **Mplus Version 8 & 9** | Yes | Yes | Validated |

## Installation

`DSEMr` compiles cleanly offline using vendored Rust crates (`src/rust/vendor.tar.xz`; requires R >= 4.2.0 and `rustc` / `cargo` >= 1.71.0):

```r
# install.packages("DSEMr") # once on CRAN
# Development version:
remotes::install_github("MarcusHarrisUConn/DSEMr")
```

## Quick examples

### Two-level location-scale DSEM with `TINTERVAL` and distal outcome (Mplus 9.30)

```r
library(DSEMr)

m_2l <- mplus_to_dsem("
  VARIABLE:
    CLUSTER = subject;
    BETWEEN = z w;
    TINTERVAL = time(1);
  ANALYSIS:
    TYPE = TWOLEVEL RANDOM;
    ESTIMATOR = BAYES;
  MODEL:
    %WITHIN%
      s | y ON y&1;
      logv | y;
    %BETWEEN%
      y s logv ON w;
      y s logv WITH y s logv;
      z ON y s logv;
")
```

### Three-level DSEM (`DSEM3`, Mplus Version 9)

```r
dat3 <- simulate_dsem(
  n = 20, clusters = c(12, 4), id = c("day", "person"),
  intercept = 1.5, ar = 0.45, sigma = 0.7,
  random_sd = c(0.35, 0.08, 0.15), seed = 2026
)

m_3l <- dsem_model("
  VARIABLE: CLUSTER = person day;
  ANALYSIS: TYPE = THREELEVEL RANDOM; ESTIMATOR = BAYES;
  MODEL:
    %WITHIN%
      s | y ON y&1;
      logv | y;
    %BETWEEN day%
      y;
    %BETWEEN person%
      y s logv WITH y s logv;
", syntax = "mplus", time = "time")

fit3 <- dsem(m_3l, dat3, chains = 4, iter = 2000, warmup = 1000, seed = 4815)
summary(fit3, standardized = TRUE)
dsem_dic(fit3)
```

## Public API

- `dsem_model()` / `mplus_to_dsem()` — dual `lavaan`-extended and Mplus v9 syntax compiler
- `dsem()` — Bayesian DSEM / RDSEM / DSEM3 / Dynamic CFA & IRT estimator
- `dsem_tinterval()` — Mplus-compatible `TINTERVAL` irregular timestamp alignment
- `dsem_standardize()` — posterior `STDYX`, `STDY`, and `STD` (including cluster-wise within-person standardization)
- `dsem_dic()` — conditional Deviance Information Criterion (`DIC`, `pD`, `D_bar`)
- `dsem_vif()` — Mplus Version 9 covariate collinearity Variance Inflation Factors
- `simulate_dsem()` / `dsem_monte_carlo()` — deterministic simulation and Monte Carlo studies
- `dsem_compute()` / `dsem_inspect()` — scheduler-aware parallelism and object inspection
- Standard S3 methods: `summary()`, `coef()`, `vcov()`, `confint()`, `logLik()`, `fitted()`, `residuals()`, `predict()`, `plot()`

## Clean-room policy & license

`DSEMr` is an independent clean-room implementation based exclusively on published methodological papers and public documentation. Licensed under GPL (>= 3).
