# DSEMr

`DSEMr` is an open-source R package for Bayesian **Dynamic Structural Equation Modeling (DSEM)** of intensive longitudinal and time-series data. It pairs a unified canonical model representation — supporting both extended `lavaan` formulas and standard section-based SEM scripts — with a compiled, offline-vendored **Rust** numerical engine (`extendr`) and a transparent R reference implementation.

## Supported DSEM Model Families

| Model family | Methodological foundation | Compiler | Rust + R Estimator | Status |
|---|---|---:|---:|---|
| N=1 Gaussian AR(p) / VAR(p), sparse lags & lagged covariates | Shumway & Stoffer (2011) | Yes | Yes | Validated |
| N=1 Dynamic CFA — Direct Autoregressive (DAFS) & White-Noise Factor Score (WNFS) | Molenaar (1985); Zhang & Nesselroade (2007) | Yes | Yes | Validated |
| N=1 Binary & Ordinal Probit DSEM / Dynamic IRT | Albert & Chib (1993); Asparouhov et al. (2018) | Yes | Yes | Validated |
| N=1 & Two-Level Tobit Censored DSEM (floor/ceiling/interval censoring) | Bayesian Tobit data augmentation | Yes | Yes | Validated |
| N=1 Multivariate Cross-Lagged Multi-Factor Dynamic SEM | Asparouhov, Hamaker, & Muthén (2018) | Yes | Yes | Validated |
| Two-Level Location-Scale DSEM (latent centering, random slopes & `logv`, between predictors, distal outcomes, short-T stationary MH) | Asparouhov et al. (2018); McNeish & Hamaker (2020); Asparouhov (2026) | Yes | Yes | Validated |
| Two-Level Multivariate Cross-Lagged VAR with Correlated Within Innovations ($\Sigma_W$) | Schuurman et al. (2016); Asparouhov et al. (2018) | Yes | Yes | Validated |
| Two-Level Single-Indicator Measurement Error, ARMA(1,1), Dynamic CFA, Dynamic IRT (cross-level invariance), & Multi-Factor Cross-Lagged SEM | Asparouhov, Hamaker, & Muthén (2018) | Yes | Yes | Validated |
| Residual DSEM (`RDSEM`: `residual(lag(y, 1))` / `y^ ON y^1`) | Asparouhov & Muthén (2020) | Yes | Yes | Validated |
| Cross-Classified DSEM (`TYPE = CROSSCLASSIFIED RANDOM`) with selective subject/occasion random slopes & time-level predictors | Asparouhov, Hamaker, & Muthén (2018) | Yes | Yes | Validated |
| Three-Level DSEM (`TYPE = THREELEVEL RANDOM` / DSEM3) with 3-level latent decomposition & simultaneous two-level block Gibbs | Asparouhov & Muthén (2026) | Yes | Yes | Validated |
| **Diagnostics & Post-Estimation:** `TINTERVAL` grid alignment, `DIC`/`pD`, cluster-averaged & cluster-specific `STDYX`/`STDY`/`STD`, Bayesian $R^2$, posterior partial correlations, derived hypothesis & Wald tests, IRFs, cluster rankings, `VIF`, `PSEM` Lasso priors, split-$\hat{R}$, PSR, bulk/tail ESS, MCSE, one-tailed $p$-values | Gelman et al. (2013); Schuurman et al. (2016); Asparouhov & Muthén (2025) | Yes | Yes | Validated |

## Installation

`DSEMr` compiles cleanly offline using vendored Rust crates (`src/rust/vendor.tar.xz`; requires R >= 4.2.0 and `rustc` / `cargo` >= 1.71.0):

```r
# install.packages("DSEMr") # once on CRAN
# Development version:
remotes::install_github("MarcusHarrisUConn/DSEMr")
```

## Quick Examples

### Two-level location-scale DSEM with `TINTERVAL` and distal outcome

```r
library(DSEMr)

m_2l <- dsem_model("
  level: 1
    s | y ~ lag(y, 1)
    logv | y ~~ y
  level: 2
    y ~ w
    s ~ w
    logv ~ w
    y ~~ s + logv
    s ~~ logv
    z ~ y + s + logv
", id = "subject", time = "time", tinterval = 1)
```

### Three-level DSEM (`DSEM3`) with simultaneous block Gibbs

```r
dat3 <- simulate_dsem(
  n = 20, clusters = c(12, 4), id = c("day", "person"),
  intercept = 1.5, ar = 0.45, sigma = 0.7,
  random_sd = c(0.35, 0.08, 0.15), seed = 2026
)

m_3l <- dsem_model("
  level: 1
    s | y ~ lag(y, 1)
    logv | y ~~ y
  level: 2
    y ~~ y
  level: 3
    y ~~ s + logv
    s ~~ logv
", id = c("day", "person"), time = "time", level_type = "threelevel")

fit3 <- dsem(m_3l, dat3, chains = 4, iter = 2000, warmup = 1000, seed = 4815)
summary(fit3, standardized = TRUE)
dsem_dic(fit3)
```

## Public API

- **Model Specification:** `dsem_model()`, `mplus_to_dsem()`, `dsem_spec()`, `as_dsem_model()` — extended `lavaan` formulas, section-based SEM script import, and declarative GUI-ready model specifications
- **Estimation & Grid Alignment:** `dsem()`, `dsem_tinterval()` — compiled Rust Bayesian MCMC estimation and order-preserving time-interval alignment
- **Standardization & Post-Estimation Analytics:**
  - `dsem_standardize()` / `dsem_r2()` — posterior `STDYX`, `STDY`, `STD` (cluster-averaged and per-cluster via `cluster = TRUE`), and Within/Between Bayesian $R^2$
  - `dsem_partial_cor()` — posterior partial and zero-order correlation networks across Between, Level-2/3, Occasion, and Within covariance blocks
  - `dsem_hypothesis()` — derived parameter posteriors (indirect effects, contrasts, variance ratios) and joint multivariate Bayesian Wald tests
  - `dsem_irf()` — orthogonalized and generalized Bayesian Impulse Response Functions with `plot()`
  - `dsem_fscomparison()` — posterior cluster random-effect rankings, exceedance probabilities, and caterpillar plots
- **Model Comparison & Diagnostics:** `dsem_dic()`, `dsem_compare()`, `anova()`, `dsem_vif()`, `dsem_inspect()`
- **Publication Reporting & Reproducibility:** `dsem_table()` (APA 7th Ed. in `markdown`, `latex`, `html`, `data.frame`), `dsem_equations()` (LaTeX equations), `dsem_diagram()` (`plot`, vector `svg`, `mermaid`, `dot`), `dsem_code()`, `dsem_to_lavaan()`, `dsem_to_mplus()`
- **Simulation & Parallelism:** `simulate_dsem()`, `dsem_monte_carlo()`, `dsem_compute()`
- **Standard S3 Methods:** `summary()`, `coef()`, `vcov()`, `confint()`, `logLik()`, `anova()`, `fitted()`, `residuals()`, `predict()`, `plot()`

## License

`DSEMr` is an independent open-source implementation of published Dynamic Structural Equation Modeling methods, licensed under GPL (>= 3).
