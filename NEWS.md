# DSEMr 0.1.0

First CRAN release providing Bayesian Dynamic Structural Equation Modeling
(DSEM) across single-level, two-level, three-level, cross-classified, residual,
and latent dynamic SEM model families with an offline-vendored Rust numerical
engine.

## Major model family capabilities

- **Three-Level DSEM (`THREELEVEL` / `DSEM3`):** full three-level latent
  variable decomposition, flexible Level-2 and Level-3 random intercepts,
  slopes, and log innovation variances, Level-2 and Level-3 predictors,
  simultaneous two-level block Gibbs updates via exact $O(C_g)$ Schur
  complement marginalization (Asparouhov & Muthén, 2026), SUR conditioning on
  correlated random effects, and full exposure of both Level-2 and Level-3
  random-effect posteriors.
- **Two-Level Location-Scale & Multivariate VAR DSEM:** latent person-mean
  centering for outcomes and exogenous covariates, arbitrary random slopes
  (including lagged exogenous covariates and sparse fixed-zero lag
  specifications), full unstructured within-level innovation covariance
  $\Sigma_W$ for multivariate two-level VAR, random log residual variances
  (`logv | y`) updated via strictly-concave Newton-Laplace Independence MH,
  exact stationary initial-state MH updates for short time series
  (Asparouhov, 2026), between-level structural regressions, unstructured
  random-effect covariances, and joint distal outcome regressions.
- **Tobit Censored DSEM (Single-Level & Two-Level):** Bayesian Tobit data
  augmentation for left-, right-, and interval-censored outcomes (`censored =`
  or `CENSORED = y (b)` / `(a)` / `(ab)`) with exact future-lag conditional
  truncation, free or random innovation variances, and observed-data Tobit DIC.
- **Cross-Classified DSEM:** subject-by-time two-way random intercepts,
  selective subject- and occasion-varying random slopes, subject- and
  time-level predictors, subject random log innovation variances via
  Newton-Laplace Independence MH, and full exposure of both subject and
  occasion effect posteriors.
- **Residual DSEM (`RDSEM`):** explicit separation of structural covariate
  effects from residual autocorrelation (`residual(lag(y, 1))` and `y^ ON y^1`).
- **Dynamic CFA, WNFS, Multi-Factor SEM, and Dynamic IRT:** multivariate Rust
  Kalman FFBS smoother, non-degenerate companion-form WNFS smoothing, binary
  and ordinal probit data augmentation, cross-level measurement invariance,
  and correlated factor innovations.

## Post-estimation analytics, reporting, comparison, and reproducible workflows

- **Standardization & Explained Variance:** `dsem_standardize()` (`STDYX`,
  `STDY`, `STD`) with cluster-averaged and cluster-specific (`cluster = TRUE`)
  within-person standardization for univariate and bivariate cross-lagged
  models (Schuurman et al., 2016), plus Within/Between Bayesian $R^2$ via
  `dsem_r2()`.
- **Posterior Analytics:**
  - `dsem_partial_cor()` — posterior partial and zero-order correlations across
    between-level, Level-2/3, occasion, and within-level covariance blocks.
  - `dsem_hypothesis()` — derived parameter posteriors (indirect effects,
    cross-lagged differences, variance ratios) and joint multivariate Bayesian
    Wald tests.
  - `dsem_irf()` — orthogonalized and generalized Bayesian Impulse Response
    Functions with posterior credible bands and `plot()` support.
  - `dsem_fscomparison()` — posterior cluster random-effect rankings,
    exceedance probabilities, pairwise comparisons, and caterpillar plots.
- **Bayesian Model Comparison & Diagnostics:** `dsem_compare()` and
  `anova.DSEMfit()` (`DIC`, `pD`, `D_bar`, `delta_dic`, DIC Akaike weights,
  comparability checks), `dsem_tinterval()`, `dsem_vif()`, `POINT = MEDIAN`
  support in `coef()` and `summary()`, Bayesian one-tailed $p$-values, split
  $\hat{R}$, PSR, bulk/tail ESS, MCSE, and Bayesian Lasso PSEM shrinkage priors
  (`prior_mode = "penalized"`).
- **Publication Reporting & Reproducible Workflows:** APA 7th Edition tables
  (`dsem_table()` in `data.frame`, `markdown`, `latex`, `html`), automatic
  LaTeX equations (`dsem_equations()`), visual DSEM path diagrams
  (`dsem_diagram()` in base R `plot`, vector `svg`, `mermaid`, `dot`), and
  declarative GUI-ready specification & code export (`dsem_spec()`,
  `as_dsem_model()`, `dsem_to_lavaan()`, `dsem_to_mplus()`, `dsem_code()`).

## CRAN and packaging hardening

- Offline CRAN Rust builds (`-j 2 --offline`) against vendored
  `src/rust/vendor.tar.xz`, reporting `rustc` and `cargo` versions and crate
  attributions in `inst/AUTHORS`.
- Complete runnable `@examples`, full S3 method registration, spelling
  wordlist (`inst/WORDLIST`), and executable vignettes.
