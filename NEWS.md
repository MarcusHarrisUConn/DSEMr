# DSEMr 0.1.0

First CRAN release providing Bayesian Dynamic Structural Equation Modeling
(DSEM) across single-level, two-level, three-level, cross-classified, residual,
and latent dynamic SEM model families with an offline-vendored Rust numerical
engine.

## Major model family capabilities

- **Three-Level DSEM (`THREELEVEL` / `DSEM3`):** full three-level latent
  variable decomposition, flexible Level-2 and Level-3 random intercepts,
  slopes, log innovation variances, and three-level random residual covariances
  (`logvf`), non-random Level-2 AR(1) dynamics (`l2_ar1_y`), short-$T$ RDSEM3,
  selective Level-2 and Level-3 predictors, simultaneous two-level block Gibbs
  updates via exact $O(C_g)$ Schur complement marginalization (Asparouhov &
  Muthén, 2026), exact PX scale group moves, PSEM Bayesian Lasso priors, and
  full exposure of both Level-2 and Level-3 random-effect posteriors.
- **Continuous-Time DSEM (`dsem_ct()`):** single-level and two-level
  Ornstein–Uhlenbeck stochastic differential equation estimation for arbitrary
  irregular observation intervals via exact Padé `[6/6]` scaling-and-squaring
  matrix exponentials $\exp(A \Delta t)$ and Kronecker continuous-time Lyapunov
  stationary covariances, with continuous-time Impulse Response Functions in
  `dsem_irf()`.
- **Two-Level Location-Scale, Random Residual Covariance, & Ordinal DSEM:**
  latent person-mean centering for outcomes and exogenous covariates, joint
  intercept-slope block Gibbs updates, arbitrary random slopes (including lagged
  exogenous covariates and sparse fixed-zero lag specifications), full
  unstructured within-level innovation covariance $\Sigma_W$, random log
  residual variances (`logv | y`) and shared-shock random residual covariances
  (`logvf | y1 ~~ y2` / `f BY y1@1 y2@1; logvf | f;`) updated via strictly-concave
  Newton-Laplace Independence MH and exact PX scale group moves, exact
  stationary initial-state MH updates for short-$T$ AR(1) and RDSEM
  (Asparouhov, 2026), two-level binary and ordinal ($K \ge 3$) probit
  augmentation, between-level structural regressions, unstructured random-effect
  covariances, and joint distal outcome regressions.
- **Exact Collapsed ARMA(1,1) & Dynamic Measurement Models:** $O(T)$ Toeplitz
  tridiagonal Cholesky collapsed ARMA(1,1) sampler with joint collapsed
  $(s, \theta, \log \sigma_e^2)$ ridge moves and exact marginal DIC; 3-way
  collapsed 1D Kalman marginal MH for single-indicator measurement-error and
  N=1 dynamic factor models; Cross-Classified Dynamic CFA with measurement
  invariance and subject-varying random factor loadings (`s1-s3 | f BY y1-y3`).
- **Scripting Extensions & Strict Monte Carlo:** `DEFINE:` inline variable
  transformations (`CLUSTER_MEAN()`, `STANDARDIZE`, arithmetic), automatic
  `MODEL CONSTRAINT:` and `MODEL TEST:` post-estimation attachment on `DSEMfit`,
  canonical `prior_mode = "diffuse"` specification, and `convergence = "strict"`
  acceptance gating in `dsem_monte_carlo()`.
- **Interactive Shiny Studio (`run_dsemr()`):** 5-tab guided `bslib` web
  interface covering curated datasets & file upload, `TINTERVAL` diagnostics,
  visual model builder with live SVG path diagrams and LaTeX equations,
  MCMC diagnostics, APA HTML tables, partial correlations, hypothesis & Wald
  tests, IRFs, cluster rankings, DIC comparison, and Quarto/R reproducibility
  bundle downloads.
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
