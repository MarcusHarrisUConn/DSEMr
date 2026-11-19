# DSEMr 0.1.0

First CRAN release achieving clean-room parity across Mplus Version 8 and
Version 9 DSEM model families with an offline-vendored Rust numerical engine.

## Major model family additions

- **Three-Level DSEM (`TYPE = THREELEVEL RANDOM` / `DSEM3`, Mplus Version 9):**
  full three-level latent variable decomposition, flexible Level-2 and Level-3
  random intercepts, slopes, and log innovation variances, Level-2 and Level-3
  predictors, and exact alternating Gibbs/MH estimation in Rust.
- **Two-Level Location-Scale DSEM (Mplus 9.30–9.32, 9.37):** latent person-mean
  centering for outcomes and exogenous covariates, arbitrary random slopes,
  random log residual variances (`logv | y`), `%BETWEEN%` structural
  regressions, unstructured random-effect covariances, and joint distal outcome
  regressions (`z ON y s logv`).
- **Cross-Classified DSEM (Mplus 9.38–9.40):** subject-by-time two-way random
  effects with within-cell autoregressive and covariate dynamics.
- **Residual DSEM (`RDSEM`):** explicit structural versus residual dynamic
  decomposition supporting both `lavaan` `residual(lag(y, 1))` and Mplus caret
  `y^ ON y^1` syntax.
- **Dynamic CFA, WNFS, Multi-Factor SEM, and Dynamic IRT (Mplus 6.26–6.28,
  9.33–9.36):** multivariate Rust Kalman FFBS smoother, non-degenerate
  companion-form WNFS smoothing, binary and ordinal probit data augmentation
  with Mplus threshold sign conventions, cross-level measurement invariance
  `(1-4)`, and correlated factor innovations.
- **Mplus Version 9 priors, fit indices, and diagnostics:** `TINTERVAL` grid
  alignment (`dsem_tinterval()`), `DIC` and `pD` (`dsem_dic()`), covariate VIFs
  (`dsem_vif()`), `STDYX`/`STDY`/`STD` standardization (`dsem_standardize()`),
  Bayesian one-tailed p-values, Mplus `PSR`, and `PSEM` Lasso shrinkage priors
  (`prior_mode = "penalized"`).

## CRAN and packaging hardening

- Updated build scripts (`configure`, `cleanup`, `src/Makevars.in`,
  `src/Makevars.win.in`, `tools/config.R`) for strict offline CRAN Rust builds
  (`-j 2 --offline`), reporting `rustc` and `cargo` versions, removing
  install-time source mutations, and adding `inst/AUTHORS` crate attributions.
- Completed runnable `@examples`, full S3 method registration (`coef`, `vcov`,
  `confint`, `logLik`, `fitted`, `residuals`, `predict`, `plot`, `summary`),
  spelling wordlist (`inst/WORDLIST`), and executable vignettes.
