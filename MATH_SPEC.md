# Mathematical specification (DSEMr 0.1.0)

## 1. Single-Level (`N=1`) VAR(p), Covariate, RDSEM, Probit, and Tobit Censored Models

For multivariate outcomes \(\mathbf{y}_t \in \mathbb{R}^D\), active lags
\(\ell \in \mathcal{L}\), and contemporaneous or lagged exogenous covariates
\(\mathbf{x}_t \in \mathbb{R}^{K_x}\):

\[
\mathbf{y}_t = \boldsymbol\alpha + \sum_{\ell \in \mathcal{L}} \Phi_\ell \mathbf{y}_{t-\ell}
+ B_x \mathbf{x}_t + \boldsymbol\epsilon_t, \qquad \boldsymbol\epsilon_t \sim N(\mathbf{0}, \Sigma).
\]

Under **RDSEM** (Asparouhov & Muthén, 2020), dynamics operate on structural residuals:

\[
\mathbf{y}_t = \boldsymbol\alpha + B_x \mathbf{x}_t + \mathbf{u}_t, \qquad
\mathbf{u}_t = \sum_{\ell \in \mathcal{L}} \Phi_\ell \mathbf{u}_{t-\ell} + \boldsymbol\epsilon_t.
\]

- **Categorical Probit:** Binary and ordinal outcomes (\(y_{tj} \in \{0, \dots, K_j - 1\}\))
  use Albert–Chib probit data augmentation \(y_{tj}^* \sim N(\mu_{tj}, 1)\) with
  thresholds \(\tau_{j,1} < \dots < \tau_{j,K_j-1}\) and threshold sign
  convention \(\tau_{j,1} = -\alpha_j\).
- **Tobit Censored Outcomes:** Left-, right-, or interval-censored continuous
  outcomes (\(y_{tj} = \min(\max(y_{tj}^*, L_j), U_j)\)) use exact truncated-normal
  Gibbs augmentation on censored cells conditioned on both contemporaneous and
  future lag likelihoods, retaining free innovation variance \(\Sigma_{jj}\).
- **Priors:** Weakly informative conjugate priors (`prior_mode = "default"`),
  diffuse priors (`"mplus"`, flat coefficients and \(IG(-1, 0)\)), and Bayesian
  Lasso scale-mixture shrinkage priors (`"penalized"`, PSEM).

## 2. Two-Level Location-Scale, Multivariate VAR, Censored, & RDSEM

Observed variables are latently decomposed into between- and within-person
components \(\mathbf{y}_{it} = \boldsymbol\mu_i + \mathbf{y}_{it}^{(w)}\) and
\(\mathbf{x}_{it} = \boldsymbol\mu_{xi} + \mathbf{x}_{it}^{(w)}\) (or group-mean
centered):

\[
\mathbf{y}_{it}^{(w)} = \sum_{\ell \in \mathcal{L}} \Phi_{i\ell} \mathbf{y}_{i,t-\ell}^{(w)}
+ B_{xi} \mathbf{x}_{it}^{(w)} + \boldsymbol\epsilon_{it},
\]

with either unstructured multivariate within-level innovation covariance
\(\boldsymbol\epsilon_{it} \sim N(\mathbf{0}, \Sigma_W)\) or cluster-specific
heteroscedastic log innovation variances \(\epsilon_{itj} \sim N(0, \exp(\omega_{ij}))\)
updated via strictly-concave Newton–Laplace Independence Metropolis–Hastings.
For pure univariate AR(1) models under latent centering, the stationary initial
observation \(y_{i,0} \sim N(\mu_i, \sigma_i^2 / (1 - \phi_i^2))\) is
integrated into the updates of \(\mu_i\), \(\omega_i\), and \(\phi_i\) (via an
exact stationary initial-state MH step) to reduce finite-\(T\) bias in short time
series (Asparouhov, 2026).

Random intercepts, random slopes, and random log innovation variances are
collected into \(\boldsymbol\theta_i = (\boldsymbol\mu_i^\top, \mathbf{s}_i^\top, \boldsymbol\omega_i^\top)^\top\):

\[
\boldsymbol\theta_i = \boldsymbol\gamma_0 + \Gamma_w \mathbf{w}_i + \boldsymbol\zeta_i,
\qquad \boldsymbol\zeta_i \sim N(\mathbf{0}, \Omega_B),
\]

with optional between-level distal outcomes \(\mathbf{z}_i = \boldsymbol\alpha_z + B_z \boldsymbol\theta_i + D_z \mathbf{w}_i + \boldsymbol\varepsilon_{zi}\)
jointly integrated in the conditional posterior of \(\boldsymbol\theta_i\).
Irregular timestamps are aligned via `TINTERVAL` (`dsem_tinterval()`) and missing
observations are Gibbs-imputed across gaps.

## 3. Three-Level DSEM (`DSEM3`)

For occasion \(t\) nested within Level-2 unit \(c\) nested within Level-3 cluster \(g\)
(Asparouhov & Muthén, 2026), variables undergo three-level latent decomposition:

\[
y_{gct} = y_g^{(3)} + y_{gc}^{(2)} + y_{gct}^{(1)}, \qquad
x_{gct} = x_g^{(3)} + x_{gc}^{(2)} + x_{gct}^{(1)}.
\]

Cluster parameters follow \(\theta_{gc,q} = \gamma_{0q} + \mathbf{w}_g^{(3)\top}\boldsymbol\gamma_{3q} + \mathbf{w}_{gc}^{(2)\top}\boldsymbol\gamma_{2q} + u_{g,q}^{(3)} + u_{gc,q}^{(2)}\)
with \(\mathbf{u}_g^{(3)} \sim N(\mathbf{0}, \Omega^{(3)})\) and
\(\mathbf{u}_{gc}^{(2)} \sim N(\mathbf{0}, \Omega^{(2)})\). Level-3 and
Level-2 intercepts (and slopes) are updated **simultaneously** in \(O(C_g)\) time
via exact block-arrow Schur complement marginalization (Asparouhov & Muthén,
2026), eliminating Level-2/Level-3 posterior ridge autocorrelation, paired with
strictly-concave Newton–Laplace Independence MH updates for random log
innovation variances.

## 4. Cross-Classified DSEM

Observations are crossed across subjects \(i = 1, \dots, N\) and occasions \(t = 1, \dots, T\):

\[
\mathbf{y}_{it} = \boldsymbol\gamma_0 + \mathbf{u}_i^{(\text{id})} + \mathbf{v}_t^{(\text{time})} + \mathbf{y}_{it}^{(w)},
\]

with selective subject-varying and/or occasion-varying random slopes, subject-level
and occasion-level predictors (\(\mathbf{w}_i^{(\text{id})}, \mathbf{w}_t^{(\text{time})}\)),
unstructured covariances \(\Omega_{\text{id}}\) and \(\Omega_{\text{time}}\),
and subject random log residual variances.

## 5. Single-Level & Two-Level Dynamic CFA, WNFS, & Dynamic IRT

For \(M\) within-level latent factors \(\boldsymbol\eta_{it}\) and \(J\) continuous
or binary/ordinal probit indicators:

\[
\mathbf{y}_{it}^{(*)} = \boldsymbol\nu_i + \Lambda_0 \boldsymbol\eta_{it} + \Lambda_1 \boldsymbol\eta_{i,t-1} + \boldsymbol\epsilon_{it}, \qquad
\boldsymbol\eta_{it} = \Phi_i \boldsymbol\eta_{i,t-1} + \boldsymbol\zeta_{it}, \qquad
\boldsymbol\zeta_{it} \sim N(\mathbf{0}, \Psi_i).
\]

- **DAFS** (\(\Lambda_1 = 0\)) uses multivariate Kalman filtering and Carter–Kohn
  backward simulation smoothing (FFBS) with Lyapunov stationary initial covariance.
- **WNFS** (\(\Lambda_1 \ne 0\); Zhang & Nesselroade, 2007) uses non-degenerate
  companion-form block-conditional backward smoothing over \((\boldsymbol\eta_t, \boldsymbol\eta_{t-1})\).
- Two-level models decompose \(\boldsymbol\nu_i = \boldsymbol\nu_0 + \Lambda_B \mathbf{f}_i^{(b)} + \mathbf{e}_i^{(b)}\)
  (with optional cross-level loading invariance \(\Lambda_0 = \Lambda_B\)) and
  jointly model \((\mathbf{f}_i^{(b)}, \mathbf{s}_i, \boldsymbol\omega_i) \sim N(\mathbf{0}, \Omega_B)\).

## 6. Fit Criteria, Standardization, Partial Correlations, IRFs, & Diagnostics

- **DIC & pD:** \(\bar{D} = E[-2 \log p(\mathbf{y} \mid \vartheta, \boldsymbol\eta)]\),
  \(p_D = \bar{D} - D(\bar{\vartheta}, \bar{\boldsymbol\eta})\), \(\text{DIC} = \bar{D} + p_D\).
- **Within-Cluster Standardization (`STDYX`):** cluster-specific slopes standardized
  by model-implied within-cluster SDs and averaged across clusters (or reported
  per cluster via `cluster = TRUE`) at each posterior draw (Schuurman et al., 2016).
- **Partial Correlations (`dsem_partial_cor()`):** precision inversion
  \(P^{(s)} = (\Sigma^{(s)})^{-1}\) at every MCMC draw giving
  \(r_{ij\cdot\text{rest}}^{(s)} = -P_{ij}^{(s)} / \sqrt{P_{ii}^{(s)} P_{jj}^{(s)}}\)
  (Asparouhov & Muthén, 2025).
- **Derived Hypotheses & Wald Tests (`dsem_hypothesis()`):** posterior distributions
  of arbitrary parameter expressions plus multivariate Bayesian Wald tests
  \(W = \bar{\boldsymbol\delta}^\top \widehat{\text{Cov}}(\boldsymbol\delta)^{-1} \bar{\boldsymbol\delta}\).
- **Impulse Response Functions (`dsem_irf()`):** orthogonalized and generalized
  dynamic response curves \(\mathbf{F}^h B_0\) with posterior credible bands.
- **Convergence & Collinearity:** rank-normalized split-\(\hat{R}\), PSR,
  bulk/tail ESS, MCSE of mean and median, Bayesian one-tailed \(p\)-values, and
  covariate Variance Inflation Factors (`dsem_vif()`).
