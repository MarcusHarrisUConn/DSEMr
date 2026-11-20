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
  diffuse priors (`prior_mode = "diffuse"`, flat coefficients and \(IG(-1, 0)\)),
  and Bayesian Lasso scale-mixture shrinkage priors (`prior_mode = "penalized"`,
  PSEM; Park & Casella, 2008; Asparouhov & Muthén, 2024).

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
\(\boldsymbol\epsilon_{it} \sim N(\mathbf{0}, \Sigma_W)\), cluster-specific
heteroscedastic log innovation variances \(\epsilon_{itj} \sim N(0, \exp(\omega_{ij}))\)
updated via strictly-concave Newton–Laplace Independence Metropolis–Hastings,
or **random residual covariances** modeled via a shared within-level shock
factor \(\boldsymbol\epsilon_{it} = \boldsymbol\lambda_f f_{it} + \mathbf{e}_{it}\)
with \(f_{it} \sim N(0, \exp(\omega_{fi}))\) (Asparouhov et al., 2018), paired
with exact Parameter-Expanded (PX) multiplicative scale group moves on
\((\boldsymbol\omega_i, \Omega_B)\) to eliminate centered Gibbs funnel
autocorrelation. For univariate AR(1) and RDSEM models under latent centering,
the stationary initial observation \(y_{i,0} \sim N(\mu_i, \sigma_i^2 / (1 - \phi_i^2))\)
is integrated into the updates of \(\mu_i\), \(\omega_i\), and \(\phi_i\) (via
an exact stationary initial-state MH step) to reduce finite-\(T\) bias in short
time series (Asparouhov, 2026). Two-level binary and ordinal ($K \ge 3$) probit
outcomes use exact future-lag truncated-normal data augmentation with ordered
threshold updates.

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
\(\mathbf{u}_{gc}^{(2)} \sim N(\mathbf{0}, \Omega^{(2)})\), supporting optional
Level-2 autoregressive dynamics \(u_{gc,j}^{(2)} = \phi_j^{(2)} u_{g,c-1,j}^{(2)} + \xi_{gc,j}\)
(Asparouhov & Muthén, 2026b), three-level random residual covariances, short-\(T\)
RDSEM3, and PSEM Bayesian Lasso priors. Level-3 and Level-2 intercepts (and
slopes) are updated **simultaneously** in \(O(C_g)\) time via exact block-arrow
Schur complement marginalization (Asparouhov & Muthén, 2026a), eliminating
Level-2/Level-3 posterior ridge autocorrelation, paired with strictly-concave
Newton–Laplace Independence MH and PX scale group moves for random log
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
  backward simulation smoothing (FFBS) with Lyapunov stationary initial covariance,
  augmented with exact 1D collapsed Kalman marginal MH moves for single-indicator
  measurement-error and N=1 dynamic factor identification ridges.
- **WNFS** (\(\Lambda_1 \ne 0\); Zhang & Nesselroade, 2007) uses non-degenerate
  companion-form block-conditional backward smoothing over \((\boldsymbol\eta_t, \boldsymbol\eta_{t-1})\).
- **Exact Collapsed ARMA(1,1):** For moving-average dynamics \(y_{it} - \mu_i = s_i (y_{i,t-1} - \mu_i) + e_{it} + \theta e_{i,t-1}\),
  latent innovations \(\mathbf{e}_i\) are integrated out analytically via \(O(T)\)
  Toeplitz tridiagonal Cholesky factorization paired with a joint collapsed
  \((s_{\text{pop}}, \{s_i\}, \theta, \log \sigma_e^2)\) ridge move.
- **Cross-Classified Dynamic CFA & Random Factor Loadings:** Two-way subject and
  occasion factor decomposition \(y_{itj} = \nu_j + \lambda_{ij} (\eta_{it} + f_i^{(\text{id})} + f_t^{(\text{time})}) + u_{ij}^{(\text{id})} + v_{tj}^{(\text{time})} + \epsilon_{itj}\)
  with measurement invariance or subject-varying random factor loadings \(\lambda_{ij} \sim N(\bar{\lambda}_j, \tau_{\lambda j}^2)\).
- Two-level models decompose \(\boldsymbol\nu_i = \boldsymbol\nu_0 + \Lambda_B \mathbf{f}_i^{(b)} + \mathbf{e}_i^{(b)}\)
  (with optional cross-level loading invariance \(\Lambda_0 = \Lambda_B\)) and
  jointly model \((\mathbf{f}_i^{(b)}, \mathbf{s}_i, \boldsymbol\omega_i) \sim N(\mathbf{0}, \Omega_B)\)
  with Newton–Laplace MH and exact PX scale moves on \(\boldsymbol\omega_i\).

## 6. Continuous-Time DSEM (`dsem_ct()`)

For irregular observation times \(t_0 < t_1 < \dots < t_{T_i}\) with elapsed
intervals \(\Delta t_{ik} = t_{ik} - t_{i,k-1} > 0\), latent centered states
evolve according to the stochastic differential equation (Oud & Jansen, 2000;
Driver & Voelkle, 2018; Asparouhov & Muthén, 2025):

\[
d\mathbf{y}_i^{(w)}(t) = A_i \mathbf{y}_i^{(w)}(t)\,dt + G_i\,d\mathbf{W}_i(t),
\]

yielding exact discrete-time transitions \(\mathbf{y}_i^{(w)}(t_k) = \exp(A_i \Delta t_{ik}) \mathbf{y}_i^{(w)}(t_{k-1}) + \boldsymbol\zeta_{ik}\)
with covariance \(Q_i(\Delta t_{ik}) = \Sigma_{\infty,i} - \exp(A_i \Delta t_{ik}) \Sigma_{\infty,i} \exp(A_i^\top \Delta t_{ik})\),
where \(A_i \Sigma_{\infty,i} + \Sigma_{\infty,i} A_i^\top + G_i G_i^\top = 0\)
is solved via exact Kronecker continuous-time Lyapunov equations and matrix
exponentials are computed via Padé `[6/6]` scaling-and-squaring. Continuous-time
impulse responses follow \(\exp(A h) B_0\) over continuous horizon \(h\).

## 7. Fit Criteria, Standardization, Partial Correlations, IRFs, & Diagnostics

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
