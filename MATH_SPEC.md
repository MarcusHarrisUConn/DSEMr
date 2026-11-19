# Mathematical specification (DSEMr 0.1.0)

## 1. Single-level (`N=1`) VAR(p), Covariate, RDSEM, and Categorical Probit

For multivariate outcomes \(\mathbf{y}_t \in \mathbb{R}^D\), lags
\(\ell \in \mathcal{L}\), and exogenous covariates \(\mathbf{x}_t \in \mathbb{R}^{K_x}\):

\[
\mathbf{y}_t = \boldsymbol\alpha + \sum_{\ell \in \mathcal{L}} \Phi_\ell \mathbf{y}_{t-\ell}
+ B_x \mathbf{x}_t + \boldsymbol\epsilon_t, \qquad \boldsymbol\epsilon_t \sim N(\mathbf{0}, \Sigma).
\]

Under **RDSEM** (Asparouhov & Muthén, 2020), dynamics operate on structural residuals:

\[
\mathbf{y}_t = \boldsymbol\alpha + B_x \mathbf{x}_t + \mathbf{u}_t, \qquad
\mathbf{u}_t = \sum_{\ell \in \mathcal{L}} \Phi_\ell \mathbf{u}_{t-\ell} + \boldsymbol\epsilon_t.
\]

Binary and ordinal outcomes (\(y_{tj} \in \{0, \dots, K_j - 1\}\)) use Albert–Chib
probit data augmentation \(y_{tj}^* \sim N(\mu_{tj}, 1)\) with thresholds
\(\tau_{j,1} < \dots < \tau_{j,K_j-1}\) and Mplus threshold sign convention
\(\tau_{j,1} = -\alpha_j\). Prior modes support weakly informative conjugate
priors (`prior_mode = "default"`), Mplus compatibility priors (`"mplus"`, flat
coefficients and \(IG(-1, 0)\)), and Mplus Version 9 penalized shrinkage priors
(`"penalized"`, Bayesian Lasso scale mixtures).

## 2. Two-Level Location-Scale DSEM & RDSEM (Mplus 9.30–9.32, 9.37)

Observed variables are latently decomposed into between- and within-person
components \(\mathbf{y}_{it} = \boldsymbol\mu_i + \mathbf{y}_{it}^{(w)}\) and
\(\mathbf{x}_{it} = \boldsymbol\mu_{xi} + \mathbf{x}_{it}^{(w)}\) (or group-mean
centered via `DEFINE: CENTER x (GROUPMEAN)`):

\[
\mathbf{y}_{it}^{(w)} = \sum_{\ell \in \mathcal{L}} \Phi_{i\ell} \mathbf{y}_{i,t-\ell}^{(w)}
+ B_{xi} \mathbf{x}_{it}^{(w)} + \boldsymbol\epsilon_{it}, \qquad
\epsilon_{itj} \sim N\!\left(0, \exp(\omega_{ij})\right).
\]

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

## 3. Three-Level DSEM (`DSEM3`, Mplus Version 9)

For occasion \(t\) nested within Level-2 unit \(c\) nested within Level-3 cluster \(g\)
(Asparouhov & Muthén, 2026), variables undergo three-level latent decomposition:

\[
y_{gct} = y_g^{(3)} + y_{gc}^{(2)} + y_{gct}^{(1)}, \qquad
x_{gct} = x_g^{(3)} + x_{gc}^{(2)} + x_{gct}^{(1)}.
\]

Cluster parameters follow \(\theta_{gc,q} = \gamma_{0q} + \mathbf{w}_g^{(3)\top}\boldsymbol\gamma_{3q} + \mathbf{w}_{gc}^{(2)\top}\boldsymbol\gamma_{2q} + u_{g,q}^{(3)} + u_{gc,q}^{(2)}\)
with \(\mathbf{u}_g^{(3)} \sim N(\mathbf{0}, \Omega^{(3)})\) and
\(\mathbf{u}_{gc}^{(2)} \sim N(\mathbf{0}, \Omega^{(2)})\), sampled via exact
alternating Gibbs for intercepts and slopes plus Metropolis–Hastings for random
log innovation variances.

## 4. Cross-Classified DSEM (Mplus 9.38–9.40)

Observations are crossed across subjects \(i = 1, \dots, N\) and occasions \(t = 1, \dots, T\):

\[
\mathbf{y}_{it} = \boldsymbol\gamma_0 + \mathbf{u}_i^{(\text{id})} + \mathbf{v}_t^{(\text{time})} + \mathbf{y}_{it}^{(w)},
\qquad \mathbf{u}_i^{(\text{id})} \sim N(\Gamma_w \mathbf{w}_i, \Omega_{\text{id}}),
\quad \mathbf{v}_t^{(\text{time})} \sim N(\mathbf{0}, \Omega_{\text{time}}).
\]

## 5. Single-Level & Two-Level Dynamic CFA, WNFS, & Dynamic IRT (6.26–6.28, 9.33–9.36)

For \(M\) within-level latent factors \(\boldsymbol\eta_{it}\) and \(J\) continuous
or binary/ordinal probit indicators:

\[
\mathbf{y}_{it}^{(*)} = \boldsymbol\nu_i + \Lambda_0 \boldsymbol\eta_{it} + \Lambda_1 \boldsymbol\eta_{i,t-1} + \boldsymbol\epsilon_{it}, \qquad
\boldsymbol\eta_{it} = \Phi_i \boldsymbol\eta_{i,t-1} + \boldsymbol\zeta_{it}, \qquad
\boldsymbol\zeta_{it} \sim N(\mathbf{0}, \Psi_i).
\]

- **DAFS** (\(\Lambda_1 = 0\)) uses multivariate Kalman filtering and Carter–Kohn
  backward simulation smoothing (FFBS) with Lyapunov stationary initial covariance.
- **WNFS** (\(\Lambda_1 \ne 0\), Mplus 6.26b) uses non-degenerate companion-form
  block-conditional backward smoothing over \((\boldsymbol\eta_t, \boldsymbol\eta_{t-1})\).
- Two-level models decompose \(\boldsymbol\nu_i = \boldsymbol\nu_0 + \Lambda_B \mathbf{f}_i^{(b)} + \mathbf{e}_i^{(b)}\)
  (with optional cross-level loading invariance \(\Lambda_0 = \Lambda_B\)) and
  jointly model \((\mathbf{f}_i^{(b)}, \mathbf{s}_i, \boldsymbol\omega_i) \sim N(\mathbf{0}, \Omega_B)\).

## 6. Fit Criteria, Standardization, & Diagnostics

- **DIC & pD:** \(\bar{D} = E[-2 \log p(\mathbf{y} \mid \vartheta, \boldsymbol\eta)]\),
  \(p_D = \bar{D} - D(\bar{\vartheta}, \bar{\boldsymbol\eta})\), \(\text{DIC} = \bar{D} + p_D\).
- **Within-Cluster Standardization (`STDYX`):** cluster-specific slopes standardized
  by model-implied within-cluster SDs and averaged across clusters at each posterior
  draw (Schuurman et al., 2016).
- **Convergence & Collinearity:** rank-normalized split-\(\hat{R}\), Mplus `PSR`,
  bulk/tail ESS, MCSE of mean and median, Bayesian one-tailed \(p\)-values, and
  covariate Variance Inflation Factors (`dsem_vif()`).
