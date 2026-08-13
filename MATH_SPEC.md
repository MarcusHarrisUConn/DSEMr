# Mathematical specification

## Foundation Gaussian AR(1)

The implemented foundation model is

\[
y_t = \alpha + \phi y_{t-1} + \mathbf{x}_t^\top\boldsymbol\beta + \epsilon_t,
\qquad \epsilon_t \sim N(0, \sigma^2).
\]

After constructing lagged rows and removing rows with missing response or
predictors, write the model as \(\mathbf{y}=\mathbf{X}\boldsymbol\theta+
\boldsymbol\epsilon\). The reference Gibbs sampler uses

\[
\boldsymbol\theta \sim N(\mathbf{m}_0, V_0), \qquad
\sigma^2 \sim \operatorname{IG}(a_0,b_0).
\]

Conditionals are

\[
V_n=(X^\top X+V_0^{-1})^{-1},\quad
m_n=V_n(X^\top y+V_0^{-1}m_0),
\]

\[
\boldsymbol\theta\mid\sigma^2,y \sim N(m_n,\sigma^2V_n),
\]

and

\[
\sigma^2\mid\boldsymbol\theta,y \sim
\operatorname{IG}\left(a_0+n/2,
b_0+(y-X\boldsymbol\theta)^\top(y-X\boldsymbol\theta)/2\right).
\]

The defaults approximate the published diffuse Mplus priors for regression
coefficients and univariate residual variance. They remain explicitly marked as
experimental until public finite-sample comparisons are complete.

## Planned two-level DSEM

For person \(i\) at time \(t\), the initial continuous model family is

\[
y_{it}=\mu_i+\phi_i(y_{i,t-1}-\mu_i)+
\mathbf{x}_{it}^\top\boldsymbol\beta_i+e_{it},
\quad e_{it}\sim N(0,\sigma_i^2),
\]

with transformed or constrained person-specific parameters modeled jointly at
the between level. The implementation must define latent centering, stationarity,
initial conditions, missing-time insertion, prior Jacobians, and covariance
parameterization before this family is enabled.

## Planned RDSEM and categorical families

RDSEM applies lagged structure to residual processes rather than silently
equating residual and observed-variable dynamics. Binary and ordinal outcomes
will use explicit latent-response/probit parameterization and threshold
identification. These sections are specifications for planned work, not current
capability claims.

