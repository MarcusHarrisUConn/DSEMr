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

## Experimental two-level Gaussian AR(1)

The implemented reference sampler currently uses

\[
y_{it}=\alpha_i+\phi_i y_{i,t-1}+e_{it},
\qquad e_{it}\sim N(0,\sigma^2),
\]

with \(\boldsymbol\theta_i=(\alpha_i,\phi_i)^\top\sim
N(\boldsymbol\mu,\Omega)\). The population mean has a diffuse normal prior,
\(\Omega\) has an inverse-Wishart prior, and \(\sigma^2\) has an inverse-gamma
prior. All person-specific posterior draws are retained and summarized.

The likelihood includes only adjacent observations at the declared regular time
interval. It never constructs a lag across an observed time gap. This model does
not yet implement latent centering, person-specific residual variances,
between-level predictors, or Mplus-compatible initial-condition integration;
those remain parity gates.

## Planned RDSEM and categorical families

RDSEM applies lagged structure to residual processes rather than silently
equating residual and observed-variable dynamics. Binary and ordinal outcomes
will use explicit latent-response/probit parameterization and threshold
identification. These sections are specifications for planned work, not current
capability claims.
