# DSEMr

`DSEMr` is an open-source R package under active development for dynamic
structural equation modeling. It combines a lavaan-inspired model language, a
clean-room Mplus-like compatibility language, a common model representation,
and a Rust numerical backend built with extendr.

> **Current status: experimental foundation release.** The validated estimator
> currently covers N=1 Gaussian AR(1) models with optional contemporaneous
> covariates. Two-level, latent-variable, categorical, RDSEM, and
> cross-classified models can be represented by the compiler in part, but are
> deliberately rejected by `dsem()` until their validation milestones pass.
> This version is not yet a replacement for Mplus and must not be used to claim
> broad Mplus parity.

## Why DSEMr?

Dynamic structural equation models combine time-series relations within people,
multilevel variation across people, and structural equation models for observed
or latent variables. DSEMr aims to make this family of models available in a
transparent, reproducible R workflow suitable for methodological research and
Monte Carlo studies.

## Installation

Source installation currently requires R 4.2 or later and a Rust toolchain with
Cargo. During private development:

```r
# install.packages("remotes")
remotes::install_github("MarcusHarrisUConn/DSEMr", auth_token = Sys.getenv("GITHUB_PAT"))
```

Rust crates are pinned and vendored in the source package, so package
installation does not contact crates.io.

## A first Gaussian AR(1) model

```r
library(DSEMr)

dat <- simulate_dsem(
  n = 200,
  intercept = 0.20,
  ar = 0.55,
  sigma = 0.70,
  seed = 2026
)

model <- dsem_model(
  "y ~ lag(y, 1)",
  syntax = "lavaan",
  time = "time"
)

fit <- dsem(
  model,
  data = dat,
  chains = 4,
  iter = 2000,
  warmup = 1000,
  seed = 4815
)

summary(fit)
coef(fit)
dsem_inspect(fit, "diagnostics")
```

The equivalent supported Mplus-like input compiles to the same internal model:

```r
mplus_model <- dsem_model("
  ANALYSIS: TYPE = BAYES;
  MODEL:
    %WITHIN%
    y ON y&1;
", syntax = "mplus", time = "time")

stopifnot(identical(model$terms, mplus_model$terms))
```

Unsupported or ambiguous Mplus statements produce an error rather than being
silently reinterpreted.

## Parallel and HPC execution

Independent chains use process-level parallelism. Core allocation is detected
with `parallelly::availableCores()`, which respects containers, CRAN checks, and
common schedulers including Slurm, PBS/TORQUE, SGE, and LSF.

```r
compute <- dsem_compute(workers = 4, backend = "psock", threads = 1)

fit <- dsem(
  model,
  dat,
  chains = 4,
  compute = compute,
  seed = 4815
)
```

Every chain and Monte Carlo replication receives a deterministic seed derived
from a master seed and stable task identity. Changing the worker count must not
change the draws for a given task.

## Public API

- `dsem_model()` compiles either syntax into a `DSEMmodel`.
- `dsem()` estimates a validated model and returns a `DSEMfit`.
- `mplus_to_dsem()` translates the supported clean-room syntax subset.
- `simulate_dsem()` generates Gaussian AR(1) data.
- `dsem_monte_carlo()` runs reproducible simulation studies.
- `dsem_compute()` describes local or scheduler-aware resources.
- `dsem_inspect()` retrieves model, draw, diagnostic, compute, and version data.

Standard `summary()`, `coef()`, `vcov()`, `fitted()`, `residuals()`, `predict()`,
and `plot()` methods are provided for fitted models.

## Compatibility roadmap

| Family | Compile | Estimate | Public parity gate |
|---|---:|---:|---|
| N=1 Gaussian AR(1) | Yes | Experimental | Internal recovery tests |
| N=1 multivariate VAR/DSEM | Partial | No | Mplus 6.23–6.28 |
| Two-level continuous DSEM | Partial | No | Mplus 9.30–9.37 |
| Dynamic CFA/SEM | Partial | No | Public latent examples |
| Binary/ordinal DSEM | Family metadata | No | Public categorical examples |
| RDSEM | Partial | No | Published RDSEM examples |
| Cross-classified DSEM | Roadmap | No | Mplus 9.38–9.40 |

The authoritative, test-level matrix is maintained in
[`docs/PARITY.md`](docs/PARITY.md).

## Clean-room policy

DSEMr is an independent implementation based on publicly available papers,
equations, examples, and outputs. Mplus names are used only to describe
compatibility targets. No proprietary source code is used. “Parity” means
compatible model semantics and statistically equivalent results within Monte
Carlo uncertainty, not identical random draws or copied implementation details.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`PROJECT_PLAN.md`](PROJECT_PLAN.md),
and [`VALIDATION.md`](VALIDATION.md). Model-family pull requests must include a
mathematical specification, reference-engine tests, simulation recovery tests,
and an update to the parity matrix.

## License

GPL-3. See [`LICENSE`](LICENSE).

