# Public compatibility matrix

Status values are `planned`, `compiler-partial`, `implemented-experimental`, or
`validated`. Only `validated` models count toward the CRAN parity gate.

| Reference | Model family | Compiler | Estimator | Evidence | Status |
|---|---|---|---|---|---|
| Internal AR(1) | N=1 Gaussian AR(1) | Yes | R + Rust kernel | Recovery and exact kernel tests | implemented-experimental |
| Internal two-level AR(1) | Gaussian random intercept and AR effect | Yes | R reference + grouped Rust kernel | Fixed recovery grid, gap tests, and exact kernel tests | implemented-experimental |
| Mplus 6.23 | N=1 continuous AR(1) | Partial | No | Official public input/output | planned |
| Mplus 6.24 | N=1 covariate AR(1) | Partial | No | Official public input/output | planned |
| Mplus 6.25–6.28 | N=1 VAR, CFA, IRT, SEM | Partial | No | Official public input/output | planned |
| Mplus 9.30–9.32 | Two-level continuous DSEM | Partial | No | Official public input/output | planned |
| Mplus 9.33–9.37 | Measurement and random-parameter DSEM | Partial | No | Official public input/output | planned |
| Mplus 9.38–9.40 | Cross-classified DSEM | No | No | Official public input/output | planned |
| Asparouhov & Muthén RDSEM | Residual dynamic SEM | Partial | No | Paper and open scripts | planned |

Every row promoted to `validated` must link to a machine-readable parity record
containing source citation, input checksum, DSEMr version/commit, seed, compute
manifest, estimates, MCSEs, decision thresholds, and reviewer sign-off.
