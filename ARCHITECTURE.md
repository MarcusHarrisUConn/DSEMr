# Architecture decisions

## One compiler representation

Both syntax frontends produce the same term table. Each row records level,
left-hand variable, operator, right-hand variable, lag, random-effect status,
residual-dynamic status, and label. Model metadata separately records family,
cluster/time identifiers, source syntax, a stable model hash, and capability
flags. Estimators consume only this canonical representation.

## Public lavaan integration

DSEMr imports lavaan and follows familiar operators, extractors, and parameter
tables, but it does not fork or depend on unexported lavaan internals. Stable
public lavaan facilities may be used as model families mature. DSEM-specific
lags, residual dynamics, centering, and Bayesian estimation remain owned by
DSEMr.

## Rust boundary

R owns parsing, validation, orchestration, user output, and process management.
Rust owns pure numerical kernels operating on copied native arrays. The R API is
never called from Rust worker threads. The extendr boundary is intentionally
small and covered by exact R-versus-Rust tests.

## Reproducibility and parallelism

Chains and simulation replications are independent tasks. Their seeds are a
stable function of master seed, model hash, task kind, and task index—not worker
number or dispatch order. Process-level parallelism is used for independent
tasks; native threads must be bounded and recorded in the compute manifest.

## Capability gating

Parsing and estimation are separate contracts. The compiler may represent a
future model, but `dsem()` must reject it until the mathematical specification,
reference implementation, recovery tests, public comparison, and diagnostics
are complete. There is no best-effort fallback that changes model meaning.

