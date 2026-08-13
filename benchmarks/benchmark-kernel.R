library(DSEMr)
set.seed(1)
x <- cbind(1, matrix(rnorm(250000), ncol = 5))
y <- rnorm(nrow(x))
stopifnot(isTRUE(all.equal(
  DSEMr:::.dsem_ar_sufficient_R(x, y),
  DSEMr:::.dsem_ar_sufficient_rust(x, y),
  tolerance = 1e-12
)))
timings <- rbind(
  R = system.time(replicate(20, DSEMr:::.dsem_ar_sufficient_R(x, y))),
  Rust = system.time(replicate(20, DSEMr:::.dsem_ar_sufficient_rust(x, y)))
)
print(timings)

