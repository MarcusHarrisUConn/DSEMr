test_that("diagnostics preserve chain axes and agree with posterior for skewed draws", {
  set.seed(103)
  draws <- array(exp(rnorm(2400)), c(300, 2, 4),
                 dimnames = list(NULL, c("alpha", "phi"), NULL))
  out <- DSEMr:::.dsem_compute_diagnostics(draws)
  for (j in 1:2) {
    m <- matrix(draws[, j, ], 300, 4)
    expect_equal(out$rhat[j], posterior::rhat(m))
    expect_equal(out$ess_bulk[j], posterior::ess_bulk(m))
    expect_equal(out$ess_tail[j], posterior::ess_tail(m))
    expect_equal(out$mcse[j], posterior::mcse_mean(m))
  }
  draws[, 1, 4] <- draws[, 1, 4] + 10
  expect_gt(DSEMr:::.dsem_compute_diagnostics(draws)$rhat[1], 1.1)
})

test_that("degenerate draws cannot indicate successful convergence", {
  expect_error(DSEMr:::.dsem_compute_diagnostics(matrix(1:4, 2)), "iteration-by")
  out <- DSEMr:::.dsem_compute_diagnostics(array(1, c(20, 1, 2)))
  expect_true(all(is.na(out$rhat)))
  expect_true(all(is.na(out$mcse)))
  expect_identical(out$parameter, "parameter1")
})
