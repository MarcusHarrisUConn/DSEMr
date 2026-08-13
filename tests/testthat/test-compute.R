test_that("compute configuration is bounded and explicit", {
  sequential <- dsem_compute()
  expect_equal(sequential$workers, 1L)
  expect_equal(sequential$backend, "sequential")
  expect_error(dsem_compute(workers = 2, backend = "sequential"),
               "requires `workers = 1`")
  expect_error(dsem_compute(threads = 0), "positive integer")
})

test_that("stable task seeds depend on task identity", {
  a <- DSEMr:::.dsem_stable_seed(100, "chain-1")
  b <- DSEMr:::.dsem_stable_seed(100, "chain-1")
  c <- DSEMr:::.dsem_stable_seed(100, "chain-2")
  expect_identical(a, b)
  expect_false(identical(a, c))
})

