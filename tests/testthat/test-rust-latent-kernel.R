test_that("latent measurement statistics match item-wise R cross-products", {
  eta <- cbind(
    factor_1 = c(-1, -0.5, 0, 0.5, 1),
    factor_2 = c(0.2, -0.1, 0.4, 0.8, -0.3)
  )
  y <- cbind(
    item_1 = c(0.1, NA, 0.8, 1.3, 1.9),
    item_2 = c(NA, -0.4, 0.2, 0.7, NA),
    item_3 = c(1.0, 1.1, 0.9, NA, 1.4)
  )
  n <- nrow(y)
  items <- ncol(y)
  factors <- ncol(eta)
  q <- factors + 1L

  rust <- DSEMr:::dsem_factor_sufficient(
    as.numeric(y), as.numeric(eta), n, items, factors
  )

  expect_identical(rust$n_obs, as.integer(colSums(!is.na(y))))
  expect_identical(rust$design_cols, q)
  for (item in seq_len(items)) {
    observed <- !is.na(y[, item])
    design <- cbind(1, eta[observed, , drop = FALSE])
    ztz_index <- (item - 1L) * q * q + seq_len(q * q)
    zty_index <- (item - 1L) * q + seq_len(q)

    expect_equal(
      matrix(rust$ztz[ztz_index], nrow = q),
      unname(crossprod(design)),
      tolerance = 1e-14
    )
    expect_equal(
      rust$zty[zty_index],
      as.numeric(crossprod(design, y[observed, item])),
      tolerance = 1e-14
    )
    expect_equal(
      rust$yty[item],
      sum(y[observed, item]^2),
      tolerance = 1e-14
    )
  }
})

test_that("latent measurement kernel rejects invalid dimensions and infinities", {
  expect_error(
    DSEMr:::dsem_factor_sufficient(c(1, 2), c(0), 2L, 1L, 1L),
    "eta dimensions"
  )
  expect_error(
    DSEMr:::dsem_factor_sufficient(c(1, Inf), c(0, 1), 2L, 1L, 1L),
    "not infinite"
  )
})
