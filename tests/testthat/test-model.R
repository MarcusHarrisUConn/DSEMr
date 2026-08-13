lavaan_ar1 <- "
level: 1
  y ~ lag(y, 1)
"

mplus_ar1 <- "
ANALYSIS: TYPE = BAYES;
MODEL:
  %WITHIN%
  y ON y&1;
"

test_that("dual syntaxes compile to equivalent terms", {
  lavaan_model <- dsem_model(lavaan_ar1, syntax = "lavaan", time = "time")
  mplus_model <- dsem_model(mplus_ar1, syntax = "mplus", time = "time")
  expect_equal(lavaan_model$terms, mplus_model$terms)
  expect_equal(lavaan_model$capabilities, mplus_model$capabilities)
})

test_that("auto syntax detection identifies Mplus sections", {
  model <- dsem_model(mplus_ar1, syntax = "auto")
  expect_equal(model$syntax, "mplus")
})

test_that("ambiguous and unsupported syntax fails loudly", {
  expect_error(dsem_model("MODEL: y XWITH z;", syntax = "mplus"),
               "Unsupported or ambiguous")
  expect_error(dsem_model("y nonsense x", syntax = "lavaan"),
               "Unsupported lavaan-style")
})

test_that("represented but unvalidated model families cannot be fit", {
  latent <- dsem_model("f =~ y1 + y2\nf ~ lag(f, 1)")
  dat <- data.frame(y1 = rnorm(20), y2 = rnorm(20), f = rnorm(20))
  expect_error(dsem(latent, dat, chains = 1, iter = 20, warmup = 10),
               "outside the validated estimator")
})

