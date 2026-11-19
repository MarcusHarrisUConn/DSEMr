# Run from the repository root after installing the current package.
# Requires developer packages yaml, digest, and jsonlite. Downloads are never
# performed by tests or installation; see dev/download-research-library.R.
library(DSEMr)
for (package in c("yaml", "digest", "jsonlite")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Install developer package: ", package)
}
fixture <- yaml::read_yaml("inst/parity/mplus-6.26.yml")
files <- c(input = "research/library/ex6.26.inp",
           data = "research/library/ex6.26.dat",
           output = "research/library/ex6.26.html")
hashes <- vapply(files, function(path) digest::digest(file = path, algo = "sha256"), character(1))
for (key in names(files)) {
  stopifnot(identical(hashes[[key]], fixture$files[[paste0(key, "_sha256")]]))
}
dat <- utils::read.table(files[["data"]], col.names = fixture$data$indicators)
stopifnot(nrow(dat) == fixture$data$observations, !anyNA(dat))
dat$time <- seq_len(nrow(dat))
model <- dsem_model(paste(readLines(files[["input"]]), collapse = "\n"),
                    syntax = "mplus", time = "time")
fit <- dsem(model, dat, chains = 4, iter = 6000, warmup = 3000,
             seed = 6262026, engine = "rust", compute = dsem_compute(workers = 1))
comparison <- do.call(rbind, lapply(fixture$parameter_targets, function(target) {
  if (isTRUE(target$fixed)) return(NULL)
  name <- if (target$parameter == "factor_innovation_variance") "factor_variance" else target$parameter
  estimate <- fit$estimates[fit$estimates$parameter == name, ]
  stopifnot(nrow(estimate) == 1)
  mat <- matrix(fit$draws[, name, ], ncol = 4)
  data.frame(parameter = name, dsemr_median = estimate$q50,
    mplus_median = target$estimate, difference = estimate$q50 - target$estimate,
    dsemr_lower = estimate$q2.5, dsemr_upper = estimate$q97.5,
    mplus_lower = target$ci95[1], mplus_upper = target$ci95[2],
    dsemr_mcse_median = as.numeric(posterior::mcse_quantile(mat, probs = 0.5)),
    dsemr_mcse_mean = estimate$mcse_mean, mplus_mcse = NA_real_,
    rhat = estimate$rhat, ess_bulk = estimate$ess_bulk, ess_tail = estimate$ess_tail,
    median_inside_public_interval = estimate$q50 >= target$ci95[1] && estimate$q50 <= target$ci95[2],
    intervals_overlap = estimate$q2.5 <= target$ci95[2] && estimate$q97.5 >= target$ci95[1])
}))
source_files <- sort(c(list.files("R", full.names = TRUE), "src/rust/src/lib.rs",
                      "src/rust/Cargo.lock", "DESCRIPTION", "dev/latent-parity.R"))
record <- list(reference = fixture$id, date_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  status = "provisional-comparison-only", strict_parity_pass = FALSE,
  blockers = c("Mplus improper priors not implemented", "Initial-state semantic match not established",
               "Public Mplus parameter MCSE unavailable", "DIC/pD not implemented", "Independent review pending"),
  source_urls = c(data = "https://www.statmodel.com/usersguide/chap6/ex6.26.dat",
                 output = fixture$source_output), input_hashes = hashes,
  base_commit = system2("git", c("rev-parse", "HEAD"), stdout = TRUE),
  source_sha256 = vapply(source_files, function(path) digest::digest(file = path, algo = "sha256"), character(1)),
  versions = fit$versions,
  rng_kind = fit$versions$rng_kind, master_seed = fit$master_seed, chain_seeds = fit$chain_seeds,
  sampler = fit$sampler, prior = fit$prior, identification = fit$identification,
  compute = fit$compute, elapsed_seconds = fit$elapsed,
  screening_converged = all(is.finite(comparison$rhat) & comparison$rhat < 1.01 &
                             comparison$ess_bulk >= 100 & comparison$ess_tail >= 100),
  comparison = comparison)
dir.create("docs/validation", recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(record, "docs/validation/mplus-6.26-audit.json",
                     pretty = TRUE, auto_unbox = TRUE, digits = 10, na = "null")
print(comparison, row.names = FALSE)
message("Convergence screening: ", record$screening_converged,
        "; strict Mplus parity: FALSE (semantic and evidence gaps).")
