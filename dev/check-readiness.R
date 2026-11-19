required <- c(
  "DESCRIPTION", "NAMESPACE", "README.md", "PROJECT_PLAN.md",
  "ARCHITECTURE.md", "MATH_SPEC.md", "VALIDATION.md", "inst/AUTHORS",
  "inst/WORDLIST", "inst/CITATION", "docs/PARITY.md",
  "research/manifest.csv", "src/rust/vendor.tar.xz"
)
missing <- required[!file.exists(required)]
if (length(missing)) stop("Missing readiness artifacts: ", paste(missing, collapse = ", "))
manifest <- read.csv("research/manifest.csv", stringsAsFactors = FALSE)
if (anyDuplicated(manifest$key)) stop("Research manifest keys must be unique.")
if (!all(nzchar(manifest$source_url))) stop("Every research source requires a URL.")
message("All DSEMr 0.1.0 CRAN readiness artifacts are present.")
