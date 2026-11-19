#' Register the DSEMr native library
#' @name DSEMr-native
#' @usage NULL
#' @useDynLib DSEMr, .registration = TRUE
NULL

utils::globalVariables(c(
  "wrap__dsem_ar_sufficient",
  "wrap__dsem_grouped_sufficient",
  "wrap__dsem_factor_sufficient",
  "wrap__dsem_rust_var_n1",
  "wrap__dsem_rust_twolevel",
  "wrap__dsem_rust_threelevel",
  "wrap__dsem_rust_crossclassified",
  "wrap__dsem_rust_dynamic_sem"
))
