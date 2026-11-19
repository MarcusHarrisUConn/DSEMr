#' Configure DSEMr computation
#'
#' Creates a scheduler-aware compute specification. The default worker count
#' respects process limits, containers, CRAN check limits, and common HPC
#' schedulers through [parallelly::availableCores()].
#'
#' @param workers Positive number of R worker processes, or `NULL` to detect.
#' @param threads Positive number of native threads per worker.
#' @param backend `"sequential"` or process-level `"psock"`.
#' @param checkpoint Optional directory for checkpoint files.
#' @param blas_threads Intended BLAS thread count recorded in the manifest.
#' @return A `DSEMcompute` object describing worker, thread, and scheduler
#'   settings.
#' @examples
#' dsem_compute()
#' dsem_compute(workers = 1, backend = "sequential")
#' @export
dsem_compute <- function(workers = NULL, threads = 1L,
                         backend = c("sequential", "psock"),
                         checkpoint = NULL, blas_threads = 1L) {
  backend <- match.arg(backend)
  detected <- as.integer(parallelly::availableCores(omit = 1L))
  chk_limit <- tolower(Sys.getenv("_R_CHECK_LIMIT_CORES_", ""))
  if (chk_limit %in% c("true", "warn")) {
    detected <- min(detected, 2L)
  }
  if (is.null(workers)) workers <- if (backend == "sequential") 1L else max(1L, detected)
  workers <- .dsem_integer(workers, "workers", 1)
  threads <- .dsem_integer(threads, "threads", 1)
  blas_threads <- .dsem_integer(blas_threads, "blas_threads", 1)
  if (backend == "sequential" && workers != 1L) {
    .dsem_abort("The sequential backend requires `workers = 1`.")
  }
  if (!is.null(checkpoint) && (!is.character(checkpoint) || length(checkpoint) != 1L || !nzchar(checkpoint))) {
    .dsem_abort("`checkpoint` must be NULL or a single directory path.")
  }
  scheduler <- .dsem_scheduler()
  out <- list(
    backend = backend, workers = workers, threads = threads,
    blas_threads = blas_threads, detected_cores = detected,
    scheduler = scheduler, checkpoint = checkpoint,
    oversubscribed = workers * threads * blas_threads > max(1L, detected)
  )
  class(out) <- c("DSEMcompute", "list")
  out
}

.dsem_scheduler <- function() {
  env <- Sys.getenv(c("SLURM_JOB_ID", "PBS_JOBID", "JOB_ID", "LSB_JOBID"), unset = "")
  keys <- c("slurm", "pbs", "sge", "lsf")
  hit <- which(nzchar(env))[1L]
  if (is.na(hit)) "local" else keys[hit]
}

#' @exportS3Method base::print
print.DSEMcompute <- function(x, ...) {
  cat("<DSEMcompute>", x$backend, "\n")
  cat(" workers:", x$workers, " native threads/worker:", x$threads, "\n")
  cat(" scheduler:", x$scheduler, " available cores:", x$detected_cores, "\n")
  if (x$oversubscribed) cat(" warning: requested parallelism may oversubscribe the allocation\n")
  invisible(x)
}
