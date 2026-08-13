#' Configure DSEMr computation
#'
#' Creates a scheduler-aware compute specification. The default worker count
#' respects process limits, containers, CRAN checks, and common HPC schedulers
#' through [parallelly::availableCores()].
#'
#' @param workers Positive number of R worker processes, or `NULL` to detect.
#' @param threads Positive number of native threads per worker.
#' @param backend `"sequential"` or process-level `"psock"`.
#' @param checkpoint Optional directory for future checkpoint files.
#' @param blas_threads Intended BLAS thread count recorded in the manifest.
#' @return A `DSEMcompute` object.
#' @export
dsem_compute <- function(workers = NULL, threads = 1L,
                         backend = c("sequential", "psock"),
                         checkpoint = NULL, blas_threads = 1L) {
  backend <- match.arg(backend)
  detected <- parallelly::availableCores(omit = 1L)
  if (is.null(workers)) workers <- if (backend == "sequential") 1L else detected
  workers <- as.integer(workers)
  threads <- as.integer(threads)
  blas_threads <- as.integer(blas_threads)
  if (length(workers) != 1L || is.na(workers) || workers < 1L) {
    .dsem_abort("`workers` must be one positive integer.")
  }
  if (length(threads) != 1L || is.na(threads) || threads < 1L) {
    .dsem_abort("`threads` must be one positive integer.")
  }
  if (length(blas_threads) != 1L || is.na(blas_threads) || blas_threads < 1L) {
    .dsem_abort("`blas_threads` must be one positive integer.")
  }
  if (backend == "sequential" && workers != 1L) {
    .dsem_abort("The sequential backend requires `workers = 1`.")
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

#' @export
print.DSEMcompute <- function(x, ...) {
  cat("<DSEMcompute>", x$backend, "\n")
  cat(" workers:", x$workers, " native threads/worker:", x$threads, "\n")
  cat(" scheduler:", x$scheduler, " available cores:", x$detected_cores, "\n")
  if (x$oversubscribed) cat(" warning: requested parallelism may oversubscribe the allocation\n")
  invisible(x)
}

