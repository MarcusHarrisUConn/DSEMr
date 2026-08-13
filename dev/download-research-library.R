manifest <- read.csv("research/manifest.csv", stringsAsFactors = FALSE)
requested <- commandArgs(trailingOnly = TRUE)
if (length(requested)) {
  unknown <- setdiff(requested, manifest$key)
  if (length(unknown)) stop("Unknown research key(s): ", paste(unknown, collapse = ", "))
  manifest <- manifest[manifest$key %in% requested, , drop = FALSE]
}
dir.create("research/library", recursive = TRUE, showWarnings = FALSE)
for (i in seq_len(nrow(manifest))) {
  destination <- file.path("research/library", manifest$local_filename[i])
  message("Downloading ", manifest$key[i], " -> ", destination)
  utils::download.file(manifest$source_url[i], destination, mode = "wb", quiet = FALSE)
}
files <- file.path("research/library", manifest$local_filename)
if (all(file.exists(files))) {
  checksums <- tools::sha256sum(files)
  print(data.frame(key = manifest$key, file = files, sha256 = unname(checksums)))
}
