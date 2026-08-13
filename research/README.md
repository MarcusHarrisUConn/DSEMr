# Local research library

`research/library/` is gitignored. Run `dev/download-research-library.R` to
download the public references listed in `research/manifest.csv`, then verify
their SHA-256 checksums against the manifest. Copyrighted PDFs and the Mplus
guide are never committed to the package repository.

The manifest is evidence metadata, not a redistribution license. Before adding
a source, record why it is needed, which equations/examples it covers, its
public access location, and the parity tests that depend on it.

