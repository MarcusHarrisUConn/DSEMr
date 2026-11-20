.dsem_parse_mplus <- function(model) {
  raw <- unlist(strsplit(model, "\n", fixed = TRUE), use.names = FALSE)
  raw <- sub("!.*$", "", raw)
  text <- paste(raw, collapse = "\n")
  sec_pat <- "(?:^|(?<=\\s|;))(TITLE|DATA|VARIABLE|DEFINE|ANALYSIS|MODEL\\s+PRIORS|MODEL\\s+CONSTRAINT|MODEL\\s+TEST|MODEL|OUTPUT|PLOT|SAVEDATA|MONTECARLO)\\s*:"
  sections <- gregexpr(sec_pat, text, ignore.case = TRUE, perl = TRUE)[[1L]]
  if (identical(sections, -1L)) {
    if (grepl("[A-Za-z]+\\s*:", text)) {
      bad <- regmatches(text, regexpr("[A-Za-z]+\\s*:", text))
      .dsem_abort("Unsupported Mplus section(s): %s.", sub("\\s*:$", "", bad))
    }
    .dsem_abort("Mplus-like syntax requires named sections.")
  }
  raw_headers <- regmatches(text, list(sections))[[1L]]
  section_names <- toupper(gsub("\\s+", " ", trimws(sub(":$", "", raw_headers))))
  if (anyDuplicated(section_names)) .dsem_abort("Duplicate Mplus sections are unsupported.")
  section_ends <- c(sections[-1L] - 1L, nchar(text))
  section_starts <- sections + attr(sections, "match.length")
  bodies <- as.list(mapply(substr, section_starts, section_ends, MoreArgs = list(x = text), USE.NAMES = FALSE))
  names(bodies) <- section_names
  if (is.null(bodies[["MODEL"]])) .dsem_abort("Mplus-like syntax requires a MODEL section.")

  symbol_order <- character()
  add_symbols <- function(syms) {
    syms <- syms[nzchar(syms)]
    symbol_order <<- unique(c(symbol_order, syms))
  }

  mplus_clusters <- character()
  mplus_categorical <- character()
  mplus_censored <- list()
  mplus_between <- character()
  mplus_between_groups <- character()
  mplus_within <- character()
  mplus_time <- NULL
  mplus_tinterval <- NULL
  mplus_centering <- NULL
  mplus_centered_vars <- character()
  mplus_level_type <- NULL
  mplus_prior_mode <- NULL
  mplus_algorithm <- NULL
  mplus_define_ops <- list()
  mplus_constraints <- character()
  mplus_tests <- character()

  # Parse ANALYSIS section
  if (!is.null(bodies[["ANALYSIS"]])) {
    stmts <- trimws(unlist(strsplit(bodies[["ANALYSIS"]], ";", fixed = TRUE)))
    for (st in stmts[nzchar(stmts)]) {
      if (grepl("^TYPE\\s*(=|IS)\\s*", st, ignore.case = TRUE)) {
        val <- toupper(trimws(sub("^TYPE\\s*(=|IS)\\s*", "", st, ignore.case = TRUE)))
        if (grepl("THREELEVEL", val)) {
          mplus_level_type <- "threelevel"
        } else if (grepl("CROSSCLASSIFIED", val)) {
          mplus_level_type <- "crossclassified"
        } else if (grepl("TWOLEVEL", val)) {
          mplus_level_type <- "twolevel"
        } else if (val %in% c("GENERAL", "RANDOM", "BAYES")) {
          # valid single-level or estimator alias
        } else {
          .dsem_abort("Unsupported ANALYSIS TYPE: `%s`.", st)
        }
      } else if (grepl("^(ESTIMATOR|PROCESSORS|BITERATIONS|FBITERATIONS|CHAINS|THIN|ALGORITHM|POINT|BSEED|BCONVERGENCE|PRIOR|CONVERGENCE|STSCALE|BSCALE)\\s*(=|IS)", st, ignore.case = TRUE)) {
        if (grepl("^PRIOR\\s*(=|IS)\\s*PSEM", st, ignore.case = TRUE)) {
          mplus_prior_mode <- "penalized"
        }
        if (grepl("^ALGORITHM\\s*(=|IS)\\s*", st, ignore.case = TRUE)) {
          mplus_algorithm <- toupper(trimws(sub("^ALGORITHM\\s*(=|IS)\\s*", "", st, ignore.case = TRUE)))
        }
      } else {
        .dsem_abort("Unsupported ANALYSIS statement: `%s`.", st)
      }
    }
  }

  # Parse VARIABLE section
  if (!is.null(bodies[["VARIABLE"]])) {
    stmts <- trimws(unlist(strsplit(bodies[["VARIABLE"]], ";", fixed = TRUE)))
    for (st in stmts[nzchar(stmts)]) {
      st_clean <- .dsem_trim(gsub("\n", " ", st))
      if (grepl("^NAMES\\s*(=|ARE|IS)\\s*", st_clean, ignore.case = TRUE)) {
        rhs <- sub("^NAMES\\s*(=|ARE|IS)\\s*", "", st_clean, ignore.case = TRUE)
        toks <- unlist(lapply(strsplit(trimws(rhs), "\\s+")[[1L]], .dsem_expand_mplus_range, symbols = symbol_order))
        add_symbols(toks)
      } else if (grepl("^USEVARIABLES\\s*(=|ARE|IS)\\s*", st_clean, ignore.case = TRUE)) {
        rhs <- sub("^USEVARIABLES\\s*(=|ARE|IS)\\s*", "", st_clean, ignore.case = TRUE)
        toks <- unlist(lapply(strsplit(trimws(rhs), "\\s+")[[1L]], .dsem_expand_mplus_range, symbols = symbol_order))
        add_symbols(toks)
      } else if (grepl("^CLUSTER\\s*(=|ARE|IS)\\s*", st_clean, ignore.case = TRUE)) {
        rhs <- sub("^CLUSTER\\s*(=|ARE|IS)\\s*", "", st_clean, ignore.case = TRUE)
        mplus_clusters <- strsplit(trimws(rhs), "\\s+")[[1L]]
      } else if (grepl("^CATEGORICAL\\s*(=|ARE|IS)\\s*", st_clean, ignore.case = TRUE)) {
        rhs <- sub("^CATEGORICAL\\s*(=|ARE|IS)\\s*", "", st_clean, ignore.case = TRUE)
        mplus_categorical <- unlist(lapply(strsplit(trimws(rhs), "\\s+")[[1L]], .dsem_expand_mplus_range, symbols = symbol_order))
      } else if (grepl("^CENSORED\\s*(=|ARE|IS)\\s*", st_clean, ignore.case = TRUE)) {
        rhs <- sub("^CENSORED\\s*(=|ARE|IS)\\s*", "", st_clean, ignore.case = TRUE)
        # Parse pairs like `y (b)`, `y1-y2 (a)`, `y (ab)` / `y (bi)` / `y (ai)`
        cm <- gregexpr("([.A-Za-z][.A-Za-z0-9_-]*(?:\\s+[.A-Za-z][.A-Za-z0-9_-]*)*)\\s*\\(\\s*([^)]+)\\s*\\)", rhs, perl = TRUE)[[1L]]
        if (!identical(cm, -1L)) {
          segs <- regmatches(rhs, list(cm))[[1L]]
          for (sg in segs) {
            pm <- regexec("^(.+?)\\s*\\(\\s*([^)]+)\\s*\\)$", trimws(sg))
            cap <- regmatches(trimws(sg), pm)[[1L]]
            if (length(cap) >= 3L) {
              vtoks <- unlist(lapply(strsplit(trimws(cap[2L]), "\\s+")[[1L]], .dsem_expand_mplus_range, symbols = symbol_order))
              flag <- tolower(gsub("\\s+", "", cap[3L]))
              dir <- if (grepl("b", flag) && grepl("a", flag)) "both" else if (grepl("a", flag)) "above" else "below"
              for (vn in vtoks) mplus_censored[[vn]] <- dir
            }
          }
        } else {
          vtoks <- unlist(lapply(strsplit(trimws(rhs), "\\s+")[[1L]], .dsem_expand_mplus_range, symbols = symbol_order))
          for (vn in vtoks) mplus_censored[[vn]] <- "below"
        }
      } else if (grepl("^BETWEEN\\s*(=|ARE|IS)\\s*", st_clean, ignore.case = TRUE)) {
        raw_rhs <- sub("^BETWEEN\\s*(=|ARE|IS)\\s*", "", st_clean, ignore.case = TRUE)
        gm <- gregexpr("\\(\\s*([.A-Za-z][.A-Za-z0-9_]*)\\s*\\)", raw_rhs, perl = TRUE)[[1L]]
        if (!identical(gm, -1L)) {
          gstrs <- regmatches(raw_rhs, list(gm))[[1L]]
          mplus_between_groups <- unique(c(mplus_between_groups, trimws(gsub("[()]", "", gstrs))))
        }
        rhs <- gsub("\\([^)]*\\)", " ", raw_rhs)
        mplus_between <- unique(c(
          mplus_between,
          unlist(lapply(strsplit(trimws(rhs), "\\s+")[[1L]], .dsem_expand_mplus_range, symbols = symbol_order))
        ))
      } else if (grepl("^WITHIN\\s*(=|ARE|IS)\\s*", st_clean, ignore.case = TRUE)) {
        rhs <- gsub("\\([^)]*\\)", " ", sub("^WITHIN\\s*(=|ARE|IS)\\s*", "", st_clean, ignore.case = TRUE))
        mplus_within <- unique(c(
          mplus_within,
          unlist(lapply(strsplit(trimws(rhs), "\\s+")[[1L]], .dsem_expand_mplus_range, symbols = symbol_order))
        ))
      } else if (grepl("^TINTERVAL\\s*(=|IS)\\s*", st_clean, ignore.case = TRUE)) {
        rhs <- sub("^TINTERVAL\\s*(=|IS)\\s*", "", st_clean, ignore.case = TRUE)
        m <- regexec("^([.A-Za-z][.A-Za-z0-9_]*)\\s*\\(\\s*([0-9.]+)\\s*\\)$", trimws(rhs))
        cap <- regmatches(trimws(rhs), m)[[1L]]
        if (!length(cap)) .dsem_abort("Invalid TINTERVAL specification: `%s`.", st_clean)
        mplus_time <- cap[2L]
        mplus_tinterval <- as.numeric(cap[3L])
      } else if (grepl("^(LAGGED|MISSING|IDVARIABLE|AUXILIARY)\\s*(=|ARE|IS)", st_clean, ignore.case = TRUE)) {
        # Valid metadata variable declarations
      } else {
        .dsem_abort("Unsupported VARIABLE statement: `%s`.", st_clean)
      }
    }
  }

  # Parse DEFINE section (CENTER, STANDARDIZE, and variable creation/transformation)
  if (!is.null(bodies[["DEFINE"]])) {
    stmts <- trimws(unlist(strsplit(bodies[["DEFINE"]], ";", fixed = TRUE)))
    for (st in stmts[nzchar(stmts)]) {
      st_clean <- .dsem_trim(gsub("\n", " ", st))
      if (grepl("^CENTER\\s+.+\\(\\s*GROUPMEAN(?:\\s+[.A-Za-z][.A-Za-z0-9_]*)?\\s*\\)$", st_clean, ignore.case = TRUE)) {
        mplus_centering <- "groupmean"
        var_part <- trimws(sub("\\([^)]*\\)$", "", sub("^CENTER\\s+", "", st_clean, ignore.case = TRUE)))
        toks <- unlist(lapply(strsplit(var_part, "\\s+")[[1L]], .dsem_expand_mplus_range, symbols = symbol_order))
        mplus_centered_vars <- unique(c(mplus_centered_vars, toks))
      } else if (grepl("^CENTER\\s+.+\\(\\s*GRANDMEAN\\s*\\)$", st_clean, ignore.case = TRUE)) {
        mplus_centering <- "grandmean"
        var_part <- trimws(sub("\\([^)]*\\)$", "", sub("^CENTER\\s+", "", st_clean, ignore.case = TRUE)))
        toks <- unlist(lapply(strsplit(var_part, "\\s+")[[1L]], .dsem_expand_mplus_range, symbols = symbol_order))
        mplus_centered_vars <- unique(c(mplus_centered_vars, toks))
      } else if (grepl("^STANDARDIZE\\s+", st_clean, ignore.case = TRUE)) {
        var_part <- trimws(sub("^STANDARDIZE\\s+", "", st_clean, ignore.case = TRUE))
        toks <- unlist(lapply(strsplit(var_part, "\\s+")[[1L]], .dsem_expand_mplus_range, symbols = symbol_order))
        mplus_define_ops[[length(mplus_define_ops) + 1L]] <- list(type = "standardize", vars = toks)
      } else if (grepl("^[.A-Za-z][.A-Za-z0-9_]*\\s*=\\s*.+$", st_clean)) {
        eq_pos <- regexpr("=", st_clean, fixed = TRUE)
        lhs_v <- trimws(substr(st_clean, 1L, eq_pos - 1L))
        rhs_e <- trimws(substr(st_clean, eq_pos + 1L, nchar(st_clean)))
        add_symbols(lhs_v)
        mplus_define_ops[[length(mplus_define_ops) + 1L]] <- list(
          type = "assign", lhs = lhs_v, expr = rhs_e
        )
      } else {
        .dsem_abort("Unsupported DEFINE statement: `%s`.", st_clean)
      }
    }
  }

  # Resolve cluster mapping for TWOLEVEL, THREELEVEL (CLUSTER = L3 L2), CROSSCLASSIFIED (CLUSTER = subject time or time subject)
  mplus_id <- NULL
  mplus_time_cluster <- NULL
  if (length(mplus_clusters) == 1L) {
    mplus_id <- mplus_clusters[1L]
    if (is.null(mplus_level_type)) mplus_level_type <- "twolevel"
  } else if (length(mplus_clusters) == 2L) {
    if (identical(mplus_level_type, "crossclassified")) {
      c1 <- mplus_clusters[1L]; c2 <- mplus_clusters[2L]
      time_like <- c("time", "occasion", "occ", "wave", "t", "day")
      c1_is_time <- tolower(c1) %in% time_like || (!is.null(mplus_time) && tolower(c1) == tolower(mplus_time))
      c2_is_time <- tolower(c2) %in% time_like || (!is.null(mplus_time) && tolower(c2) == tolower(mplus_time))
      if (c2_is_time && !c1_is_time) {
        mplus_id <- c1
        mplus_time_cluster <- c2
      } else if (c1_is_time && !c2_is_time) {
        mplus_time_cluster <- c1
        mplus_id <- c2
      } else if (tolower(c1) %in% tolower(mplus_between_groups) && !tolower(c2) %in% tolower(mplus_between_groups)) {
        mplus_id <- c1
        mplus_time_cluster <- c2
      } else {
        mplus_time_cluster <- c1
        mplus_id <- c2
      }
    } else {
      mplus_level_type <- "threelevel"
      # Three-level syntax lists highest level first: CLUSTER = level3 level2
      mplus_id <- c(mplus_clusters[2L], mplus_clusters[1L])
    }
  }

  resolve_between_header <- function(tag) {
    tag <- tolower(trimws(tag))
    if (!nzchar(tag)) return("between")
    if (identical(mplus_level_type, "threelevel")) {
      if (tag %in% c("l2", "level2", tolower(mplus_id[1L]))) return("between_2")
      if (tag %in% c("l3", "level3", tolower(mplus_id[2L]))) return("between_3")
    }
    if (identical(mplus_level_type, "crossclassified")) {
      if (!is.null(mplus_time_cluster) && tag == tolower(mplus_time_cluster)) return("time")
      if (tag == "time") return("time")
      return("between")
    }
    if (tag %in% c("l3", "level3")) return("between_3")
    if (tag %in% c("l2", "level2")) return("between_2")
    if (tag == "time") return("time")
    "between"
  }

  # Parse MODEL section
  model_body <- gsub("\n", " ", bodies[["MODEL"]])
  chunks <- unlist(strsplit(model_body, ";", fixed = TRUE), use.names = FALSE)
  terms <- .dsem_empty_terms()
  meta <- .dsem_empty_meta()
  measurement_lags <- data.frame(factor = character(), lag = integer(), stringsAsFactors = FALSE)
  level <- "within"

  for (chunk in chunks) {
    chunk <- .dsem_trim(chunk)
    if (!nzchar(chunk)) next
    # Extract any leading %WITHIN% or %BETWEEN [name]% markers sequentially
    repeat {
      if (grepl("^%WITHIN%", chunk, ignore.case = TRUE)) {
        level <- "within"
        chunk <- .dsem_trim(sub("^%WITHIN%", "", chunk, ignore.case = TRUE))
      } else if (grepl("^%BETWEEN(\\s+[A-Za-z0-9_.]+)?%", chunk, ignore.case = TRUE)) {
        m <- regexec("^%BETWEEN(?:\\s+([A-Za-z0-9_.]+))?%", chunk, ignore.case = TRUE)
        cap <- regmatches(chunk, m)[[1L]]
        level <- resolve_between_header(if (length(cap) >= 2L) cap[2L] else "")
        chunk <- .dsem_trim(sub("^%BETWEEN(?:\\s+[A-Za-z0-9_.]+)?%", "", chunk, ignore.case = TRUE))
      } else {
        break
      }
    }
    if (!nzchar(chunk)) next

    # Extract trailing parenthesized modifiers: (&1), (1-4), (&1 1-4), (a)
    eq_labels <- character()
    lag_decl <- integer()
    paren_matches <- gregexpr("\\([^)]+\\)", chunk)[[1L]]
    if (!identical(paren_matches, -1L)) {
      p_strs <- regmatches(chunk, list(paren_matches))[[1L]]
      for (ps in p_strs) {
        inner <- trimws(gsub("^\\(|\\)$", "", ps))
        ptoks <- strsplit(inner, "\\s+")[[1L]]
        for (pt in ptoks) {
          if (grepl("^&[0-9]+$", pt)) {
            lag_decl <- c(lag_decl, as.integer(sub("^&", "", pt)))
          } else if (grepl("^[0-9]+-[0-9]+$", pt)) {
            bounds <- as.integer(strsplit(pt, "-", fixed = TRUE)[[1L]])
            eq_labels <- c(eq_labels, paste0("eq_", seq.int(bounds[1L], bounds[2L])))
          } else {
            eq_labels <- c(eq_labels, pt)
          }
        }
      }
      chunk <- .dsem_trim(gsub("\\([^)]+\\)", "", chunk))
    }

    # Case 1: Bracketed intercepts / means / thresholds `[var@0]`, `[u1$1]`
    if (grepl("^\\[.+\\]$", chunk)) {
      inner <- trimws(gsub("^\\[|\\]$", "", chunk))
      toks <- strsplit(inner, "\\s+")[[1L]]
      for (tk in toks) {
        parsed_tok <- .dsem_parse_mplus_token(tk, symbol_order)
        for (vname in parsed_tok$names) {
          if (grepl("\\$", vname)) {
            sp <- strsplit(vname, "\\$")[[1L]]
            terms <- .dsem_add_term(terms, level, sp[1L], "|", paste0("t", sp[2L]))
            meta <- .dsem_add_meta(meta, parsed_tok$fixed, parsed_tok$free)
          } else {
            terms <- .dsem_add_term(terms, level, vname, "~", "1")
            meta <- .dsem_add_meta(meta, parsed_tok$fixed, parsed_tok$free)
          }
        }
      }
      next
    }

    # Case 2: Random log residual variance `logv | y` (pipe without ON/BY)
    if (grepl("\\|", chunk) && !grepl("\\b(ON|BY)\\b", chunk, ignore.case = TRUE)) {
      bits <- strsplit(chunk, "\\|")[[1L]]
      if (length(bits) != 2L) .dsem_abort("Ambiguous Mplus random-variance statement: `%s`.", chunk)
      lv_names <- .dsem_expand_mplus_list(trimws(bits[1L]), symbol_order)
      y_names <- .dsem_expand_mplus_list(trimws(bits[2L]), symbol_order)
      if (length(lv_names) != length(y_names)) {
        .dsem_abort("Random variance declaration `%s` must pair names one-to-one.", chunk)
      }
      add_symbols(lv_names)
      for (k in seq_along(y_names)) {
        terms <- .dsem_add_term(terms, level, y_names[k], "~~", y_names[k],
                                random = TRUE, label = lv_names[k])
        meta <- .dsem_add_meta(meta, NA_real_, TRUE, lv_names[k], is_logv = TRUE)
      }
      next
    }

    # Case 3: Regression `ON` (including `s | y ON y&1`, `y^ ON y^1`, `y1-y4 ON f&1`)
    if (grepl("\\bON\\b", chunk, ignore.case = TRUE)) {
      bits <- strsplit(chunk, "(?i)\\s+ON\\s+", perl = TRUE)[[1L]]
      if (length(bits) != 2L) .dsem_abort("Ambiguous Mplus ON statement: `%s`.", chunk)
      lhs_seg <- trimws(bits[1L])
      rhs_seg <- trimws(bits[2L])
      slope_names <- character()
      random <- FALSE
      if (grepl("\\|", lhs_seg)) {
        pbits <- strsplit(lhs_seg, "\\|")[[1L]]
        slope_names <- .dsem_expand_mplus_list(trimws(pbits[1L]), symbol_order)
        add_symbols(slope_names)
        lhs_seg <- trimws(pbits[2L])
        random <- TRUE
      }
      lhs_tokens <- .dsem_expand_mplus_list(lhs_seg, symbol_order)
      # Preserve `@val` and `*` modifiers on RHS tokens before `.dsem_parse_mplus_rhs_lag()`
      rhs_raw_toks <- strsplit(trimws(rhs_seg), "\\s+")[[1L]]
      rhs_tokens <- unlist(lapply(rhs_raw_toks, function(tk) {
        mod_m <- regexpr("(\\*|@-?[0-9]*\\.?[0-9]+)$", tk)
        if (mod_m > 0L) {
          base_tk <- substr(tk, 1L, mod_m - 1L)
          suf_tk <- substr(tk, mod_m, nchar(tk))
          paste0(.dsem_expand_mplus_range(base_tk, symbol_order), suf_tk)
        } else {
          .dsem_expand_mplus_range(tk, symbol_order)
        }
      }), use.names = FALSE)

      pair_idx <- 0L
      for (lhs_tk in lhs_tokens) {
        lhs_res <- grepl("\\^$", lhs_tk)
        lhs_clean <- sub("\\^$", "", lhs_tk)
        for (rhs_tk in rhs_tokens) {
          pair_idx <- pair_idx + 1L
          pt <- .dsem_parse_mplus_rhs_lag(rhs_tk)
          sname <- if (length(slope_names)) {
            slope_names[min(pair_idx, length(slope_names))]
          } else ""
          eq_l <- if (length(eq_labels)) eq_labels[min(pair_idx, length(eq_labels))] else ""
          lbl <- if (nzchar(sname)) sname else eq_l
          terms <- .dsem_add_term(terms, level, lhs_clean, "~", pt$name,
                                  lag = pt$lag, random = random,
                                  residual = lhs_res || pt$residual, label = lbl)
          meta <- .dsem_add_meta(meta, pt$fixed, pt$free, sname, FALSE, eq_l)
        }
      }
      next
    }

    # Case 4: Factor measurement `BY` (`f BY y1-y4*(&1 1-4)`, `f BY y@1(&1)`, `s1-s3 | f BY y1-y3 (&1)`)
    if (grepl("\\bBY\\b", chunk, ignore.case = TRUE)) {
      bits <- strsplit(chunk, "(?i)\\s+BY\\s+", perl = TRUE)[[1L]]
      if (length(bits) != 2L) .dsem_abort("Ambiguous BY statement: `%s`.", chunk)
      lhs_seg <- trimws(bits[1L])
      loading_names <- character()
      rand_loading <- FALSE
      if (grepl("\\|", lhs_seg)) {
        pbits <- strsplit(lhs_seg, "\\|")[[1L]]
        loading_names <- .dsem_expand_mplus_list(trimws(pbits[1L]), symbol_order)
        add_symbols(loading_names)
        lhs_seg <- trimws(pbits[2L])
        rand_loading <- TRUE
      }
      fac_name <- lhs_seg
      add_symbols(fac_name)
      if (length(lag_decl)) {
        for (lg in lag_decl) {
          measurement_lags <- rbind(measurement_lags, data.frame(
            factor = fac_name, lag = as.integer(lg), stringsAsFactors = FALSE
          ))
        }
      }
      raw_rhs <- strsplit(trimws(bits[2L]), "\\s+")[[1L]]
      item_idx <- 0L
      for (rt in raw_rhs) {
        pt <- .dsem_parse_mplus_token(rt, symbol_order)
        add_symbols(pt$names)
        for (iname in pt$names) {
          item_idx <- item_idx + 1L
          fval <- pt$fixed
          ffree <- pt$free
          if (!rand_loading && item_idx == 1L && !pt$explicit_free && is.na(fval)) {
            fval <- 1.0
            ffree <- FALSE
          }
          lname <- if (length(loading_names)) {
            loading_names[min(item_idx, length(loading_names))]
          } else ""
          eq_l <- if (length(eq_labels)) eq_labels[min(item_idx, length(eq_labels))] else ""
          lbl <- if (nzchar(lname)) lname else eq_l
          terms <- .dsem_add_term(terms, level, fac_name, "=~", iname,
                                  random = rand_loading, label = lbl)
          meta <- .dsem_add_meta(meta, fval, ffree, lname, FALSE, eq_l)
        }
      }
      next
    }

    # Case 5: Covariance `WITH` (`f2 WITH f1`, `y s logv WITH y s logv`, `sy-logv y WITH sy-logv y`)
    if (grepl("\\bWITH\\b", chunk, ignore.case = TRUE)) {
      bits <- strsplit(chunk, "(?i)\\s+WITH\\s+", perl = TRUE)[[1L]]
      if (length(bits) != 2L) .dsem_abort("Ambiguous Mplus WITH statement: `%s`.", chunk)
      lhs_list <- .dsem_expand_mplus_list(trimws(bits[1L]), symbol_order)
      rhs_tokens_raw <- strsplit(trimws(bits[2L]), "\\s+")[[1L]]
      rhs_parsed <- lapply(rhs_tokens_raw, .dsem_parse_mplus_token, symbols = symbol_order)
      rhs_list <- unlist(lapply(rhs_parsed, `[[`, "names"), use.names = FALSE)
      fval <- rhs_parsed[[length(rhs_parsed)]]$fixed
      ffree <- rhs_parsed[[length(rhs_parsed)]]$free

      seen_pairs <- character()
      for (a in lhs_list) {
        for (b in rhs_list) {
          if (a == b) {
            # Mplus all-with-all block `a b WITH a b` includes variances implicitly; skip self-duplicate if already added
            key <- paste0(a, "~~", a)
            if (!key %in% seen_pairs) {
              seen_pairs <- c(seen_pairs, key)
              terms <- .dsem_add_term(terms, level, a, "~~", a, random = level != "within")
              meta <- .dsem_add_meta(meta, fval, ffree)
            }
          } else {
            pair_sorted <- sort(c(a, b))
            key <- paste(pair_sorted, collapse = "~~")
            if (!key %in% seen_pairs) {
              seen_pairs <- c(seen_pairs, key)
              terms <- .dsem_add_term(terms, level, a, "~~", b)
              meta <- .dsem_add_meta(meta, fval, ffree)
            }
          }
        }
      }
      next
    }

    # Reject unsupported Mplus MODEL verbs explicitly before standalone variance parsing
    if (grepl("\\b(XWITH|PON|PWITH|AT)\\b", chunk, ignore.case = TRUE)) {
      .dsem_abort("Unsupported or ambiguous Mplus MODEL statement: `%s`.", chunk)
    }

    # Case 6: Standalone variance / fixed variance list (`f@1`, `fb1-fb2@1`, `y@.01`, `y s logv`)
    raw_toks <- strsplit(chunk, "\\s+")[[1L]]
    all_valid <- all(vapply(raw_toks, function(tk) {
      grepl("^[.A-Za-z][.A-Za-z0-9_]*(-[.A-Za-z][.A-Za-z0-9_]*)?(\\*|@-?[0-9]*\\.?[0-9]+)?$", tk)
    }, logical(1)))
    if (all_valid) {
      for (tk in raw_toks) {
        pt <- .dsem_parse_mplus_token(tk, symbol_order)
        for (vname in pt$names) {
          terms <- .dsem_add_term(terms, level, vname, "~~", vname,
                                  random = level != "within")
          meta <- .dsem_add_meta(meta, pt$fixed, pt$free)
        }
      }
      next
    }

    .dsem_abort("Unsupported or ambiguous Mplus MODEL statement: `%s`.", chunk)
  }

  if (!nrow(terms)) .dsem_abort("No supported MODEL terms were found.")
  if (identical(mplus_level_type, "threelevel")) {
    terms$level[terms$level == "between"] <- "between_2"
  }

  # Parse MODEL CONSTRAINT (NEW(...) definitions & derived parameter equations)
  if (!is.null(bodies[["MODEL CONSTRAINT"]])) {
    stmts <- trimws(unlist(strsplit(bodies[["MODEL CONSTRAINT"]], ";", fixed = TRUE)))
    new_vars <- character()
    for (st in stmts[nzchar(stmts)]) {
      st_clean <- .dsem_trim(gsub("\n", " ", st))
      if (grepl("^NEW\\s*\\(", st_clean, ignore.case = TRUE)) {
        inner <- sub("^NEW\\s*\\(([^)]*)\\).*$", "\\1", st_clean, ignore.case = TRUE)
        new_vars <- c(new_vars, trimws(unlist(strsplit(inner, "[\\s,]+", perl = TRUE))))
        next
      }
      if (grepl("=", st_clean, fixed = TRUE)) {
        eq_pos <- regexpr("=", st_clean, fixed = TRUE)
        lhs <- trimws(substr(st_clean, 1L, eq_pos - 1L))
        rhs <- gsub("**", "^", trimws(substr(st_clean, eq_pos + 1L, nchar(st_clean))), fixed = TRUE)
        if (nzchar(lhs) && nzchar(rhs)) {
          if (lhs %in% new_vars || grepl("^[A-Za-z_.][A-Za-z0-9_.]*$", lhs)) {
            entry <- stats::setNames(rhs, lhs)
            mplus_constraints <- c(mplus_constraints, entry)
          } else {
            mplus_constraints <- c(mplus_constraints, paste0("(", lhs, ") - (", rhs, ")"))
          }
        }
      }
    }
  }
  # Parse MODEL TEST (joint Bayesian Wald test constraints)
  if (!is.null(bodies[["MODEL TEST"]])) {
    stmts <- trimws(unlist(strsplit(bodies[["MODEL TEST"]], ";", fixed = TRUE)))
    for (st in stmts[nzchar(stmts)]) {
      st_clean <- gsub("**", "^", .dsem_trim(gsub("\n", " ", st)), fixed = TRUE)
      if (grepl("=", st_clean, fixed = TRUE)) {
        mplus_tests <- c(mplus_tests, st_clean)
      }
    }
  }

  list(
    terms = terms,
    term_meta = meta,
    warnings = character(),
    sections = bodies,
    measurement_lags = measurement_lags,
    mplus_id = mplus_id,
    mplus_time = mplus_time,
    mplus_time_cluster = mplus_time_cluster,
    mplus_tinterval = mplus_tinterval,
    mplus_categorical = mplus_categorical,
    mplus_censored = mplus_censored,
    mplus_between = mplus_between,
    mplus_within = mplus_within,
    mplus_centering = mplus_centering,
    mplus_centered_vars = if (length(mplus_centered_vars)) mplus_centered_vars else NULL,
    mplus_level_type = mplus_level_type,
    mplus_prior_mode = mplus_prior_mode,
    mplus_algorithm = mplus_algorithm,
    mplus_define_ops = mplus_define_ops,
    mplus_constraints = mplus_constraints,
    mplus_tests = mplus_tests
  )
}

.dsem_expand_mplus_range <- function(token, symbols = character()) {
  if (!grepl("-", token, fixed = TRUE)) return(token)
  # First check numeric-suffix range: e.g. y1-y4, y11-y14, fb1-fb2
  matched <- regexec("^([A-Za-z_.]+?)([0-9]+)-([A-Za-z_.]+?)([0-9]+)$", token)
  captures <- regmatches(token, matched)[[1L]]
  if (length(captures) && captures[2L] == captures[4L]) {
    from <- as.integer(captures[3L]); to <- as.integer(captures[5L])
    if (to >= from) {
      # Check if symbol table already has both endpoints in non-sequential order (e.g. s11-s21 in ex9.36!)
      lhs_sym <- paste0(captures[2L], captures[3L])
      rhs_sym <- paste0(captures[4L], captures[5L])
      if (lhs_sym %in% symbols && rhs_sym %in% symbols) {
        i1 <- match(lhs_sym, symbols); i2 <- match(rhs_sym, symbols)
        if (i1 <= i2) return(symbols[i1:i2])
      }
      return(paste0(captures[2L], seq.int(from, to)))
    }
    .dsem_abort("Descending Mplus-style ranges are unsupported: `%s`.", token)
  }
  # Second check declaration-order symbol range: e.g. sy-logv in ex9.37!
  parts <- strsplit(token, "-", fixed = TRUE)[[1L]]
  if (length(parts) == 2L && parts[1L] %in% symbols && parts[2L] %in% symbols) {
    i1 <- match(parts[1L], symbols); i2 <- match(parts[2L], symbols)
    if (i1 <= i2) return(symbols[i1:i2])
  }
  .dsem_abort("Mplus-style ranges must share a variable prefix or appear in declared order: `%s`.", token)
}

.dsem_expand_mplus_list <- function(text, symbols = character()) {
  if (!nzchar(trimws(text))) return(character())
  toks <- strsplit(trimws(text), "\\s+")[[1L]]
  unlist(lapply(toks, function(tk) {
    pt <- .dsem_parse_mplus_token(tk, symbols)
    pt$names
  }), use.names = FALSE)
}

.dsem_parse_mplus_token <- function(token, symbols = character()) {
  explicit_free <- FALSE
  fixed <- NA_real_
  free <- TRUE
  work <- token
  if (grepl("\\*$", work)) {
    explicit_free <- TRUE
    free <- TRUE
    work <- sub("\\*$", "", work)
  } else if (grepl("@-?[0-9]*\\.?[0-9]+$", work)) {
    at_pos <- regexpr("@-?[0-9]*\\.?[0-9]+$", work)
    fixed <- as.numeric(substr(work, at_pos + 1L, nchar(work)))
    free <- FALSE
    work <- substr(work, 1L, at_pos - 1L)
  }
  names <- .dsem_expand_mplus_range(work, symbols)
  list(names = names, fixed = fixed, free = free, explicit_free = explicit_free)
}

.dsem_parse_mplus_rhs_lag <- function(token) {
  fixed <- NA_real_
  free <- TRUE
  work <- token
  if (grepl("\\*$", work)) {
    work <- sub("\\*$", "", work)
  } else if (grepl("@-?[0-9]*\\.?[0-9]+$", work)) {
    at_pos <- regexpr("@-?[0-9]*\\.?[0-9]+$", work)
    fixed <- as.numeric(substr(work, at_pos + 1L, nchar(work)))
    free <- FALSE
    work <- substr(work, 1L, at_pos - 1L)
  }
  lag <- 0L
  residual <- FALSE
  if (grepl("\\^[0-9]+$", work)) {
    lag <- as.integer(sub(".*\\^", "", work))
    work <- sub("\\^[0-9]+$", "", work)
    residual <- TRUE
  } else if (grepl("&[0-9]+$", work)) {
    lag <- as.integer(sub(".*&", "", work))
    work <- sub("&[0-9]+$", "", work)
    if (grepl("\\^$", work)) {
      residual <- TRUE
      work <- sub("\\^$", "", work)
    }
  } else if (grepl("\\^$", work)) {
    residual <- TRUE
    work <- sub("\\^$", "", work)
  }
  list(name = work, lag = lag, residual = residual, fixed = fixed, free = free)
}
