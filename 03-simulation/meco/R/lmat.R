# Work with LMAT count data
# Includes some ALDEx2 helpers

require(tidyverse)
require(reshape2)
require(ALDEx2)
require(furrr)
require(assertthat)

#' Selects read counts for microbial taxa only
#'
#' @param lmat A tibble containing LMAT read counts, scores, and lineages.
#'             For example \code{lmat <- read_tsv("lmat_summary.txt")}
#'
keep_lmat_microbes <- function(lmat) {
  lmat %>%
    filter(!(kingdom %in% c("Metazoa", "Viridiplantae")) | is.na(kingdom)) %>%
    filter(!grepl("synthetic", species))
}

#' Sums read counts at the given rank
#'
#' @param lmat A tibble containing LMAT read counts, scores, and lineages
#' @param tax_rank genus, species, ... as bare words
#' @param keep_unknowns Keeps NA taxa and set taxon name to "UNKNOWN". Default, FALSE.
#'
#' For each sample, read counts for missing taxa that are seen in at least one
#' other sample are set to 0.
sum_lmat_at_rank <- function(lmat, tax_rank = genus, keep_unknowns = FALSE) {
  tax_rank <- enquo(tax_rank) # Hadley's dark magic, see vignette("programming")
  lmat_sum <- lmat %>%
    group_by(sample, !!tax_rank) %>%
    summarize(
      read_count = sum(read_count, na.rm = TRUE),
      avg_read_score = sum(total_read_score) / read_count
    ) %>%
    ungroup() %>%
    complete(sample, !!tax_rank, fill = list(read_count = 0.0, avg_read_score = -Inf))
  if (keep_unknowns) {
    lmat_sum <- lmat_sum %>%
      replace_na(list("UNKNOWN") %>% set_names(quo_name(tax_rank)))
  } else {
    lmat_sum <- lmat_sum %>% filter(!is.na(!!tax_rank))
  }
  lmat_sum
}

#' Calls aldex.clr on all samples in a long-form nested table
#'
#' @param d A long-form dataframe with samples, features, counts, [grouping variable | covariates]
#' @param formula A formula as if you were to call aldex.clr(conds = model.matrix(formula))
#' @param ... Additional aldex.clr parameters
#' @param feature Column name in `d` with feature names
#' @param sample Column name in `d` with sample names
#' @param value.var Column name in `d` with read counts
#' @param fill Value to fill for missing features
aldex_clr <- function(d, formula, ..., feature = "genus",
                      sample = "sample", value.var = "read_count", fill = 0) {
  message("aldex.glm only works with denom = 'all' and conds = model.matrix()")
  message("Working around aldex.glm requirements")

  covarterms <- attr(terms(formula), "term.labels")
  if (length(covarterms) > 1) warning("Workaround supports only one term")
  covariates <- distinct(d, !!sym(sample), !!!syms(covarterms))
  conditions <- transmute(covariates, !!sym(sample), paste(!!!syms(covarterms))) %>% deframe()

  fmla_data <- reformulate(sample, feature)
  df_tidy <-
    reshape2::dcast(d, fmla_data, value.var = value.var, fill = fill) %>%
    tibble::column_to_rownames(feature)
  df_tidy <- df_tidy[, names(conditions)] # reorder columns

  assertthat::assert_that(all(colnames(df_tidy) == names(conditions)))

  # mm <- model.matrix(formula, covariates)
  # aldex.clr(df_tidy, mm, ...)  # doesn't work with denom = 'zero'
  aldex.clr(df_tidy, conditions, ...) # doesn't work with aldex.glm
}

#' aldex.glm workaround
#'
#' @param clr An aldex.clr object
#'
#' Returns a dataframe with feature, kw.ep, glm.ep, kw.eBH, glm.eBH
aldex_glm <- function(clr) {
  glmdrop1 <- function(...) {
    glm(...) %>%
      drop1(test = "Chisq") %>%
      {
        suppressWarnings(broom::tidy(.))
      } %>%
      filter(term == "condition") %>%
      pull(p.value)
  }
  kw <- function(...) {
    kruskal.test(...) %>%
      broom::tidy() %>%
      pull(p.value)
  }
  try_glm <- possibly(glmdrop1, otherwise = NA)
  try_kw <- possibly(kw, otherwise = NA)

  # Gets a named list of length NSAMPLES containing
  # matrices of dimension NFEATURES x NINSTANCES.
  mc <- getMonteCarloInstances(clr) # has sample names
  d_cond <- enframe(clr@conds, name = "sample", value = "condition")
  # matrix row names are feature names
  d_mc <-
    mc %>%
    imap_dfr(
      ~ reshape2::melt(.x, varnames = c("feature", "replicate"), value.name = "clr"),
      .id = "sample"
    )
  d_mc <- left_join(d_mc, d_cond, by = "sample")

  fmla <- clr ~ condition

  # split-apply-combine dataframe for furrr
  oldplan_ <- future::plan()
  future::plan(future::multiprocess)
  d_mdl <- d_mc %>%
    split(.[, c("feature", "replicate")], sep = "_._") %>%
    furrr::future_map_dfr(~ tibble(
      glm_p = try_glm(data = .x, formula = fmla),
      kw_p = try_kw(formula = fmla, data = .x)
    ), .id = "grp", .progress = TRUE)
  future::plan(oldplan_)

  d_mdl <- d_mdl %>%
    tidyr::separate(grp, c("feature", "replicate"), sep = "_\\._")
  d_mdl %>%
    group_by(replicate) %>%
    mutate_at(vars(kw_p, glm_p), list(adj = ~ p.adjust(.x, method = "fdr"))) %>%
    group_by(feature) %>%
    summarize(
      kw.ep = mean(kw_p, na.rm = TRUE),
      glm.ep = mean(glm_p, na.rm = TRUE),
      kw.eBH = mean(kw_p_adj, na.rm = TRUE),
      glm.eBH = mean(glm_p_adj, na.rm = TRUE)
    )
}

#' Extracts clr point estimates to a long-from tibble
#'
#' @param clr An aldex.clr object
#' @param feature Name of the model features ("genus" or "species")
#' @param value.var Name of clr column (e.g., "clr_zero", "clr", etc...)
#'
#' Returns a dataframe with sample, feature, clr
extract_aldex_clr <- function(clr, feature = "genus", value.var = "clr") {
  # Gets a named list of length NSAMPLES containing
  # matrices of dimension NFEATURES x NINSTANCES.
  mc <- getMonteCarloInstances(clr) # has sample names
  # matrix row names are feature names
  mc %>% imap_dfr(~ rowMeans(.x) %>% enframe(name = feature, value = value.var), .id = "sample")
}
