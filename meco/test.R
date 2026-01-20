library(tidyverse)

srcs <- list.files("R", pattern = "\\.R$", full.names = TRUE)
walk(srcs, source)

lmat <- read_tsv("data/lmat_example.tsv")
samp <- tbl_df(data.frame(
  sample = unique(lmat$sample),
  treatment = sample(c("A", "B", "C"), size = length(unique(lmat$sample)), replace = TRUE)
))

lmat_microbes <- keep_lmat_microbes(lmat)

lmat_genus <- sum_lmat_at_rank(lmat_microbes, genus)
lmat_species <- sum_lmat_at_rank(lmat_microbes, species)
aldex_bytrt <-
  list(genus = lmat_genus, species = lmat_species) %>%
  map(left_join, samp, by = "sample") %>%
  imap(~{
   aldex_clr(.x, ~ treatment,
     useMC = TRUE, denom = 'zero', mc.samples = 3,
     feature = .y
   )
  })
lmat_clr <- imap(aldex_bytrt, ~ extract_aldex_clr(.x, feature = .y, value.var = "clr_zero"))

lmat_clr %>%
  hwalk(~ print(head(.x)), level = 3, tabset = TRUE)

lmat_glm <- map(aldex_bytrt, aldex_glm)

lmat_glm %>%
  hwalk(~ print(head(.x)), level = 3, tabset = TRUE)
