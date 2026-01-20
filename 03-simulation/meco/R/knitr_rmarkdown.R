# Stuff to help with notebooks

require(tidyverse)
require(glue)

#' Maps a function to a list and prints a "#" header for each element
#'
#' For use with results='asis' within an RMarkdown document
#' Based off of https://adv-r.hadley.nz/function-operators.html
hwalk <- function(.x, .f, ..., level = 4, tabset = FALSE) {
  ff <- function(f) {
    force(f)

    function(x, header, ...) {
      tbstr <- ifelse(tabset, " {.tabset .tabset-fade .tabset-pills}", "")
      cat(glue::glue(
        '\n\n\n{strrep("#", level)} {header}{tbstr}\n\n',
        .trim = FALSE
      ))
      f(x, y = header, ...)
    }
  }
  .f <- as_mapper(.f, ...)
  iwalk(.x, ff(.f), ...)
}
