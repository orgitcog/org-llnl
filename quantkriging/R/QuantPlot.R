#' Plot Univariate Quantile Data
#'
#' Plots the Quantile output from quantKrig if there is only one input.
#'
#'
#' @param QKResults Output from the quantKrig function.
#' @param X1 X values if ploting the original data in the background
#' @param Y1 Y values if ploting the original data in the background
#' @param main Plot Title defaults to Fitted Quantiles
#' @param xlab Label for x-axis defaults to X
#' @param ylab Label for y-axis defaults to Y
#' @param colors Customize colors associated with the quantiles
#'
#' @return A ggplot object
#' @export
#'
#' @examples
#' X <- seq(0,1,length.out = 20)
#' Y <- cos(5*X) + cos(X)
#' Xstar <- rep(X,each = 100)
#' Ystar <- rep(Y,each = 100)
#' Ystar <- rnorm(length(Ystar),Ystar,1)
#' Ystar <- (Ystar - mean(Ystar)) / sd(Ystar)
#' Xstar <- (Xstar - min(Xstar)/ max(Xstar) - min(Xstar))
#' lb <- c(0.0001,0.0001)
#' ub <- c(10,10)
#' Qout <- quantKrig(Xstar,Ystar, seq(0.05,0.95, length.out = 7), lower = lb, upper = ub)
#' QuantPlot(Qout, Xstar, Ystar)
#'
QuantPlot <- function(QKResults, X1 = NULL, Y1 = NULL, main = NULL, xlab = NULL, ylab = NULL, colors = NULL) {
  quants <- QKResults[[1]]
  xtest <- QKResults[[8]]
  names <- QKResults[[11]]
  if (is.null(main) == TRUE) {
    main <- "Fitted Quantiles"
  }
  if (is.null(xlab) == TRUE) {
    xlab <- "X"
  }

  if (is.null(ylab) == TRUE) {
    ylab <- "Y"
  }

  xv <- xtest[order(xtest)]
  qv <- quants[order(xtest), ]

  nquant <- dim(qv)[2]

  qv <- reshape2::melt(qv)[, c(2, 3)]
  dfqs <- data.frame(xt <- rep(xv, nquant), quant <- qv[, 2], color <- qv[, 1])
  pq <- ggplot2::ggplot()
  if (is.null(X1) == FALSE) {
    dfq <- data.frame(Xvalue <- X1, Yvalue <- Y1)
    pq <- ggplot2::ggplot(dfq, ggplot2::aes(Xvalue, Yvalue)) + ggplot2::geom_point(alpha = 1/3)
  }

  if (is.null(colors) == TRUE) {
    pq <- pq + ggplot2::geom_line(data = dfqs, ggplot2::aes(x = xt, y = quant, color = factor(color))) + ggplot2::scale_color_discrete(name = "Quantiles",
      labels = c(names)) + ggplot2::theme_bw() + ggplot2::ggtitle(main) + ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5)) +
      ggplot2::xlab(xlab) + ggplot2::ylab(ylab)
  }

  if (is.null(colors) == FALSE) {
    pq <- pq + ggplot2::geom_line(data = dfqs, ggplot2::aes(x = xt, y = quant, color = factor(color))) + ggplot2::scale_color_manual(values = colors,
      name = "Quantiles", labels = c(names)) + ggplot2::theme_bw() + ggplot2::ggtitle(main) + ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5)) +
      ggplot2::xlab(xlab) + ggplot2::ylab(ylab)
  }
  return(pq)
}
