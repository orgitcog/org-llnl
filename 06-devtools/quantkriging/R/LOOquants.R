# Copyright 2017-2019 Lawrence Livermore National Security, LLC and other
# quantkriging Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT
#' Revaluate Quantiles
#'
#' Generates Leave-One-Out predictions for each location and quantile.
#'
#' @usage LOOquants(QKResults)
#'
#'
#'
#' @param QKResults Output from the quantKrig function.
#'
#' @return Leave-one-out predictions at the input locations
#'
#' @details Returns the estimated quantiles and a plot of the leave-one-out predictions against the observed values.
#'
#' @export
#'
#' @examples
#'
#' X <- seq(0,1,length.out = 20)
#' Y <- cos(5*X) + cos(X)
#' Xstar <- rep(X,each = 100)
#' Ystar <- rep(Y,each = 100)
#' e <- rchisq(length(Ystar),5)/5 - 1
#' Ystar <- Ystar + e
#' lb <- c(0.0001,0.0001)
#' ub <- c(10,10)
#' Qout <- quantKrig(Xstar,Ystar, seq(0.05,0.95, length.out = 7), lower = lb, upper = ub)
#'
#' LOOquants(Qout)
#'
LOOquants <- function(QKResults) {
  # return the mean estimate and the quantiles if LOO! First, generate covariance matrix without one input
  quantstore <- NULL
  quantv <- QKResults[[11]]
  xstar <- QKResults[[8]]
  ystar <- QKResults[[9]]
  l <- QKResults[[4]]
  g <- QKResults[[3]]
  beta0 <- QKResults[[6]]
  nu <- QKResults[[7]]
  mult <- QKResults[[12]]
  Ki <- QKResults[[10]]
  n <- length(ystar)
  yquants <- QKResults[[2]]
  ylisto <- QKResults[[13]]
  type <- QKResults[[14]]
  nquants <- length(quantv)
  quants <- matrix(rep(0, nquants * n), nrow = nquants, ncol = n)


  C <- hetGP::cov_gen(xstar, theta = l, type = type)
  eps <- sqrt(.Machine$double.eps)
  K <- (C + g * diag(1/mult) + eps * diag(n))
  Ki <- chol2inv(chol(K))

  xtest <- xstar
  Knew <- nu * hetGP::cov_gen(X1 = xtest, X2 = xstar, theta = l, type = type)

  Kis <- Ki/nu

  meantrue <- as.vector(beta0 + Knew %*% (Kis %*% (ystar - beta0)))
  npts <- length(xtest[, 1])
  nquants <- length(quantv)
  qseq <- quantv
  quants <- matrix(rep(0, npts * nquants), nrow = npts, ncol = nquants)

  yquants <- matrix(rep(0, n * nquants), ncol = n, nrow = nquants)

  for (i in 1:nquants) {
    for (j in 1:n) {
      yquants[i, j] <- ylisto[[j]][round(mult[j] * (qseq[i]))]
    }
  }

  time1 <- Sys.time()
  for (w in 1:n) {
    xloo <- xstar[-w, ]
    yloo <- ystar[-w]

    if (is.null(dim(xloo)) == TRUE) {
      xloo <- matrix(xloo, ncol = 1)
    }

    C <- hetGP::cov_gen(xloo, theta = l, type = type)

    K <- (C + g * diag(1/mult[-w]) + eps * diag(n - 1))
    Ki <- chol2inv(chol(K))

    xtest <- xloo
    Knew <- nu * hetGP::cov_gen(X1 = xtest, X2 = xloo, theta = l, type = type)

    Kis <- Ki/nu

    meanv <- as.vector(beta0 + Knew %*% (Kis %*% (yloo - beta0)))

    # Get Quantiles
    xtest <- xstar

    K2 <- hetGP::cov_gen(xtest, X2 = xloo, theta = l, type = "Gaussian")
    Knew <- hetGP::cov_gen(X1 = xtest, X2 = xloo, theta = l, type = "Gaussian")
    Kis <- Ki
    mean <- as.vector(beta0 + Knew %*% (Kis %*% (yloo - beta0)))

    j <- w
    for (i in 1:nquants) {

      cov <- as.matrix(K2[j, ], nrow = 1)
      last <- (yquants[i, -w] - meanv)
      quants[j, i] <- mean[j] + t(cov) %*% (Ki) %*% t(t(last))

    }

  }
  quantstore <- quants
  plotdf <- cbind(reshape2::melt(quantstore)[, c(2, 3)], reshape2::melt(t(QKResults[[2]]))[, 3])
  names(plotdf) <- c("Quantile", "Prediction", "Actual")
  print(ggplot2::ggplot(plotdf, ggplot2::aes(x = plotdf$Prediction, y = plotdf$Actual, color = factor(plotdf$Quantile))) + ggplot2::geom_point() +
    ggplot2::geom_abline(slope = 1, intercept = 0) + ggplot2::scale_color_discrete(name = "Quantile", labels = quantv) +
    ggplot2::ggtitle("Parity Plot") + ggplot2::theme_bw() + ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5)))
  return(quantstore)
}

