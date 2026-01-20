# Copyright 2017-2019 Lawrence Livermore National Security, LLC and other
# quantkriging Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT
#' Revaluate Quantiles
#'
#' Generates new quantiles from a quantile object
#'
#' @usage newQuants(QKResults, quantv)
#'
#'
#' @param QKResults Output from the quantKrig function.
#' @param quantv Vector of quantile values alpha between 0 and 1,
#'
#' @return The same quantile object with new estimated quantiles.
#' @export
#'
#' @examples
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
#' Qout2 <- newQuants(Qout, c(0.025, 0.5, 0.975))
#' QuantPlot(Qout2)
newQuants <- function(QKResults, quantv) {
  # Assign QKResults to the right stuff
  ystar <- QKResults[[9]]
  xstar <- QKResults[[8]]
  mult <- QKResults[[12]]
  l <- QKResults[[4]]
  nu <- QKResults[[7]]
  beta0 <- QKResults[[6]]
  Ki <- QKResults[[10]]
  ylisto <- QKResults[[13]]
  n <- length(ystar)
  # Knew <- nu*hetGP::exp_cov_gen(X1 = xstar, X2 = xstar, theta = l, power = pwr)
  Knew <- nu * hetGP::cov_gen(X1 = xstar, X2 = xstar, theta = l, type = "Gaussian")
  Kis <- Ki/nu
  meanv <- as.vector(beta0 + Knew %*% (Kis %*% (ystar - beta0)))
  xtest <- QKResults[[8]]

  if (is.null(dim(xtest)) == TRUE) {
    xtest <- matrix(xtest, ncol = 1)
  }

  npts <- length(xtest[, 1])

  K2 <- hetGP::cov_gen(X1 = xtest, X2 = xstar, theta = l, type = "Gaussian")
  nquants <- length(quantv)
  quants <- matrix(rep(0, npts * nquants), nrow = npts, ncol = nquants)
  Knew <- hetGP::cov_gen(X1 = xtest, X2 = xstar, theta = l, type = "Gaussian")
  Kis <- Ki
  mean <- as.vector(beta0 + Knew %*% (Kis %*% (ystar - beta0)))

  yquants <- matrix(rep(0, n * nquants), ncol = n, nrow = nquants)

  for (i in 1:nquants) {
    for (j in 1:n) {
      yquants[i, j] <- ylisto[[j]][round(mult[j] * (quantv[i]))]
    }
  }

  for (j in 1:npts) {
    for (i in 1:nquants) {
      cov <- as.matrix(K2[j, ], nrow = 1)
      last <- (yquants[i, ] - meanv)
      quants[j, i] <- mean[j] + t(cov) %*% (Ki) %*% t(t(last))

    }
  }

  QKResults[[1]] <- quants
  QKResults[[11]] <- quantv

  return(QKResults)
}
