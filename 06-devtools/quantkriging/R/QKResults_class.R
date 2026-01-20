# Copyright 2017-2019 Lawrence Livermore National Security, LLC and other
# quantkriging Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT
#' QKResult Constructor
#'
#' Create Quantile Kriging Results class from list
#'
#' @usage new_QKResults(qkList)
#'
#' @param qkList  list(quants, yquants, g, l, ll <- -optparb$value, beta0, nu, xstar, ystar, Ki, quantv, mult, ylisto, type)
#'
#' @return New class QKResults
#'
#'
#' @export
#'
new_QKResults <- function(qkList = list()) {
  stopifnot(is.list(qkList))
  structure(qkList, class = "QKResults")
}


#' Revaluate Quantiles
#'
#' Quantile Predictions using Quantile Kriging model (class QKResults)
#'
#' @method predict QKResults
#'
#' @author Kevin Quinlan quinlan5@llnl.gov
#'
#' @param object Output from the quantKrig function.
#' @param xnew Locations for prediction
#' @param quantnew Quantiles for prediction, default is to keep the same as the quantile object
#' @param ... Ignore. No other arguments for this method
#'
#' @return Quantile predictions at the specified input locations
#'
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
#' predict(Qout, xnew = c(0.4, 0.5, 0.6))
#'
#' quantpreds <- predict(Qout, xnew = seq(0,1,length.out = 100), quantnew = seq(0.01,0.99,by = 0.01))
#' matplot(seq(0,1,length.out = 100), quantpreds, type = 'l')
predict.QKResults <- function(object, xnew, quantnew = NULL, ...) {
  ystar <- object[[9]]
  xstar <- object[[8]]
  mult <- object[[12]]
  l <- object[[4]]
  nu <- object[[7]]
  beta0 <- object[[6]]
  Ki <- object[[10]]
  ylisto <- object[[13]]
  type <- object[[14]]
  n <- length(ystar)

  if (is.null(quantnew) == TRUE) {
    quantv <- object[[11]]
  }

  if (is.null(quantnew) == FALSE) {
    quantv <- quantnew
  }

  if (is.null(dim(xstar)) == TRUE) {
    xstar <- matrix(xstar, ncol = 1)
  }

  if (is.null(dim(xnew)) == TRUE) {
    xnew <- matrix(xnew, ncol = 1)
  }

  Knew <- nu * hetGP::cov_gen(X1 = xstar, X2 = xstar, theta = l, type = type)
  Kis <- Ki/nu
  meanv <- as.vector(beta0 + Knew %*% (Kis %*% (ystar - beta0)))
  xtest <- xnew

  npts <- length(xtest[, 1])

  K2 <- hetGP::cov_gen(X1 = xtest, X2 = xstar, theta = l, type = type)
  nquants <- length(quantv)
  quants <- matrix(rep(0, npts * nquants), nrow = npts, ncol = nquants)
  Knew <- hetGP::cov_gen(X1 = xtest, X2 = xstar, theta = l, type = type)
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
  return(quants)
}

#' @method summary QKResults
#' @export
summary.QKResults <- function(object, ...) {
  ans <- object
  class(ans) <- "summary.QKResults"
  ans
}


#' @method print summary.QKResults
#' @export
print.summary.QKResults <- function(x, ...) {
  cat("Nugget Value: ", x[[3]], "\n")
  cat("Lengthscale Parameter(s): ", x[[4]], "\n")
  cat("\n")
  cat("Log Likelihood: ", x[[5]], "\n")
  cat("Estimated Constant Trend: ", x[[6]], "\n")
}

#' @method print QKResults
#' @export
print.QKResults <- function(x, ...) {
  cat("Nugget Value: ", x[[3]], "\n")
  cat("Lengthscale Parameter(s): ", x[[4]], "\n")
  cat("\n")
  cat("Log Likelihood: ", x[[5]], "\n")
  cat("Estimated Constant Trend: ", x[[6]], "\n")
}
