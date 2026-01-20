# Copyright 2017-2019 Lawrence Livermore National Security, LLC and other
# quantkriging Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT
# Insert Roxygen skeleton
#' Quantile Kriging
#'
#' Implements Quantile Kriging from Plumlee and Tuo (2014).
#'
#' @references  \itemize{
#' \item Matthew Plumlee & Rui Tuo (2014) Building Accurate Emulators for Stochastic Simulations via Quantile Kriging, Technometrics, 56:4, 466-473, DOI: 10.1080/00401706.2013.860919
#' \item Mickael Binois, Robert B. Gramacy & Mike Ludkovski (2018) Practical Heteroscedastic Gaussian Process Modeling for Large Simulation Experiments, Journal of Computational and Graphical Statistics, 27:4, 808-821, DOI: 10.1080/10618600.2018.1458625
#' }
#' @param x Inputs
#' @param y Univariate Response
#' @param quantv Vector of Quantile values to estimate (ex: c(0.025, 0.975))
#' @param lower Lower bound of hyperparameters, if isotropic set lengthscale then nugget, if anisotropic set k lengthscales and then nugget
#' @param upper Upper bound of hyperparameters, if isotropic set lengthscale then nugget, if anisotropic set k lengthscales and then nugget
#' @param method Either maximum likelihood ('mle') or leave-one-out cross validation ('loo') optimization of hyperparameters
#' @param type Covariance type, either 'Gaussian', 'Matern3_2', or 'Matern5_2'
#' @param rs If TRUE, rescales inputs to [0,1]
#' @param nm If TRUE, normalizes output to mean 0, variance 1
#' @param known Fixes all hyperparamters to a known value
#' @param optstart Sets the starting value for the optimization
#' @param control Control from optim function
#'
#' @return \describe{
#' \item{quants}{The estimated quantile values in matrix form}
#' \item{yquants}{The actual quantile values from the data in matrix form}
#' \item{g}{The scaling parameter for the kernel}
#' \item{l}{The lengthscale parameter(s)}
#' \item{ll}{The log likelihood}
#' \item{beta0}{Estimated linear trend}
#' \item{nu}{Estimator of the variance}
#' \item{xstar}{Matrix of unique input values}
#' \item{ystar}{Average value at each unique input value}
#' \item{Ki }{Inverted covariance matrix}
#' \item{quantv}{Vector of alpha values between 0 and 1 for estimated quantiles, it is recommended that only a small number of quantiles are used for fitting and more quantiles can be found later using newQuants}
#' \item{mult}{Number of replicates at each input}
#' }
#' @details Fits quantile kriging using a double exponential or Matern covariance function. This emulator is for a stochastic simulation and models the distribution of the results (through the quantiles), not just the mean.  The hyperparameters can be trained using maximum likelihood estimation or leave-one-out cross validation as recommended in Plumlee and Tuo (2014).  The GP is trained using the Woodbury formula to improve computation speed with replication as shown in Binois et al. (2018).  To get meaningful results, there should be sufficient replication at each input. The quantiles at a location \eqn{x0} are found using: \deqn{\mu(x0) + kn(x0)Kn^{-1}(y(i) - \mu(x)}) where \eqn{Kn} is the kernel of the design matrix (with nugget effect), \eqn{y(i)} the ordered sample closest to that quantile at each input, and \eqn{\mu(x)} the mean at each input.
#'
#'
#' @export
#'
#' @examples
#' # Simple example
#' X <- seq(0,1,length.out = 20)
#' Y <- cos(5*X) + cos(X)
#' Xstar <- rep(X,each = 100)
#' Ystar <- rep(Y,each = 100)
#' Ystar <- rnorm(length(Ystar),Ystar,1)
#' lb <- c(0.0001,0.0001)
#' ub <- c(10,10)
#' Qout <- quantKrig(Xstar,Ystar, quantv = seq(0.05,0.95, length.out = 7), lower = lb, upper = ub)
#' QuantPlot(Qout, Xstar, Ystar)
#'
#' #fit for non-normal errors
#'
#' Ystar <- rep(Y,each = 100)
#' e <- rchisq(length(Ystar),5)/5 - 1
#' Ystar <- Ystar + e
#' Qout <- quantKrig(Xstar,Ystar, quantv = seq(0.05,0.95, length.out = 7), lower = lb, upper = ub)
#' QuantPlot(Qout, Xstar, Ystar)
quantKrig <- function(x, y, quantv, lower, upper, method = "loo", type = "Gaussian", rs = TRUE, nm = TRUE,
                      known = NULL, optstart = NULL, control = list()) {
  simpdat <- hetGP::find_reps(X = x, Z = y, rescale = rs, normalize = nm)
  mult <- simpdat$mult
  xstar <- simpdat$X0
  ystar <- simpdat$Z0
  if (is.null(lower) == TRUE) {
    stop("Enter lower bound for optimization of lengthscale")
  }

  if (is.null(upper) == TRUE) {
    stop("Enter upper bound for optimization of lengthscale")
  }

  if (type != "Gaussian") {
    if (type != "Matern5_2") {
      if (type != "Matern3_2") {
        stop("type must be one of Gaussian, Matern5_2, or Matern3_2")
      }
    }
  }
  if (nm == TRUE) {
    y <- (y - mean(y))/stats::sd(y)
  }

  if (dim(xstar)[2] == 1) {
    xord <- rep(xstar, mult)
    xord <- matrix(xord, ncol = 1)
  }
  if (dim(xstar)[2] > 1) {
    xord <- NULL
    for (i in 1:length(xstar[, 1])) {
      xord <- rbind(xord, matrix(rep(xstar[i, ], mult[i]), ncol = dim(xstar)[2], byrow = TRUE))
    }
  }
  yord <- simpdat$Z
  # x <- xord y <- yord
  n <- length(mult)
  N <- length(y)

  multl <- mult
  for (i in 1:length(mult)) {
    multl[i] <- sum(mult[1:i])
  }
  multl <- c(0, multl)
  ylist <- list()
  for (j in 1:n) {
    ylist[[j]] <- yord[(multl[j] + 1):multl[j + 1]]
  }

  ylisto <- lapply(ylist, sort)
  if (length(lower) == 2) {
    if (is.null(known) == TRUE) {
      optfunc3 <- function(input) {

        N <- length(y)
        n <- length(ystar)
        l <- input[1]
        g <- input[2]
        C <- hetGP::cov_gen(xstar, theta = l, type = type)

        # ulist <- list() for(i in 1:n){ ulist[[i]] <- t(t(rep(1,mult[i]))) }
        eps <- sqrt(.Machine$double.eps)
        # U <- bdiag(ulist) An <- t(U)%*%U
        An <- diag(mult)
        K <- C + diag(eps + g/mult)
        Kc <- chol(C + diag(eps + g/mult))
        ramhead <- as.matrix(K)
        ramin <- chol2inv(chol(K))
        Ki <- ramin
        beta0 <- drop(colSums(Ki) %*% ystar/sum(Ki))
        Z <- y - beta0
        Zbar <- ystar - beta0
        vhat <- 1/N * (Z %*% Matrix::Diagonal(n = N, x = 1/g) %*% t(t(Z)) - Zbar %*% (An * 1/g) %*% t(t(Zbar)) + Zbar %*%
          ramin %*% t(t(Zbar)))

        firstterm <- -N/2 * log(vhat)
        secondterm <- -0.5 * sum(diag(An - 1) * log(g)) - 0.5 * sum(log(diag(An)))
        thirdterm <- 0.5 * -2 * sum(log(diag(Kc)))
        loglik <- as.vector(firstterm + secondterm + thirdterm)
        loglik <- loglik - N/2 - N/2 * log(2 * pi)
        return(-loglik)
      }

      n <- length(ystar)
      nquants <- length(quantv)
      yquants <- matrix(rep(0, n * nquants), ncol = n, nrow = nquants)
      for (i in 1:nquants) {
        for (j in 1:n) {
          yquants[i, j] <- ylisto[[j]][round(mult[j] * (quantv[i]))]
        }
      }
      optfuncloo <- function(input) {
        N <- length(y)
        n <- length(ystar)
        l <- input[1]
        g <- input[2]
        eps <- sqrt(.Machine$double.eps)
        C <- hetGP::cov_gen(xstar, theta = l, type = type)
        K <- (C + g * diag(1/mult) + eps * diag(n))
        Kfull <- chol2inv(chol(K))
        beta0 <- drop(colSums(Kfull) %*% ystar/sum(Kfull))
        nquants <- length(quantv)
        quantsimp <- matrix(rep(0, n * nquants), nrow = n, ncol = nquants)

        ulist <- list()
        for (i in 1:n) {
          ulist[[i]] <- t(t(rep(1, mult[i])))
        }
        eps <- sqrt(.Machine$double.eps)
        # U <- bdiag(ulist) An <- t(U)%*%U
        An <- diag(mult)
        beta0 <- drop(colSums(Kfull) %*% ystar/sum(Kfull))
        Z <- y - beta0
        Zbar <- ystar - beta0
        nu <- 1/N * (Z %*% Matrix::Diagonal(n = N, x = 1/g) %*% t(t(Z)) - Zbar %*% (An * 1/g) %*% t(t(Zbar)) + Zbar %*%
          Kfull %*% t(t(Zbar)))
        nu <- as.numeric(nu)

        for (w in 1:n) {
          xloo <- xstar[-w, ]
          yloo <- ystar[-w]

          if (is.null(dim(xloo)) == TRUE) {
          xloo <- matrix(xloo, ncol = 1)
          }

          C <- hetGP::cov_gen(xloo, theta = l, type = type)
          K <- (C + g * diag(1/mult[-w]) + eps * diag(n - 1))
          Ki <- chol2inv(chol(K))

          xtest <- xstar
          Knew <- nu * hetGP::cov_gen(X1 = xtest, X2 = xloo, theta = l, type = type)

          Kis <- Ki/nu

          meanloo <- as.vector(beta0 + Knew %*% (Kis %*% (yloo - beta0)))

          for (i in 1:nquants) {
          quantsimp[w, i] <- (Kfull[w, ] %*% (yquants[i, ] - meanloo))/Kfull[w, w]
          }
        }
        return(sum(quantsimp^2))
      }

      if (is.null(optstart) == FALSE) {
        xeval <- optstart
        xeval <- matrix(xeval, ncol = length(xeval))
        if (length(optstart) != length(lower)) {
          stop("Starting value for optimization and lower bounds have different lengths")
        }
      }

      if (is.null(optstart) == TRUE) {
        xeval <- (upper + lower)/2  #expand.grid(c(0.1,1,10,25),c(0.000001,0.1,1,5))
        xeval <- matrix(xeval, ncol = length(xeval))
      }

      optbest <- 1e+22
      for (q in 1:length(xeval[, 1])) {
        inputs <- xeval[q, ]
        lowervals <- lower
        uppervals <- upper

        if (method == "mle") {
          optpar <- stats::optim(c(inputs), optfunc3, lower = c(lowervals), upper = c(uppervals), method = "L-BFGS-B", control = control)
        }

        if (method == "loo") {
          optpar <- stats::optim(c(inputs), optfuncloo, lower = c(lowervals), upper = c(uppervals), method = "L-BFGS-B",
          control = control)
        }

        if (optpar$value < optbest) {
          optparb <- optpar
          optbest <- optpar$value
        }

      }

      l <- optparb$par[1]
      g <- optparb$par[2]
    }
    if (is.null(known) == FALSE) {
      l <- known[1]
      g <- known[2]
    }
    getparams <- function(input) {
      N <- length(y)
      n <- length(ystar)
      l <- input[1]
      g <- input[2]

      C <- hetGP::cov_gen(xstar, theta = l, type = "Gaussian")

      ulist <- list()
      for (i in 1:n) {
        ulist[[i]] <- t(t(rep(1, mult[i])))
      }
      eps <- sqrt(.Machine$double.eps)
      # U <- bdiag(ulist) An <- t(U)%*%U
      An <- diag(mult)
      K <- C + diag(eps + g/mult)
      Kc <- chol(C + diag(eps + g/mult))
      ramhead <- as.matrix(K)
      Ki <- chol2inv(chol(K))
      beta0 <- drop(colSums(Ki) %*% ystar/sum(Ki))
      Z <- y - beta0
      Zbar <- ystar - beta0
      vhat <- 1/N * (Z %*% Matrix::Diagonal(n = N, x = 1/g) %*% t(t(Z)) - Zbar %*% (An * 1/g) %*% t(t(Zbar)) + Zbar %*%
        Ki %*% t(t(Zbar)))

      firstterm <- -N/2 * log(vhat)
      secondterm <- -0.5 * sum(diag(An - 1) * log(g)) - 0.5 * sum(log(diag(An)))
      thirdterm <- 0.5 * -2 * sum(log(diag(Kc)))
      loglik <- as.vector(firstterm + secondterm + thirdterm)
      loglik <- loglik - N/2 - N/2 * log(2 * pi)
      vallist <- data.frame(nu = as.vector(vhat), beta0 = beta0, ll = -loglik)
      return(vallist)
    }
    parvals <- getparams(c(l, g))
    nu <- parvals$nu
    beta0 <- parvals$beta0
    optparb <- data.frame(value = parvals$ll)
  }

  if (dim(xstar)[2] > 1) {
    M <- dim(xstar)[2]
    if ((length(lower) - 1) == dim(xstar)[2]) {
      if (is.null(known) == TRUE) {
        optfunc3M <- function(input) {

          N <- length(y)
          n <- length(ystar)
          l <- input[1:M]
          g <- input[(M + 1)]
          C <- hetGP::cov_gen(xstar, theta = l, type = type)

          ulist <- list()
          for (i in 1:n) {
          ulist[[i]] <- t(t(rep(1, mult[i])))
          }
          eps <- sqrt(.Machine$double.eps)
          # U <- bdiag(ulist) An <- t(U)%*%U
          An <- diag(mult)
          K <- C + diag(eps + g/mult)
          Kc <- chol(C + diag(eps + g/mult))
          ramhead <- as.matrix(K)
          ramin <- chol2inv(chol(K))
          Ki <- ramin
          beta0 <- drop(colSums(Ki) %*% ystar/sum(Ki))
          Z <- y - beta0
          Zbar <- ystar - beta0
          vhat <- 1/N * (Z %*% Matrix::Diagonal(n = N, x = 1/g) %*% t(t(Z)) - Zbar %*% (An * 1/g) %*% t(t(Zbar)) +
          Zbar %*% ramin %*% t(t(Zbar)))

          firstterm <- -N/2 * log(vhat)
          secondterm <- -0.5 * sum(diag(An - 1) * log(g)) - 0.5 * sum(log(diag(An)))
          thirdterm <- 0.5 * -2 * sum(log(diag(Kc)))
          loglik <- as.vector(firstterm + secondterm + thirdterm)
          loglik <- loglik - N/2 - N/2 * log(2 * pi)
          return(-loglik)
        }

        nquants <- length(quantv)
        n <- length(ystar)
        yquants <- matrix(rep(0, n * nquants), ncol = n, nrow = nquants)
        for (i in 1:nquants) {
          for (j in 1:n) {
          yquants[i, j] <- ylisto[[j]][round(mult[j] * (quantv[i]))]
          }
        }
        optfunclooM <- function(input) {
          N <- length(y)
          l <- input[1:M]
          g <- input[(M + 1)]
          eps <- sqrt(.Machine$double.eps)
          C <- hetGP::cov_gen(xstar, theta = l, type = type)
          K <- (C + g * diag(1/mult) + eps * diag(n))
          Kfull <- chol2inv(chol(K))
          beta0 <- drop(colSums(Kfull) %*% ystar/sum(Kfull))
          quantsimp <- matrix(rep(0, n * nquants), nrow = n, ncol = nquants)

          An <- diag(mult)
          beta0 <- drop(colSums(Kfull) %*% ystar/sum(Kfull))
          Z <- y - beta0
          Zbar <- ystar - beta0
          nu <- 1/N * (Z %*% Matrix::Diagonal(n = N, x = 1/g) %*% t(t(Z)) - Zbar %*% (An * 1/g) %*% t(t(Zbar)) + Zbar %*%
          Kfull %*% t(t(Zbar)))
          nu <- as.numeric(nu)
          for (w in 1:n) {
          xloo <- xstar[-w, ]
          yloo <- ystar[-w]

          if (is.null(dim(xloo)) == TRUE) {
            xloo <- matrix(xloo, ncol = 1)
          }

          C <- hetGP::cov_gen(xloo, theta = l, type = type)

          K <- (C + g * diag(1/mult[-w]) + eps * diag(n - 1))
          Ki <- chol2inv(chol(K))
          xtest <- xstar
          Knew <- nu * hetGP::cov_gen(X1 = xtest, X2 = xloo, theta = l, type = type)

          Kis <- Ki/nu

          meanloo <- as.vector(beta0 + Knew %*% (Kis %*% (yloo - beta0)))

          for (i in 1:nquants) {
            quantsimp[w, i] <- (Kfull[w, ] %*% (yquants[i, ] - meanloo))/Kfull[w, w]
          }
          }
          return(sum(quantsimp^2))
        }

        if (is.null(optstart) == TRUE) {
          xeval <- (upper + lower)/2  #expand.grid(c(0.1,1,10,25),c(0.000001,0.1,1,5))
          xeval <- matrix(xeval, ncol = length(xeval))
        }
        if (is.null(optstart) == FALSE) {
          xeval <- optstart
          xeval <- matrix(xeval, ncol = length(xeval))
          if (length(optstart) != length(lower)) {
          stop("Starting value for optimization and lower bounds have different lengths")
          }
        }
        optbest <- 1e+22
        for (q in 1:length(xeval[, 1])) {
          inputs <- xeval[q, ]
          lowervals <- lower
          uppervals <- upper

          if (method == "mle") {
          optpar <- stats::optim(c(inputs), optfunc3M, lower = c(lowervals), upper = c(uppervals), method = "L-BFGS-B",
            control = control)
          }

          if (method == "loo") {
          optpar <- stats::optim(c(inputs), optfunclooM, lower = c(lowervals), upper = c(uppervals), method = "L-BFGS-B",
            control = control)
          }

          if (optpar$value < optbest) {
          optparb <- optpar
          optbest <- optpar$value
          }

        }

        l <- optparb$par[1:M]
        g <- optparb$par[(M + 1)]
      }
      if (is.null(known) == FALSE) {
        l <- known[1:M]
        g <- known[M + 1]
      }
      getparams <- function(input) {

        N <- length(y)
        n <- length(ystar)
        l <- input[1:M]
        g <- input[(M + 1)]

        C <- hetGP::cov_gen(xstar, theta = l, type = type)

        # ulist <- list() for(i in 1:n){ ulist[[i]] <- t(t(rep(1,mult[i]))) }
        eps <- sqrt(.Machine$double.eps)
        # U <- bdiag(ulist) An <- t(U)%*%U
        An <- diag(mult)
        K <- C + diag(eps + g/mult)
        Kc <- chol(C + diag(eps + g/mult))
        ramhead <- as.matrix(K)
        Ki <- chol2inv(chol(K))
        beta0 <- drop(colSums(Ki) %*% ystar/sum(Ki))
        Z <- y - beta0
        Zbar <- ystar - beta0
        vhat <- 1/N * (Z %*% Matrix::Diagonal(n = N, x = 1/g) %*% t(t(Z)) - Zbar %*% (An * 1/g) %*% t(t(Zbar)) + Zbar %*%
          Ki %*% t(t(Zbar)))

        firstterm <- -N/2 * log(vhat)
        secondterm <- -0.5 * sum(diag(An - 1) * log(g)) - 0.5 * sum(log(diag(An)))
        thirdterm <- 0.5 * -2 * sum(log(diag(Kc)))
        loglik <- as.vector(firstterm + secondterm + thirdterm)
        loglik <- loglik - N/2 - N/2 * log(2 * pi)
        vallist <- data.frame(nu = as.vector(vhat), beta0 = beta0, ll = -loglik)
        return(vallist)
      }
      parvals <- getparams(c(l, g))
      nu <- parvals$nu
      beta0 <- parvals$beta0
      optparb <- data.frame(value = parvals$ll)
    }
    if ((length(lower) - 1) != dim(xstar)[2]) {
      if (length(lower) != 2) {
        stop("Lower bound should be length 2 (isotropic + nugget) or length (k + 1) (anisotropic + nugget) ")
      }
    }
  }
  C <- hetGP::cov_gen(xstar, theta = l, type = type)
  eps <- sqrt(.Machine$double.eps)
  K <- (C + g * diag(1/mult) + eps * diag(n))
  Ki <- chol2inv(chol(K))

  xtest <- xstar
  Knew <- nu * hetGP::cov_gen(X1 = xtest, X2 = xstar, theta = l, type = type)

  Kis <- Ki/nu

  meanv <- as.vector(beta0 + Knew %*% (Kis %*% (ystar - beta0)))

  sd2 <- as.vector(nu - diag(Knew %*% tcrossprod(Kis, Knew)) + (1 - tcrossprod(rowSums(Kis), Knew))^2/sum(Kis))

  sd2 <- pmax(sd2, 0)

  nugs <- rep(nu * g, nrow(xtest))


  # Get Quantiles


  xtest <- xstar

  if (is.null(dim(xtest)) == TRUE) {
    xtest <- matrix(xtest, ncol = 1)
  }


  npts <- length(xtest[, 1])

  K2 <- hetGP::cov_gen(xtest, X2 = xstar, theta = l, type = type)
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
  if (method == "mle") {
    outlist <- list(quants, yquants, g, l, ll <- -optparb$value, beta0, nu, xstar, ystar, Ki, quantv, mult, ylisto, type)
  }
  if (method == "loo") {
    outlist <- list(quants, yquants, g, l, ll <- -optparb$value, beta0, nu, xstar, ystar, Ki, quantv, mult, ylisto, type)
  }
  class(outlist) <- "QKResults"
  return(outlist)

}

