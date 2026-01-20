test_that("LOOquants example", {
  set.seed(1)
  X <- seq(0,1,length.out = 20)
  Y <- cos(5*X) + cos(X)
  Xstar <- rep(X,each = 100)
  Ystar <- rep(Y,each = 100)
  e <- rchisq(length(Ystar),5)/5 - 1
  Ystar <- Ystar + e
  lb <- c(0.0001,0.0001)
  ub <- c(10,10)

  Qout <- quantKrig(Xstar,Ystar, seq(0.05,0.95, length.out = 7), lower = lb, upper = ub)
  LOO <- LOOquants(Qout)
  expect_known_value(LOO, "LOOquants.test.output")

})
