test_that("newQuants Example", {
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

  Qout2 <- newQuants(Qout, c(0.025, 0.5, 0.975))
  expect_known_value(Qout, "newQuants.test.output")
})
