cov_partition <- function(coords, cov_function, sigma, l) {
  n <- dim(coords)[1]
  p <- dim(coords)[2]
  D <- as.matrix(dist(coords[2:n,]))
  d <- rowSums(sweep(coords[2:n,], 2, coords[1,])^p)^(1/p)

  if (cov_function == "exp") {
    cov12 <- sigma^2 * exp(-d / l)
    cov22 <- sigma^2 * exp(-D / l)
  }
  
  return(list(cov12 = cov12, cov22 = cov22))
}

cond_mvn_par <- function(mu0, x, mu, sigma, l, tau, coords, cov_function) {
  cov <- cov_partition(coords, cov_function, sigma, l)
  cov11 <- sigma^2
  
  invU22 <- chol2inv(chol(cov$cov22 + diag(tau^2, dim(coords)[1] - 1)))
  cov12_invU22 <- tcrossprod(cov$cov12, invU22)

  cond_mean <- mu0 + tcrossprod((x - mu), cov12_invU22)
  cond_var <- cov11 - tcrossprod(cov$cov12, cov12_invU22)
  
  return(list(mu = c(cond_mean), sigma2 = c(cond_var)))
}

# set.seed(124)
# c <- coords[1:3, ]
# d <- as.matrix(dist(c))
# C <- sigma^2 * exp(-d / l)
# mu1 <- rnorm(3)
# x1 <- mu1 + crossprod(chol(C), rnorm(3))
# condMVNorm::condMVN(mu1, C, dependent.ind = 1, given.ind = 2:3,
# X.given = x1[2:3])
# cond_mvn_par(mu1[1], x1[2:3], mu1[2:3], sigma, l, 0, c, "exp")

#' Predict interpolation for given observed data of NN
#'
#' Internal function.
#' Predict $y(s_new, t)$ for a given sample \theta
#'
#' @param X_pred Matrix. Design matrix of predict site. Each row for
#' each time point.
#' @param coords_pred Vector. Coordinates of site to be predicted
#' @param time_pred Integer. Index of time to be predicted, number of rows to
#' number of times
#' @param y_obs Matrix. n rows by T cols 
#' @param mu_obs Matrix. 
#' @param coords_obs Matrix. 
#' @param beta Vector 
#' @param rho Numeric 
#' @param sigma Numeric 
#' @param l Numeric. 
#' @param tau Numeric. 
#' @param cov_function Character.
#' add coords <- cbind(coords_new, coords_obs) in general pred function
pred_space <- function(X_pred, coords_pred, time_pred, y_obs, mu_obs, 
                       coords_obs, beta, rho, sigma, l, tau, cov_function) {
  T <- dim(y_obs)[1]
  n <- dim(y_obs)[2]
  pred_site_index <- which(coords_pred[1] == coords_obs[,1] & 
                             coords_pred[2] == coords_obs[,2])
  
  if (length(pred_site_index) == 0) {
    max_t_krig <- min(T, time_pred) # Max time for kriging
    coords <- rbind(coords_pred, coords_obs, deparse.level = 0)
    
    pars1 <- cond_mvn_par(X_pred[1,] %*% beta, y_obs[1,], mu_obs[1,], sigma, l, 
                          tau, coords, cov_function)
    pred <- pars1$mu
    
    if (time_pred > 1) {
      y_pred_tm1 <- pred
      for (t in 2:max_t_krig) {
        pars_t <- cond_mvn_par(
          X_pred[t,] %*% beta + rho * (y_pred_tm1 - X_pred[t - 1,] %*% beta), 
          y_obs[t,], mu_obs[t,], sigma, l, tau, coords, cov_function)
        pred <- pars_t$mu
        y_pred_tm1 <- pred
      }
      if (max_t_krig > T) {
        for (t in (T + 1):time_pred) {
          D <- as.matrix(dist(coords_obs))
          L <- chol(sigma^2 * exp(-D / l))
          w <- rnorm(n) %*% L
          w_pred <- cond_mvn_par(0, w, rep(0, n), sigma, l, tau, coords, 
                                 cov_function)
          pred <- X_pred[t,] %*% beta + 
            rho * (y_pred_tm1 - X_pred[t - 1,] %*% beta) + w_pred
          y_pred_tm1 <- pred
        }
      }
    }
  } else {
    y_pred_tm1 <- y_obs[T, pred_site_index]
    for (t in (T + 1):time_pred) {
      D <- as.matrix(dist(coords_obs))
      L <- chol(sigma^2 * exp(-D / l))
      w <- rnorm(n) %*% L
      pred <- X_pred[t,] %*% beta + 
        rho * (y_pred_tm1 - X_pred[t - 1,] %*% beta) + w[pred_site_index]
      y_pred_tm1 <- pred 
    }
  }
  
  return(pred)
}

#' Predict
pred_st <- function(X_pred, coords_pred, time_pred, y_obs, mu_obs, coords_obs, 
                    beta, rho, sigma, l, tau, cov_function) {
  # For each site
  preds <- sapply(1:dim(coords_pred)[1], function(j) {
    unlist(
      # For each posterior sampling
      parallel::mclapply(1:length(sigma), function(i) pred_space(
        X_pred[[j]], coords_pred[j,], time_pred[j], y_obs, mu_obs, coords_obs, 
        as.vector(beta[i,]), rho[i], sigma[i], l[i], tau[i], cov_function), 
        mc.cores = 12)
    )
  })
  return(preds)
}
