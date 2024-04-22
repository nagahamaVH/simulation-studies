#' Construct covariance matrix
#' @param coords1 A matrix. Each row is a site and number of columns corresponds
#' to number of coordinates
#' @param coords2 (Optional), vector/matrix. Reference coordinate to be computed
#' covariance in relation to coords1
cov_matrix <- function(coords1, sigma, l, cov_function, coords2 = NULL) {
  if (is.null(coords2)) {
    D <- as.matrix(dist(coords1))
  } else {
    p <- dim(coords1)[2]
    coords2 <- if (is.vector(coords2)) matrix(coords2, ncol = p)
    D <- sqrt(rowSums(sweep(coords1, 2, coords2)^2))
  }
  
  if (cov_function == "exp") {
    cov <- sigma^2 * exp(-D / l)
  }
  
  return(cov)
}

cond_mvn_par <- function(mu1, mu2, x2, cov11, cov12, cov22) {
  invU22 <- chol2inv(chol(cov22))
  cov12_invU22 <- tcrossprod(cov12, invU22)
  
  cond_mean <- mu1 + tcrossprod(x2 - mu2, cov12_invU22)
  cond_var <- cov11 - tcrossprod(cov12, cov12_invU22)
  
  return(list(mu = c(cond_mean), sigma2 = c(cond_var)))
}

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
pred_space <- function(X_pred, coords_pred, time_pred, y_obs, mu_obs, 
                       coords_obs, nn_ind, m, beta, rho, sigma, l, tau, 
                       cov_function) {
  T <- dim(y_obs)[1]
  n <- dim(y_obs)[2]
  pred_site_index <- which(apply(coords_obs, 1, 
                                 FUN = function(x) all(x == coords_pred)))
  
  preds <- NULL
  
  if (length(pred_site_index) == 0) {
    max_t_krig <- min(T, time_pred) # Max time for kriging
    cov11 <- sigma^2
    cov22 <- (cov_matrix(coords_obs[nn_ind,], sigma, l, cov_function) +
                diag(tau^2, dim(coords_obs[nn_ind,])[1]))
    cov12 <- cov_matrix(coords_obs[nn_ind,], sigma, l, cov_function, coords_pred)
    
    Xbeta <- X_pred[1,] %*% beta
    
    # y(s0, t) = X beta + z(s0, t)
    # p(z(s0, t) | y; y(s0, t - 1), y(t - 1))
    pars1 <- cond_mvn_par(0, mu_obs[1, nn_ind], y_obs[1, nn_ind], cov11, cov12, 
                          cov22)
    preds[1] <- rnorm(1, Xbeta + pars1$mu, sqrt(pars1$sigma2))
    
    if (time_pred > 1) {
      for (t in 2:max_t_krig) {
        Xbeta <- X_pred[t,] %*% beta + 
          rho * (preds[t - 1] - X_pred[t - 1,] %*% beta)
        
        pars_t <- cond_mvn_par(0, mu_obs[t, nn_ind], y_obs[t, nn_ind], cov11, 
                               cov12, cov22)
        preds[t] <- rnorm(1, Xbeta + pars_t$mu, sqrt(pars_t$sigma2))
      }
      if (time_pred > T) {
        cov22_w <- cov_matrix(coords_obs, sigma, l, cov_function)
        cov12_w <- cov_matrix(coords_obs, sigma, l, cov_function, coords_pred)
        U_w <- chol(cov22_w)
        for (t in (T + 1):time_pred) {
          Xbeta <- X_pred[t,] %*% beta +
            rho * (preds[t - 1] - X_pred[t - 1,] %*% beta)
          
          # y(s0, t) = X beta + w(s0, t)
          # p(w(s0, t) | w)
          w <- rnorm(n) %*% U_w # Spatial field time t
          w_pars <- cond_mvn_par(0, rep(0, n), w, cov11, cov12_w, cov22_w)
          preds[t] <- rnorm(1, Xbeta + w_pars$mu, tau)
        }
      }
    }
  } else {
    y_pred_tm1 <- y_obs[T, pred_site_index]
    cov22_w <- cov_matrix(coords_obs, sigma, l, cov_function)
    U_w <- chol(cov22_w)
    for (t in (T + 1):time_pred) {
      Xbeta <- X_pred[t,] %*% beta + rho * (y_pred_tm1 - X_pred[t - 1,] %*% beta)
      
      # y(s0, t) = X beta + w(s0, t)
      w <- rnorm(n) %*% U_w # Spatial field time t
      preds[t - T] <- rnorm(1, Xbeta + w[pred_site_index], tau)
      y_pred_tm1 <- preds[t - T]
    }
  }
  
  return(preds)
}

#' Predict
pred_st <- function(X_pred, coords_pred, time_pred, y_obs, mu_obs, coords_obs,
                    nn_ind, m, beta, rho, sigma, l, tau, cov_function) {
  # For each site
  preds <- parallel::mclapply(1:dim(coords_pred)[1], function(j) {
    sapply(1:length(sigma), function(i) pred_space(
      X_pred[[j]], coords_pred[j,], time_pred[j], y_obs, mu_obs, coords_obs,
      nn_ind[j,], m, as.vector(beta[i,]), rho[i], sigma[i], l[i],
      tau[i], cov_function))
  }, mc.cores = 12)
  
  # preds <- lapply(1:dim(coords_pred)[1], function(j) {
  #   sapply(1:length(sigma), function(i) pred_space(
  #     X_pred[[j]], coords_pred[j,], time_pred[j], y_obs, mu_obs, coords_obs,
  #     nn_ind[j,], m, as.vector(beta[i,]), rho[i], sigma[i], l[i],
  #     tau[i], cov_function))
  # })
  
  # Debug
  # preds <- list()
  # for (j in 1:dim(coords_pred)[1]) {
  #   print(j)
  #   preds[[j]] <- sapply(1:length(sigma), function(i) pred_space(
  #     X_pred[[j]], coords_pred[j,], time_pred[j], y_obs, mu_obs, coords_obs,
  #     nn_ind[j,], m, as.vector(beta[i,]), rho[i], sigma[i], l[i],
  #     tau[i], cov_function))
  # }
  
  if (is.vector(preds[[1]])) preds[[1]] <- matrix(preds[[1]], nrow = 1)
  
  return(preds)
}
