#### extract distance for location i and its neighbors ####
i_dist <- function(i, neighbor_index, s) {
  dist(s[c(i, neighbor_index[[i - 1]]), ], method = "minkowski")
}

#### create matrix of distances between pairs of neighbors ####
get_NN_distM <- function(ind, ind_distM_d, M) {
  if (ind < M) {
    l <- ind
  } else {
    l <- M
  }
  M_i <- rep(0, M * (M - 1) / 2)
  if (l == 1) {} else {
    M_i[1:(l * (l - 1) / 2)] <-
      ind_distM_d[[ind]][(l + 1):(l * (l + 1) / 2)]
  }
  return(M_i)
}

#### crate matrix of distances between location i and its neighbors ####
get_NN_dist <- function(ind, ind_distM_d, M) {
  if (ind < M) {
    l <- ind
  } else {
    l <- M
  }
  D_i <- rep(0, M)
  D_i[1:l] <- ind_distM_d[[ind]][1:l]
  return(D_i)
}

get_NN_ind <- function(ind, ind_distM_i, M) {
  if (ind < M) {
    l <- ind
  } else {
    l <- M
  }
  D_i <- rep(0, M)
  D_i[1:l] <- ind_distM_i[[ind]][1:l]
  return(D_i)
}

find_nn <- function(coords, m) {
  N <- nrow(coords)
  nn_list <- GpGp::find_ordered_nn(coords, m, lonlat = F)[-1, -1] |>
    apply(MARGIN = 1, function(x) x[!is.na(x)])
  
  NN_ind <- t(sapply(1:(N - 1), get_NN_ind, nn_list, m))
  neighbor_dist <- sapply(2:N, i_dist, nn_list, coords)
  NN_distM <- t(sapply(1:(N - 1), get_NN_distM, neighbor_dist, m))
  NN_dist <- t(sapply(1:(N - 1), get_NN_dist, neighbor_dist, m))
  
  out <- list(
    coords.ord = coords,
    NN_ind = NN_ind, 
    NN_distM = NN_distM, 
    NN_dist = NN_dist
  )
  
  return(out)
}

# Merge with find_nn adding optional arg (coords_pred)
find_nn_pred <- function(coords_pred, coords_obs, m) {
  nn_list <- FNN::get.knnx(coords_obs, coords_pred, k = m)
  NN_distM <- sapply(1:nrow(coords_pred), function(i) {
    coords_obs[nn_list$nn.index[i,],] |>
      dist(method = "minkowski")
  }) |>
    t()
  
  out <- list(
    coords.ord = coords_pred,
    NN_ind = nn_list$nn.index, 
    NN_distM = NN_distM, 
    NN_dist = nn_list$nn.dist
  )
  return(out)
}
