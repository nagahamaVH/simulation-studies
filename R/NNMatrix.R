library(spNNGP) # Build neighbor index

#### distance matrix for location i and its neighbors ####
i_dist <- function(i, neighbor_index, s) {
  dist(s[c(i, neighbor_index[[i - 1]]), ])
}

#### distance matrix between neighbors of i ####
get_NN_distM <- function(ind, ind_distM_d, M) {
  if (ind < M) {
    l <- ind
  } else {
    l <- M
  }
  M_i <- rep(0, M * (M - 1) / 2)
  if (l == 1) {} else {
    M_i[1:(l * (l - 1) / 2)] <-
      c(ind_distM_d[[ind]])[(l + 1):(l * (l + 1) / 2)]
  }
  return(M_i)
}

#### distance matrix between i and its neighbors ####
get_NN_dist <- function(ind, ind_distM_d, M) {
  if (ind < M) {
    l <- ind
  } else {
    l <- M
  }
  D_i <- rep(0, M)
  D_i[1:l] <- c(ind_distM_d[[ind]])[1:l]
  return(D_i)
}

get_NN_ind <- function(ind, ind_distM_i, M) {
  if (ind < M) {
    l <- ind
  } else {
    l <- M
  }
  D_i <- rep(0, M)
  D_i[1:l] <- c(ind_distM_i[[ind]])[1:l]
  return(D_i)
}

#### NNMatrix: A wrapper of spConjNNGP to build Nearest Neighbor matrics ####
##' coords:        An n x 2 matrix of the observation coordinates in R^2
##' n.neighbors:   Number of neighbors used in the NNGP.
##' n.omp.threads: A positive integer indicating the number of threads to use
##'                for SMP parallel processing.
##' search.type:   a quoted keyword that specifies type of nearest neighbor
##'                search algorithm. Supported method key words are: "cb" and
##'                "brute". The "cb" should generally be much faster. If
##'                locations do not have identical coordinate values on the
##'                axis used for the nearest neighbor ordering (see ord
##'                argument) then "cb" and "brute" should produce identical
##'                neighbor sets. However, if there are identical coordinate
##'                values on the axis used for nearest neighbor ordering,
##'                then "cb" and "brute" might produce different, but equally
##'                valid, neighbor sets, e.g., if data are on a grid.
##' ord:           an index vector of length n used for the nearest neighbor
##'                search. Internally, this vector is used to order coords,
##'                i.e., coords[ord,], and associated data. Nearest neighbor
##'                candidates for the i-th row in the ordered coords are rows
##'                1:(i-1), with the n.neighbors nearest neighbors being those
##'                with the minimum euclidean distance to the location defined
##'                by ordered coords[i,]. The default used when ord is not
##'                specified is x-axis ordering, i.e., order(coords[,1]). This
##'                argument should typically be left blank.
NNMatrix <- function(coords, n.neighbors, n.omp.threads = 2,
                     search.type = "cb", ord = order(coords[, 1])) {
  N <- nrow(coords)
  m.c <- spConjNNGP(rep(0, N) ~ 1,
    coords = coords,
    n.neighbors = n.neighbors,
    theta.alpha = c("phi" = 5, "alpha" = 0.5),
    sigma.sq.IG = c(2, 1),
    cov.model = "exponential",
    n.omp.threads = n.omp.threads,
    search.type = search.type,
    ord = ord,
    return.neighbor.info = T, fit.rep = F,
    verbose = F
  )
  M <- n.neighbors
  NN_ind <- t(sapply(1:(N - 1), get_NN_ind, m.c$neighbor.info$n.indx[-1], M))
  neighbor_dist <- sapply(
    2:N, i_dist, m.c$neighbor.info$n.indx[-1],
    m.c$coords[m.c$neighbor.info$ord, ]
  )
  NN_distM <- t(sapply(1:(N - 1), get_NN_distM, neighbor_dist, M))
  NN_dist <- t(sapply(1:(N - 1), get_NN_dist, neighbor_dist, M))

  return(list(
    ord = m.c$neighbor.info$ord,
    coords.ord = m.c$coords[m.c$neighbor.info$ord, ],
    NN_ind = NN_ind, NN_distM = NN_distM, NN_dist = NN_dist
  ))
}
