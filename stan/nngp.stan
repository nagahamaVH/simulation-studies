/* Latent NNGP model
https://github.com/mbjoseph/gpp-speed-test
https://mc-stan.org/users/documentation/case-studies/nngp.html

The function implemented log pdf of the nngp(0, K) is computed using Cholesky
decomposition of K^{-1} = (I - A)^T * D^{-1} * (I - A). Where A is a strict
lower-triangular matrix and D is a diagonal matrix, so cheap computation of
quadractic form can be done using the non-zero elements for each location.

The log pdf can be computed in terms of vector operations and 
quadractic form
log pdf \propto -1/2 * (sum(log(d)) + u^T * 1 / d * u),

where u is equivalent to the matrix form U = (I - A) * W and
a_i = C(s_i, c_i) * C(c_i, c_i)^{-1}
d_i = C(s_i, s_i) - C(s_i, c_i) * C(c_i, c_i)^{-1} * C(s_i, c_i)^T.

Some computational optimisations can be done applying Cholesky decomposition 
on C(c_i, c_i) = L * L^T and defining b_i = L^{-1} * C(s_i, c_i), then
a_i = b_i^T * L^{-1}
d_i = C(s_i, s_i) - b_i^T * b_i.
*/
  
functions {
  real nngp_lpdf(vector w, vector X_beta, real sigmasq, real l, matrix NN_dist,
    matrix NN_distM, array[,] int NN_ind, int N, int M){
      vector[N] d;
      vector[N] wc = w - X_beta;
      vector[N] u = wc;
      int dim;
      int h;
      d[1] = 1;
      
      // For each location i compute u_i = (1 - a_i) * w_i = w_i - a_i * w_{c_i}
      for (i in 2:N) {
        matrix[i < (M + 1) ? (i - 1) : M, i < (M + 1) ? (i - 1) : M] iNNdistM;
        matrix[i < (M + 1) ? (i - 1) : M, i < (M + 1) ? (i - 1) : M] iNNCholL;
        vector[i < (M + 1) ? (i - 1) : M] iNNcorr;
        vector[i < (M + 1) ? (i - 1) : M] b_i;
        row_vector[i < (M + 1) ? (i - 1) : M] a_i;
        dim = (i < (M + 1)) ? (i - 1) : M;
        
        // Scaled covariance matrix of neighbors of i-th location - C(c_i, c_i)
        if(dim == 1){
          iNNdistM[1, 1] = 1;
        }
        else{
          h = 1;
          for (j in 1:(dim - 1)){
            for (k in (j + 1):dim){
              iNNdistM[j, k] = exp(-NN_distM[(i - 1), h] / l);
              iNNdistM[k, j] = iNNdistM[j, k];
              h += 1;
            }
          }
          for(j in 1:dim){
            iNNdistM[j, j] = 1;
          }
        }
        
        // C(c_i, c_i) = L * L^T
        iNNCholL = cholesky_decompose(iNNdistM);
        
        // Scaled covariance vector between i-th location and its neighbors 
        // C(s_i, c_i)
        iNNcorr = to_vector(exp(-NN_dist[(i - 1), 1:dim] / l));
        
        // Stan: inverse(tri(A)) * b
        // Defining the quadractic form 
        // C(s_i, c_i) * C(c_i, c_i)^{-1} * C(s_i, c_i)^T in d_i
        b_i = mdivide_left_tri_low(iNNCholL, iNNcorr);
        
        // d = Diag(D)
        d[i] = 1 - dot_self(b_i);
        
        // Stan: b * inverse(tri(A))
        // Compute a_i
        a_i = mdivide_right_tri_low(b_i', iNNCholL);
  
      // u_i = w_i - a_i * w_{c_i}
      u[i] = u[i] - a_i * wc[NN_ind[(i - 1), 1:dim]];
    }

    // u ./ d is the element-wise division
    return -0.5 * (1 / sigmasq * dot_product(u, (u ./ d)) + sum(log(d)) + 
      N * log(sigmasq));
  }
  
  real nngp_resp_lpdf(vector y, vector mu, real sigmasq, real l, real tausq, 
      matrix NN_dist, matrix NN_distM, array[,] int NN_ind, int N, int M){
      vector[N] d;
      vector[N] Ymu = y - mu;
      vector[N] u = Ymu;
      real kappa = tausq / sigmasq + 1;
      int dim;
      int h;
      d[1] = kappa;
      
      // For each location i compute u_i = (1 - a_i) * w_i = w_i - a_i * w_{c_i}
      for (i in 2:N) {
        matrix[i < (M + 1) ? (i - 1) : M, i < (M + 1) ? (i - 1) : M] iNNdistM;
        matrix[i < (M + 1) ? (i - 1) : M, i < (M + 1) ? (i - 1) : M] iNNCholL;
        vector[i < (M + 1) ? (i - 1) : M] iNNcorr;
        vector[i < (M + 1) ? (i - 1) : M] b_i;
        row_vector[i < (M + 1) ? (i - 1) : M] a_i;
        dim = (i < (M + 1)) ? (i - 1) : M;
        
        // Scaled covariance matrix of neighbors of i-th location - C(c_i, c_i)
        if(dim == 1){
          iNNdistM[1, 1] = kappa;
        }
        else{
          h = 1;
          for (j in 1:(dim - 1)){
            for (k in (j + 1):dim){
              iNNdistM[j, k] = exp(-NN_distM[(i - 1), h] / l);
              iNNdistM[k, j] = iNNdistM[j, k];
              h += 1;
            }
          }
          for(j in 1:dim){
            iNNdistM[j, j] = kappa;
          }
        }
        
        // C(c_i, c_i) = L * L^T
        iNNCholL = cholesky_decompose(iNNdistM);
        
        // Scaled covariance vector between i-th location and its neighbors 
        // C(s_i, c_i)
        iNNcorr = to_vector(exp(-NN_dist[(i - 1), 1:dim] / l));
        
        // Stan: inverse(tri(A)) * b
        // Defining the quadractic form 
        // C(s_i, c_i) * C(c_i, c_i)^{-1} * C(s_i, c_i)^T in d_i
        b_i = mdivide_left_tri_low(iNNCholL, iNNcorr);
        
        // d = Diag(D)
        d[i] = kappa - dot_self(b_i);
        
        // Stan: b * inverse(tri(A))
        // Compute a_i
        a_i = mdivide_right_tri_low(b_i', iNNCholL);
  
      // u_i = w_i - a_i * w_{c_i}
      u[i] = u[i] - a_i * Ymu[NN_ind[(i - 1), 1:dim]];
    }

    // u ./ d is the element-wise division
    return -0.5 * (1 / sigmasq * dot_product(u, (u ./ d)) + sum(log(d)) + 
      N * log(sigmasq));
  }
  
  vector rep_times(vector x, int K) {
    int N = rows(x);
    vector[N * K] y;
    int pos = 1;
    for (k in 1:K) {
      for (n in 1:N) {
        y[pos] = x[n];
        pos += 1;
      }
    }

    return y;
  }
}
