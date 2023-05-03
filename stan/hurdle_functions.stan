// Return an integer with the number of zeros in a vector
functions {
  int num_zero(real[] y) {
    int nz = 0;
    for (n in 1:size(y)) {
      if (y[n] == 0) {
        nz += 1;
      } 
    }
    return nz;
  }
  
  real nngp_w_lpdf(vector w, real sigmasq, real lsq, matrix NN_dist, 
                matrix NN_distM, int[,] NN_ind, int N, int M){
    vector[N] d;
    vector[N] u = w;
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
        h = 0;
        for (j in 1:(dim - 1)){
          for (k in (j + 1):dim){
            h = h + 1;
            iNNdistM[j, k] = exp(-NN_distM[(i - 1), h] / (2 * lsq));
            iNNdistM[k, j] = iNNdistM[j, k];
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
      iNNcorr = to_vector(exp(-NN_dist[(i - 1), 1:dim] / (2 * lsq)));
      
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
      u[i] = u[i] - a_i * w[NN_ind[(i - 1), 1:dim]];
    }
    
    // u ./ d is the element-wise division
    return -0.5 * (1 / sigmasq * dot_product(u, (u ./ d)) + sum(log(d)) + 
      N * log(sigmasq));
  }
}
