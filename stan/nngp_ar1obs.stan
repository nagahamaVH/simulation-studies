#include nngp.stan

data {
  int<lower = 1> S;
  int<lower = 1> T;
  int<lower = 1> N;
  int<lower = 1> p;
  array[T] vector[S] Y_t;
  array[T] matrix[S, p] X_t;
  int<lower = 1> M;
  array[S - 1, M] int NN_ind;
  matrix[S - 1, M] NN_dist;
  matrix[S - 1, (M * (M - 1) ./ 2)] NN_distM;
  array[S] vector[2] coords; // Station coordinates
}

parameters {
  vector[p] beta;
  real<lower = 0> tau;
  real<lower = 0> sigma;
  real<lower = 1e-6> l;
  real rho;
}

transformed parameters {
  array[T] vector[S] mu;
  
  mu[1] = X_t[1] * beta;
  for (t in 2:T) {
    mu[t] = X_t[t] * beta + rho * (Y_t[t - 1] - X_t[t - 1] * beta);
  }
}

model {
  real sigmasq = square(sigma);
  real tausq = square(tau);
  
  l ~ inv_gamma(2, 1);
  sigma ~ normal(0, 10);
  beta ~ normal(0, 10);
  tau ~ normal(0, 10);
  rho ~ uniform(-1, 1);

  for (t in 2:T){
    target += nngp_resp_lpdf(Y_t[t] | mu[t], sigmasq, l, tausq, NN_dist, 
      NN_distM, NN_ind, S, M);
  }
}

generated quantities {
  vector[N] y_sim;
  {
    matrix[S, S] cov;
    array[T] vector[S] y_sim_t;
    int h = 1;

    cov = gp_exponential_cov(coords, coords, sigma, l);
    cov = add_diag(cov, tau^2);
    // y_sim_t[1] = multi_normal_cholesky_rng(mu[1], 
    //   cholesky_decompose(cov / (1 - rho^2)));
    for (t in 1:T) {
      y_sim_t[t] = multi_normal_cholesky_rng(mu[t], cholesky_decompose(cov));
    }
    
    for (s in 1:S) {
      for (t in 1:T) {
        y_sim[h] = y_sim_t[t][s];
        h += 1;
      }
    }
  }
}
