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
  real<lower = 1e-6> tau;
  real<lower = 1e-6> sigma;
  real<lower = 1e-6> l;
  real<lower = -1, upper = 1> rho;
}

transformed parameters {
  array[T] vector[S] mu;
  
  mu[1] = X_t[1] * beta;
  for (t in 2:T) {
    // mu[t] = X_t[t] * beta + rho * Y_t[t - 1];
    mu[t] = X_t[t] * beta + rho * (Y_t[t - 1] - X_t[t - 1] * beta);
  }
}

model {
  real sigmasq = square(sigma);
  real tausq = square(tau);
  
  l ~ inv_gamma(5, 5);
  sigma ~ inv_gamma(2, 1);
  beta ~ std_normal();
  tau ~ inv_gamma(2, 1);
  rho ~ normal(0.5, 1);

  target += nngp_resp_lpdf(Y_t[1] | mu[1], sigmasq / (1 - rho^2), l, 
    tausq / (1 - rho^2), NN_dist, NN_distM, NN_ind, S, M);
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
    y_sim_t[1] = multi_normal_rng(mu[1], cov / (1 - rho^2));
    for (t in 2:T) {
      y_sim_t[t] = multi_normal_rng(mu[t], cov);
    }
    
    for (s in 1:S) {
      for (t in 1:T) {
        y_sim[h] = y_sim_t[t][s];
        h += 1;
      }
    }
  }
}
