#include nngp.stan

data {
    int<lower = 1> N;
    int<lower = 1> M;
    real<lower = 0> Y[N];
    int NN_ind[N - 1, M];
    matrix[N - 1, M] NN_dist;
    matrix[N - 1, (M * (M - 1) ./ 2)] NN_distM;
}

parameters{
    real alpha;
    real<lower = 1e-6> sigma;
    real<lower = 1e-6> l;
    real<lower = 1e-6> inverse_phi; // Gamma variance
    vector[N] w;
}

transformed parameters {
    real sigmasq = square(sigma);
    real lsq = square(l);

    vector[N] mu = exp(alpha + w);
    vector[N] beta_gamma = inverse_phi ./ mu;
}

model{
  l ~ inv_gamma(2, 1);
  sigma ~ inv_gamma(2, 1);
  alpha ~ normal(0, 1);
  inverse_phi ~ exponential(1);
  
  sum(w) ~ normal(0, 0.001 * N);
  w ~ nngp_w(sigmasq, lsq, NN_dist, NN_distM, NN_ind, N, M);
  Y ~ gamma(inverse_phi, beta_gamma);
}

generated quantities {
  real y_tilde[N];
  
  y_tilde = gamma_rng(inverse_phi, beta_gamma);
}
