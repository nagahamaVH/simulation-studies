#include hurdle_functions.stan

data {
  int<lower = 1> N;
  int<lower = 1> M;
  real<lower = 0> Y[N];
  int NN_ind[N - 1, M];
  matrix[N - 1, M] NN_dist;
  matrix[N - 1, (M * (M - 1) ./ 2)] NN_distM;
}

transformed data {
  int<lower = 0, upper = N> N0 = num_zero(Y);
  int<lower = 0, upper = N> Ngt0 = N - N0;
  real<lower = 0> y_nz[Ngt0];
  {
    int pos = 1;
    for (n in 1:N) {
      if (Y[n] != 0) {
        y_nz[pos] = Y[n];
        pos += 1;
      }
    }
  }
}

parameters{
  real alpha;
  real<lower = 1e-6> sigma;
  real<lower = 1e-6> l;
  real<lower = 1e-6> inverse_phi; // Gamma variance
  real<lower = 0, upper = 1> theta; // Zero outcome prob
  vector[Ngt0] w;
}

transformed parameters {
  real sigmasq = square(sigma);
  real lsq = square(l);
  
  vector[Ngt0] mu = exp(alpha + w);
  vector[Ngt0] beta_gamma = inverse_phi ./ mu;
}

model{
  l ~ inv_gamma(4.63, 22.07);
  sigma ~ normal(0, 3);
  alpha ~ normal(0, 1);
  inverse_phi ~ exponential(1);
  theta ~ normal(0.5, 1);

  sum(w) ~ normal(0, 0.001 * Ngt0);
  w ~ nngp_w(sigmasq, lsq, NN_dist, NN_distM, NN_ind, Ngt0, M);

  // Hurdle log likelihood
  N0 ~ binomial(N, theta);
  y_nz ~ gamma(inverse_phi, beta_gamma);
}
