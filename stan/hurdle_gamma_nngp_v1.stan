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
  int id_0[N0];
  int id_gt0[Ngt0];
  
  {
    int pos_gt0 = 1;
    int pos_0 = 1;
    for (n in 1:N) {
      if (Y[n] == 0) {
        id_0[pos_0] = n;
        pos_0 += 1;
      } else{
        y_nz[pos_gt0] = Y[n];
        id_gt0[pos_gt0] = n;
        pos_gt0 += 1;
      }
    }
  }
}

parameters{
  real alpha;
  real<lower = 1e-6> sigma;
  real<lower = 1e-6> l;
  real<lower = 1e-6> inverse_phi; // Gamma variance
  vector[N] w;
  // real alpha_hu;
}

transformed parameters {
  real sigmasq = square(sigma);
  real lsq = square(l);
  vector[Ngt0] mu;
  vector[Ngt0] b;
  vector[N0] prob;
  
  mu = exp(alpha + w[id_gt0]);
  b = inverse_phi ./ mu;
  // prob = inv_logit(alpha_hu + w[id_0]);
  mu_h = w[id_0];
}

model{
  l ~ cauchy(0, 5);
  sigma ~ normal(0, 3);
  alpha ~ normal(0, 1);
  inverse_phi ~ exponential(1);
  // alpha_hu ~ student_t(4, 0, 2);
  
  // sum(w) ~ normal(0, 0.001 * N);
  w ~ nngp_w(sigmasq, lsq, NN_dist, NN_distM, NN_ind, N, M);
  
  // Hurdle log likelihood
  for (n in 1:N) {
    if (Y[n] == 0) {
      target += bernoulli_logit_lpmf(1 | mu_h[id_ll[n]]);
    } else {
      target += bernoulli_logit_lpmf(0 | mu[id_ll[n]]) +
          gamma_lpdf(Y[n] | inverse_phi, b[id_ll[n]]);
    }
  }
}
