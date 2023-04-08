// Model:
// Y_{s, t} | mu_{s, t} ~ f(mu_{s, t}) + e_{s, t}
// mu_{s, t} = g(X_{s, t} * beta + w_{s, t})
// w_{s, t} = rho * w_{s, t - 1}

#include nngp.stan

data {
  int<lower = 1> S;
  int<lower = 1> T;
  int<lower = 1> N;
  vector[N] Y; // y_{1,.}, y_{2, .}, ..., y_{S, .}
  int<lower = 1> M;
  int NN_ind[S - 1, M];
  matrix[S - 1, M] NN_dist;
  matrix[S - 1, (M * (M - 1) ./ 2)] NN_distM;
}

parameters{
  real alpha; // Intercept
  real<lower = 1e-6> tau; // Nugget parameter
  real<lower = 1e-6> sigma; // Spatial covariance
  real<lower = 1e-6> l; // Spatial covariance
  vector[S - 1] w_raw;
  real<lower = -1, upper = 1> ar; // Temporal effect - AR(1)
  vector[T] zerr;
  real<lower = 1e-6> sigma_t;
}

transformed parameters {
  real sigmasq = square(sigma);
  real lsq = square(l);
  vector[S] w_s; // Spatial effect
  matrix[T, S] w; // Spatial-temporal effect
  matrix[T, S] mu;
  vector[T] err; // Error for temporal

  // Hard sum-to-zero constrain
  w_s = append_row(w_raw, -sum(w_raw));
  err = sigma_t * zerr;

  w[1] = to_row_vector(w_s);
  mu[1] = alpha + w[1];
  for (t in 2:T) {
    w[t] = ar * w[t - 1] + err[t];
    mu[t] = alpha + w[t];
  }
}

model {
  l ~ inv_gamma(2, 1);
  sigma ~ inv_gamma(2, 1);
  alpha ~ normal(0, 1);
  tau ~ inv_gamma(2, 1);
  ar ~ normal(0, .5);

  zerr ~ std_normal();
  w_s ~ nngp_w(sigmasq, lsq, NN_dist, NN_distM, NN_ind, S, M);
  
  for (t  in 1:T) {
    Y[t] ~ normal(mu[t], tau);
  }
}

// generated quantities {
//   vector[N] y_tilde;
// 
//   for (s in 1:S) {
//     for (t in 1:T) {
//       y_tilde[t] = normal_rng(mu[s, t], tau);
//     }
//   }
// }
