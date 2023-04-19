/* Model:
  Y_{s, t} | mu_{s, t} ~ f(mu_{s, t}) + e_{s, t}
  mu_{s, t} = g(X_{s, t} * beta + w_{s, t})
  w_{s, 1} ~ NNGP(0, C)
  w_{s, t} = rho * w_{s, t - 1} + sqrt(1 - rho^2) * e_{s, t}, t > 1
  e_{s, t} ~ NNGP(0, C)
*/

#include nngp.stan

data {
  int<lower = 1> S;
  int<lower = 1> T;
  int<lower = 1> N;
  real Y[N]; // y_{1,.}, y_{2, .}, ..., y_{S, .}
  int<lower = 1> M;
  int NN_ind[S - 1, M];
  matrix[S - 1, M] NN_dist;
  matrix[S - 1, (M * (M - 1) ./ 2)] NN_distM;
}

parameters {
  real alpha; // Intercept
  real<lower = 1e-6> tau; // Nugget parameter
  real<lower = 1e-6> sigma; // Spatial covariance
  real<lower = 1e-6> l; // Spatial covariance
  real<lower = -1, upper = 1> ar; // Temporal effect - AR(1)
  vector[S] err[T]; // w_{s, t = 1}, w_{s, t = 2}, ..., w_{s, t = T}
}

transformed parameters {
  real sigmasq = square(sigma);
  real lsq = square(l);
  matrix[T, S] w; // Spatial-temporal effect
  matrix[T, S] mu;
  
  w[1, ] = to_row_vector(err[1]);
  mu[1, ] = alpha + w[1, ];
  for (t in 2:T) {
    w[t, ] = ar * w[t - 1, ] + sqrt(1 - ar^2) * to_row_vector(err[t]);
    mu[t, ] = alpha + w[t, ];
  }
}

model {
  l ~ inv_gamma(2, 1);
  sigma ~ inv_gamma(2, 1);
  alpha ~ normal(0, 1);
  tau ~ inv_gamma(2, 1);
  ar ~ normal(0, .5);
  
  for (t in 1:T) {
    sum(err[t]) ~ normal(0, 0.001 * S);
    err[t] ~ nngp_w(sigmasq, lsq, NN_dist, NN_distM, NN_ind, S, M);
    Y[t] ~ normal(mu[t, ], tau);
  }
}

generated quantities {
  real y_tilde[N];

  for (s in 1:S) {
    for (t in 1:T) {
      y_tilde[t + (s - 1) * T] = normal_rng(mu[t, s], tau);
    }
  }
}
