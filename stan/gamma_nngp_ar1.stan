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
  real<lower = 1e-6> inverse_phi; // Inverse variance
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
  matrix[T, S]  b;
  
  w[1, ] = to_row_vector(err[1]);
  mu[1, ] = exp(alpha + w[1, ]);
  for (t in 2:T) {
    w[t, ] = ar * w[t - 1, ] + sqrt(1 - ar^2) * to_row_vector(err[t]);
    mu[t, ] = exp(alpha + w[t, ]);
  }

  b = inverse_phi ./ mu;
}

model {
  l ~ cauchy(0, 2.5);
  sigma ~ cauchy(1, 2.5);
  alpha ~ normal(0, 1);
  inverse_phi ~ exponential(1);
  ar ~ normal(0, .5);

  for (t in 1:T) {
    // sum(err[t]) ~ normal(0, 0.001 * S);
    err[t] ~ nngp_w(sigmasq, lsq, NN_dist, NN_distM, NN_ind, S, M);
  }
  
  Y ~ gamma(inverse_phi, to_vector(b));
}

generated quantities {
  real y_tilde[N];

  for (s in 1:S) {
    for (t in 1:T) {
      y_tilde[t + (s - 1) * T] = gamma_rng(inverse_phi, b[t, s]);
    }
  }
}
