// ---------------------------------------------------------------------------
// Version 2
// ---------------------------------------------------------------------------
data {
  int<lower=1> N;  // number of the entire data
  int<lower=1> T;  // number of time point
  int<lower=1> S; // number of locations or dimension of space
  int y[N];    // observation
  real x1[S]; // location x
  real x2[S];  // location y
}

// transform data
transformed data {
  row_vector[2] x_tot[S];
  for (s in 1:S) {
    x_tot[s, 1] = x1[s];
    x_tot[s, 2] = x2[s];
  }
}

parameters {
  real<lower=0> length_scale; // scale parameter
  real<lower=0> alpha; // variance
  real beta; // intercept
  real<lower=0, upper=1> rho;  // temporal correlation parameter
  real<lower=1e-6> sigma; // spatio-temporal process
  row_vector[S] S_init;
  matrix[T, S] zerr;
}

transformed parameters {
  matrix[S, S] K;  // the covariance matrix
  matrix[T, S] S_x_t; // spatio-temporal process
  matrix[T, S] mu_mat;

  K = cov_exp_quad(x_tot, alpha, length_scale);
  for (s in 1:S) {
    K[s, s] = K[s, s] + 1e-12;
  }

  S_x_t[1,] = S_init;
  mu_mat[1, ] = beta + S_x_t[1, ];
  for (t in 2:T) {
    S_x_t[t, ] = rho * S_x_t[t - 1,] + sigma * zerr[t,];
    mu_mat[t, ] = beta + S_x_t[t,];
  }
}

model {
  length_scale ~ inv_gamma(2, 1);
  alpha ~ inv_gamma(2, 1);
  beta ~ std_normal();
  rho ~ normal(0, 0.5);
  sigma ~ inv_gamma(2, 1);
  S_init ~ multi_normal(rep_vector(0, S), K);

  for (t in 1:T) {
    zerr[t,] ~ normal(0, 1);
    y ~ poisson_log(mu_mat[t, ]);
  }
}
