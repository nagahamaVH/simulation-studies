data {
    int<lower = 1> T;
    real<lower = 0> Y[T];
}

parameters {
    real alpha;
    real<lower = -1, upper = 1> rho;
    vector[T] zerr;
    real<lower = 1e-5> sigma_ar;
    real<lower = 1e-6> inverse_phi; // Gamma precision parameter
}

transformed parameters {
    vector[T] mu;
    real u[T];
    vector[T] err;
    vector[T] beta_gamma; // Gamma inverse scale parameter
    err = sigma_ar * zerr;
    u[1] = err[1];
    mu[1] = exp(alpha);
    beta_gamma[1] = inverse_phi / mu[1];
    for (t in 2:T) {
      u[t] = rho * err[t - 1] + err[t];
      mu[t] = exp(alpha + u[t]);
      beta_gamma[t] = inverse_phi / mu[t];
    }
}

model {
    alpha ~ normal(1, 2);
    rho ~ normal(0, .5);
    zerr ~ normal(0, 1);
    inverse_phi ~ exponential(1);
    
    Y ~ gamma(inverse_phi, beta_gamma);
}

generated quantities {
  real y_tilde[T];

  y_tilde = gamma_rng(inverse_phi, beta_gamma);
}
