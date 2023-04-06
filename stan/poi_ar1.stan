data {
    int<lower = 1> T;
    int<lower = 0> Y[T];
}

parameters {
    real alpha;
    real<lower = -1, upper = 1> rho;
    vector[T] zerr;
    real<lower = 1e-5> sigma_ar;
}

transformed parameters {
    vector[T] mu;
    real u[T];
    vector[T] err;
    
    err = sigma_ar * zerr;
    mu[1] = exp(alpha);
    u[1] = err[1];
    for (t in 2:T) {
      u[t] = rho * err[t - 1] + err[t];
      mu[t] = exp(alpha + u[t]);
    }
}

model {
    alpha ~ normal(3, 2);
    rho ~ normal(0, .5);
    zerr ~ normal(0, 1);
    Y ~ poisson(mu);
}

generated quantities {
  real y_tilde[T];

  y_tilde = poisson_rng(mu);
}
