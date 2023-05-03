library(dplyr)
library(tidyr)
library(rstan)
library(posterior)
library(bayesplot)

source("./R/NNMatrix.R")

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# simulate data
n <- 500

# Spatial
sigma <- 1
l <- 2.3

alpha <- 1
prec <- 1.2
prop <- 0.3 # Proportion of zeros

set.seed(1239)
coords <- cbind(runif(n), runif(n))
ord <- order(coords[,1])
coords <- coords[ord,]

# Simulate zero values
theta <- rbinom(n, 1, 1 - prop)

d <- dist(coords[theta == 1,]) |>
  as.matrix()
w <- mvtnorm::rmvnorm(1, sigma = sigma^2 * exp(-d / l^2)) |>
  c()

mu <- exp(alpha + w)

# Simulate nonzero values
y_aux <- rgamma(length(w), shape = prec, rate = prec / mu)

y <- theta
y[theta == 1] <- y_aux

hist(y)

# -----------------------------------------------------------------------------
m <- 5

nn <- NNMatrix(coords, n.neighbors = m)

data_stan <- list(
  N = n,
  Y = y,
  M = m,
  NN_ind = nn$NN_ind,
  NN_dist = nn$NN_dist,
  NN_distM = nn$NN_distM
)

model <- stan_model("./stan/hurdle_gamma_nngp.stan")

result_stan <- sampling(
  model,
  data = data_stan,
  chains = 4,
  iter = 3000)

result_stan |>
  mcmc_trace(pars = c("alpha", "sigma", "l", "inverse_phi", "theta")) +
  scale_color_discrete()

result_stan |>
  mcmc_dens_overlay(pars = c("alpha", "sigma", "l", "inverse_phi", "theta")) +
  scale_color_discrete()

result_stan |>
  mcmc_pairs(pars = c("alpha", "sigma", "l", "inverse_phi", "theta"))

stan2_df <- as_draws_df(result_stan)
stan2_summary <- stan2_df |>
  select_at(vars("alpha", "sigma", "l", "inverse_phi", "theta")) |>
  summarise_draws()
stan2_summary
