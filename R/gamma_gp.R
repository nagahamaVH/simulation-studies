library(dplyr)
library(tidyr)
library(rstan)
library(posterior)
library(bayesplot)

source("./R/NNMatrix.R")

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# simulate data
n <- 200

# Spatial
sigma <- 1
l <- 2.3

# Intercept
alpha <- 1
prec.par <- 1.2

set.seed(1239)
coords <- cbind(runif(n), runif(n))
ord <- order(coords[,1])
coords <- coords[ord,]

d <- dist(coords) |>
  as.matrix()

w <- mvtnorm::rmvnorm(1, sigma = sigma^2 * exp(-d / l^2)) |>
  c()

mu <- exp(alpha + w)

y2 <- rgamma(n, shape = prec.par, rate = prec.par / mu)

hist(y2)

# -----------------------------------------------------------------------------
m <- 5

nn <- NNMatrix(coords, n.neighbors = m)

data_stan <- list(
  N = n,
  Y = y2,
  M = m,
  NN_ind = nn$NN_ind,
  NN_dist = nn$NN_dist,
  NN_distM = nn$NN_distM
)

model <- stan_model("./stan/gamma_nngp.stan")

result_stan <- sampling(
  model,
  data = data_stan,
  chains = 4,
  iter = 3000)

result_stan |>
  mcmc_trace(pars = c("alpha", "sigma", "l", "inverse_phi")) +
  scale_color_discrete()

result_stan |>
  mcmc_dens_overlay(pars = c("alpha", "sigma", "l", "inverse_phi")) +
  scale_color_discrete()

result_stan |>
  mcmc_pairs(pars = c("alpha", "sigma", "l", "inverse_phi"))

stan2_df <- as_draws_df(result_stan)
stan2_summary <- stan2_df |>
  select_at(vars("alpha", "sigma", "l", "inverse_phi")) |>
  summarise_draws()
stan2_summary

# Posterior predictive distribution
y_tilde <- stan2_df |>
  slice_sample(n = 100, replace = F) |>
  select_at(vars(starts_with("y_tilde"), ".draw")) |>
  pivot_longer(!".draw")

ggplot(y_tilde, aes(x = value, group = .draw)) +
  geom_density(col = "grey60") +
  geom_density(data = data.frame(y = y2), aes(x = y, group = 1))
