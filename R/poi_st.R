library(dplyr)
library(tidyr)
library(rstan)
library(posterior)
library(bayesplot)

source("./R/NNMatrix.R")

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# simulate data
S <- 20
TT <- 100
n <- S * TT

# Spatial
sigma <- 2
l <- 1

# Temporal
rho <- 0.6

# Intercept
alpha <- 3

set.seed(1233)
coords <- cbind(runif(S), runif(S))

ord <- order(coords[,1])
coords <- coords[ord,]

d <- dist(coords) |>
  as.matrix()
C <- sigma^2 * exp(-d / (2 * l^2))

plot(d[lower.tri(C)], C[lower.tri(C)])

w_s <- mvtnorm::rmvnorm(1, sigma = C) |>
  c()

w_mat <- matrix(nrow = TT, ncol = S)
w_mat[1,] <- w_s
for (t in 2:TT) {
  w_s <- mvtnorm::rmvnorm(1, sigma = C) |>
    c()
  w_mat[t,] <- rho * w_mat[t - 1,] + sqrt(1 - rho^2) * w_s
}
w <- as.vector(w_mat)

mu <- exp(alpha + w)

y2 <- rpois(n, mu)
plot(1:TT, w_mat[, 1], "l")
hist(y2)

# -----------------------------------------------------------------------------
m <- 5

nn <- NNMatrix(coords, n.neighbors = m)

data_stan <- list(
  S = S,
  T = TT,
  N = n,
  Y = y2,
  M = m,
  NN_ind = nn$NN_ind,
  NN_dist = nn$NN_dist,
  NN_distM = nn$NN_distM
)

model <- stan_model("./stan/poi_nngp_ar1.stan")

result_stan <- sampling(
  model,
  data = data_stan,
  chains = 4,
  iter = 1000)

result_stan |>
  mcmc_trace(pars = c("alpha", "ar", "sigma", "l")) +
  scale_color_discrete()

result_stan |>
  mcmc_dens_overlay(pars = c("alpha", "ar", "sigma", "l")) +
  scale_color_discrete()

result_stan |>
  mcmc_pairs(pars = c("alpha", "ar", "sigma", "l"))

stan2_df <- as_draws_df(result_stan)
stan2_summary <- stan2_df |>
  select_at(vars("alpha", "ar", "sigma", "l")) |>
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
