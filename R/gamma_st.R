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
sigma <- 1
l <- .6

# Temporal
rho <- 0.8
prec <- 10

# Intercept
alpha <- 1

set.seed(1239)
coords <- cbind(runif(S), runif(S))

ord <- order(coords[,1])
coords <- coords[ord,]
d <- dist(coords) |>
  as.matrix()
C <- sigma^2 * exp(-d / l^2)

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

a <- prec
b <- prec / mu

y2 <- rgamma(n, shape = a, rate = b)
plot(1:TT, w_mat[, 1], "l")
hist(y2)

df <- tibble(
  s = rep(1:S, each = TT),
  t = rep(1:TT, S),
  c1 = rep(coords[,1], each = TT),
  c2 = rep(coords[,2], each = TT),
  y = y2,
  w = w
)

# -----------------------------------------------------------------------------
m <- 3

nn <- NNMatrix(coords, n.neighbors = m)

data_stan <- list(
  S = S,
  T = TT,
  N = n,
  Y = y2,
  M = m,
  NN_ind = nn$NN_ind,
  NN_dist = nn$NN_dist,
  NN_distM = nn$NN_distM)

model <- stan_model("./stan/gamma_nngp_ar1.stan")

result_stan <- sampling(
  model,
  data = data_stan,
  chains = 4,
  iter = 1500)
np <- nuts_params(result_stan)

result_stan |>
  mcmc_trace(
    pars = c("alpha", "ar", "sigma", "l", "inverse_phi"), np = np) +
  scale_color_discrete()

result_stan |>
  mcmc_dens_overlay(pars = c("alpha", "ar", "sigma", "l", "inverse_phi")) +
  scale_color_discrete()

result_stan |>
  mcmc_pairs(
    pars = c("alpha", "ar", "sigma", "l", "inverse_phi"), off_diag_fun = "hex", np = np)

stan2_df <- as_draws_df(result_stan)
stan2_summary <- stan2_df |>
  select_at(vars("alpha", "ar", "sigma", "l", "inverse_phi")) |>
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

w_df <- stan2_df |>
  select(starts_with("err")) |>
  summarise_draws() |>
  mutate(
    s = rep(1:S, each = TT),
    t = rep(1:TT, S),
  )

w_df <- w_df |>
  left_join(df, by = c("t", "s")) |>
  mutate(
    resid = median - w,
    resid_std = resid / sd(resid)
  )

w_df |>
  ggplot(mapping = aes(x = 1:nrow(w_df), y = resid_std)) +
  geom_point()
