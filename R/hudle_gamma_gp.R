library(dplyr)
library(tidyr)
library(rstan)
library(posterior)
library(bayesplot)

source("./R/NNMatrix.R")
source("./R/predict_nngp.R")

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# simulate data
n <- 100

# Spatial
sigma <- 1
l <- 2.3

alpha <- 1
prec <- 1.2
alpha_h <- 2

set.seed(129)
coords <- cbind(runif(n), runif(n))
ord <- order(coords[,1])
coords <- coords[ord,]

plot(coords)

d <- dist(coords) |>
  as.matrix()
w <- mvtnorm::rmvnorm(1, sigma = sigma^2 * exp(-d / l^2)) |>
  c()

mu <- exp(alpha + w)
prob <- 1 / (1 + exp(-(alpha_h + w)))

theta <- rbinom(n, 1, prob) # Simulate non-zero values
y <- theta * rgamma(n, shape = prec, rate = prec / mu)

prop_test <- .2
test_id <- sample(1:n, size = round(n * prop_test))

hist(y)
table(y == 0) |>
  prop.table()

df <- tibble(
  id = 1:n,
  y,
  c1 = coords[,1],
  c2 = coords[,2],
  w,
  test = id %in% test_id
)

train <- df |>
  filter(!test)
test <- df |>
  filter(test)
# -----------------------------------------------------------------------------
m <- 5

nn <- NNMatrix(coords[!df$test,], n.neighbors = m)

data_stan <- list(
  N = nrow(train),
  Y = train$y,
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
  iter = 4000)

result_stan |>
  mcmc_trace(pars = c("alpha", "sigma", "l", "inverse_phi", "alpha_hu")) +
  scale_color_discrete()

result_stan |>
  mcmc_dens_overlay(pars = c("alpha", "sigma", "l", "inverse_phi", "alpha_hu")) +
  scale_color_discrete()

result_stan |>
  mcmc_pairs(
    pars = c("alpha", "sigma", "l", "inverse_phi", "alpha_hu"),
    transformations = list(sigma = "log", l = "log", inverse_phi = "log"))

stan2_df <- as_draws_df(result_stan)
stan2_summary <- stan2_df |>
  select_at(vars("alpha", "sigma", "l", "inverse_phi", "alpha_hu")) |>
  summarise_draws()
stan2_summary

n_samples <- 100
set.seed(28)
idx <- sample(1:nrow(stan2_df), n_samples)

mu <- stan2_df |>
  slice(idx) |>
  select(starts_with("mu[")) |>
  as.matrix()
inverse_phi <- stan2_df |>
  slice(idx) |>
  pull(inverse_phi)

prob <- stan2_df |>
  select(starts_with("prob")) |>
  summarise_draws()
prob

# Posterior predictive distribution
set.seed(248)
pp <- sapply(1:n_samples, FUN = function(x){
  rgamma(y[y > 0], inverse_phi[x], rate = inverse_phi[x] / mu[x, ])
}) |>
  t() |>
  as_tibble() |>
  mutate(sample = 1:n_samples) |>
  pivot_longer(!sample)

ggplot(pp, aes(x = value, group = sample)) +
  geom_density(col = "grey60") +
  geom_density(data = filter(train, y > 0), aes(x = y, group = 1))

# Predict for new observations
parms_hat <- stan2_df |>
  select(l, sigma, alpha, alpha_hu) |>
  summarise_draws("median")

sigma_hat <- parms_hat |>
  filter(variable == "sigma") |>
  pull(median)
l_hat <- parms_hat |>
  filter(variable == "l") |>
  pull(median)
alpha_hat <- parms_hat |>
  filter(variable == "alpha") |>
  pull(median)
alpha_h_hat <- parms_hat |>
  filter(variable == "alpha_hu") |>
  pull(median)

w_hat <- lapply(1:nrow(test), function(i) {
  w_hat <- predict_nngp(coords_new = coords[df$test, ][i,], train$w, sigma_hat, l_hat, coords[!df$test,], m)
  tibble(mean = w_hat$mean, var = w_hat$var[1])
}) |>
  bind_rows()

prob_hat <- 1 - 1 / (1 + exp(-(alpha_h_hat + w_hat$mean))) # prob non-zero
mu_hat <- exp(alpha_hat + w_hat$mean)

test <- test |>
  mutate(
    prob_hat,
    mu_hat,
    w_hat = w_hat$mean,
    pred = prob_hat * mu_hat,
    resid = y - pred
  )

prob_hat <- 1 - 1 / (1 + exp(-(alpha_h_hat + w_hat$mean))) # prob non-zero
mu_hat <- exp(alpha_hat + w_hat$mean)

ggplot(test, aes(x = y)) +
  geom_density() +
  geom_density(aes(x = pred), col = "grey")


# Predictive distribution for new point
n_samples <- 500
set.seed(28)
idx <- sample(1:nrow(stan2_df), n_samples)

df_sampled <- stan2_df |>
  select_at(vars("alpha", "sigma", "l", "alpha_hu")) |>
  slice(idx)

df_hat <- tibble()
for (j in 1:nrow(test)) {
  coords_new <- coords[df$test, ][j,]
  y_hat <- sapply(1:n_samples, function(i){
    temp <- df_sampled[i,]
    w_hat <- predict_nngp(
      coords_new = coords_new, train$w, temp$sigma, temp$l, coords[!df$test,], m)
    
    prob_hat <- 1 - 1 / (1 + exp(-(alpha_h_hat + w_hat$mean))) # prob non-zero
    mu_hat <- exp(alpha_hat + w_hat$mean)
    prob_hat * mu_hat
  })
  temp <- tibble(y = y_hat, id = test$id[j], draw = 1:n_samples)
  df_hat <- bind_rows(df_hat, temp)
}

df_hat |>
  group_by(id) |>
  summarise(
    y = mean(y)
  )
