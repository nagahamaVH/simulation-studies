library(dplyr)
library(tidyr)
library(cmdstanr)
library(bayesplot)
library(ggplot2)
library(ggforce)

source("./R/nearest_neighbor_functions.R")
source("./R/nngp_ar1obs_varslopes_pred.R")

model_board <- pins::board_folder("models", versioned = T)

set.seed(1521)

# simulate data
S <- 100
TT <- 20
n <- S * TT

# Spatial
sigma <- 2
l <- 0.3

# Temporal
rho <- 0.6
sigma_e <- 0.1

beta <- c(-1, 4)
sigma_beta <- 1
beta_s <- rnorm(S, beta[2], sd = sigma_beta)

set.seed(1233)
coords <- cbind(runif(S), runif(S))
ord <- order(coords[,1])
coords <- coords[ord,]

d <- dist(coords) |>
  as.matrix()
C <- sigma^2 * exp(-d / l)

plot(d[lower.tri(C)], C[lower.tri(C)])

Sigma <- C + diag(sigma_e^2, nrow = S)

df <- tibble(
  s = rep(1:S, each = TT),
  t = rep(1:TT, times = S),
  c1 = rep(coords[,1], each = TT),
  c2 = rep(coords[,2], each = TT),
  x = rnorm(n)
)

X_t <- lapply(split(df, ~t), function(df) model.matrix(~x, df))

set.seed(251)
y_t <- matrix(NA, nrow = TT, ncol = S)
y_t[1,] <- mvtnorm::rmvnorm(1, beta[1] * X_t[[1]][,1] + beta_s * X_t[[1]][,2], 
                            Sigma / (1 - rho^2))
for (t in 2:TT) {
  e_t <- y_t[t - 1,] - (beta[1] * X_t[[t - 1]][,1] + beta_s * X_t[[t - 1]][,2])
  mu_t <- beta[1] * X_t[[t]][,1] + beta_s * X_t[[t]][,2] + rho * e_t
  y_t[t,] <- mvtnorm::rmvnorm(1, mu_t, Sigma)
}
df$y <- as.vector(y_t)

hist(y_t)

ggplot(df, aes(x = x, y = y)) +
  geom_point()

ggplot(df, aes(x = x, y = y)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~s)

set.seed(215)
sampled_stations <- sample(unique(df$s), 20)
ggplot(filter(df, s %in% sampled_stations), aes(x = t, y = y)) +
  geom_line() +
  facet_wrap(~s, scales = "free")

# Out Of Sample validation
test_station <- sample(1:S, 10)
test_time <- 3

train <- filter(df, !(s %in% test_station) & (t <= (max(t) - test_time)))
test <- filter(df, s %in% test_station) |>
  rbind(filter(df, t > max(t) - test_time & !(s %in% test_station)))

# test <- filter(df, (t > (max(t) - test_time)) & !(s %in% test_station))
# test <- filter(df, s %in% test_station)
# -----------------------------------------------------------------------------
m <- 3

y_t <- lapply(split(train, ~t), function(df) df$y)
X_t <- lapply(split(train, ~t), model.matrix, object = ~x)

coords_train <- unique(train[, c("c1", "c2")]) |>
  as.matrix()
nn <- find_nn(coords_train, m)

data <- list(
  N = nrow(train),
  S = n_distinct(train$s),
  T = n_distinct(train$t),
  Y_t = y_t,
  X_t = X_t,
  p = ncol(X_t[[1]]),
  M = m,
  NN_ind = nn$NN_ind,
  NN_dist = nn$NN_dist,
  NN_distM = nn$NN_distM,
  coords = coords_train
)

model_file <- "./stan/nngp_ar1obs_varslopes.stan"

model <- cmdstan_model(model_file, compile = F)
model$compile(cpp_options = list(stan_threads = TRUE), include_paths = "./stan",
              dir = "./stan_compiled")

n_chain <- 4
n_iter <- 500
seed <- 295

pins::pin_read(model_board, "st_ar1obs_varslopes")
fit <- model$sample(
  data = data,
  chains = n_chain,
  parallel_chains = n_chain, 
  threads_per_chain = n_chain,
  iter_sampling = floor(n_iter / 2),
  iter_warmup = floor(n_iter / 2),
  refresh = round(n_iter / 10),
  seed = seed
)

# pins::pin_write(
#   model_board, fit, 
#   name = "st_ar1obs_varslopes", 
#   type = "rds")

# Convergence diagnostics
np <- nuts_params(fit)

fit$draws(variables = c("tau", "sigma", "l", "beta", "rho", "tau_beta")) |>
  mcmc_dens_overlay() +
  scale_color_discrete()

beta_s[c(1, 5, 12)]
fit$draws(variables = c("beta_s[1]", "beta_s[5]", "beta_s[12]")) |>
  mcmc_dens_overlay() +
  scale_color_discrete()

fit$draws(variables = c("tau", "sigma", "l", "rho")) |>
  summary()

fit$draws(variables = c("tau", "sigma", "l", "rho", "beta")) |>
  mcmc_pairs(
    transformations = list(sigma = "log", l = "log", tau = "log"),
    off_diag_fun = "hex",
    np = np)

# Posterior predictive checking
pp <- fit$draws(variables = "y_sim", format = "matrix") |>
  apply(MARGIN = 2, function(x) c(mean(x), quantile(x, probs = c(0.1, 0.9))))

# Fitted, residuals
fit_summary <- train |>
  mutate(
    pred = pp[1,],
    lb = pp[2,],
    ub = pp[3,],
    resid = y - pred,
    resid_std = resid / sd(resid)
  )

# Fitted vs observed
ggplot(fit_summary, aes(x = pred, y = y)) +
  geom_point() +
  geom_ribbon(aes(ymin = lb, ymax = ub), fill = "blue", alpha = .1) +
  geom_abline(slope = 1, col = "red")

# Fitted TS
set.seed(5230)
sampled_station <- sample(unique(fit_summary$s), 30)
fit_summary |>
  filter(s %in% sampled_station) |>
  ggplot(mapping = aes(x = t, y = y)) +
  geom_ribbon(aes(ymin = lb, ymax = ub), fill = "blue", alpha = .1) +
  geom_point() +
  geom_line(aes(y = pred), col = "blue") +
  facet_wrap(~s, scales = "free_y")

# -----------------------------------------------------------------------------
# Predict
post_sigma <- fit$draws(variables = "sigma", format = "matrix")
post_l <- fit$draws(variables = "l", format = "matrix")
post_tau <- fit$draws(variables = "tau", format = "matrix")
post_beta <- fit$draws(variables = "beta", format = "matrix")
post_beta_s <- fit$draws(variables = "beta_s", format = "matrix")
post_rho <- fit$draws(variables = "rho", format = "matrix")

coords_pred <- as.matrix(test[, c("c1", "c2")])
nn_pred <- find_nn_pred(coords_pred, coords_train, m)

X_pred <- list()
pointer <- 1
for (i in 1:dim(test)[1]) {
  X_pred[[pointer]] <- model.matrix(
    ~x, df[df$s == test$s[i] & df$t <= test$t[i],])
  pointer <- pointer + 1
}

time_pred <- test$t
y_obs <- matrix(train$y, nrow = max(train$t)) # dim = S x TT
mu_obs <- matrix(fit_summary$pred, nrow = max(fit_summary$t))

pins::pin_read(model_board, "st_ar1obs_varslopes_predictions")

pp_pred <- pred_st(X_pred, coords_pred, time_pred, y_obs, mu_obs, coords_train, 
                   post_beta, post_beta_s, post_rho, post_sigma, post_l, 
                   post_tau, "exp")
pp_pred <- simplify2array(pp_pred)

# pins::pin_write(
#   model_board, pp_pred, 
#   name = "st_ar1obs_varslopes_predictions", 
#   type = "rds")

pp_pred_summary <- apply(pp_pred, 2, function(x) 
  c(mean(x), quantile(x, probs = c(0.1, 0.9))))

test_summary <- test |>
  mutate(
    pred = pp_pred_summary[1,],
    lb = pp_pred_summary[2,],
    ub = pp_pred_summary[3,],
    resid = y - pred,
    resid_std = resid / sd(resid)
  )

# Fitted vs observed
ggplot(test_summary, aes(x = pred, y = y)) +
  geom_point() +
  geom_ribbon(aes(ymin = lb, ymax = ub), fill = "blue", alpha = .1) +
  geom_abline(slope = 1, col = "red")

# Fitted TS
filter(test_summary, s %in% test_station) |>
  ggplot(mapping = aes(x = t, y = y)) +
  geom_ribbon(aes(ymin = lb, ymax = ub), fill = "blue", alpha = .1) +
  geom_point() +
  geom_line(aes(y = pred), col = "blue") +
  facet_wrap(~s, scales = "free_y")

# Forecast only: visualising fitted and forecasted TS
temp <- test_summary |>
  filter(!(s %in% test_station)) |>
  mutate(type = "test") |>
  bind_rows(fit_summary) |>
  mutate(type = ifelse(is.na(type), "train", type))

n_col <- 3
n_row <- 4
req_pages <- ceiling(n_distinct(temp$s) / (n_col * n_row))
for (i in 1:req_pages) {
  p <- ggplot(temp, mapping = aes(x = t, y = y)) +
    geom_ribbon(aes(ymin = lb, ymax = ub, fill = type), alpha = .2) +
    geom_point() +
    geom_line(aes(y = pred, col = type)) +
    facet_wrap_paginate(~s, ncol = n_col, nrow = n_row, page = i, 
                        scales = "free_y")
  print(p)
}
