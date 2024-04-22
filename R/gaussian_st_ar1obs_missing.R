library(dplyr)
library(tidyr)
library(cmdstanr)
library(bayesplot)
library(ggplot2)
library(ggforce)

source("./R/nearest_neighbor_functions.R")
source("./R/nngp_ar1obs_pred.R")

# simulate data
S <- 100
TT <- 20
n <- S * TT

# Spatial
sigma <- 2
l <- 0.3

# Temporal
rho <- 0.9
sigma_e <- 0.1

beta <- c(-1, 6)

set.seed(1233)
coords <- cbind(runif(S), runif(S))
ord <- order(coords[,1])
coords <- coords[ord,]

d <- dist(coords) |>
  as.matrix()
C <- sigma^2 * exp(-d / l)

hist(d[lower.tri(d)])
plot(d[lower.tri(C)], C[lower.tri(C)])

Sigma <- C + diag(sigma_e^2, nrow = S)

df <- tibble(
  s = rep(1:S, each = TT),
  t = rep(1:TT, times = S),
  c1 = rep(coords[,1], each = TT),
  c2 = rep(coords[,2], each = TT),
  x = rnorm(n)
) |>
  arrange(t, s)

X_t <- lapply(split(df, ~t), function(df) model.matrix(~x, df))

set.seed(251)
y_t <- matrix(NA, nrow = TT, ncol = S)
y_t[1,] <- mvtnorm::rmvnorm(1, X_t[[1]] %*% beta, Sigma / (1 - rho^2))
for (t in 2:TT) {
  mu_t <- X_t[[t]] %*% beta + rho * (y_t[t - 1,] - X_t[[t - 1]] %*% beta)
  y_t[t,] <- mvtnorm::rmvnorm(1, mu_t, Sigma)
}
df$y_all <- as.vector(t(y_t))

hist(y_t)

ggplot(df, aes(x = t, y = y_all)) +
  geom_point()

ggplot(df, aes(x = x, y = y_all)) +
  geom_point()

df_t1 <- filter(df, t == 1)
sp::coordinates(df_t1) <- ~ c1 + c2
vgm1 <- gstat::variogram(y_all ~ x, df_t1)
vgm_exp <- gstat::fit.variogram(vgm1, model = gstat::vgm("Exp"))
pred_vgm <- gstat::variogramLine(vgm_exp, maxdist = max(vgm1$dist))
ggplot(vgm1, aes(x = dist, y = gamma)) + 
  geom_point(aes(size = np)) +
  geom_line(data = pred_vgm)

set.seed(215)
sampled_stations <- sample(unique(df$s), 20)
ggplot(filter(df, s %in% sampled_stations), aes(x = t, y = y_all)) +
  geom_line() +
  facet_wrap(~s, scales = "free")

# Out Of Sample validation
test_station <- sample(1:S, 10)
test_time <- 3
missing_prop <- 0.1

train <- filter(df, !(s %in% test_station) & (t <= (max(t) - test_time)))
missing_id <- sample(c(1, 0), dim(train)[1], c(missing_prop, 1 - missing_prop), 
                     replace = TRUE)
train$y <- ifelse(missing_id == 1, NA, train$y_all)

test <- filter(df, s %in% test_station) |>
  rbind(filter(df, t > max(t) - test_time & !(s %in% test_station))) |>
  arrange(s, t)
# -----------------------------------------------------------------------------
m <- 3

id_mis <- which(is.na(train$y))
id_obs <- which(!is.na(train$y))
y_obs <- train$y[id_obs]
X_t <- lapply(split(train, ~t), model.matrix, object = ~x)

coords_train <- unique(train[, c("c1", "c2")]) |>
  as.matrix()
nn <- find_nn(coords_train, m)

data <- list(
  N = nrow(train),
  S = n_distinct(train$s),
  T = n_distinct(train$t),
  n_obs = length(id_obs),
  n_mis = length(id_mis),
  id_obs = id_obs,
  id_mis = id_mis, 
  Y_obs = y_obs, # Sorted by time
  X_t = X_t,
  p = ncol(X_t[[1]]),
  n_coords = 2,
  M = m,
  NN_ind = nn$NN_ind,
  NN_dist = nn$NN_dist,
  NN_distM = nn$NN_distM,
  coords = coords_train
)

model_file <- "./stan/nngp_ar1obs_missing.stan"

model <- cmdstan_model(model_file, compile = F)
model$compile(cpp_options = list(stan_threads = TRUE), include_paths = "./stan",
              dir = "./stan_compiled")

n_chain <- 4
n_iter <- 500
seed <- 295

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

# Convergence diagnostics
np <- nuts_params(fit)

fit$draws(variables = c("tau", "sigma", "l", "beta", "rho")) |>
  mcmc_dens_overlay() +
  scale_color_discrete()

fit$draws(variables = c("Y_mis[1]", "Y_mis[2]", "Y_mis[3]", "Y_mis[4]")) |>
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
  apply(MARGIN = 2, function(x) c(mean(x), quantile(x, probs = c(0.025, 0.975))))

# Imputation ( FIX COMMON INDEX FOR ALL DFs- s, t)
pp_mis <- fit$draws(variables = "Y_mis", format = "matrix") |>
  apply(MARGIN = 2, function(x) c(mean(x), quantile(x, probs = c(0.025, 0.975))))

mean_pred <- rep(NA, dim(train)[1])
mean_pred[id_obs] <- pp[1, id_obs]
mean_pred[id_mis] <- pp_mis[1,]

lb <- rep(NA, dim(train)[1])
lb[id_obs] <- pp[2, id_obs]
lb[id_mis] <- pp_mis[2,]

ub <- rep(NA, dim(train)[1])
ub[id_obs] <- pp[3, id_obs]
ub[id_mis] <- pp_mis[3,]

# Fitted, residuals
fit_summary <- train |>
  mutate(
    pred = mean_pred,
    lb = lb,
    ub = ub,
    resid = y_all - pred,
    resid_std = resid / sd(resid)
  )

# Fitted vs observed
ggplot(fit_summary, aes(x = pred, y = y_all)) +
  geom_point() +
  geom_ribbon(aes(ymin = lb, ymax = ub), fill = "blue", alpha = .2) +
  geom_abline(slope = 1, col = "red")

ggplot(fit_summary, aes(x = pred, y = y_all)) +
  geom_point() +
  geom_ribbon(aes(ymin = lb, ymax = ub), fill = "blue", alpha = .2) +
  geom_abline(slope = 1, col = "red") +
  facet_wrap(~is.na(y))

ggplot(fit_summary, aes(x = t, y = resid)) +
  geom_point()

# Fitted TS
set.seed(5230)
sampled_station <- sample(unique(fit_summary$s), 30)
fit_summary |>
  filter(s %in% sampled_station) |>
  ggplot(mapping = aes(x = t, y = y_all)) +
  geom_ribbon(aes(ymin = lb, ymax = ub), fill = "blue", alpha = .2) +
  geom_point() +
  geom_line(aes(y = pred), col = "blue") +
  facet_wrap(~s, scales = "free_y")

# -----------------------------------------------------------------------------
# Predict
post_sigma <- fit$draws(variables = "sigma", format = "matrix")
post_l <- fit$draws(variables = "l", format = "matrix")
post_tau <- fit$draws(variables = "tau", format = "matrix")
post_beta <- fit$draws(variables = "beta", format = "matrix")
post_rho <- fit$draws(variables = "rho", format = "matrix")

# test_BK <- test
# test <- test_BK[1:40,]
# test <- test_BK[450:470,]
# test <- test_BK

test_maxt <- test |>
  group_by(s) |>
  filter(t == max(t)) |>
  ungroup()

coords_pred <- as.matrix(test_maxt[, c("c1", "c2")])
nn_pred <- find_nn_pred(coords_pred, coords_train, m)

X_pred <- list()
pointer <- 1
for (i in 1:dim(test_maxt)[1]) {
  X_pred[[pointer]] <- model.matrix(
    ~x, df[df$s == test_maxt$s[i] & df$t <= test_maxt$t[i],])
  pointer <- pointer + 1
}

fit_summary$y_filled <- ifelse(is.na(fit_summary$y), fit_summary$pred, 
                               fit_summary$y) # Sorted by time, station
time_pred <- test_maxt$t
# dim = TT x S
y_obs <- matrix(fit_summary$y_filled, nrow = max(train$t), byrow = TRUE)
mu_obs <- matrix(fit_summary$pred, nrow = max(fit_summary$t), byrow = TRUE)

pp_pred <- pred_st(X_pred, coords_pred, time_pred, y_obs, mu_obs, 
                   coords_train, nn_pred$NN_ind, m, post_beta, post_rho, 
                   post_sigma, post_l, post_tau, "exp")

summary_stats <- function(x) c("mean" = mean(x), 
                               quantile(x, probs = c(0.025, 0.975)))
pp_pred_summary <- lapply(pp_pred, FUN = apply, MARGIN = 1, summary_stats)

test_summary <- test |>
  mutate(
    pred = unlist(lapply(pp_pred_summary, "[", 1,)),
    lb = unlist(lapply(pp_pred_summary, "[", 2,)),
    ub = unlist(lapply(pp_pred_summary, "[", 3,)),
    resid = y_all - pred,
    resid_std = resid / sd(resid)
  )

# Fitted vs observed
ggplot(test_summary, aes(x = pred, y = y_all)) +
  geom_point() +
  geom_ribbon(aes(ymin = lb, ymax = ub), fill = "blue", alpha = .2) +
  geom_abline(slope = 1, col = "red")

# Fitted TS
filter(test_summary, s %in% test_station) |>
  ggplot(mapping = aes(x = t, y = y_all)) +
  geom_ribbon(aes(ymin = lb, ymax = ub), fill = "blue", alpha = .2) +
  geom_point() +
  geom_line(aes(y = pred), col = "blue") +
  facet_wrap(~s, scales = "free_y")

# Forecast only: visualising fitted and forecasted TS
temp <- test_summary |>
  filter(!(s %in% test_station)) |>
  mutate(type = "test") |>
  bind_rows(fit_summary) |>
  mutate(type = ifelse(is.na(type), "train", type))

n_col <- 4
n_row <- 4
req_pages <- ceiling(n_distinct(temp$s) / (n_col * n_row))
for (i in 1:req_pages) {
  p <- ggplot(temp, mapping = aes(x = t, y = y_all)) +
    geom_ribbon(aes(ymin = lb, ymax = ub, fill = type), alpha = .2) +
    geom_point() +
    geom_line(aes(y = pred, col = type)) +
    facet_wrap_paginate(~s, ncol = n_col, nrow = n_row, page = i, 
                        scales = "free_y")
  print(p)
}
