library(dplyr)
library(tidyr)
library(cmdstanr)
# library(posterior)
library(bayesplot)
library(ggplot2)

source("./R/NNMatrix.R")

# simulate data
S <- 100
TT <- 30
n <- S * TT

# Spatial
sigma <- 2
l <- 0.3

# Temporal
rho <- 0.9
sigma_e <- 0.1

# Intercept
alpha <- 3

set.seed(1233)
coords <- cbind(runif(S), runif(S))
ord <- order(coords[,1])
coords <- coords[ord,]

d <- dist(coords) |>
  as.matrix()
C <- sigma^2 * exp(-d / l)

plot(d[lower.tri(C)], C[lower.tri(C)])

Sigma <- C + diag(sigma_e^2, nrow = S)

set.seed(251)
y_t <- mvtnorm::rmvnorm(TT, rep(alpha, S), Sigma / (1 - rho^2)) # dim = TT x S
for (t in 2:TT) {
  mu_t <- alpha + rho * (y_t[t - 1,] - alpha)
  y_t[t,] <- mvtnorm::rmvnorm(1, mu_t, Sigma)
}

hist(y_t)

df <- tibble(
  s = rep(1:S, each = TT),
  t = rep(1:TT, times = S),
  c1 = rep(coords[,1], each = TT),
  c2 = rep(coords[,2], each = TT),
  y = as.vector(y_t)
)

df_t1 <- filter(df, t == 1)
sp::coordinates(df_t1) <- ~ c1 + c2
vgm1 <- gstat::variogram(y ~ 1, df_t1)
vgm_exp <- gstat::fit.variogram(vgm1, model = gstat::vgm("Exp"))
pred_vgm <- gstat::variogramLine(vgm_exp, maxdist = max(vgm1$dist))
ggplot(vgm1, aes(x = dist, y = gamma)) + 
  geom_point(aes(size = np)) +
  geom_line(data = pred_vgm)

set.seed(215)
sampled_stations <- sample(unique(df$s), 20)
ggplot(filter(df, s %in% sampled_stations), aes(x = t, y = y)) +
  geom_line() +
  facet_wrap(~s, scales = "free")

# -----------------------------------------------------------------------------
m <- 3

nn <- NNMatrix(coords, n.neighbors = m)

data <- list(
  N = nrow(df),
  S = n_distinct(df$s),
  T = n_distinct(df$t),
  Y_t = lapply(split(df, ~t), function(x) x$y),
  X_t = lapply(1:TT, function(x) as.matrix(rep(1, S))),
  p = ncol(X),
  M = m,
  NN_ind = nn$NN_ind,
  NN_dist = nn$NN_dist,
  NN_distM = nn$NN_distM,
  coords = coords
)

model_file <- "./stan/nngp_ar1obs.stan"

model <- cmdstan_model(model_file, compile = F)
model$compile(cpp_options = list(stan_threads = TRUE), include_paths = "./stan",
              dir = "./stan_compiled")

n_chain <- 4
n_iter <- 500
seed <- 295

t_init <- proc.time()
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
t_total <- proc.time() - t_init

# Convergence diagnostics
np <- nuts_params(fit)

fit$draws(variables = c("tau", "sigma", "l", "beta", "rho")) |>
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
fit_summary <- df |>
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
