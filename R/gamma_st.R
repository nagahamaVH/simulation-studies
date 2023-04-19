library(dplyr)
library(tidyr)
library(bmstdr)
library(rstan)
library(posterior)
library(bayesplot)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# f2 <- y8hrmax~xmaxtemp+xwdsp+xrh
# M2 <- Bsptime(
#   model="separable", formula=f2, data=nysptime, coordtype="utm", coords=4:5, 
#   package = "stan", N = 1, burn.in = 0)
# get_stanmodel(M2$fit)

# simulate data
S <- 5
TT <- 100
n <- S * TT

# Spatial
sigma <- 1
l <- 2.3

# Temporal
rho <- 0.8
prec <- 10
prec.par <- 2

# Intercept
alpha <- 1

# w_s | theta ~ GP(0, C)
# u_t | theta ~ AR(1)
# eta_{s, t} = alpha + w_s + u_t
# y_{s, t} | eta ~ Gamma(shape, scale)
set.seed(1239)
coords <- cbind(runif(S), runif(S))

ord <- order(coords[,1])
coords <- coords[ord,]

d <- dist(coords) |>
  as.matrix()
w <- mvtnorm::rmvnorm(1, sigma = sigma^2 * exp(-d / l^2)) |>
  c()

u <- as.vector(arima.sim(list(order = c(1, 0, 0), ar = rho), n = TT, sd = sqrt(1 / prec)))

# mu <- exp(alpha + rep(w, each = TT) + rep(u, times = S))
mu <- exp(alpha + rep(w, each = TT) + rep(u, times = S))

# a <- mu^2 / prec.par
# b <- mu / prec.par

a <- prec.par
b <- prec.par / mu

y2 <- rgamma(n, shape = a, rate = b)

# -----------------------------------------------------------------------
# INLA sim 2
# -----------------------------------------------------------------------
data_inla <- tibble(
  y = y2, 
  loc = rep(1:S, each = TT), 
  t = rep(1:TT, times = S), 
)

result_inla <- inla(
  y ~ 1 + f(loc, model = "dmatern", locations = coords) + 
    f(t, model = "ar1", hyper = list(prec = list(param = c(10, 100)))),
  data = data_inla, family = "gamma", control.compute = list(config = T)
)

summary(result_inla)

bri.hyperpar.plot(result_inla, together = F)
bri.fixed.plot(result_inla, together = F)

# Posterior predictive distribution
sim <- inla.posterior.sample(100, result_inla)

pp_df <- tibble()
for (i in 1:length(sim)) {
  df_i <- tibble(
    draw = i,
    y = exp(sim[[i]]$latent[,1])
  )
  pp_df <- pp_df |>
    bind_rows(df_i)
}

ggplot(pp_df, aes(x = y, group = draw)) +
  geom_density(col = "grey60") +
  geom_density(data = data_inla, aes(x = y, group = 1)) +
  xlim(0, 10)

# -----------------------------------------------------------------------
# Stan
# ------------------------------------------------------------------------
data_stan2 <- list(T = length(y2), Y = y2)

model <- stan_model("./stan/gamma_nngp_ar1.stan")

result_stan2 <- sampling(
  model,
  data = data_stan2,
  chains = 4,
  iter = 3000)

result_stan2 |>
  mcmc_dens_overlay(pars = c("alpha", "rho", "sigma_ar", "inverse_phi")) +
  scale_color_discrete()

stan2_df <- as_draws_df(result_stan2)
stan2_summary <- stan2_df |>
  select_at(vars("alpha", "rho", "sigma_ar", "inverse_phi", starts_with("u"))) |>
  summarise_draws()

stan2_summary

stan2_summary |>
  filter(stringr::str_detect(variable, "^u")) |>
  ggplot(aes(x = median)) +
  geom_density()

stan2_summary |>
  filter(stringr::str_detect(variable, "^u")) |>
  ggplot(aes(x = 1:100, y = median)) +
  geom_line()

# Posterior predictive distribution
y_tilde <- stan2_df |>
  slice_sample(n = 100, replace = F) |>
  select_at(vars(starts_with("y_tilde"), ".draw")) |>
  pivot_longer(!".draw")

ggplot(y_tilde, aes(x = value, group = .draw)) +
  geom_density(col = "grey60") +
  geom_density(data = data.frame(y = y2), aes(x = y, group = 1))
