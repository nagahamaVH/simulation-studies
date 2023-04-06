library(dplyr)
library(tidyr)
library(INLA)
library(brms)
library(rstan)
library(posterior)
library(bayesplot)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# simulate data
n <- 100
rho <- 0.8
prec <- 10
alpha <- 1
prec.par <- 4

# u_i | theta ~ AR(1)
# eta_i = alpha + u_i
# y_i | eta ~ Gamma(., .)
set.seed(1239)
x <- as.vector(
  arima.sim(list(order = c(1, 0, 0), ar = rho), n = n, sd = sqrt(1 / prec)))
mu <- exp(alpha + x)

a <- prec.par
b <- prec.par / mu

y2 <- rgamma(n, shape = a, rate = b)

# -----------------------------------------------------------------------
# INLA sim 2
# -----------------------------------------------------------------------
data_inla2 <- list(y = y2, z = 1:n)

## fit the model
formula <- y ~ 1 + f(z, model = "ar1")
result_inla2 <- inla(formula, family = "gamma", data = data_inla2)

summary(result_inla2)

pp <- posterior_predict(result_brms2)
pp <- as_tibble(pp) |>
  mutate(
    draw = 1:n()
  ) |>
  slice_sample(n = 100) |>
  pivot_longer(!"draw")

ggplot(pp, aes(x = value, group = draw)) +
  geom_density(col = "grey60") +
  geom_density(data = data_brms2, aes(x = y2, group = 1))

# -----------------------------------------------------------------------
# BRMS sim 2
# ------------------------------------------------------------------------
data_brms2 <- data.frame(y = y2)

# get_prior(y ~ 1 + ar(p = 1), family = Gamma(link = "log"), data = data_brms2)
prior_brms <- c(
  prior_string("normal(0, .5)", class = "ar"),
  prior_string("normal(1, 1)", class = "Intercept"),
  prior_string("inv_gamma(2, 1)", class = "sderr"),
  prior_string("inv_gamma(2, 1)", class = "shape")
)

result_brms2 <- brm(
  y ~ 1 + ar(p = 1), family = Gamma(link = "log"), data = data_brms2, 
  prior = prior_brms)

plot(result_brms2)
summary(result_brms2)
result_brms2$fit@stanmodel

# -----------------------------------------------------------------------
# Stan
# ------------------------------------------------------------------------
data_stan2 <- list(T = length(y2), Y = y2)

model <- stan_model("./stan/gamma_ar1.stan")

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
  ggplot(aes(x = 1:n, y = median)) +
  geom_line()

# Posterior predictive distribution
y_tilde <- stan2_df |>
  slice_sample(n = 100, replace = F) |>
  select_at(vars(starts_with("y_tilde"), ".draw")) |>
  pivot_longer(!".draw")

ggplot(y_tilde, aes(x = value, group = .draw)) +
  geom_density(col = "grey60") +
  geom_density(data = data.frame(y = y2), aes(x = y, group = 1))
