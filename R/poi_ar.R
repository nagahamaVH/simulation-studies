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
alpha <- 3
## note that the marginal precision would be
marg.prec <- prec * (1 - rho^2) # marg.prec = 3.6, var = 0.2778

# u_i | theta ~ AR(1)
# mu_i = exp(alpha + u_i)
# y_i | eta ~ Poisson(mu_i)
set.seed(1239)
x <- as.vector(arima.sim(list(order = c(1, 0, 0), ar = rho), n = n, sd = sqrt(1 / prec)))
y <- rpois(n, exp(x))
y2 <- rpois(n, exp(alpha + x))

# -----------------------------------------------------------------------
# INLA sim 1
# -----------------------------------------------------------------------
data_inla <- list(y = y, z = 1:n)

formula <- y ~ 1 + f(z, model = "ar1")
result_inla1 <- inla(formula, family = "poisson", data = data_inla)

summary(result_inla1)

# -----------------------------------------------------------------------
# INLA sim 2
# -----------------------------------------------------------------------
data_inla2 <- list(y = y2, z = 1:n)

## fit the model
formula <- y ~ 1 + f(z, model = "ar1")
result_inla2 <- inla(formula, family = "poisson", data = data_inla2)

summary(result_inla2)

# -----------------------------------------------------------------------
# BRMS sim 1
# ------------------------------------------------------------------------
data_brms <- data.frame(y = y)
result_brms <- brm(y ~ 1 + ar(p = 1), family = poisson(), data = data_brms)

summary(result_brms)

# -----------------------------------------------------------------------
# BRMS sim 2
# ------------------------------------------------------------------------
data_brms2 <- data.frame(y = y2)
result_brms2 <- brm(y ~ 1 + ar(p = 1), family = poisson(), data = data_brms2)

summary(result_brms2)
result_brms2$fit@stanmodel
# -----------------------------------------------------------------------
# Stan
# ------------------------------------------------------------------------
data_stan2 <- list(T = length(y2), Y = y2)

model <- stan_model("./stan/poi_ar1.stan")

result_stan2 <- sampling(
  model,
  data = data_stan2,
  chains = 4,
  iter = 3000)

result_stan2 |>
  mcmc_dens_overlay(pars = c("alpha", "rho", "sigma_ar")) +
  scale_color_discrete()

stan2_df <- as_draws_df(result_stan2)
stan2_summary <- stan2_df |>
  select_at(vars("alpha", "rho", "sigma_ar", starts_with("u"))) |>
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
