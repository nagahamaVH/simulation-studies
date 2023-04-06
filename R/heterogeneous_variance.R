library(dplyr)
library(nlme)

# Simulating data
set.seed(256)
n <- 1000
n_group <- 3

df <- tibble(
  group = sample(LETTERS[1:n_group], n, replace = T)
) %>%
  mutate(
    y = case_when(
      group == "A" ~ rnorm(n, 7, 3.2),
      group == "B" ~ rnorm(n, 0.5, 1),
      group == "C" ~ rnorm(n, -2, 2)
    )
  )

boxplot(y ~ group, df) # notice the heteroscedasticity between groups

# Homoscedastic and heteroscedastic variance model
model0 <- lm(y ~ group, df)
model1 <- gls(y ~ group, weights = varIdent(form = ~ 1 | group), df)

par(mfrow = c(1,2))
boxplot(resid(model0) ~ df$group); abline(h = 0)
boxplot(resid(model1) ~ df$group); abline(h = 0)
par(mfrow = c(1,1))

# Comparing models
anova(model1, model0)

# Marginal effect
me0 <- c(coef(model0)[1], coef(model0)[-1] + coef(model0)[1])
me1 <- c(coef(model0)[1], coef(model1)[-1] + coef(model1)[1])

# Estimation of variance parameters
## Homogeneous model
summary(model0)$sigma

## Heterogeneous model
model1$modelStruct$varStruct
model1$sigma

# Confidence interval of fitted values
# predict(model0, interval = "confidence") %>%
#   as_tibble()

# Bootstrap CI
# predict(model1, interval = "confidence") %>%
#   as_tibble()
