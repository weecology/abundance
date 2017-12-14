library(tidyverse)
library(rstan)
set.seed(1)

`%+%` = function(a, b){t(t(a) + b)}

n_freq = 1000
N_bottleneck = 10

meta = read.csv("metadata.csv")
year = meta$year

x_raw = read_csv("x.csv") %>% 
  filter(year == 2011) %>% 
  select(bio5, bio15, ndvi_sum, elevs) %>% 
  scale()
y = read_csv("y.csv") %>% 
  filter(year == 2011)

# Bandwidth/scale/lengthscale parameter, to be optimized via cross-validation.
# One rule of thumb is to use the median distance between points.
# Another rule of thumb is 1 standard deviation.
bw = median(dist(x_raw))

# Randomly-sampled frequencies in each dimension.
# Using a zero-mean Gaussian distribution of frequencies lets us approach what
# we'd get from the "squared exponential" kernel, which is infinitely smooth.
omega = matrix(rnorm(n_freq * ncol(x_raw)), nrow = ncol(x_raw))

make_features = function(x_raw, omega){
  # Multiply the data by scaled frequencies
  x = x_raw %*% omega * bw
  
  # Most papers use random phase shifts. Supposedly, using a sine-cosine pair
  # for each frequency works better. Note that this doubles the number of 
  # features.
  sqrt(2 / (ncol(x) * 2)) * cbind(sin(x), cos(x))
}

x = make_features(x_raw, omega)

data = list(
  x = x,
  y = as.list(y),
  N = nrow(x),
  N_x = ncol(x),
  N_y = ncol(y),
  N_bottleneck = N_bottleneck
)

model = stan_model("fourier.stan")

fit = vb(model, data = data)

W1 = matrix(colMeans(as.matrix(fit, "W1")), n_freq * 2)
W2 = matrix(colMeans(as.matrix(fit, "W2_mu")), N_bottleneck)
intercept = colMeans(as.matrix(fit, "intercept_mu"))
theta = colMeans(as.matrix(fit, "NB_theta"))

# Plot the curve for a few random species's response to one column, 
# holding the other columns constant at zero.
col = 4 # Index for the nonzero column
xx = x_raw * 0
xx[,col] = seq(min(x_raw[,col]), max(x_raw[,col]), length.out = nrow(xx))

matplot(
  xx[,col], 
  make_features(xx, omega) %*% W1 %*% W2[,sample.int(ncol(y), 6)], 
  type = "l", 
  lty = 1
)
