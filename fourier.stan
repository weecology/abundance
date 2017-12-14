data {
  int N;
  int N_x;
  int N_y;
  int N_bottleneck;
  matrix[N, N_x] x;
  
  // Array dimensions are switched vs. matrices for efficient indexing
  int y[N_y, N];
}

parameters {
  matrix[N_x, N_bottleneck] W1;
  matrix[N_bottleneck, N_y] W2_mu;
  real<lower=0> NB_theta[N_y];
  
  vector[N_y] intercept_mu;
}

model {
  matrix[N, N_y] XW_mu; // Linear predictor apart from intercepts
  matrix[N_x, N_y] W_mu = W1 * W2_mu; // Full weight matrix
  
  ////
  // Priors
  ////
  for (i in 1:N_x) {
    // Need to confirm that no Jacobian adjustment is necessary: See question
    // at http://discourse.mc-stan.org/t/jacobian-for-matrix-product/2756
    W_mu[i] ~ normal(0.0, 1.0);
  }
  NB_theta ~ lognormal(2, 1);
  intercept_mu ~ normal(0, 3);
  
  ////
  // Conditional likelihood
  ////
  
  XW_mu = x * W_mu;
  for (i in 1:N_y) {
    vector[N] mu = exp(intercept_mu[i] + XW_mu[:,i]);
    y[i] ~ neg_binomial_2(mu, NB_theta[i]);  
  }
}

