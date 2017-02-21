"""Functions for zero-inflated negative binomial model in tensorflow"""

import numpy as np
import tensorflow as tf

# Negative binomial & its helpers
def log1p(x):
    """log(1+x) when x is close to zero; based on npy_log1p in npy_math.c.src"""

    one_plus_x = 1.0 + x
    # 1+x-1 should just be x, but apparently this is more numerically stable?
    # I'm just doing what numpy does.
    # No idea why we multiply by x and then divide by it again, either.
    other_x = one_plus_x - 1.0
    value = tf.log(one_plus_x) * x / other_x

    # return x wherever x is +Inf, zero wherever x is zero
    value = tf.where(tf.is_inf(x), x, value)
    value = tf.where(tf.equal(x, 1.), tf.zeros_like(value), value)
    return value

def xlog1py(x, y):
    """return x*log1p(y), unless x==0; based on scipy.special.xlog1py"""
    value = x * log1p(y)
    return tf.where(tf.equal(x, 0.), tf.zeros_like(value), value)

def nbinom_ll(x, n, p):
    """modified from scipy.stats.nbinom._lpmf"""
    with tf.variable_scope("negative_binomial"):
        coeff = tf.lgamma(n+x) - tf.lgamma(x+1) - tf.lgamma(n)
        out = coeff + n*tf.log(p) + xlog1py(x, -p)
    return out

def log_sum_exp(x, y):
    """numerically stable version of log(exp(x) + exp(y))"""
    larger_values = tf.maximum(x, y)
    return larger_values + tf.log(tf.exp(x - larger_values) + tf.exp(y - larger_values))

# Zero-inflated negative binomial likelihood
def zi_nbinom_ll(x, n, p, zi_p):
    """zero-inflated negative binomial likelihood"""
    with tf.variable_scope("zero_inflation"):
        is_zero = tf.cast(tf.equal(x, tf.zeros_like(x)), tf.float32)
        count_ll = nbinom_ll(x, n, p)

        # Zero could be zero-inflated, _or_ a non-inflated while _also_ matching count distribution
        zeros_lls = log_sum_exp(tf.log(zi_p), tf.log(1 - zi_p) + count_ll)
        # Nonzero can't be zero-inflated, _and_ must match the count distribution
        nonzeros_lls = tf.log(1 - zi_p) + count_ll

        out = zeros_lls * is_zero + nonzeros_lls * (1-is_zero)
        return out

# Equation 10 from Kingma & Welling
def kl(mu, sigma):
    """KL divergence: penalty for deviations from the prior"""
    return -0.5 * tf.reduce_sum(1 + tf.log(tf.square(sigma)) - tf.square(mu) - tf.square(sigma))

def gaussian_loss(x, mean, sd, clip=0):
    """Gaussian negative-log-likelihood, up to an additive constant"""
    loss = tf.square((x-mean)/sd) / 2.0
    if clip > 0:
        loss = tf.nn.relu(loss - clip)
    return tf.reduce_sum(loss)

def make_weights(n_in, n_out):
    """Weight matrix with Xavier Glorot's initialization"""
    return tf.Variable(tf.random_normal([n_in, n_out], stddev=np.sqrt(3. / (n_in + n_out))))

def logit(p):
    """log odds for tf tensors"""
    with tf.variable_scope("logit"):
        out = tf.log(p/(1-p))
        return out

def kumar_lpdf(x, a, b):
    """log probability density of the Kumaraswamy distribution"""
    lpdf = tf.log(a) +\
             tf.log(b) +\
             (a - 1) * tf.log(x) +\
             (b - 1) * log1p(-x**a)
    return tf.reduce_sum(lpdf)
