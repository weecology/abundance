
# coding: utf-8

# # Imports

# In[ ]:

import numpy as np
import scipy

import tensorflow as tf

import edward as ed
from edward.models import RandomVariable, Normal


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

print(tf.__version__)


# # Data

# In[2]:

N = 1000 # Number of data points
mini_N = 101 # Minibatch size

# One uniformly-distributed predictor, one Gaussian-distributed predictor
x_train = np.column_stack([np.linspace(-2, 2, num=N), np.random.normal(0, 1, N)]).astype(np.float32)

# Only the first predictor matters
w = np.array([[3, -3, 0], [0,0,0]])

# "True" latent variable" for each data point
true_z = np.random.normal(0, .5, N)

# Expected abundance
mu = np.exp(3 + np.sin(np.dot(x_train, w)) + np.column_stack([true_z, true_z, true_z]))

# Observed abundances
y_train = np.random.poisson(mu).astype(np.float32)


# In[3]:

# Mu vs x
plt.scatter(x_train[:,0], mu[:,0], s=15, alpha = 0.25);
plt.scatter(x_train[:,0], mu[:,1], s=15, alpha = 0.25);
plt.scatter(x_train[:,0], mu[:,2], s=15, alpha = 0.25);


# In[4]:

# Mu vs true_z
plt.scatter(true_z, mu[:,0], s=15, alpha = 0.25);
plt.scatter(true_z, mu[:,1], s=15, alpha = 0.25);
plt.scatter(true_z, mu[:,2], s=15, alpha = 0.25);


# # Model

# ### Latent variables

# In[5]:

n_z = 2

# Observation-level random effects: prior distribution
z = Normal(loc=tf.zeros([N, n_z]), scale=tf.ones([N,n_z]))


# ### Neural network

# In[6]:

# Running this twice will throw an error because it can't overwrite the variables in
# the layers' scope
# Note that the regularizers in tf.layers have no effect on Edward models!
with tf.variable_scope("network"):
    # Concatenate the inputs to the neural net
    xz = tf.concat([x_train, z], 1)

    # These hidden layer(s) are just basis expansions---not intended to be interpreted---so no names
    h = tf.layers.dense(xz, 100, activation=tf.nn.elu)
    #h = tf.layers.dense(h, 50, activation=tf.nn.elu)

    # Low-dimensional "environment", to which all species respond in a generalized linear way.
    env = tf.layers.dense(h, 10, activation=tf.nn.elu, name="env")

    # Expected abundances. Initialize the biases so they tend to produce values with the right geometric mean
    yhat = tf.layers.dense(env, 
                           y_train.shape[1], 
                           activation=tf.exp, 
                           name="out", 
                           bias_initializer=tf.constant_initializer(np.log(np.mean(y_train, 0)).astype(np.float32)))


# ### Outputs

# In[7]:

# Placeholder telling which rows to use (e.g. for minibatching)
row_ph = tf.placeholder(tf.int32, [None])

y = ed.models.Poisson(tf.gather(yhat, row_ph))


# ### Approximate posterior distribution (variational approximation)
# 
# Approximate posteriors are denoted by adding `q` at the beginning of the name

# In[8]:

with tf.variable_scope("posterior"):
    # Variational approximation to the posterior for observation-level random effects
    qz = Normal(loc = tf.Variable(0.0 * tf.ones([N,n_z])), 
                scale = tf.nn.softplus(tf.Variable(0.55 * tf.ones([N,n_z]))))

    # Other approximate posteriors (with qz filled in for z like in `inference` below)
    qenv = ed.copy(env, {z: qz})
    qyhat = ed.copy(yhat, {z: qz})


# # Boilerplate for model fitting

# In[9]:

# Count how many hill-climbing steps have been taken
global_step = tf.Variable(0, trainable=False)

inference = ed.KLqp({z: qz}, 
                    {y: tf.gather(y_train, row_ph)})
inference.initialize(var_list=tf.trainable_variables(), 
                     n_samples=1,
                     global_step = global_step,
                     kl_scaling={z: N / mini_N},
                     optimizer=tf.train.AdamOptimizer(learning_rate=.001))

# Start a session, then initialize all the variables
sess = ed.get_session()
tf.global_variables_initializer().run()


# In[ ]:




# # Run

# In[ ]:

row_d = tf.contrib.data.Dataset.range(N).shuffle(N).batch(mini_N).repeat()
iter_d = row_d.make_one_shot_iterator()

for _ in range(2000):
    rows = iter_d.get_next().eval()
    if len(rows) != mini_N:
        # kl_scaling won't work properly if the number of rows varies.
        # This will happen at the end of each epoch if the N % mini_N != 0
        rows = iter_d.get_next().eval()
    inference.update(feed_dict={row_ph: rows})


# # Diagnostics

# In[ ]:

print(global_step.eval())
print(inference.loss.eval(feed_dict={row_ph: np.arange(N), y:y_train}))


# In[ ]:

which_species = 0

# Predictions using the prior on z
plt.figure(figsize=(10, 10))
plt.scatter(x_train[:,0], np.log(mu[:,which_species]));
for _ in range(20):
    plt.scatter(x_train[:,0], np.log(yhat.eval()[:,which_species]), alpha=0.1, c="black", s=10);


# In[ ]:

# Predictions using the approximate posterior on z
plt.figure(figsize=(10, 10))
plt.scatter(x_train[:,0], np.log(mu[:,which_species]));
for _ in range(20):
    plt.scatter(x_train[:,0], np.log(qyhat.eval()[:,which_species]), alpha=0.1, c="darkred", s=10);


# In[ ]:

plt.hist(qz.stddev().eval(), bins=50);


# In[ ]:

# Approximate posterior distributions of z, given x. Should be no big (low-frequency?) trends or gaps
for i in range(n_z):
    for _ in range(25):
        plt.scatter(x_train[:,0], qz.eval()[:,i], alpha=0.1, c="darkred", s=5)
    plt.show()


# In[ ]:

nn = 1000
xx = 0
qxx = 0
for _ in range(nn):
    xx += yhat.eval() / nn
    qxx += qyhat.eval() / nn
    
# Predicted mean versus x
plt.scatter(x_train[:,0], np.log(mu)[:,which_species])
plt.scatter(x_train[:,0], np.log(xx)[:,which_species], s=10, alpha=0.5, c="black");
plt.show()

# Inferred mu versus "true" mu
plt.scatter(np.log(qxx[:,which_species].flatten()), np.log(mu[:,which_species]).flatten(), c="darkred");


# In[ ]:

# "Residual correlation" among species
np.corrcoef(np.log(yhat.eval() / xx).T)


# # Other outputs

# In[ ]:

# List of parameters trained by the network.
tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network")


# In[ ]:

# Get the weights of the final layer (i.e. species-level coefficients)
tf.get_default_graph().get_tensor_by_name('network/out/kernel:0').eval()


# # To do:
# 
# * Put priors/weight decay on coefficients
# * Fix first dimension of tensors:
#     * predicting with different N
#     * minibatching
# * Extend the response distribution
#     * overdispersion
#     * zero-inflation
# * Repeated measures for sites & observers
# * Penalties to discourage structure in `qz`?
# * Imputation/missing values?
#     * Maybe only after model has already been trained, like Mistnet's figure 5

# In[ ]:

q = qz.variance().eval()
plt.scatter(x_train[:,0], q[:,0], alpha=0.25)
plt.scatter(x_train[:,0], q[:,1], alpha=0.25)


# In[ ]:




# In[ ]:



