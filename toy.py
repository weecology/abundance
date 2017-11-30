
# coding: utf-8

# # Imports

# In[1]:

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

# One uniformly-distributed predictor, one Gaussian-distributed predictor
x_train = np.column_stack([np.linspace(-2, 2, num=N), np.random.normal(0, 1, N)]).astype(np.float32)

# Only the first predictor matters
w = np.array([[3, -3], [0,0]])

# "True" latent variable" for each data point
true_z = np.random.normal(0, .5, N)

# Expected abundance
mu = np.exp(3 + np.sin(np.dot(x_train, w)) + np.column_stack([true_z, true_z]))

# Observed abundances
y_train = np.random.poisson(mu).astype(np.int32)


# In[3]:

# Mu vs x
plt.scatter(x_train[:,0], mu[:,0], s=15, edgecolors='none');
plt.scatter(x_train[:,0], mu[:,1], s=15, edgecolors='none');


# In[4]:

# Mu vs true_z
plt.scatter(true_z, mu[:,0], s=15, edgecolors='none');
plt.scatter(true_z, mu[:,1], s=15, edgecolors='none');


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
# Note that the regularizers in tf.layers have no effect on Edward models
with tf.variable_scope("network"):
    # Concatenate the inputs to the neural net
    xz = tf.concat([x_train, z], 1)

    # These hidden layer(s) are just basis expansions---not intended to be interpreted---so no names
    h = tf.layers.dense(xz, 100, activation=tf.nn.elu)
    #h = tf.layers.dense(h, 50, activation=tf.nn.elu)

    # Low-dimensional "environment", to which all species respond in a generalized linear way.
    env = tf.layers.dense(h, 10, activation=tf.nn.elu, name="env")

    # Expected abundances
    yhat = tf.layers.dense(env, 2, activation=tf.exp, name="out")


# ### Outputs

# In[7]:

y = ed.models.Poisson(yhat)


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


# # Fitting

# ### Initialization

# In[9]:

# Count how many hill-climbing steps have been taken
global_step = tf.Variable(0, trainable=False)

inference = ed.KLqp({z: qz}, data={y: y_train})
inference.initialize(var_list=tf.trainable_variables(), 
                     n_samples=10,
                     global_step = global_step,
                     kl_scaling={z: 1})

# Start a session, then initialize all the variables
sess = ed.get_session()
tf.global_variables_initializer().run()


# ### Run

# In[10]:

for _ in range(25000):
    inference.update()


# # Diagnostics

# In[11]:

print(global_step.eval())
print(inference.loss.eval())


# In[12]:

plt.figure(figsize=(10, 10))
plt.scatter(x_train[:,0], np.log(mu[:,0]));
for _ in range(20):
    plt.scatter(x_train[:,0], np.log(yhat.eval()[:,0]), alpha=0.1, c="black", s=10);


# In[13]:

plt.figure(figsize=(10, 10))
plt.scatter(x_train[:,0], np.log(mu[:,0]));
for _ in range(20):
    plt.scatter(x_train[:,0], np.log(qyhat.eval()[:,0]), alpha=0.1, c="darkred", s=10);


# In[14]:

plt.hist(qz.stddev().eval(), bins=50);


# In[26]:

# Approximate posterior distributions of z, given x. Should be no big trends or gaps
for _ in range(n_z):
    for _ in range(25):
        plt.scatter(x_train[:,0], qz.eval()[:,which_latent], alpha=0.1, c="darkred", s=5)
    plt.show()


# In[27]:

nn = 1000
xx = 0
qxx = 0
for _ in range(nn):
    xx += yhat.eval() / nn
    qxx += qyhat.eval() / nn
    
# Predicted mean versus x
plt.scatter(x_train[:,0], np.log(mu)[:,1])
plt.scatter(x_train[:,0], np.log(xx)[:,1], s=10, alpha=0.5, c="black");
plt.show()

# Inferred mu versus "true" mu
plt.scatter(np.log(qxx[:,0].flatten()), np.log(mu[:,0]).flatten(), c="darkred");


# # Other outputs

# In[24]:

# List of parameters trained by the network.
tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network")


# In[18]:

# Get the weights of the final layer (i.e. species-level coefficients)
tf.get_default_graph().get_tensor_by_name('network/out/kernel:0').eval()


# In[ ]:




# In[ ]:




# In[ ]:



