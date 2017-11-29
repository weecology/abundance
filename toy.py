
# coding: utf-8

# # Imports

# In[1]:

import numpy as np
import scipy

import tensorflow as tf
from tensorflow.contrib.distributions import Distribution

import edward as ed
from edward.models import RandomVariable, Normal

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# # Data

# In[2]:

N = 1000
x_train = np.linspace(-6, 6, num=N)
color = np.random.randint(0, 2, N)
y_train = np.random.poisson(np.exp(2 + 2 * np.cos(x_train) * (0.5 - color)))
x_train = x_train.astype(np.float32).reshape((N, 1))
y_train = y_train.astype(np.int32)

plt.scatter(x_train, y_train, c=color, s=15, edgecolors='none');


# # Model

# ### Latent variables

# In[3]:

n_z = 1

# Observation-level random effects: prior distribution
z = Normal(loc=tf.zeros([N, n_z]), scale=tf.ones([N,n_z]))


# ### Neural network

# In[4]:

# Running this twice will throw an error because it can't overwrite the variables in
# the layers' scope
class network(object):
    def f(self, x_object, z_object):
        with tf.variable_scope("network"):
            # Concatenate the inputs to the neural net
            self.xz = tf.concat([x_object, z_object], 1)
            
            # These hidden layer(s) are just basis expansions---not intended to be interpreted---so no names
            self.h = tf.layers.dense(self.xz, 50, activation=tf.nn.elu)
            self.h = tf.layers.dense(self.h, 50, activation=tf.nn.elu)

            # Low-dimensional "environment", to which all species respond in a generalized linear way.
            self.env = tf.layers.dense(self.h, 10, activation=None, name="env")

            # Expected abundances
            self.out = tf.layers.dense(self.env, 1, activation=tf.exp, name="out")[:,0]

            return self.env, self.out
net = network()


# ### Outputs

# In[5]:

env, yhat = net.f(x_train, z)
y = ed.models.Poisson(yhat)


# ### Variational approximation
# 
# Approximate posteriors are denoted by adding `q` at the beginning of the name

# In[6]:

with tf.variable_scope("posterior"):
    # Variational approximation to the posterior for observation-level random effects
    qz = Normal(loc = tf.Variable(0.0 * tf.ones([N,n_z])), 
                scale = tf.nn.softplus(tf.Variable(0.55 * tf.ones([N,n_z]))))

    # Other approximate posteriors (with qz filled in for z like in `inference` below)
    qenv = ed.copy(env, {z: qz})
    qyhat = ed.copy(yhat, {z: qz})


# # Fitting

# ### Initialization

# In[86]:

# Count how many hill-climbing steps have been taken
global_step = tf.Variable(0, trainable=False)

learning_rate = 0.01

inference = ed.KLqp({z: qz}, data={y: y_train})
inference.initialize(var_list=tf.trainable_variables(), 
                     n_samples=5,
                     global_step = global_step,
                     kl_scaling={z: 1}, 
                     optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate))

# Start a session, then initialize all the variables
sess = ed.get_session()
tf.global_variables_initializer().run()


# ### Run

# In[111]:

for _ in range(8):
    for _ in range(5000):
        inference.update()
    learning_rate = learning_rate * .96


# # Diagnostics

# In[112]:

print(global_step.eval())
print(inference.loss.eval())
learning_rate


# In[125]:

plt.figure(figsize=(10, 10))
plt.scatter(x_train, y_train);
for _ in range(20):
    plt.scatter(x_train, yhat.eval(), alpha=0.1, c="black", s=10);


# In[121]:

plt.figure(figsize=(10, 10))
plt.scatter(x_train, y_train);
for _ in range(20):
    plt.scatter(x_train, qyhat.eval(), alpha=0.1, c="darkred", s=10);


# In[115]:

plt.hist(qz.stddev().eval().flatten(), bins=50);


# In[127]:

plt.figure(figsize=(10, 10))
for _ in range(50):
    plt.scatter(x_train, qz.eval(), alpha=0.1, c="black", s=5)


# # Other outputs

# In[117]:

# List of parameters trained by the network.
tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network")


# In[118]:

# Get the weights of the final layer (i.e. species-level coefficients)
tf.get_default_graph().get_tensor_by_name('network/out/kernel:0').eval()


# In[ ]:




# In[ ]:




# In[ ]:



