# coding: utf-8

# ## Imports

# In[1]:

import numpy as np

import tensorflow as tf
from tensorflow.contrib.distributions import Distribution

# Edward doesn't work with the most recent standalone keras,
# so I'm specifying tensorflow.contrib.keras and giving it a special name
import tensorflow.contrib.keras as tfk
from tensorflow.contrib.keras import regularizers


import edward as ed
from edward.models import RandomVariable, Normal

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# # Data

# In[2]:

N = 1000
x_train = np.linspace(-2, 2, num=N)
color = np.random.randint(0, 2, N)
y_train = np.cos(x_train) * (0.5 - color) + np.random.normal(0, 0.1, size=N)
x_train = x_train.astype(np.float32).reshape((N, 1))
y_train = y_train.astype(np.float32)

plt.scatter(x_train, y_train, c=color, s=15, edgecolors='none');


# In[3]:

# Optionally, give the network the color labels
#x_train = np.concatenate([x_train, color.astype(np.float32).reshape((N,1))], 1)


# ## Latent variables

# In[4]:

# Observation-level random effects: prior distribution
z = Normal(loc=tf.zeros([N, 1]), scale=tf.ones([N,1]))

# Variational approximation to the posterior for observation-level random effects
qz = Normal(loc = tf.Variable(tf.zeros([N,1])), 
            scale = tf.nn.softplus(tf.Variable(0.55 * tf.ones([N,1]))))


# ## Neural network

# In[5]:

# If this code block throws an error about z not being a standard tensor,
# it's a version incompatibility between Keras and Edward. See
# https://github.com/fchollet/keras/issues/6979
class network(object):
    def f(self, x_object, z_object):
        z_layer = tfk.layers.Dense(5, activation='sigmoid', kernel_regularizer=regularizers.l2(.001))(z_object)
        self.out = tfk.layers.Dense(50, activation='elu', kernel_regularizer=regularizers.l2(.001))(tf.concat([x_object, z_layer], 1))
        self.out = tfk.layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(.001))(self.out)[:,0]
        return self.out

# Running this line twice will throw an error because it can't overwrite the variables in
# the network scope
with tf.variable_scope("network"):
    net = network()
    yhat = net.f(x_train, z)
with tf.variable_scope("network", reuse=True):
    qyhat = net.f(x_train, qz)


# ## Objective function

# In[6]:

global_step = tf.Variable(0, trainable=False)
temperature = 1 + 5E5 / (1 + 1E4 + tf.cast(global_step, tf.float32))
y = Normal(loc=yhat, scale=tf.sqrt(0.01 * temperature))


# In[ ]:




# ## Initialization

# In[ ]:

inference = ed.KLqp({z: qz}, data={y: y_train})
inference.initialize(var_list=tf.trainable_variables(), 
                     n_samples=25,
                     global_step = global_step,
                     n_iter = 10)

# Initialize a TF session, then initialize all the variables
sess = ed.get_session()
tf.global_variables_initializer().run()


# ## Run the model

# In[ ]:

maxit = 100000
chunk_size = 1000
prog = tf.contrib.keras.utils.Progbar(maxit)
for _ in range(maxit // chunk_size):
    for _ in range(chunk_size):
        inference.update()
    prog.add(chunk_size)
    print(" ", inference.loss.eval())


# ## Diagnostics

# In[28]:

print(inference.t.eval(), "training iterations")
print("Loss:", inference.loss.eval())
print("temperature:", temperature.eval())


# In[33]:

# Distribution of y's mean in the training set
for i in np.arange(np.floor(1E4 / N)):
    plt.scatter(x_train[:,0], qyhat.eval(), s=4, c="darkred", alpha=0.05)
#plt.scatter(x_train[:,0], y_train, c=color, s=25, edgecolors='black');
plt.show()

# Distribution of y's mean in the test set
for i in np.arange(np.floor(1E4 / N)):
    plt.scatter(x_train[:,0], yhat.eval(), s=4, c="darkblue", alpha=0.05)
#plt.scatter(x_train[:,0], y_train, c=color, s=25, edgecolors='black');
plt.show()


# In[30]:

plt.figure(figsize=(8, 8))
for i in np.arange(np.floor(1E5 / N)):
    plt.scatter(x_train[:,0].flatten(), 
                qz.eval(), s=10, c="darkred", alpha=0.05, edgecolors='none')
plt.scatter(x_train[:,0], qz.mean().eval(), c="black", s = 20, alpha = 0.25, edgecolors='none');


# In[31]:

plt.scatter(x_train[:,0], qz.mean().eval());
plt.show()
plt.scatter(x_train[:,0], qz.variance().eval());


# In[ ]:
