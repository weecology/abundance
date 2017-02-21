
# coding: utf-8

# In[1]:

import pandas as pd
import scipy
import numpy as np
from scipy import stats, special

import tensorflow as tf
print(tf.__version__)
sess = tf.InteractiveSession()

import tfnb

import matplotlib
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# # Hyperparameters

# In[2]:

N_z = 50  # Latent variables
N_h = 100 # Neurons in non-final hidden layer(s)
N_bottleneck = 50 # Neurons in final hidden layer


# # Data

# In[3]:

# Data
x_array = np.array(pd.read_csv("x.csv"))
y_array = np.array(pd.read_csv("y.csv"))
site_array = (np.array(pd.read_csv("site.csv"))) - 1 # Zero-indexing versus R's one-indexing
obs_array = (np.array(pd.read_csv("obs.csv"))) - 1   # Zero-indexing

# Sizes
N_x = x_array.shape[1]
N_y = y_array.shape[1]

N_s = int(np.max(site_array[:,1])) + 1 # Include zero as a column
N_o = int(np.max(obs_array[:,1])) + 1  # Include zero as a column
N_rows = x_array.shape[0]

N = tf.placeholder(tf.int32, shape=[]) # Minibatch size
Y = tf.placeholder(tf.float32, shape=[None, N_y], name = "abundance") # Response variables
X = tf.placeholder(tf.float32, shape=[None, N_x], name = "environment") # Predictor variables
SO = tf.sparse_placeholder(tf.float32, shape=[None, N_s + N_o], name = "sites-observers") # Random effect design matrix; sites then observers
training = tf.placeholder(tf.bool, shape=[]) # Are we in training mode or test mode?


# # Layer 0: calculate random effects & concatenate with X
# 
# I'm doing this part manually because of sparseness of the `SO` matrix.  A higher-level way to do this probably exists using one-hot features or something similar.

# In[4]:

with tf.variable_scope("latent"):
    # Declare variables for layer 0
    W0_mu = tfnb.make_weights(N_s + N_o,N_z)
    W0_sigma = tfnb.make_weights(N_s + N_o,N_z)
    b0_mu = tf.Variable(tf.zeros(N_z))
    b0_sigma = tf.Variable(0.25 * tf.ones(N_z)) 

    # Sample from latent density (Z)
    mu0 = tf.sparse_tensor_dense_matmul(SO,W0_mu) + b0_mu
    sigma0 = tf.nn.softplus(tf.sparse_tensor_dense_matmul(SO,W0_sigma) + b0_sigma)
    epsilon = tf.random_normal(tf.shape(mu0))
    Z = mu0 + sigma0 * epsilon

# Define layer 1's inputs with X and Z
XZ = tf.concat([X,Z], 1)


# # Hidden layers

# In[5]:

# Feed forward
H1 = tf.layers.dense(XZ, 
                     units=N_h, 
                     activation=tf.nn.elu,
                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                     kernel_regularizer = tf.contrib.layers.l2_regularizer(0.01),
                     name="layer1")
H1_normalized = tf.contrib.layers.layer_norm(inputs=H1)
H2 = tf.layers.dense(H1_normalized, 
                     units=N_bottleneck, 
                     activation=tf.nn.elu,
                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                     kernel_regularizer = tf.contrib.layers.l2_regularizer(0.01),
                     name="layer2")
H2_normalized = tf.contrib.layers.layer_norm(inputs=H2)
HN = H2_normalized


# # Output layer

# In[6]:

def mu_initializer(*args, **kwargs):
    out = np.log(np.mean(y_array,axis=0)) + 1
    return out
def size_initializer(*args, **kwargs):
    out = scipy.special.logit(np.mean(y_array!=0, axis=0)) / 2 - 1
    return out
def zi_p_initializer(*args, **kwargs):
    out = scipy.special.logit((np.mean(y_array == 0, axis=0))) / 5.0
    return out

nb_mu = tf.layers.dense(HN,
                        units=N_y,
                        activation=tf.nn.softplus,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.2),
                        bias_initializer = mu_initializer,
                        name="negbin-mean")
nb_size = tf.layers.dense(HN,
                          units=N_y,
                          activation=tf.nn.softplus,
                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                          kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
                          bias_initializer = size_initializer,
                          name="negbin-size")
zi_p = tf.layers.dense(HN,
                       units = N_y,
                       activation=tf.nn.sigmoid,
                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                       kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0),
                       bias_initializer = zi_p_initializer,
                       name="zero-inflation-prob")

# Switch to N,p parameterization for negative binomial
with tf.variable_scope("negbin-prob"):
    nb_p = nb_size / (nb_mu + nb_size)


# # Loss & Optimizer

# In[7]:

# For fitting loop
rows = np.arange(y_array.shape[0])
n_steps = 0.
t = tf.placeholder(dtype=tf.float32)

def make_minibatch(n,is_training):
    rows = np.arange(N_rows)
    if n!=N_rows:
        np.random.shuffle(rows)
    rows = rows[range(n)]
    
    # Coordinates of nonzero entries in sparse design matrix.
    # one site and one observer per row
    i = np.concatenate((np.arange(n), np.arange(n)))
    j = np.concatenate((site_array[rows,1], N_s + obs_array[rows,1]))

    return {X:x_array[rows,:], 
           Y:y_array[rows,:], 
           SO:(np.vstack((i,j)).transpose(), [1] * (2 * n), [n, N_s + N_o]), 
           N:n, 
           t:n_steps,
           training:is_training}
full_feed = make_minibatch(N_rows,False)


with tf.variable_scope("lossses"):
    prediction_loss = -tf.reduce_sum(tfnb.zi_nbinom_ll(Y, nb_size, nb_p, zi_p))

    # Put a prior on negative binomial's "p" and the zero-inflation parameter, 
    # which both cause numerical problems at 0 or 1
    prior_loss = tfnb.gaussian_loss(tfnb.logit(nb_p), 0, 10) +                    tfnb.gaussian_loss(tfnb.logit(zi_p), 0, 10) +                    tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        
    variational_loss = tfnb.kl(mu0, sigma0)
    
    # Don't pay too much for the KL/entropy term until a few thousand iterations in, when
    # the network has started to settle down a bit.
    loss = prediction_loss +            prior_loss +            variational_loss * (1 - tf.exp(-t / 5000.))
    
adam = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, 
                              use_locking=False, name='Adam');
train_step = adam.minimize(loss);


# In[8]:

# Session incantations 
init = tf.global_variables_initializer()
sess.run(init)

# Save output for tensorboard
train_writer = tf.summary.FileWriter('./train', sess.graph)
train_writer.close()


# # Model-fitting loop

# In[9]:

# Fit the model
n = 32

for i in range(1000000):
    if n_steps % 1000 == 0:
        raw_losses = sess.run([prediction_loss, prior_loss, variational_loss, loss], 
                              feed_dict=full_feed)
        print(np.round([loss / N_rows for loss in raw_losses], 4))
    sess.run(train_step, feed_dict=make_minibatch(n,True))
    n_steps += 1
print(n_steps)


# # Downstream analyses

# In[10]:

mus, sigmas = sess.run((mu0, sigma0), feed_dict=make_minibatch(N_rows, False))
plt.scatter(range(N_z), np.mean(mus**2, axis = 0));
np.var(mus)


# In[11]:

plt.hist(np.log10(np.ndarray.flatten(sess.run(nb_mu, feed_dict=make_minibatch(N_rows, False)))), bins = "fd");


# In[12]:

plt.hist(sess.run(tfnb.logit(zi_p), feed_dict=make_minibatch(N_rows,False)).flatten(), bins = "fd");


# In[13]:

plt.hist(np.log10(sess.run(nb_size, feed_dict=make_minibatch(N_rows,False)).flatten()), bins = "fd");


# In[14]:

plt.hist(scipy.special.logit(sess.run(nb_p, feed_dict=make_minibatch(N_rows, False)).flatten()), bins = "fd");


# In[15]:

#print(np.sqrt(np.mean(sess.run(tf.square(W0_mu)))))
#print(np.sqrt(np.mean(sess.run(tf.square(W0_sigma)))))
#print(np.sqrt(np.mean(sess.run(tf.square(WN_mu)))))
#print(np.sqrt(np.mean(sess.run(tf.square(WN_size)))))
#print(np.sqrt(np.mean(sess.run(tf.square(WN_zi_p)))))


# In[16]:

#plt.scatter(np.log(np.mean(y_array, axis=0) + 1), 
#            bN_mu.eval());


# In[17]:

plt.scatter(sess.run(tf.log(nb_mu[:,3]), feed_dict=make_minibatch(N_rows,False)),
             sess.run(tfnb.logit(nb_size[:,3]), feed_dict=make_minibatch(N_rows,False)));


# In[ ]:




# In[18]:

#all_mu, all_zip = sess.run((nb_mu, zi_p), feed_dict=full_feed)
#np.savetxt("mu.csv", all_mu, delimiter=",")
#np.savetxt("zip.csv", all_zip, delimiter=",")


# In[19]:

print(n_steps)


# In[20]:

tf.get_collection(tf.GraphKeys.WEIGHTS)


# In[21]:

with tf.variable_scope("layer1", reuse=True):
    plt.scatter(range(N_z + N_x), np.sum(tf.get_variable("kernel").eval()**2, axis=1))


# In[22]:

plt.scatter(range(N_s + N_o),
            np.var(sess.run(W0_mu, feed_dict=make_minibatch(N_rows, True)), axis=1));


# In[23]:

plt.scatter(range(N_s + N_o),
            np.var(sess.run(W0_sigma, feed_dict=make_minibatch(N_rows, True)), axis=1));

