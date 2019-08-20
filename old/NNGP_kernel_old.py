#LD_LIBRARY_PATH=/local/home/valleperez/cuda/lib64/
# import os
# os.environ["LD_LIBRARY_PATH"]="/local/home/valleperez/cuda/lib64/"
# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os.path
import time

import numpy as np
import tensorflow as tf

# tf.enable_eager_execution()

# import gpr
import load_dataset
# import nngp
from gpflow import settings
import tqdm

m=20000
(train_image, train_label, valid_image, valid_label, test_image,
    test_label) = load_dataset.load_mnist(
       num_train=m,
       mean_subtraction=True,
       random_roated_labels=False)


# %%
def pi(y,f):
    return 1/(1+tf.exp(-y*f))

def GP_prob_iteration(fs_old,ys,K):
    N=ys.get_shape().as_list()[0]
    # print(N,type(N))
    # W = tf.diag([pi(1,fs_old[i])*(1-pi(1,fs_old[i])) for i,y in enumerate(ys)])
    W = tf.diag( tf.map_fn(lambda f: pi(1,f)*(1-pi(1,f)), fs_old) )
    B = tf.eye(N,dtype="float64") + tf.matmul(W**0.5,tf.matmul(K,W**0.5))
    L = tf.cholesky(B)
    b = tf.tensordot(W,fs_old,axes=[[1],[0]])
    # print(b)
    b += tf.map_fn(lambda x: (x[0]+1)/2 - pi(1,x[1]), (ys,fs_old), dtype=tf.float64)
    # print(b)
    b = tf.expand_dims(b,-1)
    a = b - tf.matmul(W**0.5,tf.matrix_triangular_solve(tf.transpose(L),tf.matrix_triangular_solve( L, tf.matmul(W**0.5,tf.matmul(K,b)) ),lower=False))
    fs = tf.matmul(K,a)
    objective = -0.5*tf.matmul(tf.transpose(a),fs)+ tf.reduce_sum(tf.map_fn(lambda x: tf.log(pi(x[0],x[1])), (ys,fs), dtype=tf.float64))
    # print(a)
    logProb = objective - tf.reduce_sum(tf.log(tf.diag_part(L)))
    return [tf.squeeze(fs),tf.squeeze(objective),tf.squeeze(logProb)]

# tfe = tf.contrib.eager

def GP_prob(K,ys,tolerance=0.01):
    N = len(ys)
    fs = np.zeros(N,dtype="float64")
    # fs = np.random.randn(N).astype("float64")

    fs_old_plh = tf.placeholder(tf.float64,N)
    ys_plh = tf.placeholder(tf.float64,N)
    K_plh = tf.placeholder(tf.float64,(N,N))

    fs_node, objective_node, logProb_node = GP_prob_iteration(fs_old_plh,ys_plh,K_plh)
    # test = tfe.py_func(GP_prob_iteration, [fs_old_plh,ys_plh,K_plh], tf.float64)
    # sess.run(test, feed_dict = {fs_old_plh:fs ,ys_plh:ys ,K_plh:K})
    # fs_new,objective_new

    sess = tf.Session()

    fs,objective_new = sess.run([fs_node, objective_node], feed_dict = {fs_old_plh:fs ,ys_plh:ys ,K_plh:K})
    objective_old = objective_new
    fs,objective_new,logProb = sess.run([fs_node, objective_node,logProb_node], feed_dict = {fs_old_plh:fs ,ys_plh:ys ,K_plh:K})

    while abs(objective_old - objective_new) > tolerance:
        print( abs(objective_old - objective_new) )
        objective_old = objective_new
        fs,objective_new,logProb = sess.run([fs_node, objective_node, logProb_node], feed_dict = {fs_old_plh:fs ,ys_plh:ys ,K_plh:K})
    #return np.exp(logProb)
    sess.close()
    return logProb


ys = [int((np.argmax(labels)>5))*2.0-1 for labels in train_label]
ys=[int(xx)*2.0-1 for xx in list("11011101111111110100110011111111111111111111111111011111111111111111111111111111110111011111111111111111111111111101111111111111")]
N = len(ys)
input_dim = train_image.shape[1]
input_dim = 7

# sigmaw=28*28
# sigmab=1/np.sqrt(28*28)

sigmaw=1.3
sigmab=0.2

number_layers = 2

def kern(X1,X2):
    # X1 = tf.placeholder("float32",(None,input_dim))
    # X2 = tf.placeholder("float32",(None,input_dim))
    N = X1.get_shape().as_list()[0]
    if X2 is None:
        K = sigmab**2 + sigmaw**2 * tf.matmul(X1,tf.transpose(X1))/input_dim
        for l in range(number_layers):
            K_diag = tf.diag_part(K)
            K_diag = tf.expand_dims(K_diag,-1)
            K1 = tf.tile(K_diag,[1,N])
            K2 = tf.tile(tf.transpose(K_diag),[N,1])

            K12 = K1 * K2
            costheta = K / tf.sqrt(K12)
            theta = tf.acos(costheta)
            K = sigmab**2 + (sigmaw**2/(2*np.pi))*tf.sqrt(K12)*(tf.sin(theta)+(np.pi-theta)*costheta)

        # K_symm=K
        return K
    else:
        K = sigmab**2 + sigmaw**2 * tf.matmul(X1,tf.transpose(X2))/input_dim
        K1_diag = sigmab**2 + sigmaw**2 * tf.reduce_sum(X1*X1,axis=1, keepdims=True)/input_dim
        K2_diag = sigmab**2 + sigmaw**2 * tf.reduce_sum(X2*X2,axis=1, keepdims=True)/input_dim
        for l in range(number_layers):
            # K1_diag = tf.expand_dims(K1_diag,-1)
            K1 = tf.tile(K1_diag,[1,N])
            # K2_diag = tf.expand_dims(K2_diag,-1)
            K2 = tf.tile(tf.transpose(K2_diag),[N,1])

            K12 = K1 * K2
            costheta = K / tf.sqrt(K12)
            theta = tf.acos(costheta)
            K = sigmab**2 + (sigmaw**2/(2*np.pi))*tf.sqrt(K12)*(tf.sin(theta)+(np.pi-theta)*costheta)

            K1_diag = sigmab**2 + (sigmaw**2/2)*K1_diag
            K2_diag = sigmab**2 + (sigmaw**2/2)*K2_diag

        # K_cross=K
        return K

n_max = 10000
n_max = min(n_max,N)
slices = list((slice(j, j+n_max), slice(i, i+n_max))
    for j in range(0, N, n_max)
    for i in range(j, N, n_max))

# tf.Session().run(K1,feed_dict={X1:X[slices[0][0]], X2:X[slices[0][0]]})

# type(settings.float_type)

n_gpus = 1
X = train_image
X = np.array([[float(l) for l in "{0:07b}".format(i)] for i in range(0,2**input_dim)])
with tf.Session() as sess:
    K_ops = []
    for i in range(n_gpus):
        with tf.device("gpu:{}".format(i)):
            X1 = tf.placeholder(settings.float_type, [n_max, X.shape[1]], "X1")
            X2 = tf.placeholder(settings.float_type, X1.shape, "X2")
            K_cross = kern(X1, X2)
            K_symm = kern(X1, None)
            K_ops.append((X1, X2, K_cross, K_symm))

    out = np.zeros((N, N), dtype=settings.float_type)
    for j in tqdm.trange(0, len(slices), n_gpus):
        feed_dict = {}
        ops = []
        for (X1, X2, K_cross, K_symm), (j_s, i_s) in (
                zip(K_ops, slices[j:j+n_gpus])):
            print((j_s, i_s))
            if j_s == i_s:
                feed_dict[X1] = X[j_s]
                ops.append(K_symm)
            else:
                feed_dict[X1] = X[j_s]
                feed_dict[X2] = X[i_s]
                ops.append(K_cross)
        results = sess.run(ops, feed_dict=feed_dict)
        for r, (j_s, i_s) in zip(results, slices[j:j+n_gpus]):
            out[j_s, i_s] = r
            if j_s != i_s:
                out[i_s, j_s] = r.T
    # kernel = sess.run(K,feed_dict={xs:train_image})

out.shape

out

np.save(open("kernel_20k_mnist_fc.p","wb"),out)

np.save(open("train_label_20k_mnist.p","wb"),train_label)

# %%
ys=[int(xx)*2.0-1 for xx in list("01001101010011110000000011101110010101010000001001000000111111110100111101001111000011001110111001000000000000000100110001011111")]

logPU = GP_prob(out,ys,0.001)

m=128
delta = 2**-10
print( (-logPU+2*np.log(m)+1-np.log(delta))/m )
