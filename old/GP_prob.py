import numpy as np
import tensorflow as tf

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

def GP_prob(K,ys,tolerance=0.01,cpu=False):
    N = len(ys)
    fs = np.zeros(N,dtype="float64")
    # fs = np.random.randn(N).astype("float64")

    fs_old_plh = tf.placeholder(tf.float64,N)
    ys_plh = tf.placeholder(tf.float64,N)
    K_plh = tf.placeholder(tf.float64,(N,N))

    if cpu:
        with tf.device('/cpu:0'):
            fs_node, objective_node, logProb_node = GP_prob_iteration(fs_old_plh,ys_plh,K_plh)
    else:
        fs_node, objective_node, logProb_node = GP_prob_iteration(fs_old_plh,ys_plh,K_plh)
    # test = tfe.py_func(GP_prob_iteration, [fs_old_plh,ys_plh,K_plh], tf.float64)
    # sess.run(test, feed_dict = {fs_old_plh:fs ,ys_plh:ys ,K_plh:K})
    # fs_new,objective_new

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    fs,objective_new = sess.run([fs_node, objective_node], feed_dict = {fs_old_plh:fs ,ys_plh:ys ,K_plh:K})
    objective_old = objective_new
    fs,objective_new,logProb = sess.run([fs_node, objective_node,logProb_node], feed_dict = {fs_old_plh:fs ,ys_plh:ys ,K_plh:K})

    while abs(objective_old - objective_new) > tolerance:
        #print( abs(objective_old - objective_new) )
        objective_old = objective_new
        fs,objective_new,logProb = sess.run([fs_node, objective_node, logProb_node], feed_dict = {fs_old_plh:fs ,ys_plh:ys ,K_plh:K})
    #return np.exp(logProb)
    sess.close()
    return logProb
