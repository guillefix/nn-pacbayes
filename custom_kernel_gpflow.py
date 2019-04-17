import tensorflow as tf
import gpflow
import numpy as np
# import deep_ckern as dkern
# from save_kernels import compute_big_K,mnist_1hot_all
# from paramz.caching import Cache_this
#
# from cnn_kernel import kernel_matrix
#
# image_size=28
# number_channels=1
# filter_sizes=[[5, 5], [2, 2], [5, 5], [2, 2]]
# padding=["VALID", "SAME", "VALID", "SAME"]
# strides=[[1, 1]] * 4
# sigmaw=1.0
# sigmab=1.0
#
# with tf.device("cpu:0"):
#     kern = dkern.DeepKernel(
#         [number_channels, image_size, image_size],
#         filter_sizes=filter_sizes,
#         recurse_kern=dkern.ExReLU(multiply_by_sqrt2=True),
#         var_weight=sigmaw**2,
#         var_bias=sigmab**2,
#         padding=padding,
#         strides=strides,
#         data_format="NCHW",
#         skip_freq=-1,
#         )

# def compute_big_K(kern, n_max, X, X2=None, n_gpus=1):
#     """
#     Compute the kernel matrix between `X` and `X2`.
#     """
#     N = X.get_shape().as_list()[0]
#     N2 = N if X2 is None else X2.get_shape().as_list()[0]
#
#     # Make a list of all the point kernel matrices to be computed
#     if X2 is None or X2 is X:
#         diag_symm = True
#         slices = list((slice(j, j+n_max), slice(i, i+n_max))
#                       for j in range(0, N, n_max)
#                       for i in range(j, N2, n_max))
#     else:
#         diag_symm = False
#         slices = list((slice(j, j+n_max), slice(i, i+n_max))
#                       for j in range(0, N, n_max)
#                       for i in range(0, N2, n_max))
#
#     # Make the required kernel ops and placeholders for each GPU
#     ops = []
#     for i in tqdm.trange(0, len(slices), n_gpus):
#         for (slice1,slice2) in slices[i:i+n_gpus]:
#             with tf.device("gpu:{}".format(i)):
#                 X_ph = X1[slice1]
#                 X2_ph = X2[slice2]
#                 # X2_ph = tf.placeholder(settings.float_type, X_ph.shape, "X2_ph")
#                 K_cross = kern.K(X_ph, X2_ph)
#                 if diag_symm:
#                     K_symm = kern.K(X_ph, None)
#                 else:
#                     K_symm = None
#                 if j_s == i_s and diag_symm:
#                     ops.append(K_symm)
#                 else:
#                     ops.append(K_cross)
#
#     # Execute on all GPUs concurrently
#     # out = tf.zeros((N, N2), dtype=settings.float_type)
#     # for j in tqdm.trange(0, len(slices), n_gpus):
#     #     feed_dict = {}
#     #     ops = []
#     #     for (X_ph, X2_ph, K_cross, K_symm), (j_s, i_s) in (
#     #             zip(K_ops, slices[j:j+n_gpus])):
#     #         if j_s == i_s and diag_symm:
#     #             # feed_dict[X_ph] = X[j_s]
#     #             ops.append(K_symm)
#     #         else:
#     #             # feed_dict[X_ph] = X[j_s]
#     #             if X2 is None:
#     #                 # feed_dict[X2_ph] = X[i_s]
#     #             else:
#     #                 # feed_dict[X2_ph] = X2[i_s]
#                 # ops.append(K_cross)
#         # results = sess.run(ops, feed_dict=feed_dict)
#         out = tf.zeros((N,N2))
#         for j in range(n_gpus):
#             for o, (slice1,slice2) in zip(ops, slices[j:j+n_gpus]):
#                 out[slice1, slice2] = o
#                 if slice1 != slice2 and diag_symm:
#                     out[slice2, slice1] = tf.transpose(o)
#     return out


class CustomMatrix(gpflow.kernels.Kernel):
    def __init__(self,input_dim,X,K,active_dims=None):
    # def __init__(self,input_dim,active_dims=None):
        super(CustomMatrix, self).__init__(input_dim, active_dims, 'custom_matrix')
        self.Kmatrix = tf.constant(K) #Param('matrix', K)
        self.X = tf.constant(X)
        # self.link_parameters(self.K)
    # @Cache_this(limit=3, ignore_args=())
    def K(self,X1,X2=None):

        # return compute_big_K(kern,400,X1,X2)
        # return kernel_matrix(X,X2)
        # if X2 is None and np.all(X1 == self.X):
        #     return tf.constant(self.Kmatrix)

        def index1d(t):
            return tf.reduce_min(tf.where(tf.reduce_all(tf.equal(t,self.X),-1)))

        indices1 = tf.map_fn(index1d, X1, dtype=tf.int64)

        if X2 is None:
            X2 = X1
            indices2 = indices1
        else:
            indices2 = tf.map_fn(index1d, X2, dtype=tf.int64)
            # indices2 = np.prod(np.isin(self.X,X2),-1).nonzero()[0]
        indices1 = tf.reshape(indices1,(-1,1))
        indices2 = tf.reshape(indices2,(1,-1))
        reps1 = tf.concat([tf.constant([1]),tf.expand_dims(tf.shape(indices2)[1],-1)],0)
        reps2 = tf.concat([tf.expand_dims(tf.shape(indices1)[0],-1),tf.constant([1])],0)
        combined_indices = tf.stack([tf.tile(indices1,reps1),tf.tile(indices2,reps2)],-1)
        return tf.gather_nd(self.Kmatrix,combined_indices)

        # if np.all(np.isin(X2,self.X)):
        #     if len(indices2) != X2.shape[0] or len(indices1) != X1.shape[0]:
        #         return NotImplementedError("Some elements of X2 or X1 appear more than once in X")
        #     else:
        #         return tf.constant(self.Kmatrix[indices1[:, None], indices2])
        # else:
        #     # return kernel_matrix(X1,X2)
        #     # return tf.constant([1])
        #     return NotImplementedError("Some elements of X2 are not in X")
        #     return NotImplementedError
    # @Cache_this(limit=3, ignore_args=())
    def Kdiag(self,X):
        # K = kernel_matrix(X,X2)
        if np.all(np.isin(X2,self.X)):
            indices = np.prod(np.isin(self.X,X),-1).nonzero()[0]
            return tf.constant(np.diag(self.Kmatrix)[indices])
        else:
            return np.diag(kernel_matrix(X))
    def update_gradients_full(self, dL_dK, X, X2):
        pass
    def update_gradients_diag(self, dL_dK, X, X2):
        pass
