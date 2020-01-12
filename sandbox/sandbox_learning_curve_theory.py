import numpy as np
%matplotlib
#%%
n=15
def to_binary(i,n):
    cosa = bin(i)[2:]
    return cosa.zfill(n)
# len(to_binary(100,10))
boolean_inputs = np.array([[2*int(x)-1 for x in to_binary(i,n)] for i in range(2**n)])

#%%
sigmab=0.1
sigmaw=1.0

num_hidden_layers = 4

def phi_raw(x):
    K12 = sigmab**2+sigmaw**2*(x)
    K11 = sigmab**2+sigmaw**2
    for i in range(num_hidden_layers):
        theta = np.arccos(K12/K11)
        K12 = sigmab**2 + sigmaw**2/(2*np.pi)*K11*(np.sin(theta) + (np.pi - theta)*np.cos(theta))
        K11 = sigmab**2 + sigmaw**2/(2)*K11
    return K12

phi = np.vectorize(phi_raw)

input_sums = np.sum(boolean_inputs, axis=1)
def eig_for_degree(S):
    prods = np.prod(boolean_inputs[:,:S], axis=1)
    return np.sum(prods*phi(input_sums/n))

# S=3
eig_for_degree(1)

from scipy.special import comb
# comb(3,2)
def num_per_degree(S):
    return comb(n,S)


Ss = range(n)

reduced_eigs = np.array(list(map(eig_for_degree, Ss)))
num_eigs = np.array(list(map(num_per_degree, Ss)))

# m=1000

from utils import load_kernel_by_filename

K = load_kernel_by_filename("kernels/mnist_msweep_kernels__fc_mnist_"+str(8192)+"_0.0_0.0_True_False_True_2_1.41_0.1_None_00_max_kernel.npy")
eigs = np.linalg.eigh(K)[0]
n=13
2**13
ms = list(map(lambda x: 2**x, range(n)))
epsilons = []
for m in ms:
    epsilon = 1
    sigma=1e-1
    for i in range(1000):
        epsilon = np.sum(eigs/(1+eigs*m/(sigma**2+epsilon)))
        # epsilon = np.sum(num_eigs*reduced_eigs/(1+reduced_eigs*m/(sigma**2+epsilon)))

    epsilons.append(epsilon)

import matplotlib.pyplot as plt
plt.plot(ms, epsilons)
# plt.scatter(ms, epsilons, label=str(num_hidden_layers)+" hidden layers")
plt.scatter(ms, epsilons)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Number of training samples")
plt.ylabel("MSE error")

plt.legend()
# plt.savefig("boolean_fc_Sollich_LC_learning_curve_well_specified.png")
plt.savefig("mnist_fc_Sollich_LC_learning_curve_well_specified_lambda_estimate_8192.png")
