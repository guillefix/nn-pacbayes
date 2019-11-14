import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib

# d = pd.read_csv("results/index_funs_probs_cnn_sample__mnist_cnn_4_none_0000_sigmab1.0.txt", header=None, names=["index","fun","ent"],delim_whitespace=True)
d = pd.read_csv("results/index_funs_probs_cnn_sample__mnist_cnn_4_none_0000_sigmab1.0.txt", header=None, names=["index","fun","ent"],delim_whitespace=True)
d = pd.read_csv("results/index_funs_probs_perceptron_sample__1.0_boolean_fc_0_none_.txt", header=None, names=["index","fun","ent","n1s"],delim_whitespace=True)
d = pd.read_csv("results/index_funs_probs_2perceptron_sample__1.0_boolean_fc_0_none_.txt", header=None, names=["index","fun","ent","n1s"],delim_whitespace=True)
d = pd.read_csv("results/index_funs_probs_0_centered_perceptron_sample__1.0_boolean_fc_0_none_.txt", header=None, names=["index","fun","ent","n1s"],delim_whitespace=True)
d = pd.read_csv("results/index_funs_probs_cnn_sample_sigmab0.0_CIFAR10__1.0_cifar_cnn_4_none_0000.txt", header=None, names=["index","fun","ent","n1s"],delim_whitespace=True)
d = pd.read_csv("results/index_funs_probs_0_cnn_sample_sigmab0.0_CIFAR10_uncentered__1.0_cifar_cnn_4_none_0000.txt", header=None, names=["index","fun","ent","n1s"],delim_whitespace=True)
# d = pd.concat([d,d2],axis=0)

#%%
len(d)
d = d[:100000]
num_ones = lambda x: len([c for c in x if c=="1"])

d["T"] = d.apply(lambda x: num_ones(x["fun"]),axis=1)

# hist_data = plt.hist(d["T"], bins=100)
hist_data = plt.hist(d["T"], bins=len(d["T"].unique()))

len(hist_data[0])


# plt.bar(hist_data[1][:-1],hist_data[0]/np.sum(hist_data[0]),width=1, log=True)
plt.bar(hist_data[1][:-1],hist_data[0]/np.sum(hist_data[0]),width=1, log=False)
plt.xlabel("T")
plt.ylabel("Probability")
plt.savefig("T_dist_cnn_sample1e5__mnist_cnn_4_none_0000_sigmab1.0.png")
# plt.savefig("T_dist_perceptron_sample__1.0_boolean_fc_0_none_.png")
# plt.savefig("T_dist_2perceptron_sample__1.0_boolean_fc_0_none_.png")
# plt.savefig("T_dist_0_centered_perceptron_sample__1.0_boolean_fc_0_none_.png")
# plt.savefig("T_dist_cnn_sample_sigmab0.0_CIFAR10__1.0_cifar_cnn_4_none_0000.png")
# plt.savefig("T_dist_0_cnn_sample_sigmab0.0_CIFAR10_uncentered__1.0_cifar_cnn_4_none_0000.png")

# d["ent"].plot.bar(density=True, bins=100, log=True)
