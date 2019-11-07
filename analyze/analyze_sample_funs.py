import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib

d = pd.read_csv("results/index_funs_probs_cnn_sample__mnist_cnn_4_none_0000.txt", header=None, names=["index","fun","ent"],delim_whitespace=True)

num_ones = lambda x: len([c for c in x if c=="1"])

d["T"] = d.apply(lambda x: num_ones(x["fun"]),axis=1)

hist_data = plt.hist(d["T"], bins=100)

len(hist_data[0])


plt.bar(hist_data[1][:-1],hist_data[0]/np.sum(hist_data[0]),width=1, log=True)
plt.xlabel("T")
plt.ylabel("Probability")
plt.savefig("T_dist_cnn_sample1e5__mnist_cnn_4_none_0000.png")

d["ent"].plot.bar(density=True, bins=100, log=True)
