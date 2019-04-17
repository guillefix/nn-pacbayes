import pickle

csr_data=pickle.load(open("CSRs__cnn_4_none_0000.p", "rb"))
csr_data=pickle.load(open("CSRs__cnn_4_max_1111.p", "rb"))
csr_data=pickle.load(open("CSRs_0_03c_cnn_4_none_0000.p", "rb"))
csr_data=pickle.load(open("old/CSRs_test_cnn_4_none_0000.p", "rb"))

prob_data = pickle.load(open("probs_0_03c_cnn_4_none_0000.p", "rb"))

import pandas as pd

d=pd.read_csv("/local/home/valleperez/36ve/ai/CSR/old/index_funs_probs_test_cnn_4_none_0000.txt", header=None, delim_whitespace=True,names=["index","fstring","logP"])

d

csr_data

csr_data = sum(csr_data, [])
prob_data = sum(prob_data, [])

d[d["index"]==4]["logP"]

prob_data = [d[d["index"]==x[0]]["logP"][x[0]] for x in csr_data]
csr_data = [x[1] for x in csr_data]

len(csr_data)
len(prob_data)

import numpy as np
total_samples = 1000
prob_data = [-(x*total_samples-2*np.log(total_samples)-1+np.log(2**-10)) for x in prob_data]

import matplotlib.pyplot as plt

%matplotlib

plt.scatter(csr_data,prob_data)
# plt.yscale("log")

plt.hist(csr_data)

plt.xlabel("CSR")
plt.ylabel("Frequency")
plt.ylabel("logP")
# plt.savefig("CSRs_0_03c_cnn_4_none_0000.png")
plt.savefig("CSR_logP_test_cnn_4_none_0000.png")

# plt.savefig("CSRs__cnn_4_none_0000.png")
plt.savefig("CSRs__cnn_4_max_1111.png")

fstrings=pickle.load(open("fstrings_0_03c_cnn_4_none_0000.p", "rb"))

csr_data
