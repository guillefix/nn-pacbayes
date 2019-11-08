import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib

train_set="01110001001001110110011110110000010110001111100011100110000011011010001000001010010101110110100100111101100111111010101100001001"
target_fun="00000000000000001111111111111111000000000000000011111111111111110000000000000000111111111111111100000000000000001111111111111110"

# train_set_indices = [i for i in range(len(train_set)) if train_set[i]=="1"]
train_set_indices = [i for i in range(len(train_set)-1) if train_set[i+1]=="1"]
# test_set_indices = [i for i in range(len(train_set)) if train_set[i]=="0"]
test_set_indices = [i for i in range(len(train_set)-1) if train_set[i+1]=="0"]

abi_sample = pd.read_csv("results/abi_sample_0_35.0_2_7_40_1_1.000000_0.000000_fun_samples.txt",header=None,delim_whitespace=True, names=["fun","generror","ent","freq"], comment="#")
sgd_sample = pd.read_csv("results/sgd_perceptron_64_512_0_8_sgd_ce_1_combined_nn_train_functions_counted.txt", header=None, delim_whitespace=True, names=["freq","testfun"])

sgd_tot_samples = sum(sgd_sample["freq"])
abi_tot_samples = sum(abi_sample["freq"])

# abi_sample["testfun"] = abi_sample["fun"].apply(lambda fun: "".join([x for i,x in enumerate(fun) if i in test_set_indices]))
abi_sample["testfun"] = abi_sample["fun"].apply(lambda fun: "".join([x for i,x in enumerate(fun[1:]) if i in test_set_indices]))

abi_sample["testfun"].iloc[0] in abi_sample["testfun"].unique()
sgd_sample["testfun"].iloc[0] in abi_sample["testfun"].unique()

abi_freqs = []
sgd_freqs = []
for test_fun in sgd_sample["testfun"]:
    if test_fun in abi_sample["testfun"].unique():
        abi_freqs.append(abi_sample[abi_sample["testfun"]==test_fun].iloc[0]["freq"])
        sgd_freqs.append(sgd_sample[sgd_sample["testfun"]==test_fun].iloc[0]["freq"])

plt.scatter(np.array(abi_freqs)/abi_tot_samples, np.array(sgd_freqs)/sgd_tot_samples)
plt.yscale("log")
plt.xscale("log")
plt.xlim([1e-5,1e0])
plt.ylim([1e-6,1e0])
