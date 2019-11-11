import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib

#1st experiment with perceptron
# train_set="01110001001001110110011110110000010110001111100011100110000011011010001000001010010101110110100100111101100111111010101100001001"
# target_fun="00000000000000001111111111111111000000000000000011111111111111110000000000000000111111111111111100000000000000001111111111111110"
#2nd batch of experiments
# train_set="11100010000011101100111101100000101100011111000111001100000110110100100000010100110101101101100001111011101111110110100101001000"
train_set="01100010000011100000110101000000101100001010000101001000000110010100000000010000000000000001000000101001100000010010000101000000"
target_fun="00110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011"

train_set_indices = [i for i in range(len(train_set)) if train_set[i]=="1"]
# train_set_indices = [i for i in range(len(train_set)-1) if train_set[i+1]=="1"]
test_set_indices = [i for i in range(len(train_set)) if train_set[i]=="0"]
# test_set_indices = [i for i in range(len(train_set)-1) if train_set[i+1]=="0"]

# abi_sample = pd.read_csv("results/abi_sample_0_35.0_2_7_40_1_1.000000_0.000000_fun_samples.txt",header=None,delim_whitespace=True, names=["fun","generror","ent","freq"], comment="#")
# sgd_sample = pd.read_csv("results/sgd_perceptron_64_512_0_8_sgd_ce_1_combined_nn_train_functions_counted.txt", header=None, delim_whitespace=True, names=["freq","testfun"])
# abi_sample = pd.read_csv("results/abi_samples_sgd_vs_bayes_2hlfc_counts_sorted.txt",header=None,delim_whitespace=True, names=["freq", "fun", "generror"], comment="#")
# sgd_sample = pd.read_csv("results/sgd_fc_64___8_sgd_ce_1_train_functions_counts_sorted.txt", header=None, delim_whitespace=True, names=["freq","testfun"])
abi_sample = pd.read_csv("results/m32_abi_sample_net_2hl_counts_sorted.txt",header=None,delim_whitespace=True, names=["freq", "fun", "generror"], comment="#")
sgd_sample = pd.read_csv("results/sgd_fc_32___8_sgd_ce_1_counts_sorted.txt", header=None, delim_whitespace=True, names=["freq","testfun"])

abi_sample.sort_values("freq",ascending=False)

sgd_sample

sgd_tot_samples = sum(sgd_sample["freq"])
abi_tot_samples = sum(abi_sample["freq"])

abi_sample["testfun"] = abi_sample["fun"].apply(lambda fun: "".join([x for i,x in enumerate(fun) if i in test_set_indices]))
# abi_sample["testfun"] = abi_sample["fun"].apply(lambda fun: "".join([x for i,x in enumerate(fun[1:]) if i in test_set_indices]))

abi_sample = abi_sample.groupby("testfun").sum()

# abi_sample["testfun"].iloc[0] in abi_sample["testfun"].unique()
# sgd_sample["testfun"].iloc[0] in abi_sample["testfun"].unique()

# abi_sample = abi_sample.set_index("testfun")
sgd_sample = sgd_sample.set_index("testfun")

# abi_sample.index
#
# abi_sample.loc[test_fun_abi_sample_unique[0]]["freq"]

# test_fun_abi_sample_unique = abi_sample["testfun"].unique()

abi_freqs = []
sgd_freqs = []
test_funs = []
# for index,row in list(sgd_sample[["testfun","freq"]].iterrows())[:10]:
#     print(row["freq"])

for test_fun,row in sgd_sample.iterrows():
    # test_fun = row["testfun"]
    sgd_freq = row["freq"]
    if test_fun in abi_sample.index:
        if sgd_freq > 3:
            abi_freq = abi_sample.loc[test_fun]["freq"]
            if abi_freq > 3:
                abi_freqs.append(abi_freq)
                sgd_freqs.append(sgd_freq)
                test_funs.append(test_fun)
#     else:
#         abi_freqs.append(1)
#         sgd_freqs.append(sgd_sample[sgd_sample["testfun"]==test_fun].iloc[0]["freq"])
#
# for test_fun in abi_sample["testfun"]:
#     if test_fun not in sgd_sample["testfun"].unique():
#         abi_freqs.append(abi_sample[abi_sample["testfun"]==test_fun].iloc[0]["freq"])
#         sgd_freqs.append(1)

normalized_abi_freqs = np.array(abi_freqs)/abi_tot_samples
normalized_sgd_freqs = np.array(sgd_freqs)/sgd_tot_samples
#%%

len(logPUs_GP) == len(normalized_sgd_freqs)

len(normalized_sgd_freqs)
plt.scatter(normalized_abi_freqs, normalized_sgd_freqs)
# plt.scatter(logPUs_GP, normalized_sgd_freqs)
plt.yscale("log")
plt.xscale("log")
plt.xlim([normalized_abi_freqs.min()*0.5,normalized_abi_freqs.max()*1.5])
# plt.xlim([min(logPUs_GP)*1.1,max(logPUs_GP)*0.1])
plt.ylim([normalized_sgd_freqs.min()*0.5,normalized_sgd_freqs.max()*1.5])
plt.xlabel("ABI probabilities")
plt.ylabel("SGD probabilities")
# plt.savefig("sgd_fc_64___8_sgd_ce_vs_abi_7_2x40_1_above5.png")
plt.savefig("sgd_fc_32___8_sgd_ce_vs_abi_7_2x40_1_above5.png")

# plt.scatter(logPUs_GP, normalized_sgd_freqs)
plt.scatter(logPUs_GP, normalized_abi_freqs)
plt.yscale("log")
plt.xlim([min(logPUs_GP)*1.1,max(logPUs_GP)*0.1])
# plt.ylim([normalized_sgd_freqs.min()*0.5,normalized_sgd_freqs.max()*1.5])
plt.ylim([normalized_abi_freqs.min()*0.5,normalized_abi_freqs.max()*1.5])
plt.xlabel("GPEP probabilities")
# plt.ylabel("SGD probabilities")
plt.ylabel("ABI probabilities")
# plt.savefig("sgd_fc_32___8_sgd_ce_vs_gpep_7_2x40_1_above4.png")
plt.savefig("abi_fc_32___8_sgd_ce_vs_gpep_7_2x40_1_above4.png")

H, xedges, yedges = np.histogram2d(normalized_abi_freqs, normalized_sgd_freqs)

H


h


# h,_,_,_ = plt.hist2d(np.log(normalized_abi_freqs), np.log(normalized_sgd_freqs), weights=np.maximum(1,normalized_abi_freqs), bins=30)
# h,xedges,yedges,_ = plt.hist2d(np.log(normalized_abi_freqs), np.log(normalized_sgd_freqs), weights=normalized_abi_freqs, bins=30)
h,xedges,yedges,_ = plt.hist2d(np.log10(normalized_abi_freqs), np.log10(normalized_sgd_freqs), weights=normalized_sgd_freqs, bins=30)
h,_,_,_ = plt.hist2d(np.log(normalized_abi_freqs), np.log(normalized_sgd_freqs), bins=30)
h = h/np.maximum(1e-6,h.max(axis=1, keepdims=True))
# h = h/np.maximum(1e-6,h.max(axis=0, keepdims=True))
plt.imshow(np.rot90(h))
tick_places = list(map(lambda x: np.argmin(np.abs(xedges+x)), range(5,0,-1)))
# tick_places = range(3,30,5)
plt.xticks(tick_places,["$10^{{{0:.0f}}}$".format(x) for i,x in enumerate(xedges) if i in tick_places])
tick_places = list(map(lambda x: np.argmin(np.abs(yedges+x)), range(0,6)))
plt.yticks(tick_places,["$10^{{{0:.0f}}}$".format(x) for i,x in enumerate(yedges) if i in tick_places])
plt.xlabel("ABI probabilities")
plt.ylabel("SGD probabilities")
# plt.savefig("sgd_fc_32___8_sgd_ce_vs_abi_7_2x40_1_above3_sgd_weighted_column_normalized.png")
plt.savefig("sgd_fc_32___8_sgd_ce_vs_abi_7_2x40_1_above3_sgd_weighted_row_normalized.png")
# plt.savefig("sgd_fc_32___8_sgd_ce_vs_abi_7_2x40_1_above3_unweighted_column_normalized.png")
plt.savefig("sgd_fc_32___8_sgd_ce_vs_abi_7_2x40_1_above3_unweighted_row_normalized.png")
# plt.savefig("sgd_fc_32___8_sgd_ce_vs_abi_7_2x40_1_above3_abi_weighted_column_normalized.png")
plt.hist2d(np.log(normalized_abi_freqs), np.log(normalized_sgd_freqs))
from matplotlib.colors import LogNorm
plt.hist2d(np.log(normalized_abi_freqs), np.log(normalized_sgd_freqs), bins=40, norm=LogNorm())

import seaborn as sns

%matplotlib
# %matplotlib inline

sns.jointplot(x=np.log(normalized_abi_freqs), y=np.log(normalized_sgd_freqs), kind="hex", color="k");
sns.jointplot(x=np.log(normalized_abi_freqs), y=np.log(normalized_sgd_freqs), kind="reg", color="k");

# sns.violinplot(np.log(normalized_abi_freqs), np.log(normalized_sgd_freqs))

############################GP

#%%

from utils import load_data_by_filename
filename = "sgd_fc_32___8_sgd_ce_1__fc_boolean_32_0.0_0.0_True_False_False_False_-1_True_False_False_none_data.h5"
train_images,flat_data,ys,test_images,test_ys = load_data_by_filename("data/"+filename)

# from utils import load_data,load_model,load_kernel
# train_images,flat_train_images,ys,test_images,test_ys = load_data(FLAGS)
input_dim = train_images.shape[1]
num_channels = train_images.shape[-1]
# tp_order = np.concatenate([[0,len(train_images.shape)-1], np.arange(1, len(train_images.shape)-1)])
# train_images = tf.constant(train_images)
import numpy as np
X = np.stack([x.flatten() for x in train_images])
X_test = np.stack([x.flatten() for x in test_images])

# test_images = test_images[:500]
# test_ys = test_ys[:500]

Xfull =  np.concatenate([X,X_test])
ys2 = [[y] for y in ys]
ysfull = ys2 + [[y] for y in test_ys]
Yfull = np.array(ysfull)
Y = np.array(ys2)

# Xfull.shape
# X.shape

from nngp_kernel.fc_kernel import kernel_matrix

Kfull = kernel_matrix(Xfull, number_layers=2,sigmaw=1.0,sigmab=1.0)
K = kernel_matrix(X, number_layers=2,sigmaw=1.0,sigmab=1.0)
from GP_prob.GP_prob_gpy import GP_prob

kern_multiplier = 100

#%%

# test_ys
#
# test_ys = np.array([float(x) for x in test_funs[0]])

GP_prob_train_set = GP_prob(kern_multiplier*K,X,Y, method="EP")

logPUs_GP = []
for test_fun in test_funs:
    ysfull = ys2 + [[float(x)] for x in test_fun]
    Yfull = np.array(ysfull)
    logPU = GP_prob(kern_multiplier*Kfull,Xfull,Yfull, method="EP") - GP_prob_train_set
    logPUs_GP.append(logPU)

# logPU
import pickle
pickle.dump(logPUs_GP,open("logPUs_GP_7_2x40_1_fc_32___8_sgd_ce_kernmult100.p","wb"))

logPUs_GP = pickle.load(open("logPUs_GP_7_2x40_1_fc_32___8_sgd_ce_kernmult100.p","rb"))
