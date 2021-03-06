#%%
from fc_kernel import kernel_matrix
import numpy as np
import sys
# sys.path.insert(0, 'gp/nngp/')
# imp.reload(GP_prob_gpy)
from GP_prob_gpy import GP_prob as logGPProb

import pickle

target_comp=84.0
input_dim=7
hidden_neurons=40
hidden_layers=2
num_iters=150000
num_inits_per_task=100
size=100
m=118
train_acc=1.0

#%%

#
inputs = np.array([[float(x) for x in "{0:b}".format(i).zfill(input_dim)] for i in range(0,2**input_dim)])

target_fun = "11110011011100010111000100110001000000000000000000000000000000000011000101110001000100000011000000000000000000000000000000000000" #84.0
target_fun = "11011101111111111111110111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111" #84.0
# target_fun = "00000000100000000000000010100000000000000000000000000000000000000000000010100000000000001011000000000000000000000000000000000000" #63.0

input_dim = 7
number_layers=2
sigmaw=1.0
sigmab=1.0
K = kernel_matrix(inputs,number_layers=number_layers,sigmaw=sigmaw,sigmab=sigmab)
# K=pickle.load(open("data/K_"+str(input_dim)+"_"+str(hidden_layers)+"_"+str(sigmaw)+"_"+str(sigmab)+".p","rb"))
#
# import pickle
# pickle.dump(K,open("K_"+str(input_dim)+"_"+str(number_layers)+"_"+str(sigmaw)+"_"+str(sigmab)+".p","wb"))
target_ys=np.array([[int(xx)] for xx in list(target_fun)])

#%%

'''COLLECTING SGD DATA'''

algo = "adam"
algo = "advsgd"
different_training_sets = "one"
different_training_sets = "many"
learning_data_folder = "../../learning/84.0/Adam_7_40_40_1/"
learning_data_folder = "../../learning/63.0/Adam_7_40_40_1_single_train_set/"
learning_data_folder = "../../learning/63.0/advSGD_7_40_40_1_single_train_set/"
learning_data_folder = "../../learning/84.0/depth_sweep/"
# funs = pickle.load(open("../../learning/7_40_40_1/final_funs_advSGD_38.5_7_40_2_150000_100000.p","rb"))


signature_str_suffix="_"+algo+"_"+different_training_sets+"_"+str(target_comp)+"_"+str(input_dim)+"_"+str(hidden_neurons)+"_"+str(hidden_layers)+"_"+str(num_iters)+"_"+str(num_inits_per_task*size)+"_"+str(m)+"_no_replace_"+str(train_acc)
# signature_str_suffix="_AdamSGD_"+str(target_comp)+"_"+str(input_dim)+"_"+str(hidden_neurons)+"_"+str(hidden_layers)+"_"+str(num_iters)+"_"+str(num_inits_per_task*size)+"_"+str(m)+"_no_replace_"+str(train_acc)

funss = []
for train_set_index in range(10):
    funs = []
    signature_str=str(train_set_index)+signature_str_suffix
    for init in range(1000):
        funs += pickle.load(open(learning_data_folder+str(init)+"_final_funs_"+signature_str+".p","rb"))
    funss += [funs]

# []+[2]+[3]
len(funss)
len(funss[1])

training_sets = []
for train_set_index in range(10):
    signature_str=str(train_set_index)+signature_str_suffix
    #for init in range(100):
    init=0 # all training sets are same for different inits..
    training_sets += [pickle.load(open(learning_data_folder+str(init)+"_train_sets_"+signature_str+".p","rb"))]

len(training_sets)
#training_setss[2]
number_training_sets = len(training_sets)

''' COMPUTING GP PROBABILITIES '''

from collections import Counter
funs_flat = sum(funss,[])
cnt = Counter(funs_flat)

tot_samples = len(funs_flat)

# training_set[2]

# OldPUs = PUs
#
# ys_tmp = np.array([[(x[0]+1)/2] for x in training_set[2]])
# ys_tmp = np.array([[np.random.choice([-1,1])] for x in training_set[2]])
# ys_tmp2 = np.array(training_set[2])
#
# logGPProb(K_train,np.array(training_set[1]),ys_tmp2)
# logGPProb(K_train,np.array(training_set[1]),ys_tmp)

# Counter(funss[95])[fun]

#%%

t_set_str = "10010001010101010100101100011100110101111101100100100101101010011011111111000000100100101011010101110000001110001011001100110110"
m=64
indices = [i for i in range(128) if t_set_str[i]=="1"]

indices = np.random.choice(range(128), size=m,replace=False)
x_train = np.array([inputs[i] for i in indices])
y_train = np.array([target_ys[i] for i in indices])
# test_indices = list(np.random.choice([ii for ii in range(2**input_dim) if ii not in indices],size=test_size,replace=False))
# inputs_with_test.append(np.array([inputs[ii] for ii in test_indices]))
columns = np.matlib.repmat(indices,m,1)
# columns_with_test = np.matlib.repmat(indices+test_indices,m+test_size,1)
K_train=K[columns.T, columns]
# K_train_test=K[columns_with_test.T, columns_with_test]
# Ks_train_test.append(K_train_test)
PU = np.exp(logGPProb(K_train,x_train,y_train))
# PU = logGPProb(K_train,x_train,y_train)/np.log(10)
PU

len(cnt)

# target_ys[0]

PUs = []
# Ks_train_test = []
# test_size = 5
# inputs_with_test = []
for training_set in training_sets:
    indices = list(training_set[0])
    x_train = np.array([inputs[i] for i in indices])
    y_train = np.array([target_ys[i] for i in indices])
    # test_indices = list(np.random.choice([ii for ii in range(2**input_dim) if ii not in indices],size=test_size,replace=False))
    # inputs_with_test.append(np.array([inputs[ii] for ii in test_indices]))
    columns = np.matlib.repmat(indices,m,1)
    # columns_with_test = np.matlib.repmat(indices+test_indices,m+test_size,1)
    K_train=K[columns.T, columns]
    # K_train_test=K[columns_with_test.T, columns_with_test]
    # Ks_train_test.append(K_train_test)
    PU = np.exp(logGPProb(K_train,x_train,y_train))
    PU
    PU
    PUs.append(PU)

len(PUs)
tot_distinct_funs = len(cnt)

# np.array(training_set[1])
#
# indices

# fun = cnt.most_common()[0][0]
#
# ys=np.array([[int(xx)] for xx in list(fun)])
# GP_prob = 0
# how_many_t_sts =0
# Pf = np.exp(logGPProb(K,inputs,ys))
# for i,training_set in enumerate(training_sets):
#     how_many_t_sts+=1
#     if np.all([fun[j]==target_fun[j] for j in training_set[0]]):
#         GP_prob += Pf/PUs[i]
#         print(Pf/PUs[i])
#
# how_many_t_sts
#
# GP_prob/number_training_sets

# len([kek for kek,freq in cnt.most_common() if freq ==1])

# cnt.most_common()

len(cnt)
GP_probs = []
SGD_probs = []
for fun_index,(fun,freq) in enumerate(cnt.most_common()):
    # if freq == 1:
    #     continue
    SGD_probs.append(freq/tot_samples)
    ys=np.array([[int(xx)] for xx in list(fun)])
    Pf = np.exp(logGPProb(K,inputs,ys))
    GP_prob = 0
    for i,training_set in enumerate(training_sets):
        if np.all([fun[j]==target_fun[j] for j in training_set[0]]):
            GP_prob += Pf/PUs[i]
    if fun_index%10==0:
        print(fun_index,fun)
        print(GP_prob/number_training_sets)
    GP_probs.append(GP_prob/number_training_sets)

# fun_index
#freq/tot_sample \approx Exp[Ind_f,S1 + Ind_f,S2+ ... + Ind_f,SN]/N \approx? (p_f,S1 + ... + p_f,SN ) /N ; N = Num of training sets, Si
#tot_sample = N * (number_init_per_tasks)

# SGD_probs = []
# GP_probs = []
# NN_probs = []
# for i,training_set in training_sets:
#     for fun,freq in cnt.most_common():
#         SGD_probs.append(freq/tot_samples)
#         GP_probs.append(GPprob(fun))


import matplotlib.pyplot as plt

%matplotlib

import matplotlib
font = {'size'   : 14}

matplotlib.rc('font', **font)

# SGD_probs = list(np.array(SGD_probs)[np.array(SGD_probs)>1/tot_samples])

plt.scatter(SGD_probs,GP_probs)
plt.xscale("log")
plt.yscale("log")
nonzeroGP = list(np.array(GP_probs)[np.array(GP_probs)>0])
plt.xlim([0.7*min(SGD_probs+nonzeroGP),1.3*max(SGD_probs+GP_probs)])
plt.ylim([0.7*min(SGD_probs+nonzeroGP),1.3*max(SGD_probs+GP_probs)])
# plt.xlim([0.7*min(SGD_probs+nonzeroGP),1e0])
# plt.ylim([0.7*min(SGD_probs+nonzeroGP),1e0])
# plt.xlim([1e-4,1e0])
# plt.ylim([1e-4,1e0])

# plt.ylim()
plt.xlabel(algo+" probabilities")
plt.ylabel("GP probabilities (EP approx)")
ax = plt.gca()
plt.plot(ax.get_xlim(), ax.get_xlim(), 'k-', alpha=0.75, zorder=10)
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
# plt.savefig("SGD_prob_EPapprox_no_single_sample_vs"+signature_str_suffix+"_"+str(sigmaw)+"_"+str(sigmab)+".png")
plt.savefig("SGD_prob_EPapprox_vs"+signature_str_suffix+"_"+str(sigmaw)+"_"+str(sigmab)+".png")

# sig_str = "_AdamSGD_"+str(target_comp)+"_"+str(input_dim)+"_"+str(hidden_neurons)+"_"+str(hidden_layers)+"_"+str(num_iters)+"_"+str(num_inits_per_task*size)+"_"+str(m)+"_no_replace_"+str(train_acc)

pickle.dump(SGD_probs,open("sgd_probs"+signature_str_suffix+"_"+str(sigmaw)+"_"+str(sigmab)+".p","wb"))
pickle.dump(GP_probs,open("GP_probs"+signature_str_suffix+"_"+str(sigmaw)+"_"+str(sigmab)+".p","wb"))

#%%

''' DISTS '''

distss = []
for train_set_index in range(10):
    dists = []
    signature_str=str(train_set_index)+signature_str_suffix
    for init in range(1000):
        dists += pickle.load(open(learning_data_folder+str(init)+"_dists_"+signature_str+".p","rb"), encoding="latin1")
    distss += [dists]

dists_flat = sum(distss,[])

np.mean(dists_flat)


''' correlation coefficients '''

algo = "adam"
algo = "advsgd"
sigmab=sigmaw=10.0
m=118
signature_str_suffix="_"+algo+"_"+different_training_sets+"_"+str(target_comp)+"_"+str(input_dim)+"_"+str(hidden_neurons)+"_"+str(hidden_layers)+"_"+str(num_iters)+"_"+str(num_inits_per_task*size)+"_"+str(m)+"_no_replace_"+str(train_acc)


SGD_probs = pickle.load(open("sgd_probs"+signature_str_suffix+"_"+str(sigmaw)+"_"+str(sigmab)+".p","rb"))
GP_probs = pickle.load(open("GP_probs"+signature_str_suffix+"_"+str(sigmaw)+"_"+str(sigmab)+".p","rb"))

from scipy.stats import pearsonr

pearsonr(SGD_probs,GP_probs)
