import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib

quantities = ['m','n_filters','n_layers','n_inits','sigmaw','sigmab','label_corruption','training_error','generror','weights_std','biases_std','weights_norm_mean','weights_norm_std','biases_norm_mean','biases_norm_std','mean_iters']
network = "cnn"
dataset="mnist"
dataset="mnist-fashion"
dataset="cifar"
pretty_names = {"mnist":"MNIST","mnist-fashion":"fashion-MNIST","cifar":"CIFAR"}
colors = ['C0','C1','C2']
M_MAX=40000
for i,dataset in enumerate(["mnist","mnist-fashion","cifar"]):
    # d_bounds = pd.read_csv("results/results_msweep/"+network+"_"+dataset+"_bounds.txt",header=None,delimiter="\t",names=["m","number_layers","sigmaw","sigmab","label_corruption","bound"])
    d_train_data = pd.read_csv("results/results_msweep/"+network+"_"+dataset+"_nn_training_results.txt",header=None,delimiter="\t",names=quantities)
    d_train_data = d_train_data[d_train_data["m"]<=M_MAX]
    # d_bounds = d_bounds[d_bounds["m"]<=M_MAX]

    plt.plot(d_train_data["m"], d_train_data["generror"], '--o',c=colors[i], label=pretty_names[dataset]+" Mean test error")
    # plt.plot(d_bounds["m"], d_bounds["bound"],'-^', c=colors[i], label=pretty_names[dataset]+" PAC-Bayes bound")

for i,dataset in enumerate(["mnist","mnist-fashion","cifar"]):
    d_bounds = pd.read_csv("results/results_msweep/"+network+"_"+dataset+"_bounds.txt",header=None,delimiter="\t",names=["m","number_layers","sigmaw","sigmab","label_corruption","bound"])
    # d_train_data = pd.read_csv("msweep/"+network+"_"+dataset+"_nn_training_results.txt",header=None,delimiter="\t",names=quantities)
    # d_train_data = d_train_data[d_train_data["m"]<=M_MAX]
    d_bounds = d_bounds[d_bounds["m"]<=M_MAX]
    # d_bounds["new_bound"] = d_bounds["bound"]
    m = d_bounds["m"]
    d_bounds["new_bound"] = 1-np.exp(-(d_bounds["bound"]*m+10*np.log(2))/(m-1))

    # plt.plot(d_train_data["m"], d_train_data["generror"], '--o',c=colors[i], label=pretty_names[dataset]+" Mean test error")
    plt.plot(d_bounds["m"], d_bounds["new_bound"],'-^', c=colors[i], label=pretty_names[dataset]+" PAC-Bayes bound")

# plt.scatter(d_bounds["number_layers"], [b-0.3 for b in d_bounds["bound"]])

plt.xlabel("Training set size",fontsize=12)
plt.ylabel("Generalization error",fontsize=12)
# plt.legend(loc="center right",fontsize=7)
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=9)

plt.subplots_adjust(left=0.1, right=0.9, top=0.83, bottom=0.1)

plt.savefig("msweeps_mnist_fashion-mnist_cifar.png")
# plt.savefig("err_vs_layers_cnn.png")
# plt.savefig("err_vs_m_"+dataset+".png")