import pandas as pd
import numpy as np

# number_layers = 2

############ new_data
rootfolder = "results/results_compsweep/"
quantities = ['m','n_filters','n_layers','n_inits','sigmaw','sigmab','label_corruption','training_error','generror','weights_std','biases_std','weights_norm_mean','weights_norm_std','biases_norm_mean','biases_norm_std','mean_iters']

data = {"cnn":{},"fc":{},"resnet":{}}
bounds = {"cnn":{},"fc":{},"resnet":{}}
for network in ['cnn','fc']:
    for i,dataset in enumerate(["mnist","mnist-fashion","cifar"]):

        data[network][dataset] = pd.read_csv(rootfolder+network+"_"+dataset+"_nn_training_results.txt",sep="\t",\
            header=None, names=quantities).drop_duplicates("label_corruption")
        bounds[network][dataset] = pd.read_csv(rootfolder+network+"_"+dataset+"_bounds.txt",\
            sep="\t",header=None, names=['m','sigmaw','sigmab','label_corruption','bound']).drop_duplicates("label_corruption")

        data[network][dataset] = data[network][dataset].groupby(['m','sigmaw', 'sigmab','label_corruption'],as_index=False).mean()

for network in ['cnn','fc']:
    for i,dataset in enumerate(["mnist","mnist-fashion","cifar"]):
        print(network, dataset, data[network][dataset][data[network][dataset]["label_corruption"]==0.0]["generror"].iloc[0])
        print(network, dataset, 1-np.exp(-bounds[network][dataset][bounds[network][dataset]["label_corruption"]==0.0]["bound"].iloc[0]))

import matplotlib.pyplot as plt

# %matplotlib
# network = 'cnn'
# dataset = 'cifar'

delta=0.5**10
m=10000
# fig, ax1 = plt.subplots()
# # These are in unitless percentages of the figure size. (0,0 is bottom left)
# left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
# ax2 = fig.add_axes([left, bottom, width, height])
# i=0
# ax1.plot(data[network][dataset]["label_corruption"], data[network][dataset]["generror"], '--o', c=colors[i],label=dataset+" Mean error")
# ax1.plot(bounds[network][dataset]["label_corruption"], 1-np.exp(-bounds[network][dataset]["bound"]), '-^', c=colors[i], label=dataset+" PAC-Bayes bound")
# ax2.plot(bounds[network][dataset]["label_corruption"], -m*bounds[network][dataset]["bound"]-np.log(1/delta)-2*np.log(m)-1)
# ax2.set_yscale("log")

# ax1.legend

colors = ['C0','C1','C2']
for network in ['cnn','fc']:
    fig, ax1 = plt.subplots()
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.22, 0.62, 0.235, 0.235]
    # left, bottom, width, height = [1-0.205-0.2, 1-0.6-0.23, 0.22, 0.22]
    ax2 = fig.add_axes([left, bottom, width, height])
    for i,dataset in enumerate(["mnist","mnist-fashion","cifar"]):
        logPU = -m*bounds[network][dataset]["bound"]-np.log(1/delta)-2*np.log(m)-1
        ax1.plot(data[network][dataset]["label_corruption"], data[network][dataset]["generror"], '--o', c=colors[i],label=dataset+" Mean error")
        # ax1.plot(bounds[network][dataset]["label_corruption"], 1-np.exp(-bounds[network][dataset]["bound"]), '-^', c=colors[i], label=dataset+" PAC-Bayes bound")
        ax1.plot(bounds[network][dataset]["label_corruption"], 1-np.exp(-(-logPU+np.log(1/delta)+np.log(2*np.sqrt(m)))/m), '-^', c=colors[i], label=dataset+" PAC-Bayes bound")
        ax2.plot(bounds[network][dataset]["label_corruption"], logPU, c=colors[i])
    ax1.set_xlabel("Label corruption", fontsize=16)
    ax1.set_ylabel("Generalization error", fontsize=16)
    ax1.legend()
    ax1.set_ylim([0.0, 1.0])
    ax2.set_ylim([-9000,-1500])
    # plt.ylim([0.0, 0.8])
    if network == 'cnn':
        plt.savefig("new_bound_insets_"+network+"_sigmaw_1_sigmab_1_MNIST_fashionMNIST_CIFAR_generror_vs_labelcorruption.png")
    if network == 'fc':
        plt.savefig("new_bound_insets_"+network+"_sigmaw_10_sigmab_10_MNIST_fashionMNIST_CIFAR_generror_vs_labelcorruption.png")
    plt.close()



################

# d = pd.read_csv("generalization_errors_"+str(number_layers)+".txt", sep="\t", names=['m','num_layers','n_inits','sigmaw', 'sigmab', 'label_corruption', 'training_err', 'generror', 'bound'])
#
# averaged_data = d.groupby(['m','num_layers','n_inits','sigmaw', 'sigmab','label_corruption', 'bound'],as_index=False).mean()

# averaged_data["label_corruption"]
# averaged_data["generror"]
# averaged_data["bound"]


# imp.reload(plots)
# from plots import shaded_std_plot_double_scatter,shaded_std_plot
#
# list(d[d["label_corruption"]==0.0]["generror"])
#
# m = 30
# number_layers = 2
# sigmaw = 100.0
# sigmab = 100.0
#
# signature_string = str(m)+"_"+str(sigmaw)+"_"+str(sigmab)+"_"+str(number_layers)
# shaded_std_plot(averaged_data["label_corruption"],[list(d[d["label_corruption"]==c]["generr"]) for c in averaged_data["label_corruption"]],filename="test"+signature_string+".png",xlabel="Label corruption", ylabel="generalization error")

## VARIANCE SEEMS VERY SMALL
