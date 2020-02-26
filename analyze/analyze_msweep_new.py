import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib

#%%

#main NNGP big run (up to 4k training set size:)
# training_data = pd.read_csv("results/new_mother_of_all_msweeps_nn_training_results.txt", sep="\t", comment="#")
bounds = pd.read_csv("results/new_mother_of_all_msweeps_bounds1000.txt", sep="\t", comment="#")

##resnet50 training results:
training_data = pd.read_csv("results/2jade_msweep_nn_training_results.txt", sep="\t", comment="#")
# training_data = pd.read_csv("results/3gpu_msweep_nn_training_results.txt", sep="\t", comment="#")

#Other stuff:
# training_data = pd.read_csv("results/gpu_msweep_nn_training_results.txt", sep="\t", comment="#")

# training_data = pd.read_csv("results/2gpu_msweep_nn_training_results.txt", sep="\t", comment="#")
# training_data = pd.read_csv("results/gpu_msweep_nn_training_results_cnn.txt", sep="\t", comment="#")
# training_data = pd.read_csv("results/2jade_new_msweep_nn_training_results.txt", sep="\t", comment="#")
# bounds = pd.read_csv("results/new_mother_of_all_msweeps_bounds.txt", sep="\t", comment="#")
# bounds = pd.read_csv("results/new_mother_of_all_msweeps_bounds100.txt", sep="\t", comment="#")
# bounds = pd.read_csv("results/new_mother_of_all_msweeps_bounds50.txt", sep="\t", comment="#")

# bounds = pd.read_csv("results/2new_mother_of_all_msweeps_bounds.txt", sep="\t", comment="#")
# bounds = pd.read_csv("results/gpu_msweep_bounds.txt", sep="\t", comment="#")

# bounds = pd.read_csv("results/2gpu_msweep_bounds.txt", sep="\t", comment="#")

# bounds = pd.read_csv("results/gpu_msweep_bounds_cnn.txt", sep="\t", comment="#")
# bounds = pd.read_csv("results/2jade_new_msweep_bounds.txt", sep="\t", comment="#")


nets = training_data["network"].unique()
datasets = training_data["dataset"].unique()

#sort by m
training_data = training_data.sort_values("m")
bounds = bounds.sort_values("m")
bounds = bounds[bounds["m"]>=50]
training_data = training_data[training_data["m"]>=50]
# bounds = bounds[bounds["kern_mult"]==10000]
bounds = bounds.groupby(["m","network","dataset","pooling"],as_index=False).mean()
# bounds = bounds[(bounds["sigmaw"]==10.0) & (bounds["sigmab"]==10.0)]

#PLOT
#%%
# net="cnn"
# net="fc"
# net="resnetv2_50"
# net="densenet201"
# net="resnext101"
# net="resnetv2_101"
# net="nasnet"
# net="vgg16"
# net="resnext101"
# net="densenet121"
# net="resnet50"
# net="mobilenetv2"
dataset="mnist"
# dataset="EMNIST"
# dataset="KMNIST"
# dataset="cifar"
# colors = np.random.rand(len(nets),3)
import matplotlib
cmap = matplotlib.cm.get_cmap('Spectral')
for i, net in enumerate(nets):
# for i,dataset in enumerate(datasets):
# for ii in [1]:
    # pool="None"
    # pool="avg"
    pool="max"
    if net=="fc":
        pool="None"
    bdata=bounds[(bounds["network"]==net) & (bounds["dataset"]==dataset) & (bounds["pooling"]==pool)]
    # bdata=bounds[(bounds["network"]==net) & (bounds["dataset"]==dataset)]
    tdata=training_data[(training_data["network"]==net) & (training_data["dataset"]==dataset) & (training_data["pooling"]==pool)]
    # bounds.dtypes

    # bounds["bound"] = pd.to_numeric(bounds["bound"])
    tdata.columns
    bdata.columns

    color = cmap(i/len(nets))
    # color = cmap(i/len(datasets))
    # plt.plot(bdata["m"], bdata["bound"], c=color, label="PAC-Bayes bound "+net+" "+dataset+" "+pool)
    # plt.plot(bdata["m"], -bdata["logP"]/bdata["m"], c=color, label="PAC-Bayes bound "+net+" "+dataset+" "+pool)
    # plt.plot(bdata["m"], -bdata["logP"]/bdata["m"], c=color)
    plt.plot(tdata["m"], tdata["test_error"], "--", c=color, label="Test error "+net+" "+dataset+" "+pool)
    # plt.plot(tdata["m"], tdata["train_acc"], label="Training error "+net+" "+dataset+" "+pool)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("m")
    plt.ylabel("generalization error")

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
plt.legend()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig("learning_curve_fc_mnist_km10k_1_41_0.png")
# plt.savefig("learning_curve_resnet50_max_mnist_second_training_set_sample__1_41_0.png")
# plt.savefig("learning_curve_resnet50_max_mnist_combined_training_set_samples__1_41_0.png")
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
#
# MPI.COUNT


#%%
###### exploring the standard deviation

training_data["test_acc_std"] = training_data["test_acc_std"] + training_data["test_acc"] - training_data["test_acc"]**2

training_data = training_data.sort_values("test_acc_std")

np.min(training_data["number_inits"])

np.max(training_data["test_acc_std"]/training_data["test_acc"])

training_data["relative_test_acc_error"]=training_data["test_acc_std"]/training_data["test_acc"]

training_data = training_data.sort_values("relative_test_acc_error")

training_data[["test_acc","relative_test_acc_error"]]

np.min(training_data[training_data["relative_test_acc_error"]>0.01]["test_acc"])

np.sort(training_data["test_acc_std"]/training_data["test_acc"])[-40:]

training_data[["test_acc","test_acc_std"]]
