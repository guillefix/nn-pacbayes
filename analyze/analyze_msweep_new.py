import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib

training_data = pd.read_csv("results/new_mother_of_all_msweeps_nn_training_results.txt", sep="\t", comment="#")
bounds = pd.read_csv("results/new_mother_of_all_msweeps_bounds.txt", sep="\t", comment="#")

nets = training_data["network"].unique()
datasets = training_data["dataset"].unique()

#%%
# net="fc"
net="resnetv2_50"
dataset="EMNIST"
# for net in nets:
# for dataset in datasets:
for ii in [1]:
    pool="avg"
    if net=="fc":
        pool="None"
    bdata=bounds[(bounds["network"]==net) & (bounds["dataset"]==dataset) & (bounds["pooling"]==pool)]
    # bdata=bounds[(bounds["network"]==net) & (bounds["dataset"]==dataset)]
    tdata=training_data[(training_data["network"]==net) & (training_data["dataset"]==dataset) & (training_data["pooling"]==pool)]
    # bounds.dtypes

    # bounds["bound"] = pd.to_numeric(bounds["bound"])
    tdata.columns
    bdata.columns

    color = np.random.rand(3,)
    plt.plot(bdata["m"], bdata["bound"], c=color)
    # plt.plot(bdata["m"], -bdata["logP"]/bdata["m"], c=color)
    plt.plot(tdata["m"], tdata["test_error"], "--", c=color)
    plt.plot(tdata["m"], tdata["train_acc"])
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("m")
    plt.ylabel("generalization error")
