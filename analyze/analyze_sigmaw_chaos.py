import pandas as pd

training_results = pd.read_csv("results/sigmaw_chaos/sigmaw_chaos_nn_training_results_new.txt",delimiter="\t", comment="#")
bounds = pd.read_csv("results/sigmaw_chaos/sigmaw_chaos_bounds_new.txt",delimiter="\t", comment="#")

reduced_training_results = training_results.groupby(by=["sigmaw","number_layers","test_error"],as_index=False).mean()
reduced_bounds = bounds.groupby(by=["sigmaw","number_layers","bound"],as_index=False).mean()

import matplotlib.pyplot as plt

sigmaws = training_results["sigmaw"].unique()
number_layerss = training_results["number_layers"].unique()

cmap = plt.cm.get_cmap('Spectral')

reduced_training_results = reduced_training_results.sort_values(by=["number_layers", "sigmaw"])
reduced_bounds = reduced_bounds.sort_values(by=["number_layers", "sigmaw"])

%matplotlib

for i,sigmaw in enumerate(sigmaws):
    t = reduced_training_results[reduced_training_results["sigmaw"]==sigmaw]
    b = reduced_bounds[reduced_bounds["sigmaw"]==sigmaw]
    c=cmap(1.0*i/len(sigmaws))
    plt.plot(t["number_layers"],t["test_error"], '-', c=c, label=sigmaw)
    plt.plot(b["number_layers"],b["bound"], '--', c=c, label=sigmaw)

plt.legend(bbox_to_anchor=(1.12, 1.0))

for i,number_layers in enumerate(number_layerss):
    t = reduced_training_results[reduced_training_results["number_layers"]==number_layers]
    b = reduced_bounds[reduced_bounds["number_layers"]==number_layers]
    c=cmap(1.0*i/len(number_layerss))
    plt.plot(t["sigmaw"],t["test_error"], '-', c=c)
    plt.plot(b["sigmaw"],b["bound"], '--', c=c)

######

reduced_training_results.columns

for i,sigmaw in enumerate(sigmaws):
    t = reduced_training_results[reduced_training_results["sigmaw"]==sigmaw]
    b = reduced_bounds[reduced_bounds["sigmaw"]==sigmaw]
    c=cmap(1.0*i/len(sigmaws))
    # plt.plot(t["number_layers"],t["weights_norm_mean"], '-', c=c, label=sigmaw)
    # plt.plot(t["number_layers"],t["mean_iters"], '-', c=c, label=sigmaw)
    plt.plot(t["number_layers"],t["weights_norm_mean"], '-', c=c, label=sigmaw)
    # plt.plot(t["number_layers"],1000*t["test_error"], '*-', c=c, label=sigmaw)
    # plt.plot(b["number_layers"],b["bound"], '--', c=c, label=sigmaw)

plt.legend(bbox_to_anchor=(1.12, 1.0))
