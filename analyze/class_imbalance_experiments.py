import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
filename="unbalanced_mnist_sens_nn_training_results.txt"
dataset="unbalancedt1_mnist"
dataset="unbalancedt1_emnist"
dataset="unbalanced_boolean"
dataset="unbalancedt9_cifar"
dataset="unbalancedt9_mnist"
dataset="init_dist_test"
filename=dataset+"_nn_training_results.txt"
# filename="unbalancedt1_emnist_nn_training_results.txt"
filename="boolfunnn_training_results.txt"
filename="imbalanced_boolean_nn_training_results.txt"
filename="oversampling_test_nn_training_results.txt"
# filename="unbalanced_boolean_nn_training_results.txt"
d=pd.read_csv(filename, sep="\t",comment="#")
# d["train_acc"].plot.hist()

# d["number_layers"]

import numpy as np
d.groupby(["centering","number_layers","sigmab"],as_index=False).count()[["centering","number_layers","sigmab","test_error","test_sensitivity"]]
d1 = d[d["train_acc"]==1].groupby(["threshold","init_dist","centering","number_layers","sigmab"],as_index=False).mean()[["threshold","init_dist","centering","number_layers","sigmab","test_error","test_acc","train_acc","test_sensitivity"]]
# d1 = d[d["train_acc"]==1].groupby(["init_dist","centering","number_layers","sigmab"],as_index=False).mean()[["init_dist","centering","number_layers","sigmab","test_error","test_acc","train_acc","test_sensitivity"]]
# d1 = d[d["train_acc"]==1].groupby(["centering","number_layers","sigmab"],as_index=False).mean()[["centering","number_layers","sigmab","test_error","test_acc","train_acc","test_sensitivity"]]

# d1 = d.groupby(["centering","number_layers","sigmab"],as_index=False).mean()[["centering","number_layers","sigmab","test_error","test_acc","train_acc","test_sensitivity"]]
# # d1 = d[d["train_acc"]==1].groupby(["centering","number_layers","sigmab"],as_index=False).mean()[["centering","number_layers","sigmab","test_error","train_acc"]]
# d[d["train_acc"]==1][["centering","number_layers","sigmab","test_error","test_acc","train_acc","test_sensitivity"]].groupby(["centering","number_layers","sigmab"],as_index=True).std()[["test_error","test_acc","train_acc","test_sensitivity"]]
# d[d["train_acc"]==1][["centering","number_layers","sigmab","test_error","train_acc"]].groupby(["centering","number_layers","sigmab"],as_index=True).std()[["test_error","train_acc"]]
d1


d1.groupby(["centering"]).mean()["test_acc"].plot.line()
d1.groupby(["number_layers"]).mean()["test_acc"].plot.line()
d1.groupby(["number_layers"]).mean()["test_sensitivity"].plot.line()
d1.groupby(["sigmab"]).mean()["test_sensitivity"].plot.line()
%matplotlib

# d1.groupby(["centering"]).mean()["test_sensitivity"].plot.line()
#%%
d1.groupby(["sigmab"]).mean()["test_acc"].plot.line()
plt.xlabel("$\\sigma_b$")
plt.ylabel("Accuracy")
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
plt.savefig("acc_sigmab_all_marginal_"+dataset+".png")

#%%
L=8
for L in range(1,7):
    d1[(d1["centering"]) & (d1["number_layers"]==L)][["sigmab","test_acc"]].plot.line("sigmab","test_acc", legend=False)
    plt.xlabel("$\\sigma_b$")
    plt.ylabel("Accuracy")
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.savefig("img/acc_sigmab_centering_L"+str(L)+"_"+dataset+".png")
    # d1[(~d1["centering"]) & (d1["number_layers"]==8)][["sigmab","test_acc","train_acc"]].plot.line("sigmab",["test_acc","train_acc"])
    d1[(~d1["centering"]) & (d1["number_layers"]==L)][["sigmab","test_acc","train_acc"]].plot.line("sigmab",["test_acc"],legend=False)
    plt.xlabel("$\\sigma_b$")
    plt.ylabel("Accuracy")
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.savefig("img/acc_sigmab_nocentering_L"+str(L)+"_"+dataset+".png")
#%%
plt.close()
d1[(d1["centering"]) & (d1["sigmab"]==0.0)][["number_layers","test_acc"]].plot.line("number_layers","test_acc")
d1[(~d1["centering"]) & (d1["sigmab"]==2.0)][["number_layers","test_acc"]].plot.line("number_layers","test_acc")
