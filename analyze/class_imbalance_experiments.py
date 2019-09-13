import pandas as pd
%matplotlib inline
filename="unbalanced_mnist_sens_nn_training_results.txt"
filename="unbalancedt1_mnist_nn_training_results.txt"
filename="unbalancedt1_emnist_nn_training_results.txt"
d=pd.read_csv(filename, sep="\t",comment="#")
d

import numpy as np
d.groupby(["centering","number_layers","sigmab"],as_index=False).count()[["centering","number_layers","sigmab","test_error","test_sensitivity"]]
d1 = d[d["train_acc"]==1].groupby(["centering","number_layers","sigmab"],as_index=False).mean()[["centering","number_layers","sigmab","test_error","test_acc","train_acc","test_sensitivity"]]
d[d["train_acc"]==1][["centering","number_layers","sigmab","test_error","test_acc","train_acc","test_sensitivity"]].groupby(["centering","number_layers","sigmab"],as_index=True).std()[["test_error","test_acc","train_acc","test_sensitivity"]]



d1.groupby(["centering"]).mean()["test_acc"].plot.line()
d1.groupby(["number_layers"]).mean()["test_acc"].plot.line()
d1.groupby(["sigmab"]).mean()["test_acc"].plot.line()


d1[(d1["centering"]) & (d1["number_layers"]==2)][["sigmab","test_acc"]].plot.line("sigmab","test_acc")
# d1[(~d1["centering"]) & (d1["number_layers"]==8)][["sigmab","test_acc","train_acc"]].plot.line("sigmab",["test_acc","train_acc"])
d1[(~d1["centering"]) & (d1["number_layers"]==1)][["sigmab","test_acc","train_acc"]].plot.line("sigmab",["test_acc"])
d1[(d1["centering"]) & (d1["sigmab"]==0.0)][["number_layers","test_acc"]].plot.line("number_layers","test_acc")
d1[(~d1["centering"]) & (d1["sigmab"]==2.0)][["number_layers","test_acc"]].plot.line("number_layers","test_acc")
