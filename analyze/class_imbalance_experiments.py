impor_ pandas as pd
filename="unbalanced_mnist_sens_nn_training_results.txt"
filename="unbalancedt1_mnist_nn_training_results.txt"
filename="unbalancedt1_emnist_nn_training_results.txt"
d=pd.read_csv(filename, sep="\t",comment="#")
d.groupby(["centering","number_layers","sigmab","test_error","test_sensitivity"],as_index=False).mean()[["centering","number_layers","sigmab","test_error","test_sensitivity"]]

