import pandas as pd
import os

#for f in $(ls results/sgd_vs_bayes/generrs/); do sed '1s/^.//' -i results/sgd_vs_bayes/generrs/$f; done


# training_data.columns
generrors_dir = {}
trainaccs_dir = {}
iterss_dir = {}
files = [f for f in os.listdir('results/sgd_vs_bayes/generrs/') if os.path.isfile("results/sgd_vs_bayes/generrs/"+f)]
for filename in files:
    print(filename)
    comp = filename.split("_")[0]
    training_data = pd.read_csv("results/sgd_vs_bayes/generrs/"+filename, sep="\t", comment="#")
    generrors = list(training_data["test_error"])
    trainaccs = list(training_data["train_acc"])
    iterss = list(training_data["mean_iters"])
    generrors_dir[comp] = generrors
    trainaccs_dir[comp] = trainaccs
    iterss_dir[comp] = iterss

import pickle
# pickle.dump(generrors_dir,open("generrors_dir.p","wb"))
#scp generrors_dir.p office:/local/home/valleperez/36ve/ai
#in office: python3 analyze_abi.py
#scp office:/local/home/valleperez/36ve/ai/ABI_vs_SGD_generror_vs_comp.png.html .

import numpy as np
%matplotlib
import matplotlib.pyplot as plt
for comp in sorted([float(x) for x in generrors_dir.keys()]):
    print(comp)
    # plt.hist(generrors_dir[str(comp)],label=str(comp))
    # plt.hist(trainaccs_dir[str(comp)],label=str(comp))
    print(trainaccs_dir[str(comp)])
    # print(iterss_dir[str(comp)])
    # print(len(generrors_dir[str(comp)]))
    print(np.mean(generrors_dir[str(comp)]))


plt.legend()
