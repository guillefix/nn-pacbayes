import pandas as pd
import numpy as np

results_folder = "results/results_confsweep/"
results_folder = "./"
# prefix = "wronglabel1"
prefix = "1"
prefix = "smalldatabigconfusion" # has wrong (opposite) label too.
prefix = "randomlabelconfusion"
prefix = "randomlabelbiggerdata"
prefix = "new_sigmas_"

training_results = pd.read_csv(results_folder+prefix+"nn_training_results.txt",comment="#", header='infer',sep="\t")
bounds = pd.read_csv(results_folder+prefix+"bounds.txt",comment="#", header='infer',sep="\t")

training_results.columns
bounds.columns
# bounds

# training_results["test_error"].unique()
# training_results["m"].unique()
# bounds["bound"]

#%%

import matplotlib.pyplot as plt
%matplotlib

network = "cnn"
network = "fc"
dataset = "mnist"
training_samples = 10000
# training_samples = 500
number_layers = 1
# number_layers = 32
# training_results[(training_results["dataset"]==dataset) & (training_results["network"]==network)]
# training_results
# for dataset in ["mnist"]:#,"mnist-fashion", "cifar"]:
filtered_true_errors = training_results[(training_results["dataset"]==dataset) & (training_results["network"]==network) & (training_results["m"]==training_samples)  & (training_results["number_layers"]==number_layers)]
filtered_bounds = bounds[(bounds["dataset"]==dataset) & (bounds["network"]==network) & (bounds["m"]==training_samples) & (bounds["number_layers"]==number_layers)]
filtered_true_errors = filtered_true_errors.sort_values(by="confusion").groupby("confusion",as_index=False).mean()
filtered_bounds = filtered_bounds.sort_values(by="confusion").groupby("confusion",as_index=False).mean()

plt.plot(filtered_true_errors["confusion"],filtered_true_errors["test_error"], label=dataset+" "+network+" error")
#%%

plt.plot(filtered_bounds["confusion"],filtered_bounds["bound"], label=dataset+" "+network+" Looser PAC-Bayes bound")
# plt.plot(filtered_bounds["confusion"],1-np.exp(-filtered_bounds["bound"]), label=dataset+" "+network+" PAC-Bayes bound")
### BOUND PROCESSING FOR OLDER SCRIPTS:
#%%
c = filtered_bounds["confusion"]
p = c/(1.0+c)
B1 = filtered_bounds["bound"]
m = training_samples
B = B1*(1-p)+0.5*p
# B = ((B)*m-2*np.log(m))/m
Bexp = 1-np.exp(-B)
Bexp2 = (Bexp - 0.5*p)/(1-p) #to correct for the confusion changing the training data distribution!
# B2 = 1-p+(2*p-1)*(B-p**2-(1-p)**2)/(4*p*(1-p**2))
# B3 = 1-p + B*(2*p-1)
# B3 = 1-p + (np.exp(-B))*(2*p-1)
# B4 = (1-np.exp(-B)-p)/(1-2*p)
#this is the correct bound the other one is not completely correct (though they all agree for small values of bound)
plt.plot(filtered_bounds["confusion"],Bexp2, label=dataset+" "+network+" PAC-Bayes bound")
#%%

# plt.plot(filtered_bounds["confusion"],Bexp, label=dataset+" "+network+" PAC-Bayes bound")
plt.plot(filtered_bounds["confusion"],1-np.exp(-(B)), label=dataset+" "+network+" PAC-Bayes bound")
plt.plot(filtered_bounds["confusion"],p, label=dataset+" "+network+" fraction of confusion data")
#lower/upper bound when confusion data has _strictly wrong_ labels (i.e. the opposite of the true label for binary data)
plt.plot(filtered_bounds["confusion"],B4, label=dataset+" "+network+" PAC-Bayes upper/lower bound") # upper bound below p=0.5, upper bound above it
# plt.plot(filtered_bounds["confusion"],B2, label=dataset+" "+network+" PAC-Bayes bound")
# plt.plot(filtered_bounds["confusion"],B3, label=dataset+" "+network+" PAC-Bayes bound")
plt.plot(filtered_bounds["confusion"],filtered_bounds["bound"], label=dataset+" "+network+" Looser PAC-Bayes bound")
# bounds with formula to correct previous typo in the code:
plt.plot(filtered_bounds["confusion"],1-np.exp(-((1-filtered_bounds["confusion"]**2)*filtered_bounds["bound"] + 0.5* filtered_bounds["confusion"]**2)), label=dataset+" "+network+" PAC-Bayes bound")
# plt.plot(filtered_bounds["confusion"],((1-filtered_bounds["confusion"]**2)*filtered_bounds["bound"] + 0.5* filtered_bounds["confusion"]**2), label=dataset+" "+network+" Looser PAC-Bayes bound")



plt.legend()
plt.xlabel("Confusion")
plt.ylabel("Generalization error")
plt.savefig("generror_confusion_"+prefix+network+"_"+dataset+".png")
# filtered_bounds
#
# 1-np.exp(-filtered_bounds["bound"])

###############

network = "cnn"; database = "mnist"

training_results = pd.read_csv("results_compsweep/"+network+"_"+database+"_nn_training_results.txt", header=None,delim_whitespace=True)
bounds = pd.read_csv("results_compsweep/"+network+"_"+database+"_bounds.txt", header=None,delim_whitespace=True)

plt.plot(training_results[6],training_results[8])
# plt.plot(bounds[4],bounds[5])
plt.plot(bounds[4],1-np.exp(-bounds[5]))
