import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results_folder = "results/results_confsweep/"
results_folder = "./"
results_folder = "results/"
# prefix = "wronglabel1"
# prefix = "1"
# prefix = "smalldatabigconfusion" # has wrong (opposite) label too.
# prefix = "randomlabelconfusion"
# prefix = "randomlabelbiggerdata"
# prefix = "new_sigmas_"
prefix = "layer_sweep"
# prefix = "arch_sweep"
prefix = "test_"
prefix = "newer_arch_sweep_"
prefix = "new_pool_sweep"

training_results = pd.read_csv(results_folder+prefix+"nn_training_results.txt",comment="#", header='infer',sep="\t")
bounds = pd.read_csv(results_folder+prefix+"bounds.txt",comment="#", header='infer',sep="\t")

training_results.iloc[-1]
training_results.columns
bounds.columns
# bounds

training_results["dataset"].unique()
training_results["network"].unique()
training_results["number_layers"].unique()
training_results["m"].unique()
training_results["train_acc"].unique()
bounds["sigmaw"].unique()
bounds["sigmab"].unique()
# training_results["optimizer"].unique()

####POOLING######

%matplotlib

training_results[["pooling","test_error"]]
bounds[["pooling","bound"]]

# d = pd.concat([training_results,bounds],axis=1)
d = training_results.merge(bounds,on="pooling")
import seaborn as sns
# sns.barplot(x="pooling",y=["bound","test_error"],data=d)
d=d.sort_values("test_error")
def subcategorybar(xname, valnames, d, labels=None, width=0.8):
    X=d[xname]
    if labels == None: labels=valnames
    vals = list(map(lambda x: d[x],valnames))
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        plt.bar(_X - width/2. + i/float(n)*width, vals[i],
                width=width/float(n), align="edge",label=labels[i])
    plt.xticks(_X, X)
    plt.legend()

subcategorybar("pooling",["bound","test_error"],d, labels=["PAC-Bayes bound","Test error"])
plt.savefig("bound_test_error_vs_poolingtype_CNN_4_layers_KMNIST_m1000_sigmawEP50_sigmabEP0.png")


# training_results["m"].unique()
# bounds["bound"]

training_results.groupby(["network"],as_index=False).mean()[["network", "test_error"]]

#%%
###### CONFUSION  #######

%matplotlib

network = "cnn"
# network = "fc"
dataset = "mnist"
training_samples = 10000
# training_samples = 500
# number_layers = 1
number_layers = 4
# number_layers = 32
# training_results[(training_results["dataset"]==dataset) & (training_results["network"]==network)]
# training_results
# for dataset in ["mnist"]:#,"mnist-fashion", "cifar"]:
filtered_true_errors = training_results[(training_results["dataset"]==dataset) & (training_results["network"]==network) & (training_results["m"]==training_samples)  & (training_results["number_layers"]==number_layers)]
filtered_bounds = bounds[(bounds["dataset"]==dataset) & (bounds["network"]==network) & (bounds["m"]==training_samples) & (bounds["number_layers"]==number_layers)]

filtered_true_errors = filtered_true_errors[filtered_true_errors["confusion"]<=3].sort_values(by="confusion").groupby("confusion",as_index=False).mean()
filtered_bounds = filtered_bounds.sort_values(by="confusion").groupby("confusion",as_index=False).mean()

plt.plot(filtered_true_errors["confusion"],filtered_true_errors["test_error"], label=dataset+" "+network+" error")
#%%

plt.plot(filtered_bounds["confusion"],filtered_bounds["bound"], label=dataset+" "+network+" PAC-Bayes bound")
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

#%%
### LAYERS

%matplotlib

network = "cnn"
dataset = "mnist"
training_samples = 10000
training_results["pooling"].unique()
bounds["pooling"].unique()
training_results["number_layers"].unique()
pooling = "none"
filtered_true_errors = training_results[(training_results["dataset"]==dataset) & (training_results["network"]==network) & (training_results["m"]==training_samples) & (training_results["pooling"]==pooling)]
filtered_bounds = bounds[(bounds["dataset"]==dataset) & (bounds["network"]==network) & (bounds["m"]==training_samples) & (bounds["pooling"]==pooling)]
filtered_true_errors = filtered_true_errors.sort_values(by="number_layers").groupby("number_layers",as_index=False).mean()
filtered_bounds = filtered_bounds.sort_values(by="number_layers").groupby("number_layers",as_index=False).mean()
filtered_true_errors["train_acc"]
plt.plot(filtered_true_errors["number_layers"],filtered_true_errors["test_error"], label=dataset+" "+network+" error")
plt.plot(filtered_bounds["number_layers"],filtered_bounds["bound"], label=dataset+" "+network+" PAC-Bayes bound")
plt.xlim([1,9])
plt.xlabel("Number of layers")
plt.ylabel("Generalization error")
plt.legend()
plt.savefig("generror_bound_vs_layers_cnn_max_mnist.png")

#%%

### ARCHS
%matplotlib

training_results["network"].unique()
bounds["network"].unique()
bounds["sigmab"]
bounds["sigmaw"]
training_results["value"] = "True error"
bounds["value"] = "Bound"
bounds["test_error"] = bounds["bound"]
combined_data = pd.concat([training_results,bounds])

plt.bar(bounds["network"].unique(),bounds["bound"])
import seaborn as sns
g = sns.catplot(x="network", y="test_error", hue="value", data=combined_data,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Generalization error")


###############

network = "cnn"; database = "mnist"

training_results = pd.read_csv("results_compsweep/"+network+"_"+database+"_nn_training_results.txt", header=None,delim_whitespace=True)
bounds = pd.read_csv("results_compsweep/"+network+"_"+database+"_bounds.txt", header=None,delim_whitespace=True)

plt.plot(training_results[6],training_results[8])
# plt.plot(bounds[4],bounds[5])
plt.plot(bounds[4],1-np.exp(-bounds[5]))
