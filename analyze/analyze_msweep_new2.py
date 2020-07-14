import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib

#%%

#main NNGP big run (up to 4k training set size:)
training_data = pd.read_csv("results/new_mother_of_all_msweeps_nn_training_results.txt", sep="\t", comment="#")
# training_data[training_data["batch_size"]==32]["m"].tolist()
training_data[(training_data["batch_size"] == 32) & (training_data["dataset"] == "KMNIST") & (training_data["network"] == "fc")][["m","train_acc","test_error"]]
#removing the old batch 32 data which was done with a different learning rate or something
training_data = training_data[~(training_data["batch_size"] == 32)]


training_data2 = pd.read_csv("results/newer_mother_of_all_msweeps_nn_training_results.txt", sep="\t", comment="#")
training_data2[training_data2["batch_size"] == 32][["m","dataset","train_acc","test_error","network"]]
# training_data = training_data[~((training_data["network"] == "fc") & (training_data["dataset"] == "cifar"))]
training_data = training_data.append(training_data2)

#this one has the batch 32 data for sizes >= 15k
training_data3 = pd.read_csv("results/grandmother_of_all_msweeps_nn_training_results.txt", sep="\t", comment="#")
training_data3[(training_data3["batch_size"] == 32) & (training_data3["dataset"] == "KMNIST") & (training_data3["network"] == "fc")][["m","train_acc","test_error"]]
training_data3[(training_data3["batch_size"] == 32) & (training_data3["train_acc"]==1) & (training_data3["network"] == "resnet50")][["m","dataset","train_acc","test_error"]].head(40)
training_data = training_data.append(training_data3)

#this one has the batch 32 data for sizes < 15k
training_data4 = pd.read_csv("results/new_grandmother_of_all_msweeps_nn_training_results.txt", sep="\t", comment="#")
# training_data4[training_data4["batch_size"] == 32]["m"].unique()
training_data4[(training_data4["batch_size"] == 32) & (training_data4["dataset"] == "KMNIST") & (training_data4["network"] == "fc")][["m","train_acc","test_error"]]
training_data = training_data.append(training_data4)
# training_data = training_data[~((training_data["network"] == "fc") & (training_data["dataset"] == "cifar"))]
# training_data = training_data[~((training_data["network"] == "fc") & (training_data["dataset"] == "EMNIST"))]
# training_data2 = pd.read_csv("results/newer_mother_of_all_msweeps_nn_training_results.txt", sep="\t", comment="#")
# training_data = training_data.append(training_data2)

# bounds = pd.read_csv("results/new_mother_of_all_msweeps_bounds10000.txt", sep="\t", comment="#")
bounds = pd.read_csv("results/newer_mother_of_all_msweeps_bounds.txt", sep="\t", comment="#")
bounds2 = pd.read_csv("results/grandmother_of_all_msweeps_bounds.txt", sep="\t", comment="#")
bounds = bounds.append(bounds2)
# bounds = pd.read_csv("results/new_mother_of_all_msweeps_bounds.txt", sep="\t", comment="#")

##resnet50 training results:
# training_data = pd.read_csv("results/2jade_msweep_nn_training_results.txt", sep="\t", comment="#")
# training_data = pd.read_csv("results/3gpu_msweep_nn_training_results.txt", sep="\t", comment="#")

#Other stuff:
# more data on fc/cnn
# training_data = pd.read_csv("results/gpu_msweep_nn_training_results.txt", sep="\t", comment="#")
# training_data = pd.read_csv("results/new_small_b_3jade_new_msweep_nn_training_results.txt", sep="\t", comment="#")
# training_data[training_data["network"]=="fc"]["m"]
# bounds[bounds["network"]=="fc"]["dataset"]
# training_data = pd.read_csv("results/2gpu_msweep_nn_training_results.txt", sep="\t", comment="#")
# bounds = pd.read_csv("results/gpu_msweep_bounds.txt", sep="\t", comment="#")
# bounds = pd.read_csv("results/2gpu_msweep_bounds.txt", sep="\t", comment="#")
# # bounds = pd.read_csv("results/2jade_new_msweep_bounds.txt", sep="\t", comment="#")

# bounds2 = pd.read_csv("results/new_small_b_3jade_new_msweep_bounds.txt", sep="\t", comment="#")
# bounds = bounds.append(bounds2)

# training_data = pd.read_csv("results/2gpu_msweep_nn_training_results.txt", sep="\t", comment="#")
# training_data = pd.read_csv("results/gpu_msweep_nn_training_results_cnn.txt", sep="\t", comment="#")
# training_data = pd.read_csv("results/2jade_new_msweep_nn_training_results.txt", sep="\t", comment="#")
# bounds = pd.read_csv("results/new_mother_of_all_msweeps_bounds.txt", sep="\t", comment="#")
# bounds = pd.read_csv("results/new_mother_of_all_msweeps_bounds100.txt", sep="\t", comment="#")
# bounds = pd.read_csv("results/new_mother_of_all_msweeps_bounds50.txt", sep="\t", comment="#")

# bounds = pd.read_csv("results/2new_mother_of_all_msweeps_bounds.txt", sep="\t", comment="#")


# bounds = pd.read_csv("results/gpu_msweep_bounds_cnn.txt", sep="\t", comment="#")
#missing nasnet

training_data = training_data.append(pd.Series({"m":40000, "dataset":"mnist","network":"vgg16","pooling":"avg","test_error":1-0.9884999990463257,"train_acc":1.0,"batch_size":200}), ignore_index=True)
training_data = training_data.append(pd.Series({"m":40000, "dataset":"mnist","network":"vgg19","pooling":"avg","test_error":1-0.9898999929428101,"train_acc":1.0,"batch_size":200}), ignore_index=True)
training_data = training_data.append(pd.Series({"m":40000, "dataset":"mnist-fashion","network":"vgg19","pooling":"avg","test_error":1-0.9465999603271484,"train_acc":1.0,"batch_size":200}), ignore_index=True)
training_data = training_data.append(pd.Series({"m":40000, "dataset":"mnist-fashion","network":"vgg16","pooling":"avg","test_error":1-0.9443999528884888,"train_acc":1.0,"batch_size":200}), ignore_index=True)
training_data = training_data.append(pd.Series({"m":40000, "dataset":"mnist-fashion","network":"cnn","pooling":"avg","test_error":1-0.9502,"train_acc":1.0,"batch_size":200}), ignore_index=True)
training_data = training_data.append(pd.Series({"m":40000, "dataset":"mnist-fashion","network":"cnn","pooling":"None","test_error":1-0.9473,"train_acc":1.0,"batch_size":200}), ignore_index=True)
training_data = training_data.append(pd.Series({"m":40000, "dataset":"mnist-fashion","network":"cnn","pooling":"max","test_error":1-0.9484,"train_acc":1.0,"batch_size":200}), ignore_index=True)
training_data = training_data.append(pd.Series({"m":40000, "dataset":"cifar","network":"cnn","pooling":"max","test_error":1-0.7894,"train_acc":1.0,"batch_size":200}), ignore_index=True)
training_data = training_data.append(pd.Series({"m":40000, "dataset":"cifar","network":"cnn","pooling":"None","test_error":1-0.7635,"train_acc":1.0,"batch_size":200}), ignore_index=True)
training_data = training_data.append(pd.Series({"m":40000, "dataset":"cifar","network":"cnn","pooling":"avg","test_error":1-0.8392,"train_acc":1.0,"batch_size":200}), ignore_index=True)
training_data = training_data.append(pd.Series({"m":40000, "dataset":"cifar","network":"vgg16","pooling":"avg","test_error":1-0.7134,"train_acc":1.0,"batch_size":200}), ignore_index=True)
training_data = training_data.append(pd.Series({"m":40000, "dataset":"cifar","network":"vgg19","pooling":"avg","test_error":1-0.7816,"train_acc":1.0,"batch_size":200}), ignore_index=True)
training_data = training_data.append(pd.Series({"m":40000, "dataset":"mnist","network":"cnn","pooling":"None","test_error":1-0.9874,"train_acc":1.0,"batch_size":200}), ignore_index=True)
training_data = training_data.append(pd.Series({"m":40000, "dataset":"mnist","network":"cnn","pooling":"max","test_error":1-0.9909,"train_acc":1.0,"batch_size":200}), ignore_index=True)
training_data = training_data.append(pd.Series({"m":40000, "dataset":"mnist","network":"cnn","pooling":"avg","test_error":1-0.9918,"train_acc":1.0,"batch_size":200}), ignore_index=True)

ms=[1, 3, 11, 36, 122, 407, 1357, 4516, 15026]
for m,acc in zip(ms,[0.5139, 0.5139, 0.5585, 0.6271, 0.6563, 0.9088, 0.9235, 0.9679, 0.9824]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"mnist","network":"vgg19","pooling":"avg","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)

for m,acc in zip(ms,[0.5000, 0.5128, 0.5163, 0.5212, 0.5448, 0.5681, 0.5693, 0.6329, 0.6654]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"cifar","network":"vgg16","pooling":"avg","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)
for m,acc in zip(ms,[0.5000, 0.5148, 0.5086, 0.5141, 0.5391, 0.5463, 0.5816, 0.6835, 0.7116]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"cifar","network":"vgg19","pooling":"avg","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)

for m,acc in zip(ms,[0.5000, 0.4291, 0.7743, 0.8618, 0.8595, 0.8919, 0.9031, 0.9221, 0.9302]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"mnist-fashion","network":"vgg16","pooling":"avg","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)
for m,acc in zip(ms,[0.5000, 0.4879, 0.6764, 0.8620, 0.8561, 0.8831, 0.9100, 0.9205, 0.9294]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"mnist-fashion","network":"vgg19","pooling":"avg","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)


ms=[122, 407, 1357, 4516, 15026]
for m,acc in zip(ms,[0.7039, 0.9153, 0.9469, 0.9717, 0.9786]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"mnist","network":"vgg16","pooling":"avg","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)
for m,acc in zip(ms,[0.7848, 0.8971, 0.9468, 0.9693, 0.9788]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"mnist","network":"cnn","pooling":"None","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)
for m,acc in zip([407, 1357, 4516, 15026],[0.7914, 0.8980, 0.9476, 0.9683]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"mnist","network":"cnn","pooling":"avg","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)


ms=[36, 122, 407, 1357, 4516, 15026]
for m,acc in zip(ms,[0.5417, 0.6163, 0.7703, 0.8402, 0.8848, 0.9177]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"KMNIST","network":"vgg16","pooling":"avg","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)
for m,acc in zip(ms,[0.5418, 0.5942, 0.7593, 0.8345, 0.8906, 0.9230]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"KMNIST","network":"vgg16","pooling":"avg","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)


ms=[122, 407, 1357, 4516, 15026]
for m,acc in zip(ms,[0.8660, 0.8952, 0.9001, 0.9191, 0.9310]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"mnist-fashion","network":"cnn","pooling":"None","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)
for m,acc in zip(ms,[0.8505, 0.8766, 0.9071, 0.9241, 0.9360]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"mnist-fashion","network":"cnn","pooling":"max","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)
for m,acc in zip(ms,[0.8380, 0.8747, 0.8897, 0.9265, 0.9358]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"mnist-fashion","network":"cnn","pooling":"avg","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)

ms=[122, 407, 1357, 4516, 15026]
for m,acc in zip(ms,[0.7349, 0.7949, 0.8151, 0.8477, 0.8703]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"EMNIST","network":"vgg16","pooling":"avg","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)
for m,acc in zip(ms,[0.7361, 0.7786, 0.8088, 0.8592, 0.8694]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"EMNIST","network":"vgg19","pooling":"avg","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)
for m,acc in zip(ms,[0.7179, 0.7774, 0.7962, 0.8212, 0.8476 ]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"EMNIST","network":"cnn","pooling":"None","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)
for m,acc in zip([15026],[0.8744]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"EMNIST","network":"cnn","pooling":"max","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)
for m,acc in zip([407, 1357, 4516],[0.7930, 0.8303, 0.8582]):
    training_data = training_data.append(pd.Series({"m":m, "dataset":"EMNIST","network":"cnn","pooling":"avg","test_error":1-acc,"train_acc":1.0,"batch_size":256}), ignore_index=True)


### training_data = training_data.append(pd.Series({"m":40000, "dataset":"EMNIST","network":"cnn","pooling":"None","test_error":1-0.9032,"train_acc":1.0,"batch_size":200}), ignore_index=True)
### training_data = training_data.append(pd.Series({"m":40000, "dataset":"EMNIST","network":"cnn","pooling":"max","test_error":1-0.9228,"train_acc":1.0,"batch_size":200}), ignore_index=True)

bounds["network"].unique()
training_data["network"].unique()
nets = training_data["network"].unique()
datasets = training_data["dataset"].unique()

# batch_size=256
batch_size=32

#sort by m
training_data = training_data.sort_values("m")
bounds = bounds.sort_values("m")
bounds = bounds[bounds["m"]>=50]
training_data = training_data[training_data["m"]>=50]
# training_data = training_data[training_data["batch_size"]==32]
# training_data = training_data[training_data["batch_size"]>32]
# training_data = training_data[training_data["batch_size"]>=200]
training_data = training_data[training_data["batch_size"]==batch_size]
training_data = training_data[training_data["train_acc"]==1.0]
# bounds = bounds[bounds["kernel_mult"]==10000]
# bounds = bounds[bounds["kern_mult"]==10000]
# bounds = bounds.groupby(["m","network","dataset","pooling"],as_index=False).mean()
# training_data = training_data.groupby(["m","network","dataset","pooling"],as_index=False).mean()
bounds = bounds.groupby(["m","network","dataset","pooling"],as_index=False).min()
training_data = training_data.groupby(["m","network","dataset","pooling"],as_index=False).mean()
# bounds = bounds[(bounds["sigmaw"]==10.0) & (bounds["sigmab"]==10.0)]


# training_data[(training_data["network"]=="fc") & (training_data["dataset"]=="EMNIST") & (training_data["pooling"]=="None")][["m","test_error"]]

# nets

# training_data[(training_data["network"]==net)]

#PLOT FOR NETS SWEEP
#%%
datasets=["mnist","mnist-fashion","EMNIST","KMNIST","cifar"]

from pylab import rcParams
rcParams['figure.figsize'] = 9,6
fig, axx = plt.subplots(nrows=2, ncols=3)

net="cnn"
pools=["avg"]
net="fc"
# net="resnet50"
# net="resnet101"
# net="resnet152"
# net="resnetv2_50"
# net="resnetv2_101"
# net="resnetv2_152"
# net="resnext50"
# net="resnext101"
# net="densenet121"
# net="densenet169"
# net="densenet201"
# net="mobilenetv2"
# net="nasnet"
# net="vgg16"
# net="vgg19"

dataset="mnist"
# dataset="mnist-fashion"
# dataset="EMNIST"
# dataset="KMNIST"
dataset="cifar"
# colors = np.random.rand(len(nets),3)
import matplotlib
# cmap = matplotlib.cm.get_cmap('rainbow')
cmap = matplotlib.cm.get_cmap('tab20')


# j=0
sweep="nets"
# sweep="datasets"
if sweep=="nets":
    things1 = datasets
    # things2 = nets
    # things2 = ["fc"]
    # things2 = ["fc","cnn"]
    # things2 = ["resnet50","resnet101","resnet152"]
    # things2 = ["resnetv2_50","resnetv2_101","resnetv2_152"]
    # things2 = ["resnet50","resnet101","resnet152","resnetv2_50","resnetv2_101","resnetv2_152","resnext50","resnext101"]
    # things2 = ["densenet121","densenet169","densenet201"]
    # things2 = ["mobilenetv2"]
    # things2 = ["vgg16","vgg19"]
    things2 = ["cnn"]
    # things2 = ["fc","resnet50","densenet121","mobilenetv2"]
    # things2 = ["fc","resnet50","densenet121","mobilenetv2"]
    # things2 = ["fc","cnn","resnet50","densenet121","mobilenetv2"]
else:
    things1 = nets
    things2 = datasets

# fig.delaxes(axx[2][1])
# fig.delaxes(axx[0][2])
# fig.delaxes(axx[1][2])
# fig.delaxes(axx[2][2])
fig.delaxes(axx[1][2])
for nya,thing1 in enumerate(things1):
# for thing1 in ["fc"]:
    plotto=axx[nya//3][nya%3]
    ii=0
    try:
        if sweep=="datasets":
            if thing1=="cnn":
                pools=["None","avg","max"]
            if net=="fc":
                pools=["None"]
            elif net!="cnn":
                pools=["avg"]
        else:
            pools=["hi"]
        for pool1 in pools:
            for i, thing2 in enumerate(things2):
                if sweep=="nets":
                    dataset=thing1
                    net=thing2
                else:
                    net=thing1
                    dataset=thing2

            # for i,dataset in enumerate(datasets):
            # for ii in [1]:
                # if i!=j: break
                # plt.close()
                # pool="None"
                # pool="avg"
                # pool="max"
                if sweep=="nets":
                    if net=="cnn":
                        pools=["None","avg","max"]
                        # pools=["None"]
                else:
                    pools=[pool1]
                if net=="fc":
                    pools=["None"]
                elif net!="cnn":
                    pools=["avg"]

                for pool in pools:
                    # if net!="cnn":
                    #     continue
                    # if pool!="avg":
                    #     continue
                    # if dataset not in ["KMNIST"]:
                    #     continue
                    # if dataset not in ["cifar","mnist","EMNIST"]:
                    #     continue
                    bdata=bounds[(bounds["network"]==net) & (bounds["dataset"]==dataset) & (bounds["pooling"]==pool)]
                    # bdata=bounds[(bounds["network"]==net) & (bounds["dataset"]==dataset)]
                    tdata=training_data[(training_data["network"]==net) & (training_data["dataset"]==dataset) & (training_data["pooling"]==pool)]
                    # bounds.dtypes
                    # print(net, len(bdata))
                    print(net, dataset)
                    print(net, bdata[["m","bound"]])
                    # print(net, len(tdata))
                    print(net,tdata[["m","train_acc"]])
                    print(net,tdata[["m","test_error"]])

                    # bounds["bound"] = pd.to_numeric(bounds["bound"])
                    tdata.columns
                    bdata.columns

                    color = cmap(ii/(len(things2)+1))
                    if sweep=="nets":
                        color = cmap(ii/(len(things2)+3))
                    # color = cmap(i/(len(datasets)))
                    if net != "cnn":
                        plotto.plot(bdata["m"], bdata["bound"], c=color, label="PAC-Bayes bound "+net+" "+dataset)
                        # plt.plot(bdata["m"], -bdata["logP"]/bdata["m"], c=color, label="PAC-Bayes bound "+net+" "+dataset+" "+pool)
                        # plt.plot(bdata["m"], -bdata["logP"]/bdata["m"], c=color)
                        plotto.plot(tdata["m"], tdata["test_error"], "--", c=color, label="test error "+net+" "+dataset)
                        # plt.plot(tdata["m"], tdata["train_acc"], label="Training error "+net+" "+dataset)
                    else:
                        plotto.plot(bdata["m"], bdata["bound"], c=color, label="PAC-Bayes bound "+net+" "+dataset+" "+pool)
                        # plt.plot(bdata["m"], -bdata["logP"]/bdata["m"], c=color, label="PAC-Bayes bound "+net+" "+dataset+" "+pool)
                        # plt.plot(bdata["m"], -bdata["logP"]/bdata["m"], c=color)
                        plotto.plot(tdata["m"], tdata["test_error"], "--", c=color, label="test error "+net+" "+dataset+" "+pool)
                        # plt.plot(tdata["m"], tdata["train_acc"], label="Training error "+net+" "+dataset+" "+pool)
                    ii+=1
                    plotto.set_yscale("log")
                    if nya in [2,3,4]:
                        plotto.set_xlabel("m", fontsize=12)
                    plotto.set_xscale("log")
                    if nya%3==0:
                        plotto.set_ylabel("generalization error", fontsize=12)



            # ax = plotto.gca()
            ax = plotto
            box = ax.get_position()
            ax.set_title(dataset)
            ax.set_position([box.x0, box.y0*1.1, box.width * 0.95, box.height])

            # Put a legend to the right of the current axis
            if nya==4:
                plotto.legend()
                # ax.legend(loc='center left', bbox_to_anchor=(1.1, -0.5), prop={'size': 8})
                # ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), prop={'size': 8})
                ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.4), prop={'size': 8})

            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(10)
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small')
                # tick.label.set_rotation('vertical')
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(10)
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small')
                # tick.label.set_rotation('vertical')

            if sweep=="nets":
                min_error = np.min(training_data[training_data["dataset"]==dataset]["test_error"])
                # min_error = np.min(training_data["test_error"])
            else:
                min_error = np.min(training_data["test_error"])
            # min_error = np.min(training_data["test_error"])

            plotto.set_ylim([min_error*0.8,1.1e0])
            plotto.set_xlim([1e2,50000])

            # if sweep=="datasets":
            #     plt.savefig("img/msweep/nets/learning_curve_sweep_"+sweep+"_"+net+"_"+pool+"_"+str(batch_size)+".png")
            #     # plt.savefig("img/msweep/learning_curve_sweep_"+sweep+"_"+net+"_"+pool+"_"+str(batch_size)+"+200.png")
            # else:
            #     plt.savefig("img/msweep/datasets/learning_curve_sweep_"+sweep+"_"+dataset+"_"+str(batch_size)+".png")
            #     # plt.savefig("img/msweep/learning_curve_sweep_"+sweep+"_"+dataset+"_"+str(batch_size)+"+200.png")
            #
            # plt.close()
    except Exception as e:
        print(e)
        plt.close()
        continue

# plt.savefig("learning_curve_sweep_nets_"+str(batch_size)+".png")
# plt.savefig("learning_curve_sweep_resnets_"+str(batch_size)+".png")
# plt.savefig("learning_curve_sweep_densenets_"+str(batch_size)+".png")
# plt.savefig("learning_curve_sweep_somenets_"+str(batch_size)+".png")
# plt.savefig("learning_curve_sweep_vggs_"+str(batch_size)+".png")
# plt.savefig("learning_curve_sweep_cnns_"+str(batch_size)+".png")
# plt.savefig("learning_curve_sweep_mobilenet_"+str(batch_size)+".png")

# plt.savefig("learning_curve_fc_dataset_all_1_41_0.png")
# plt.savefig("learning_curve_fc_dataset_selection_1_41_0.png")
# plt.savefig("learning_curve_resnet50_dataset_selection_1_41_0.png")
# plt.savefig("learning_curve_resnet50_v2_dataset_selection_1_41_0.png")
# plt.savefig("learning_curve_fc_mnist_km10k_1_41_0.png")
# plt.savefig("learning_curve_resnet50_max_mnist_second_training_set_sample__1_41_0.png")
# plt.savefig("learning_curve_resnet50_max_mnist_combined_training_set_samples__1_41_0.png")
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
#
# MPI.COUNT

#for repeated entries keep last one in both training_data and bounds
#for data from jade, because the non-last one could have had problems with two jobs trying to compute the same thing and the later overriding data/kernel of the former
# %% # dataset sweeps for all architectures


from pylab import rcParams
rcParams['figure.figsize'] = 9,9
# fig, axx = plt.subplots(nrows=6, ncols=3)
fig, axx = plt.subplots(nrows=3, ncols=3)

fig.subplots_adjust(top=0.9)

# colors = np.random.rand(len(nets),3)
import matplotlib
cmap = matplotlib.cm.get_cmap('rainbow')
# j=0
# sweep="nets"
# nets
nets=["fc","cnn","resnet50","resnet101","resnet152","resnetv2_50","resnetv2_101","resnetv2_152","resnext50","resnext101","densenet121","densenet169","densenet201","mobilenetv2","vgg16","vgg19","nasnet"]


sweep="datasets"
if sweep=="nets":
    things1 = datasets
    things2 = nets
else:
    things1 = nets
    things2 = datasets

# fig.delaxes(axx[5][2])
# fig.delaxes(axx[5][1])
fig.delaxes(axx[2][2])

half=0
# half=1
# for nya,thing1 in enumerate(things1):
if half==0:
    things1=things1[:8]
elif half==1:
    things1=things1[8:]
for nya,thing1 in enumerate(things1):
# for thing1 in ["fc"]:
    plotto=axx[nya//3][nya%3]
    ii=0
    try:
        if sweep=="datasets":
            if thing1=="cnn":
                pools=["None","avg","max"]
            if net=="fc":
                pools=["None"]
            elif net!="cnn":
                pools=["avg"]
        else:
            pools=["hi"]
        for pool1 in pools:
            for i, thing2 in enumerate(things2):
                if sweep=="nets":
                    dataset=thing1
                    net=thing2
                else:
                    net=thing1
                    dataset=thing2

            # for i,dataset in enumerate(datasets):
            # for ii in [1]:
                # if i!=j: break
                # plt.close()
                # pool="None"
                # pool="avg"
                # pool="max"
                if sweep=="nets":
                    if net=="cnn":
                        pools=["None","avg","max"]
                else:
                    pools=[pool1]
                if net=="fc":
                    pools=["None"]
                elif net!="cnn":
                    pools=["avg"]

                for pool in pools:
                    # if net!="cnn":
                    #     continue
                    # if pool!="avg":
                    #     continue
                    # if dataset not in ["KMNIST"]:
                    #     continue
                    # if dataset not in ["cifar","mnist","EMNIST"]:
                    #     continue
                    bdata=bounds[(bounds["network"]==net) & (bounds["dataset"]==dataset) & (bounds["pooling"]==pool)]
                    # bdata=bounds[(bounds["network"]==net) & (bounds["dataset"]==dataset)]
                    tdata=training_data[(training_data["network"]==net) & (training_data["dataset"]==dataset) & (training_data["pooling"]==pool)]
                    # bounds.dtypes
                    # print(net, len(bdata))
                    print(net, dataset)
                    print(net, bdata[["m","bound"]])
                    # print(net, len(tdata))
                    print(net,tdata[["m","train_acc"]])
                    print(net,tdata[["m","test_error"]])

                    # bounds["bound"] = pd.to_numeric(bounds["bound"])
                    tdata.columns
                    bdata.columns

                    color = cmap(ii/(len(things2)+1))
                    if sweep=="nets":
                        color = cmap(ii/(len(things2)+3))
                    # color = cmap(i/(len(datasets)))
                    if net=="fc":
                        plotto.plot(bdata["m"], bdata["bound"], c=color, label="bound"+net+" "+dataset)
                        # plt.plot(bdata["m"], -bdata["logP"]/bdata["m"], c=color, label="PAC-Bayes bound "+net+" "+dataset+" "+pool)
                        # plt.plot(bdata["m"], -bdata["logP"]/bdata["m"], c=color)
                        plotto.plot(tdata["m"], tdata["test_error"], "--", c=color, label="error"+net+" "+dataset)
                        # plt.plot(tdata["m"], tdata["train_acc"], label="Training error "+net+" "+dataset)
                    else:
                        plotto.plot(bdata["m"], bdata["bound"], c=color, label="bound "+net+" "+dataset+" "+pool)
                        # plt.plot(bdata["m"], -bdata["logP"]/bdata["m"], c=color, label="PAC-Bayes bound "+net+" "+dataset+" "+pool)
                        # plt.plot(bdata["m"], -bdata["logP"]/bdata["m"], c=color)
                        plotto.plot(tdata["m"], tdata["test_error"], "--", c=color, label="error "+net+" "+dataset+" "+pool)
                        # plt.plot(tdata["m"], tdata["train_acc"], label="Training error "+net+" "+dataset+" "+pool)
                    ii+=1
                    plotto.set_yscale("log")
                    # if nya==15 or nya==14 or nya==13:
                    if nya==7 or nya==6 or nya==5:
                        plotto.set_xlabel("m", fontsize=12)
                    plotto.set_xscale("log")
                    if nya%3==0:
                        plotto.set_ylabel("generalization error", fontsize=12)



            # ax = plotto.gca()
            ax = plotto
            box = ax.get_position()
            if net!="cnn":
                ax.set_title(net)
            else:
                ax.set_title(net+" "+pool)
            ax.set_position([box.x0, box.y0*1.05, box.width * 0.95, box.height])

            # Put a legend to the right of the current axis
            if nya==7:
            # if nya==15:
                plotto.legend()
                ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), prop={'size': 8})

            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(10)
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small')
                # tick.label.set_rotation('vertical')
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(10)
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small')
                # tick.label.set_rotation('vertical')

            if sweep=="nets":
                # min_error = np.min(training_data[training_data["dataset"]==dataset]["test_error"])
                min_error = np.min(training_data["test_error"])
            else:
                min_error = np.min(training_data["test_error"])
            # min_error = np.min(training_data["test_error"])

            plotto.set_ylim([min_error*0.8,1.1e0])
            plotto.set_xlim([1e2,50000])

            # if sweep=="datasets":
            #     plt.savefig("img/msweep/nets/learning_curve_sweep_"+sweep+"_"+net+"_"+pool+"_"+str(batch_size)+".png")
            #     # plt.savefig("img/msweep/learning_curve_sweep_"+sweep+"_"+net+"_"+pool+"_"+str(batch_size)+"+200.png")
            # else:
            #     plt.savefig("img/msweep/datasets/learning_curve_sweep_"+sweep+"_"+dataset+"_"+str(batch_size)+".png")
            #     # plt.savefig("img/msweep/learning_curve_sweep_"+sweep+"_"+dataset+"_"+str(batch_size)+"+200.png")
            #
            # plt.close()
    except Exception as e:
        print(e)
        plt.close()
        continue

plt.savefig("learning_curve_sweep_datasets_"+str(batch_size)+"_"+str(half)+".png")

#%% # dataset sweeps for some architectures

from pylab import rcParams
rcParams['figure.figsize'] = 9,9
# fig, axx = plt.subplots(nrows=6, ncols=3)
# fig, axx = plt.subplots(nrows=2, ncols=3)
fig, axx = plt.subplots(nrows=2, ncols=2)

fig.subplots_adjust(top=0.9)

# colors = np.random.rand(len(nets),3)
import matplotlib
cmap = matplotlib.cm.get_cmap('rainbow')
# j=0
# sweep="nets"
# nets
nets=["fc","resnet50","densenet121"]


sweep="datasets"
if sweep=="nets":
    things1 = datasets
    things2 = nets
else:
    things1 = nets
    things2 = datasets

# fig.delaxes(axx[5][2])
# fig.delaxes(axx[5][1])
fig.delaxes(axx[1][1])
# fig.delaxes(axx[0][2])
# fig.delaxes(axx[1][2])
for nya,thing1 in enumerate(things1):
# for thing1 in ["fc"]:
    plotto=axx[nya//2][nya%2]
    ii=0
    try:
        if sweep=="datasets":
            if thing1=="cnn":
                pools=["None","avg","max"]
            if net=="fc":
                pools=["None"]
            elif net!="cnn":
                pools=["avg"]
        else:
            pools=["hi"]
        for pool1 in pools:
            for i, thing2 in enumerate(things2):
                if sweep=="nets":
                    dataset=thing1
                    net=thing2
                else:
                    net=thing1
                    dataset=thing2

            # for i,dataset in enumerate(datasets):
            # for ii in [1]:
                # if i!=j: break
                # plt.close()
                # pool="None"
                # pool="avg"
                # pool="max"
                if sweep=="nets":
                    if net=="cnn":
                        pools=["None","avg","max"]
                else:
                    pools=[pool1]
                if net=="fc":
                    pools=["None"]
                elif net!="cnn":
                    pools=["avg"]

                for pool in pools:
                    # if net!="cnn":
                    #     continue
                    # if pool!="avg":
                    #     continue
                    # if dataset not in ["KMNIST"]:
                    #     continue
                    # if dataset not in ["cifar","mnist","EMNIST"]:
                    #     continue
                    bdata=bounds[(bounds["network"]==net) & (bounds["dataset"]==dataset) & (bounds["pooling"]==pool)]
                    # bdata=bounds[(bounds["network"]==net) & (bounds["dataset"]==dataset)]
                    tdata=training_data[(training_data["network"]==net) & (training_data["dataset"]==dataset) & (training_data["pooling"]==pool)]
                    # bounds.dtypes
                    # print(net, len(bdata))
                    print(net, dataset)
                    print(net, bdata[["m","bound"]])
                    # print(net, len(tdata))
                    print(net,tdata[["m","train_acc"]])
                    print(net,tdata[["m","test_error"]])

                    # bounds["bound"] = pd.to_numeric(bounds["bound"])
                    tdata.columns
                    bdata.columns

                    color = cmap(ii/(len(things2)+1))
                    if sweep=="nets":
                        color = cmap(ii/(len(things2)+3))
                    # color = cmap(i/(len(datasets)))
                    if net=="fc":
                        plotto.plot(bdata["m"], bdata["bound"], c=color, label="bound"+net+" "+dataset)
                        # plt.plot(bdata["m"], -bdata["logP"]/bdata["m"], c=color, label="PAC-Bayes bound "+net+" "+dataset+" "+pool)
                        # plt.plot(bdata["m"], -bdata["logP"]/bdata["m"], c=color)
                        plotto.plot(tdata["m"], tdata["test_error"], "--", c=color, label="error"+net+" "+dataset)
                        # plt.plot(tdata["m"], tdata["train_acc"], label="Training error "+net+" "+dataset)
                    else:
                        plotto.plot(bdata["m"], bdata["bound"], c=color, label="bound "+net+" "+dataset+" "+pool)
                        # plt.plot(bdata["m"], -bdata["logP"]/bdata["m"], c=color, label="PAC-Bayes bound "+net+" "+dataset+" "+pool)
                        # plt.plot(bdata["m"], -bdata["logP"]/bdata["m"], c=color)
                        plotto.plot(tdata["m"], tdata["test_error"], "--", c=color, label="error "+net+" "+dataset+" "+pool)
                        # plt.plot(tdata["m"], tdata["train_acc"], label="Training error "+net+" "+dataset+" "+pool)
                    ii+=1
                    plotto.set_yscale("log")
                    # if nya==15 or nya==14 or nya==13:
                    if nya==2 or nya==1:
                        plotto.set_xlabel("m", fontsize=12)
                    plotto.set_xscale("log")
                    if nya%2==0:
                        plotto.set_ylabel("generalization error", fontsize=12)



            # ax = plotto.gca()
            ax = plotto
            box = ax.get_position()
            if net!="cnn":
                ax.set_title(net)
            else:
                ax.set_title(net+" "+pool)
            ax.set_position([box.x0, box.y0*1.05, box.width * 0.95, box.height])

            # Put a legend to the right of the current axis
            if nya==2:
            # if nya==15:
                plotto.legend()
                ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), prop={'size': 10})

            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(10)
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small')
                # tick.label.set_rotation('vertical')
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(10)
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small')
                # tick.label.set_rotation('vertical')

            if sweep=="nets":
                # min_error = np.min(training_data[training_data["dataset"]==dataset]["test_error"])
                min_error = np.min(training_data["test_error"])
            else:
                min_error = np.min(training_data["test_error"])
            # min_error = np.min(training_data["test_error"])

            plotto.set_ylim([min_error*0.8,1.1e0])
            plotto.set_xlim([1e2,50000])

            # if sweep=="datasets":
            #     plt.savefig("img/msweep/nets/learning_curve_sweep_"+sweep+"_"+net+"_"+pool+"_"+str(batch_size)+".png")
            #     # plt.savefig("img/msweep/learning_curve_sweep_"+sweep+"_"+net+"_"+pool+"_"+str(batch_size)+"+200.png")
            # else:
            #     plt.savefig("img/msweep/datasets/learning_curve_sweep_"+sweep+"_"+dataset+"_"+str(batch_size)+".png")
            #     # plt.savefig("img/msweep/learning_curve_sweep_"+sweep+"_"+dataset+"_"+str(batch_size)+"+200.png")
            #
            # plt.close()
    except Exception as e:
        print(e)
        plt.close()
        continue

plt.savefig("learning_curve_sweep_datasets_"+str(batch_size)+"_main.png")

# dataset="mnist-fashion"
# bounds[(bounds["network"]==net) & (bounds["dataset"]==dataset) & (bounds["pooling"]==pool)]
# training_data[(training_data["network"]==net) & (training_data["dataset"]==dataset) & (training_data["pooling"]==pool)][["m","batch_size", "test_error"]]
# training_data[training_data["dataset"]==dataset][["m","batch_size", "test_error"]]
# UwU

#%% #bound_vs_errors
%matplotlib
dataset = "EMNIST"
dataset = "cifar"
dataset = "mnist"
dataset = "KMNIST"
dataset = "mnist-fashion"
# pool = "avg"
bounds["m"].unique()
# m=40000
# m=15026
# m=4516
# m=1357
#
# bdata=bounds[(bounds["m"]==4516) & (bounds["dataset"]==dataset) & (bounds["pooling"]==pool)]
for dataset in datasets:
    for m in [1357, 4516, 15026, 40000]:
        try:
            bdata=bounds[(bounds["m"]==m) & (bounds["dataset"]==dataset)]
            # tdata=training_data[(training_data["m"]==4516) & (training_data["dataset"]==dataset) & (training_data["pooling"]==pool)]
            tdata=training_data[(training_data["m"]==m) & (training_data["dataset"]==dataset)]
            # tdata=training_data[(training_data["dataset"]==dataset)]
            len(bdata)

            boundss = []
            test_errorss = []
            netss=[]
            pools=[]
            nets=bounds["network"].unique()

            for net in nets:
                # if net in ["vgg19","vgg16"]:
                #     continue
                if net != "cnn" and net != "fc":
                # if net != "cnn":
                    if len(bdata[bdata["network"]==net]) == 1 and len(tdata[tdata["network"]==net]) == 1:
                        boundss.append(bdata[bdata["network"]==net]["bound"].iloc[0])
                        test_errorss.append(tdata[tdata["network"]==net]["test_error"].iloc[0])
                        netss.append(net)
                        pools.append("avg")
                        print(net, boundss[-1], test_errorss[-1])
                # elif net=="fc":
                #     if len(bdata[bdata["network"]==net]) == 1 and len(tdata[tdata["network"]==net]) == 1:
                #         boundss.append(bdata[bdata["network"]==net]["bound"].iloc[0])
                #         test_errorss.append(tdata[tdata["network"]==net]["test_error"].iloc[0])
                #         netss.append(net)
                #         pools.append("None")
                #         print(net, boundss[-1], test_errorss[-1])
                # else:
                #     for pool in ["None","max","avg"]:
                #     # for pool in ["max"]:
                #     # for pool in ["None"]:
                #         if len(bdata[(bdata["network"]==net) & (bdata["pooling"]==pool)]) == 1 and len(tdata[(tdata["network"]==net) & (tdata["pooling"]==pool)]) == 1:
                #             boundss.append(bdata[(bdata["network"]==net) & (bdata["pooling"]==pool)]["bound"].iloc[0])
                #             test_errorss.append(tdata[(tdata["network"]==net) & (tdata["pooling"]==pool)]["test_error"].iloc[0])
                #             netss.append(net)
                #             pools.append(pool)
                #             print(net, boundss[-1], test_errorss[-1])

            ii=0
            fig, ax = plt.subplots()
            for bound,error,net,pool in zip(boundss,test_errorss,netss,pools):
                color = cmap(ii/(len(nets)+1))
                if net!="cnn":
                    plt.scatter(bound, error,c=color, label=net)
                    ax.annotate(net, (bound, error))
                else:
                    plt.scatter(bound, error,c=color, label=net+"_"+pool)
                    ax.annotate(net+"_"+pool, (bound, error))
                ii+=1

            plt.xlim([0.9*np.min(boundss), 1.1*np.max(boundss)])
            plt.ylim([0.9*np.min(test_errorss), 1.1*np.max(test_errorss)])
            # plt.legend()
            plt.xlabel("PAC-Bayes bound")
            plt.ylabel("Test error")
            plt.savefig("img/msweep/bound_vs_error/bound_vs_error_"+dataset+"_"+str(m)+"_"+str(batch_size)+".png")
            plt.close()
        except Exception as e:
            plt.close()
            print(e)

# plt.scatter(boundss,test_errorss)
# plt.

# %%
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
