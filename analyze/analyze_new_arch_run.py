import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib
import tensorflow as tf

data_folder = "data/"
arch_folder = "archs/"
kernel_folder = "kernels/"
results_folder = "results/"

prefix = "newer_arch_sweep_ce_sgd_"
training_results = pd.read_csv(results_folder+prefix+"nn_training_results.txt",comment="#", header='infer',sep="\t")

networks = training_results["network"].unique()

get_test_error = lambda net: sum(training_results[training_results["network"]==net]["test_error"])

# training_results[training_results["network"]==net]["test_error"]

from GP_prob.GP_prob_gpy import GP_prob as GP_prob1
from GP_prob.GP_prob_gpy2 import GP_prob as GP_prob2
from GP_prob.GP_prob_MC import GP_prob as GP_prob_MC
from GP_prob.GP_prob_VI import GP_prob as GP_prob_VI
from GP_prob.GP_prob_regression import GP_prob as GP_prob_regresison
# import imp;import GP_prob.average_error_estimate; imp.reload(GP_prob.average_error_estimate)
from GP_prob.average_error_estimate import average_error

def logP(K,X,Y):
    logdet = np.sum(np.log(np.linalg.eigh(Kfull)[0]))
    thing = -0.5*(-np.matmul(Y.T,np.matmul(np.linalg.inv(Kfull),Y)) - m*np.log(np.pi) - logdet)
    return thing[0,0]

def average_error_wrapper(K):
    sigma=1e-2
    return average_error(K,sigma)

#%%

# methods = ["EP","Laplace","importance_sampling","HMC", "Metropolis_EPproposal","variational", "logP_regression"]
methods = ["EP","exact_PB","logP_regression"]
methods = ["average_error_estimate"]
funs={"average_error_estimate":lambda K,X,Y: average_error_wrapper(K)}
funs = {"EP": lambda K,X,Y: -GP_prob1(K,X,Y, method="EP"),
        "exact_PB": lambda K,X,Y: -GP_prob2(K,X,Y,using_exactPB=True),
        "Laplace": lambda K,X,Y: -GP_prob1(K,X,Y,method="Laplace"),
        "importance_sampling": lambda K,X,Y: -GP_prob_MC(K,X,Y,method="importance_sampling"),
        "HMC": lambda K,X,Y: -GP_prob_MC(K,X,Y,"HMC"),
        "Metropolis_EPproposal": lambda K,X,Y: -GP_prob_MC(K,X,Y,method="Metropolis_EPproposal"),
        "variational":lambda K,X,Y: -GP_prob_VI(K,X,Y),
        "average_error_estimate":lambda K,X,Y: average_error_wrapper(K),
        "logP_regression": logP}

results_df = pd.DataFrame(columns=["net", "test_error"]+methods)

# things = []
# for net in ["densenet121","densenet169","densenet201","mobilenetv2","nasnet","resnet50","vgg16","vgg19"]:
for net in networks:
    if net == "nasnet": # haven't got its NTK yet
        continue
    print(net)
    filename = "newer_arch_sweep_ce_sgd__"+net+"_EMNIST_1000_0.0_0.0_True_False_False_False_-1_True_False_False_data.h5"

    from utils import load_data_by_filename
    train_images,flat_data,ys,test_images,test_ys = load_data_by_filename("data/"+filename)
    #%%

    input_dim = train_images.shape[1]
    num_channels = train_images.shape[-1]
    # tp_order = np.concatenate([[0,len(train_images.shape)-1], np.arange(1, len(train_images.shape)-1)])
    # train_images = tf.constant(train_images)
    tp_order = np.concatenate([[0,len(train_images.shape)-1], np.arange(1, len(train_images.shape)-1)])
    flat_data = np.transpose(train_images, tp_order)  # NHWC -> NCHW # this is because the cnn GP kernels assume this (tho we are not calculating kernels here so meh)
    X = np.stack([x.flatten() for x in train_images])
    X_test = np.stack([x.flatten() for x in test_images])

    test_images = test_images[:500]
    test_ys = test_ys[:500]

    Xfull =  np.concatenate([X,X_test])
    ys2 = [[y] for y in ys]
    ysfull = ys2 + [[y] for y in test_ys]
    Yfull = np.array(ysfull)
    Y = np.array(ys2)

    #%%

    # filename = net+"_KMNIST_1000_0.0_0.0_True_False_True_4_3.0_0.0_None_0000_max_kernel.npy"
    # filename = "newer_arch_sweep_ce_sgd__"+str(net)+"_EMNIST_1000_0.0_0.0_True_False_True_4_1.414_0.0_None_0000_max_kernel.npy"
    filename = "newer_arch_sweep_ce_sgd__"+str(net)+"_EMNIST_1000_0.0_0.0_True_False_True_4_1.414_0.0_None_0000_max_NTK_kernel.npy"

    from utils import load_kernel_by_filename
    try:
        Kfull = load_kernel_by_filename("kernels/"+filename)
    except FileNotFoundError:
        print("File not found :P")
        continue
    # m = 1000
    # Kfull.max()

    K = Kfull/Kfull.max()
    K *= 1000
    # K = Kfull
    # K
    # Y = Y*2-1
    # Kfull.shape
    test_error = get_test_error(net)
    results = {"net":net, "test_error": test_error}
    for method in methods:
        results[method] = funs[method](K,X,Y)

    results_df = results_df.append(results,ignore_index=True)

# results
# results_df.append(results,ignore_index=True)

# for net in ["densenet121","densenet169","densenet201","mobilenetv2","nasnet","resnet50","vgg16","vgg19"]:
#     results_df.loc[results_df["net"]==net,"test_error"] = get_test_error(net)
#
# for net in ["densenet121","densenet169","densenet201","mobilenetv2","nasnet","resnet50","vgg16","vgg19"]:
#     results_df.loc[results_df["net"]==net,"logP_regression"] = results_df.loc[results_df["net"]==net,"logP_regression"].iloc[0][0,0]


# results_df["logP_regression"]

# results_df["EP"] *= -1
# results_df["exact_PB"] *= -1
# results_df["variational"] *= -1

results_df

# results_df.plot()

%matplotlib
plot_multi(results_df,results_df.columns[1:])
plt.savefig("ntk_expected_error_prediction.png")


def plot_multi(data, cols=None, spacing=.1, **kwargs):

    from pandas import plotting

    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return
    # colors = getattr(plotting, '_get_standard_colors')(num_colors=len(cols))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0])
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)])
        ax_new.set_ylabel(ylabel=cols[n])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    return ax

plot_multi(results_df,results_df.columns[1:])


pd.plotting._get_standard_colors
import matplotlib
