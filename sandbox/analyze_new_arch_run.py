import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib
import tensorflow as tf

data_folder = "data/"
arch_folder = "archs/"
kernel_folder = "kernels/"

training_results = pd.read_csv(results_folder+prefix+"nn_training_results.txt",comment="#", header='infer',sep="\t")

get_test_error = lambda net: training_results[training_results["network"]==net]["test_error"]

from GP_prob.GP_prob_gpy import GP_prob as GP_prob1
from GP_prob.GP_prob_MC import GP_prob as GP_prob_MC
from GP_prob.GP_prob_VI import GP_prob as GP_prob_VI
from GP_prob.GP_prob_regression import GP_prob as GP_prob_regresison

def logP(K,X,Y):
    logdet = np.sum(np.log(np.linalg.eigh(Kfull)[0]))
    thing = -0.5*(-np.matmul(Y.T,np.matmul(np.linalg.inv(Kfull),Y)) - m*np.log(np.pi) - logdet)
    return thing

methods = ["EP","Laplace","importance_sampling","HMC", "Metropolis_EPproposal","variational", "logP"]
funs = {"EP": lambda K,X,Y: GP_prob1(K,X,Y, method="EP"),
        "Laplace": lambda K,X,Y: GP_prob1(K,X,Y,method="Laplace"),
        "importance_sampling": lambda K,X,Y: GP_prob_MC(K,X,Y,method="importance_sampling"),
        "HMC": lambda K,X,Y: GP_prob_MC(K,X,Y,"HMC"),
        "Metropolis_EPproposal": lambda K,X,Y: GP_prob_MC(K,X,Y,method="Metropolis_EPproposal"),
        "variational":lambda K,X,Y: GP_prob_VI(K,X,Y),
        "logP_cont": logP}

results_df = pd.DataFrame(["net", "test_error"]+methods)

# things = []
for net in ["densenet121","densenet169","densenet201","mobilenetv2","nasnet","resnet50","vgg16","vgg19"]:
    filename = "newer_arch_sweep_ce_sgd__"+net+"_EMNIST_1000_0.0_0.0_True_False_False_False_-1_True_False_False_data.h5"

    from utils import load_data_by_filename
    train_images,flat_data,ys,test_images,test_ys = load_data_by_filename("data/"+filename)
    #%%

    input_dim = train_images.shape[1]
    num_channels = train_images.shape[-1]
    # tp_order = np.concatenate([[0,len(train_images.shape)-1], np.arange(1, len(train_images.shape)-1)])
    # train_images = tf.constant(train_images)
    X = np.stack([x.flatten() for x in train_images])
    X_test = np.stack([x.flatten() for x in test_images])

    test_images = test_images[:500]
    test_ys = test_ys[:500]

    #%%

    Xfull =  np.concatenate([X,X_test])
    ys2 = [[y] for y in ys]
    ysfull = ys2 + [[y] for y in test_ys]
    Yfull = np.array(ysfull)
    Y = np.array(ys2)

    #%%

    filename = net+"_KMNIST_1000_0.0_0.0_True_False_True_4_3.0_0.0_None_0000_max_kernel.npy"

    from utils import load_kernel_by_filename
    Kfull = load_kernel_by_filename("kernels/"+filename)
    m = 1000
    # Kfull.max()

    # K = Kfull/Kfull.max()
    Kfull = K
    # K
    # Y = Y*2-1
    # Kfull.shape
    test_error = get_test_error(net)
    results = {"net":net, "test_error": test_error}
    for method in methods:
        results[method] = funs[method](K,X,Y)

    results_df.append(results)

results_df
