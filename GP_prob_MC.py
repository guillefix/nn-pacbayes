import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import GPy
from custom_kernel_matrix.custom_kernel_matrix import CustomMatrix
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def GP_prob(K,X,Y,FLAGS):
    globals().update(FLAGS)
    # link_fun = GPy.likelihoods.link_functions.Heaviside()
    # lik = GPy.likelihoods.Bernoulli(gp_link=link_fun)
    lik = GPy.likelihoods.Bernoulli()

    inference_method = GPy.inference.latent_function_inference.expectation_propagation.EP(parallel_updates=True)
    # inference_method = GPy.inference.latent_function_inference.laplace.Laplace()

    model = GPy.core.GP(X=X,
                    Y=Y,
                    kernel=CustomMatrix(X.shape[1],X,K),
                    inference_method=inference_method,
                    likelihood=lik)

    mean, cov = model._raw_predict(X, full_cov=True)
    mean *= 1
    mean = mean.flatten()
    cov *= cov_mult
    log_norm_ratio = np.sum(np.log((np.linalg.eigh(cov)[0] / np.linalg.eigh(K)[0])))/2
    covinv = np.linalg.inv(cov)
    Kinv = np.linalg.inv(K)
    tot = 0
    shift = m*np.log(2)*0.3

    #parallelizing tasks with mpi rank
    num_tasks = num_post_samples
    num_tasks_per_job = num_tasks//size
    tasks = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))
    if rank < num_tasks%size:
        tasks.append(size*num_tasks_per_job+rank)

    sample = np.random.multivariate_normal(mean, cov, len(tasks)).T

    import sys
    for i in range(len(tasks)):
        # print(i)
        # if (i%(len(tasks)/100)) == (len(tasks)/100)-1:
        #     print(str(int(100*i/len(tasks)))+"%")
        f = sample[:,i]
        PQratio = np.exp(shift-0.5*(np.matmul(f.T, np.matmul(Kinv, f)) - np.matmul((f-mean).T, np.matmul(covinv, (f-mean))) ) + log_norm_ratio)
        if np.prod((f.T>0) == Y.T):
            print(PQratio)
            tot += PQratio

    tots = comm.gather(tot,root=0)
    if rank == 0:
        tot = sum(tots)
        logPU = np.log(tot) - np.log(num_post_samples)
        return logPU
    else:
        return None
