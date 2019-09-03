#!/bin/bash
#for dataset in 'mnist' 'mnist-fashion' 'cifar'; do
#	python NNGP_kernel.py 10000 0.0 $dataset 1.0 1.0 resnet $L bound 1
#	python NN_train.py 10000 2 0.0 32 1.0 1.0 resnet $dataset
#done


expname="randomlabelbiggerdata"
m=10000

#time eval './run_experiment --network fc --number_layers 4 --m $m --dataset '{cifar,mnist,mnist-fashion}' --confusion 0.'{7..9}' --compute_bound --prefix $expname;'
#time eval './run_experiment --network fc --number_layers 4 --m $m --dataset '{cifar,mnist,mnist-fashion}' --confusion '{3..5}' --compute_bound --prefix $expname;'
time eval './run_experiment --network fc --number_layers 4 --m $m --dataset '{mnist,mnist-fashion}' --confusion '{1..3}' --compute_bound --prefix $expname;'

#time eval './run_experiment --network cnn --number_layers 4 --m $m --dataset '{cifar,mnist,mnist-fashion}' --confusion 0.'{0..9}' --compute_bound --prefix $expname;'
#time eval './run_experiment --network cnn --number_layers 4 --m $m --dataset '{cifar,mnist,mnist-fashion}' --confusion '{1..9}' --compute_bound --prefix $expname;'
#
#time eval './run_experiment --network resnet --number_layers 32 --m $m --dataset '{cifar,mnist,mnist-fashion}' --confusion 0.'{0..9}' --compute_bound --prefix $expname;'
#time eval './run_experiment --network resnet --number_layers 4 --m $m --dataset '{cifar,mnist,mnist-fashion}' --confusion '{1..9}' --compute_bound --prefix $expname;'
#
#time eval './run_experiment --network fc --number_layers 4 --m $m --dataset '{cifar,mnist,mnist-fashion}' --confusion '{10..20}' --compute_bound --prefix $expname;'
#
#time eval './run_experiment --network cnn --number_layers 4 --m $m --dataset '{cifar,mnist,mnist-fashion}' --confusion '{10..20}' --compute_bound --prefix $expname;'
#
#time eval './run_experiment --network resnet --number_layers 32 --m $m --dataset '{cifar,mnist,mnist-fashion}' --confusion '{10..20}' --compute_bound --prefix $expname;'



#eval './run_experiment --network cnn --number_layers 4 --m 1000 --dataset mnist --gamma 1.0 --confusion 0.0 --prefix test;'


