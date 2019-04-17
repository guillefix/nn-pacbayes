#!/bin/bash

#source ~/anaconda/bin/activate root 
#~/anaconda3/bin/python compute_CSR.py 
#~/anaconda3/bin/python compute_probs.py 
#python sample_funs.py $@
#python compute_CSR.py $@
#python compute_probs.py $@
#py=/usr/bin/python3 
#py=~/anaconda3/bin/python3
py=python3

m=500
dataset=mnist
net=cnn
L=4
#net=resnet
#L=32
prefix=new_run
num_samples=1000

max=max
#$py generate_NN_arch.py --dataset $dataset --network $net --number_layers $L --pooling $max
#$py generate_inputs_sample.py --prefix $prefix --m $m --dataset $dataset --network $net --number_layers $L --training --pooling $max
mpiexec -n 4 $py compute_kernel_and_bound.py --prefix $prefix --m $m --dataset $dataset --network $net --number_layers $L --ngpus 1 --compute_bound --pooling $max --n_samples_repeats 2.0
#$py NN_train.py --prefix $prefix --m $m --dataset $dataset --network $net --number_layers $L --pooling $max

#for dataset in cifar mnist mnist-fashion; do
#    for net in cnn 'fc' resnet; do
#        if [ $net == 'resnet' ]; then
#            L=32
#        else
#            L=4
#        fi
#        #$py generate_NN_arch.py --dataset $dataset --network $net --number_layers $L
#        #$py generate_inputs_sample.py --prefix $prefix --m $m --dataset $dataset --network $net --number_layers $L --whitening --compute_kernel
#        addqueue -n 250 -m 5 $py sample_funs.py --prefix $prefix --number_samples 1000 --m $m --dataset $dataset --network $net --number_layers $L --whitening -n_gpus 0
#    done
#done


#$py generate_NN_arch.py --dataset $dataset --network $net --number_layers $L
#$py generate_inputs_sample.py --prefix $prefix --m $m --dataset $dataset --network $net --number_layers $L --whitening --compute_kernel
#$py sample_funs.py --prefix $prefix --number_samples 2 --m $m --dataset $dataset --network $net --number_layers $L --whitening

#mpiexec -n 8 $py sample_funs.py --prefix $prefix --number_samples $num_samples --m $m --dataset $dataset --network $net --number_layers $L --whitening --n_gpus 8
#mpiexec -n 8 $py compute_CSR.py --prefix $prefix --number_samples $num_samples --m $m --dataset $dataset --network $net --number_layers $L --whitening --n_gpus 8

#mpiexec -n $(($(nproc)<$num_samples?$(nproc):$num_samples)) $py sample_funs.py --prefix $prefix --number_samples $num_samples --m $m --dataset $dataset --network $net --number_layers $L --whitening --n_gpus 8

#addqueue -n 250 -m 5 $py sample_funs.py --prefix $prefix --number_samples 1000 --m $m --dataset $dataset --network $net --number_layers $L --whitening
#addqueue -n 10 -m 5 $py sample_funs.py --prefix $prefix --number_samples 10 --m $m --dataset $dataset --network $net --number_layers $L --whitening

