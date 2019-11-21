#!/bin/bash

m=100
dataset=mnist
#dataset=EMNIST
#dataset=$1
#dataset=cifar
#dataset=calabiyau
#dataset=boolean
#boolfun=00001110110011111001111111001111000000000000000000000000000000000000000001001100000000001101110100000000000000000000000000000000
#boolfun=00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000
#boolfun=$1
boolfun_comp=84.0
#net=cnn
net=fc
#L=4
L=1
#optimizer=adam
optimizer=sgd
loss=ce
sigmaw=1.41
#sigmaw=100.0
sigmab=0.0
pool=none
c=0.0
epochs_after_fit=1
t=-1
if [ "$dataset" == "boolean" ]; then
t=1
elif [ "$dataset" == "EMNIST" ]; then
#t=61
t=1
else
t=1
#t=9
#t=5
fi
#prefix=new_unbalancedt${t}_${dataset}
#prefix=test_${t}_${dataset}
#prefix=GPEP2_logPs_
#prefix=shifted_init_sweep_
#prefix=shifted_init_sweep_${t}_${dataset}_
#prefix=test_shifted_init_sweep_${t}_${dataset}_

n_gpus=1
export n_gpus=$n_gpus
export n_procs=10
number_inits=10
number_samples=100

#for shifted_init_shift in `seq -5.0 1.0 5.0`; do
for shifted_init_shift in `seq -1.0 0.2 1.0`; do
#for shifted_init_shift in `seq -10.0 2 10.0`; do
#for shifted_init_shift in `seq 200.0 -50 -200.0`; do
#for shifted_init_shift in 1 -5 5 -4 4 -3 3 -2 2 -1 0 0.5 1.5; do
#for shifted_init_shift in -1.5 -0.5 0.5 1.5; do
echo $shifted_init_shift
prefix=tanh_centered_fc_shifted_init_sweep_${shifted_init_shift}_${t}_${dataset}_

  #./run_experiment3 --prefix $prefix --m $m --threshold $t --dataset $dataset --network $net --number_layers $L --training --sigmaw $sigmaw --sigmab $sigmab --confusion $c --n_gpus $n_gpus --pooling $pool --number_inits $number_inits --n_samples_repeats 2.0 --boolfun_comp $boolfun_comp --boolfun $boolfun --epochs_after_fit $epochs_after_fit -loss $loss --use_shifted_init --shifted_init_shift $shifted_init_shift
  ./run_experiment3 --prefix $prefix --m $m --threshold $t --dataset $dataset --network $net --number_layers $L --training --sigmaw $sigmaw --sigmab $sigmab --confusion $c --n_gpus $n_gpus --pooling $pool --number_inits $number_inits --n_samples_repeats 2.0 --boolfun_comp $boolfun_comp --boolfun $boolfun --epochs_after_fit $epochs_after_fit -loss $loss --shifted_init_shift $shifted_init_shift --number_samples $number_samples --centering
  #./run_experiment3 --prefix $prefix --m $m --threshold $t --dataset $dataset --network $net --number_layers $L --training --sigmaw $sigmaw --sigmab $sigmab --confusion $c --n_gpus $n_gpus --pooling $pool --number_inits $number_inits --n_samples_repeats 2.0 --boolfun_comp $boolfun_comp --boolfun $boolfun --epochs_after_fit $epochs_after_fit -loss $loss --shifted_init_shift $shifted_init_shift --number_samples $number_samples 
  
  #./run_experiment3 --prefix $prefix --m $m --threshold $t --dataset $dataset --network $net --number_layers $L --training --sigmaw $sigmaw --sigmab $sigmab --confusion $c --n_gpus $n_gpus --pooling $pool --number_inits $number_inits --n_samples_repeats 2.0 --boolfun_comp $boolfun_comp --boolfun $boolfun --epochs_after_fit $epochs_after_fit -loss $loss $@

done
