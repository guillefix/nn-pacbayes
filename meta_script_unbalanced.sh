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
net=cnn
L=4
#optimizer=adam
optimizer=sgd
loss=ce
sigmaw=1.41
#sigmaw=100.0
sigmab=0.1
pool=none
c=0.0
number_inits=20
epochs_after_fit=1

t=-1
if [ "$dataset" == "boolean" ]; then
t=1
elif [ "$dataset" == "EMNIST" ]; then
t=61
else
t=1
#t=9
#t=5
fi
#prefix=new_unbalancedt${t}_${dataset}
#prefix=test_${t}_${dataset}
#prefix=GPEP2_logPs_
#prefix=shifted_init_sweep_
prefix=shifted_init_sweep_${t}_${dataset}_
#prefix=test_shifted_init_sweep_${t}_${dataset}_

n_gpus=0
export n_gpus=$n_gpus
export n_procs=20

for shifted_init_shift in `seq -2.0 0.5 5.0`; do
#for shifted_init_shift in 1.5; do
echo $shifted_init_shift
prefix=shifted_init_sweep_${shifted_init_shift}_${t}_${dataset}_

  ./run_experiment3 --prefix $prefix --m $m --threshold $t --dataset $dataset --network $net --number_layers $L --training --sigmaw $sigmaw --sigmab $sigmab --confusion $c --n_gpus $n_gpus --pooling $pool --number_inits $number_inits --n_samples_repeats 2.0 --boolfun_comp $boolfun_comp --boolfun $boolfun --epochs_after_fit $epochs_after_fit -loss $loss --use_shifted_init --shifted_init_shift $shifted_init_shift
  #./run_experiment3 --prefix $prefix --m $m --threshold $t --dataset $dataset --network $net --number_layers $L --training --sigmaw $sigmaw --sigmab $sigmab --confusion $c --n_gpus $n_gpus --pooling $pool --number_inits $number_inits --n_samples_repeats 2.0 --boolfun_comp $boolfun_comp --boolfun $boolfun --epochs_after_fit $epochs_after_fit -loss $loss $@

done
