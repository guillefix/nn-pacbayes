#!/bin/bash -l
echo =========================================================   
echo Job submitted  date = Mon Apr  6 18:43:21 BST 2020      
date_start=`date +%s`
echo $SLURM_JOB_NUM_NODES nodes \( $SMP processes per node \)        
echo $SLURM_JOB_NUM_NODES hosts used: $SLURM_JOB_NODELIST      
echo Job output begins                                           
echo -----------------                                           
echo   
#hostname

# Need to set the max locked memory very high otherwise IB can't allocate enough and fails with "UCX  ERROR Failed to allocate memory pool chunk: Input/output error"
ulimit -l unlimited

# To allow mvapich to run ok
export MV2_SMP_USE_CMA=0

#which mpirun
export OMP_NUM_THEADS=1
 nice -n 10 /users/guillefix/nn-pacbayes/./run_experiment_sgd_bayes --prefix 6_sgd_vs_bayes_gpmse___6_32_40_2_1_sgd_mse_0_0.001_ --m 32 --dataset boolean --network fc --number_layers 2 --training --sigmaw 1.0 --sigmab 1.0 --n_gpus 0 --pooling none --loss mse --optimizer sgd --number_inits 1000000 --ignore_non_fit --batch_size 1 --layer_width 40 --epochs_after_fit 0 --zero_one --boolfun 00110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011 --booltrain_set 01100010000011100000110101000000101100001010000101001000000110010100000000010000000000000001000000101001100000010010000101000000 --learning_rate 0.001
# If we've been checkpointed
#if [ -n "${DMTCP_CHECKPOINT_DIR}" ]; then
  if [ -d "${DMTCP_CHECKPOINT_DIR}" ]; then
#    echo -n "Job was checkpointed at "
#    date
#    echo 
     sleep 1
#  fi
   echo -n
else
  echo ---------------                                           
  echo Job output ends                                           
  date_end=`date +%s`
  seconds=$((date_end-date_start))
  minutes=$((seconds/60))
  seconds=$((seconds-60*minutes))
  hours=$((minutes/60))
  minutes=$((minutes-60*hours))
  echo =========================================================   
  echo PBS job: finished   date = `date`   
  echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
  echo =========================================================
fi
if [ ${SLURM_NTASKS} -eq 1 ]; then
  rm -f $fname
fi
