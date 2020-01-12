#!/bin/bash -l
echo =========================================================   
echo Job submitted  date = Mon Nov 11 21:14:03 GMT 2019      
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
 nice -n 10 /users/guillefix/nn-pacbayes/./run_experiment_sgd_bayes_perceptron_langevin --prefix sgd_fc_32___1_sgd_mse_1_ --m 32 --dataset boolean --network fc --number_layers 2 --training --sigmaw 1.0 --sigmab 1.0 --n_gpus 0 --pooling none --loss mse --optimizer sgd --number_inits 600000 --ignore_non_fit --batch_size 1 --layer_width 40 --epochs_after_fit 1 --zero_one --boolfun 00110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011
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
