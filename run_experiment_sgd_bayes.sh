#!/bin/bash -l
echo =========================================================   
echo Job submitted  date = Thu Nov  7 20:10:48 GMT 2019      
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
 nice -n 10 /users/guillefix/nn-pacbayes/./run_experiment_sgd_bayes --prefix sgd_perceptron_64_512_0_8_adam_ce_1_ --m 64 --dataset boolean --network fc --number_layers 0 --training --sigmaw 1.0 --sigmab 0.0357 --n_gpus 0 --pooling none --loss ce --optimizer adam --number_inits 600000 --ignore_non_fit --batch_size 8 --layer_width 512 --epochs_after_fit 1 --zero_one
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
