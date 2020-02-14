#!/bin/bash -l
echo =========================================================   
echo Job submitted  date = Thu Feb 13 18:40:33 GMT 2020      
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
 nice -n 10 /users/guillefix/nn-pacbayes/./run_experiment_sgd_bayes --prefix 6_sgd_vs_bayes_32_40_2_8_sgd_ce_0_ --m 32 --dataset boolean --network fc --number_layers 2 --training --sigmaw 1.0 --sigmab 1.0 --n_gpus 0 --pooling none --loss ce --optimizer sgd --number_inits 1000000 --ignore_non_fit --batch_size 8 --layer_width 40 --epochs_after_fit 0 --zero_one --boolfun 00110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011 --booltrain_set 01001100010100101100001000100010000010010011000010000010001010010000000001100000010000100000000000100010010000011000100011000000
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
