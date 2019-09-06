#!/bin/bash -l
echo =========================================================   
echo Job submitted  date = Fri Sep  6 14:06:11 BST 2019      
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
 nice -n 10 /users/guillefix/nn-pacbayes/./run_experiment --prefix new_comp_sweep_ --m 1000 --dataset mnist --network cnn --number_layers 4 --training -sigmaw 1.0 -sigmab 0.0 --label_corruption 0.1 --n_gpus 1 --compute_bound --pooling none --n_samples_repeats 2.0
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
