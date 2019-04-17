#!/bin/bash -l
echo =========================================================   
echo Job submitted  date = Thu Feb 21 06:11:43 GMT 2019      
date_start=`date +%s`
echo $SLURM_JOB_NUM_NODES nodes \( $SMP processes per node \)        
echo $SLURM_JOB_NUM_NODES hosts used: $SLURM_JOB_NODELIST      
echo Job output begins                                           
echo -----------------                                           
echo   
#hostname
#ulimit -l
#which mpirun
export OMP_NUM_THEADS=1
 /usr/local/shared/slurm/bin/srun -n 250 --mpi=pmi2 --mem-per-cpu=5000 nice -n 10 /users/guillefix/anaconda3/bin/python3 sample_funs.py --prefix new_run --number_samples 1000 --m 1000 --dataset mnist-fashion --network resnet --number_layers 32 --whitening -n_gpus 0
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
