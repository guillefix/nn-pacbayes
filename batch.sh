#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH -J layer_sweep

#small for 1 gpu, big for 4 or 8
#SBATCH --partition=big

# set number of GPUs
#SBATCH --gres=gpu:8

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=guillefix@gmail.com

##SBATCH --array=1-15

#Launching the commands within script.sh

#archs=(vgg19 vgg16 resnet50 resnet101 resnet152 resnetv2_50 resnetv2_101 resnetv2_152 resnext50 resnext101 densenet121 densenet169 densenet201 mobilenetv2 nasnet)

#echo ${archs[$SLURM_ARRAY_TASK_ID]}.sh
#rm ${archs[$SLURM_ARRAY_TASK_ID]}.sh
#echo '#!/bin/bash' > ${archs[$SLURM_ARRAY_TASK_ID]}.sh
#echo './meta_script_arch_sweep '${archs[$SLURM_ARRAY_TASK_ID]} >> ${archs[$SLURM_ARRAY_TASK_ID]}.sh
#chmod +x ${archs[$SLURM_ARRAY_TASK_ID]}.sh

#/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c ./densenet201.sh
#/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c ./meta_script
/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c ./meta_script_layer_sweep
#/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c ${archs[$SLURM_ARRAY_TASK_ID]}.sh
