#!/bin/bash

#$ -l rt_F=2
#$ -l h_rt=0:10:00
#$ -j y
#$ -cwd
#$ -l USE_SSH=1
#$ -v SSH_PORT=2299

source /etc/profile.d/modules.sh
# NOTE Same versions as JAX was built with
# https://jax.readthedocs.io/en/latest/installation.html#pip-installation-gpu-cuda-installed-locally-harder
module load cuda/12.3
module load cudnn/8.9
module load nccl/2.19
module load hpcx/2.12

source ~/miniforge3/etc/profile.d/conda.sh 
conda activate octo

cd ~/octo

export WANDB_MODE=disabled

# Make first node the coordinator
export COORDINATOR_ADDRESS=`head -1 $SGE_JOB_HOSTLIST`:12345

mpirun -npernode 1 -hostfile $SGE_JOB_HOSTLIST \
    python scripts/finetune.py --config.pretrained_path=hf://rail-berkeley/octo-small --debug