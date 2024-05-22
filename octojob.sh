#!/bin/bash

#$ -l rt_F=1
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

#export WANDB_MODE=disabled
#export NCCL_DEBUG=INFO

# Make first node the coordinator
export COORDINATOR_ADDRESS=`head -1 $SGE_JOB_HOSTLIST`:12345

# NOTE JAX expectes 1 process per GPU in SLURM/OpenMPI
export NUM_GPUS=`nvidia-smi -L | wc -l`

# NOTE up to 64 samples/gpu in V100 (~13G) before OOM
# NOTE --debug disables wandb

export NUM_NODES=`cat $SGE_JOB_HOSTLIST | wc -l`
# NOTE Default values in config
export TOTAL_STEPS=`echo "50000 * 256" | bc`
export MAX_SAMPLES_GPU=64
export BATCH_SIZE=`echo "$MAX_SAMPLES_GPU * $NUM_GPUS * $NUM_NODES" | bc`
export NUM_STEPS=`echo "$TOTAL_STEPS / $BATCH_SIZE" | bc`

echo "NUM_NODES=$NUM_NODES"
echo "TOTAL_STEPS=$TOTAL_STEPS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "NUM_STEPS=$NUM_STEPS"

mpirun -npernode $NUM_GPUS -hostfile $SGE_JOB_HOSTLIST \
    python scripts/finetune.py --config.pretrained_path=hf://rail-berkeley/octo-small --config.batch_size=$BATCH_SIZE --config.num_steps=$NUM_STEPS --config.save_dir="savedir" --config.wandb.group="mpi"
    #python scripts/finetune.py --config.pretrained_path=hf://rail-berkeley/octo-small --config.batch_size=256 --config.num_steps=50000 --config.save_dir="savedir" --config.wandb.group="mpi"
