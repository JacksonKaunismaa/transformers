#!/bin/bash
set -e
PROJECT_DIR=/h/jackk/transformers
cd $PROJECT_DIR
# This is the script to perform training, the goal is that code in
# this script can be safely preempted. Jobs in slurm queue are scheduled
# and preempted according to their priorities. Note that even the job with
# deadline queue can be preempted over time so it is crucial that you
# checkpoint your program state for every fixed interval: e.g 10 mins.

# Vector provides a fast parallel filesystem local to the GPU nodes,  dedicated
# for checkpointing. It is mounted under /checkpoint. It is strongly
# recommended that you keep your intermediary checkpoints under this directory
# i.e. /checkpoint/${USER}/${SLURM_JOB_ID}

# We also recommend users to create a symlink of the checkpoint dir so your
# training code stays the same with regards to different job IDs and it would
# be easier to navigate the checkpoint directory
ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $PWD/checkpoint


# In the future, the checkpoint directory will be removed immediately after the
# job has finished. If you would like the file to stay longer, and create an
# empty "delay purge" file as a flag so the system will delay the removal for
# 48 hours
touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE

# prepare the environment, here I am using environment modules, but you could
# select the method of your choice (but note that code in ~/.bash_profile or
# ~/.bashrc will not be executed with a new job)
module purge && module use /h/${USER}/.environment_modules/ && module load pytorch-2-python-3.10-cuda-118 


# log in to wandb
python3 -m wandb login $(cat wandb_api_key)
# Then we run our training code, using the checkpoint dir provided the code
# demonstrates how to perform checkpointing in pytorch, please navigate to the
# file for more information.
# if [[ "$1" == "--debug" ]]; then
# 	python3 -m pdb run_experiment.py
# else   # forward arguments passed to script to python 
#python3 run_experiment.py $@
# echo $SLURM_NNODES $SLURM_GPUS_PER_NODE $SLURM_CPUS_PER_GPU $MASTER_ADDR $MASTER_PORT
# if [[ $SLURM_NNODES -eq 1 ]]; then
# 	echo "running single"
# 	torchrun --nproc-per-node=$SLURM_GPUS_PER_NODE --standalone \
# 			 run_experiment.py --num_workers=$SLURM_CPUS_PER_GPU
# else
# 	echo "running multi"
# 	torchrun --nproc-per-node=$SLURM_GPUS_PER_NODE --rdzv-id=$SLURM_JOB_ID \
# 		 	--rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
# 			run_experiment.py --num_workers=$SLURM_CPUS_PER_GPU
# fi
# would have used torchrun for this, but it just didn't work for some reason (couldn't get NCCL to not complain about
# duplicate GPU use)
python3 run_experiment.py --local-world-size=$SLURM_GPUS_PER_NODE --nnodes=$1 --rank=$2 --num-workers=$SLURM_CPUS_PER_GPU --id=$SLURM_JOB_ID --resume-path=$3

