#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aman.kukde@visitaors.fht.org
#SBATCH --partition=gpuq
#SBATCH --time=04-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --output=./logs/logfiles/fit_%j.log
#SBATCH --error=./logs/errfiles/fit_%j.log
#SBATCH --mem=64GB

source ~/.bashrc
export PATH="/home/aman.kukde/miniforge3/condabin/:$PATH"
mamba activate StateSpace
cd /home/aman.kukde/Projects/arch/FourierImageTransformer/
srun python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --job_id $SLURM_JOB_ID --num_nodes $SLURM_JOB_NUM_NODES --dataset MNIST --loss sum --model_type mamba --w_phi 1000 --batch_size 32 --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 20
