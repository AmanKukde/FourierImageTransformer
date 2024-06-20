#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aman.kukde@visitors.fht.org
#SBATCH --partition=gpuq
#SBATCH --time=06-00:00:00
#SBATCH --nodes=2
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
