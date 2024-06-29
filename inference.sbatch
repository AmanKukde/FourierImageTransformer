#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aman.kukde@fht.org
#SBATCH --partition=gpuq
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=inf_%j.log
#SBATCH --mem=128GB
source ~/.bashrc
export PATH="/home/aman.kukde/miniforge3/condabin/:$PATH"
mamba activate StateSpace
cd /home/aman.kukde/Projects/main/FourierImageTransformer/
#SUM + MNIST
echo "total_gpus: $SLURM_NTASKS"
srun python3 /home/aman.kukde/Projects/main/FourierImageTransformer/inference.py 