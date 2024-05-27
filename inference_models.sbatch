#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aman.kukde@fht.org
#SBATCH --partition=gpuq
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --output=fit_%j.log
#SBATCH --mem=64GB

source ~/.bashrc
export PATH="/home/aman.kukde/miniforge3/condabin/:$PATH"
mamba activate StateSpace
cd /home/aman.kukde/Projects/FourierImageTransformer/

srun python3 /home/aman.kukde/Projects/FourierImageTransformer/inference.py --job_id $SLURM_JOB_ID --num_nodes $SLURM_JOB_NUM_NODES --ckpt_path '/home/aman.kukde/Projects/FourierImageTransformer/models/MNIST/mamba/sum/Mamba_MNIST_wp_1000_sum_L_8_H_8_s_10_subset_False_26-05_21-34-54/epoch=970-step=417530.ckpt'