#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aman.kukde@fht.org
#SBATCH --partition=gpuq
#SBATCH --time=10-00:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --output=fit_%j.log
#SBATCH --mem=64GB
source ~/.bashrc
export PATH="/home/aman.kukde/miniforge3/condabin/:$PATH"
mamba activate StateSpace
cd /home/aman.kukde/Projects/FourierImageTransformer/
#SUM + MNIST
# srun python3 /home/aman.kukde/Projects/FourierImageTransformer/main.py --job_id $SLURM_JOB_ID --num_nodes $SLURM_JOB_NUM_NODES --dataset "MNIST" --loss "sum" --model_type "mamba" --w_phi 1000 --batch_size 32 --wandb --d_query 32 --n_layers 8 --n_heads 32 
# srun python3 /home/aman.kukde/Projects/FourierImageTransformer/main.py --job_id $SLURM_JOB_ID --num_nodes $SLURM_JOB_NUM_NODES --dataset "MNIST" --loss "sum" --model_type "fast" --w_phi 1000 --batch_size 32 --wandb --d_query 8 --n_layers 8 --n_heads 8 

# SUM + CELEBA
# srun python3 /home/aman.kukde/Projects/FourierImageTransformer/main.py --job_id $SLURM_JOB_ID --num_nodes $SLURM_JOB_NUM_NODES --dataset "CelebA" --loss "sum" --model_type "mamba" --w_phi 1000 --batch_size 8 --wandb --d_query 32 --n_layers 8 --n_heads 32 
# srun python3 /home/aman.kukde/Projects/FourierImageTransformer/main.py --job_id $SLURM_JOB_ID --num_nodes $SLURM_JOB_NUM_NODES --dataset "CelebA" --loss "sum" --model_type "fast" --w_phi 1000 --batch_size 8 --wandb --d_query 8 --n_layers 8 --n_heads 8 


#PROD+ "MNIST"
# srun python3 /home/aman.kukde/Projects/FourierImageTransformer/main.py --job_id $SLURM_JOB_ID --num_nodes $SLURM_JOB_NUM_NODES --dataset "MNIST" --loss "prod" --model_type "mamba" --w_phi 1000 --batch_size 32 --wandb --d_query 8 --n_layers 8 --n_heads 8 
# srun python3 /home/aman.kukde/Projects/FourierImageTransformer/main.py --job_id $SLURM_JOB_ID --num_nodes $SLURM_JOB_NUM_NODES --dataset "MNIST" --loss "prod" --model_type "fast" --w_phi 1000 --batch_size 32 --wandb --d_query 8 --n_layers 8 --n_heads 8 

#PROD + CELEBA
# srun python3 /home/aman.kukde/Projects/FourierImageTransformer/main.py --job_id $SLURM_JOB_ID --num_nodes $SLURM_JOB_NUM_NODES --dataset "CelebA" --loss p"rod" --model_type "mamba" --w_phi 1000 --batch_size 32 --wandb --d_query 8 --n_layers 8 --n_heads 8 
# srun python3 /home/aman.kukde/Projects/FourierImageTransformer/main.py --job_id $SLURM_JOB_ID --num_nodes $SLURM_JOB_NUM_NODES --dataset "CelebA" --loss "prod" --model_type "fast" --w_phi 1000 --batch_size 32 --wandb --d_query 8 --n_layers 8 --n_heads 8 
















































































