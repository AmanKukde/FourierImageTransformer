#!/bin/bash

# SSH variables
remote_user="aman.kukde"
remote_host="hpclogin.fht.org"
remote_path="/home/aman.kukde/Projects/arch/FourierImageTransformer/job_scripts"

# Commands to run
commands=(
    #MAMBA
    # "srun python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --job_id \$SLURM_JOB_ID --num_nodes \$SLURM_JOB_NUM_NODES --dataset MNIST --loss sum --model_type mamba --w_phi 1000 --batch_size 32 --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 1"
    # "srun python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --job_id \$SLURM_JOB_ID --num_nodes \$SLURM_JOB_NUM_NODES --dataset CelebA --loss sum --model_type mamba --w_phi 1000 --batch_size 32 --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 1"
    # "srun python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --job_id \$SLURM_JOB_ID --num_nodes \$SLURM_JOB_NUM_NODES --dataset MNIST --loss sum --model_type fast --w_phi 1000 --batch_size 32 --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 1"
    # "srun python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --job_id \$SLURM_JOB_ID --num_nodes \$SLURM_JOB_NUM_NODES --dataset CelebA --loss sum --model_type fast --w_phi 1000 --batch_size 32 --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 1"
    # "srun python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --job_id \$SLURM_JOB_ID --num_nodes \$SLURM_JOB_NUM_NODES --dataset MNIST --loss sum --model_type mamba --w_phi 1000 --batch_size 32 --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 20"
    "srun python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --job_id \$SLURM_JOB_ID --num_nodes \$SLURM_JOB_NUM_NODES --dataset CelebA --loss sum --model_type mamba --w_phi 1000 --batch_size 16 --d_query 64 --n_layers 12 --n_heads 12 --no_of_sectors 10"
    "srun python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --job_id \$SLURM_JOB_ID --num_nodes \$SLURM_JOB_NUM_NODES --dataset MNIST --loss sum --model_type fast --w_phi 1000 --batch_size 32 --d_query 64 --n_layers 12 --n_heads 12 --no_of_sectors 10"
    "srun python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --job_id \$SLURM_JOB_ID --num_nodes \$SLURM_JOB_NUM_NODES --dataset CelebA --loss sum --model_type fast --w_phi 1000 --batch_size 16 --d_query 64 --n_layers 12 --n_heads 12 --no_of_sectors 10"
    "srun python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --job_id \$SLURM_JOB_ID --num_nodes \$SLURM_JOB_NUM_NODES --dataset MNIST --loss sum --model_type mamba --w_phi 1000 --batch_size 32 --d_query 64 --n_layers 12 --n_heads 12 --no_of_sectors 10"
    # "srun python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --job_id \$SLURM_JOB_ID --num_nodes \$SLURM_JOB_NUM_NODES --dataset CelebA --loss sum --model_type mamba --w_phi 1000 --batch_size 32 --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 20"
    # "srun python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --job_id \$SLURM_JOB_ID --num_nodes \$SLURM_JOB_NUM_NODES --dataset MNIST --loss sum --model_type fast --w_phi 1000 --batch_size 32 --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 20"
    # "srun python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --job_id \$SLURM_JOB_ID --num_nodes \$SLURM_JOB_NUM_NODES --dataset CelebA --loss sum --model_type fast --w_phi 1000 --batch_size 32 --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 10"
    )


# Function to create and submit a job script
create_and_submit_job() {
    local command="$1"
    local script_name="job_script.sbatch"
    
    # Create job script locally
    {
        cat /home/aman.kukde/Projects/arch/FourierImageTransformer/base_job_script.sbatch
        echo "$command"
    } > $script_name
    
    # Transfer the job script to the remote server
    scp $script_name $remote_user@$remote_host:$remote_path/
    
    # Submit the job script via SSH
    ssh $remote_user@$remote_host "sbatch $remote_path/$script_name"
}

# Ensure the remote job script directory exists
ssh $remote_user@$remote_host "mkdir -p $remote_path"

# Create and submit each command as a separate job
for command in "${commands[@]}"; do
    create_and_submit_job "$command"
done
