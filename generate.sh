#!/bin/bash

# SSH variables
remote_user="aman.kukde"
remote_host="hpclogin.fht.org"
remote_path="/home/aman.kukde/Projects/arch/FourierImageTransformer/job_scripts"
log_dir="/home/aman.kukde/Projects/arch/FourierImageTransformer/logs"

# Ensure the remote log directory exists
ssh $remote_user@$remote_host "mkdir -p $remote_path"
ssh $remote_user@$remote_host "mkdir -p $log_dir"

# Commands to run
commands=(
    "python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --dataset MNIST --loss sum --model_type mamba --w_phi 1000 --batch_size 32 --wandb --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 1"
    "python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --dataset CelebA --loss sum --model_type mamba --w_phi 1000 --batch_size 32 --wandb --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 1"
    "python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --dataset MNIST --loss sum --model_type fast --w_phi 1000 --batch_size 32 --wandb --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 1"
    "python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --dataset CelebA --loss sum --model_type fast --w_phi 1000 --batch_size 32 --wandb --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 1"
    "python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --dataset MNIST --loss sum --model_type mamba --w_phi 1000 --batch_size 32 --wandb --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 5"
    "python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --dataset CelebA --loss sum --model_type mamba --w_phi 1000 --batch_size 32 --wandb --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 5"
    "python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --dataset MNIST --loss sum --model_type fast --w_phi 1000 --batch_size 32 --wandb --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 5"
    "python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --dataset CelebA --loss sum --model_type fast --w_phi 1000 --batch_size 32 --wandb --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 5"
    "python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --dataset MNIST --loss sum --model_type mamba --w_phi 1000 --batch_size 32 --wandb --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 10"
    "python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --dataset CelebA --loss sum --model_type mamba --w_phi 1000 --batch_size 32 --wandb --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 10"
    "python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --dataset MNIST --loss sum --model_type fast --w_phi 1000 --batch_size 32 --wandb --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 10"
    "python3 /home/aman.kukde/Projects/arch/FourierImageTransformer/main.py --dataset CelebA --loss sum --model_type fast --w_phi 1000 --batch_size 32 --wandb --d_query 32 --n_layers 8 --n_heads 8 --no_of_sectors 10"
)

# Function to create and submit a job script
create_and_submit_job() {
    local command="$1"
    #local job_name=$(echo $command | awk '{print $4 "_" $8 "_" $10 "_" $14 "_" $20}')  # Extract some identifiers for the job name
    local script_name="job_script.sbatch"
    local remote_script_path="${remote_path}/${script_name}"

    # Create job script locally
    {
        cat <<EOL
#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aman.kukde@fht.org
#SBATCH --partition=gpuq
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --output=./logs/logfiles/fit_%j.log
#SBATCH --error=$./logs/errfiles/fit_%j.err
#SBATCH --mem=64GB

source ~/.bashrc
export PATH="/home/aman.kukde/miniforge3/condabin/:$PATH"
mamba activate StateSpace
cd /home/aman.kukde/Projects/arch/FourierImageTransformer/

srun $command
EOL
    } > $script_name

    # Transfer the job script to the remote server
    scp $script_name $remote_user@$remote_host:$remote_path/

    # Submit the job script via SSH
    ssh $remote_user@$remote_host "sbatch $remote_script_path"
}

# Create and submit each command as a separate job
for command in "${commands[@]}"; do
    create_and_submit_job "$command"
done
