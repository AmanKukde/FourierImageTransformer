#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aman.kukde@fht.org
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --output=fit_%j.log
#SBATCH --mem=16GB
source ~/.bashrc
export PATH="/home/aman.kukde/miniforge3/bin:$PATH"
conda activate StateSpace
cd /home/aman.kukde/Projects/FourierImageTransformer/

declare -a params

# Read input parameters from file
while IFS= read -r line; do
    # Skip empty lines and lines starting with '#' (comments)
    if [[ -n $line && $line != "#"* ]]; then
        params+=("$line")
    fi
done < /home/aman.kukde/Projects/FourierImageTransformer/config.txt

python3 /home/aman.kukde/Projects/FourierImageTransformer/main.py "${params[@]}"


























































































