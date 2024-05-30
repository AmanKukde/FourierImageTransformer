#!/bin/bash

# SSH variables
remote_user="aman.kukde"
remote_host="hpclogin.fht.org"
# remote_script_path="~/Projects/FourierImageTransformer/bash"  # Path to the script on the remote server
remote_script_path="~/batches"  # Path to the script on the remote server

# Local variables
bash_script="inference.sbatch"  # Name of your bash script

# Submit the bash script using sbatch via SSH
ssh "$remote_user@$remote_host" "sbatch $remote_script_path/$bash_script"
