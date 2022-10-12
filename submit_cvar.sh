#!/bin/bash

#SBATCH --job-name=CVAR    # Submit a job named "example"
#SBATCH --gres=gpu:1             # Use 1 GPU
#SBATCH --time=0-12:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=32768              # cpu memory size
#SBATCH --cpus-per-task=64       # cpu 개수
#SBATCH --output=./slurm_log/S-%x.%j.out

eval "$(conda shell.bash hook)"
conda activate py38

COMMAND="bash run_cvar.sh"
srun $COMMAND
