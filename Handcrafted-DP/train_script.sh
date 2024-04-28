#!/bin/bash
    
# To be submitted to the SLURM queue with the command:
# sbatch batch-submit.sh

# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Set output file destinations (optional)
# By default, output will appear in a file in the submission directory:
# slurm-$job_number.out
# This can be changed:
#SBATCH --output=/u2/s4mokhta/DP-Benchmarks/Handcrafted-DP/script_outputs/JOB%j.out # File to which STDOUT will be written
#SBATCH --error=/u2/s4mokhta/DP-Benchmarks/Handcrafted-DP/script_outputs/JOB%j_err.out # File to which STDERR will be written
 
# Load up your conda environment
# Set up environment on watgpu.cs or in interactive session (use `source` keyword instead of `conda`)
source activate ffcv
 
# Task to run
python3 /u2/s4mokhta/DP-Benchmarks/Handcrafted-DP/cnns.py