#!/bin/sh

#Test script for slurm on FSH page
#BDS 4/28/25

#SBATCH --job-name=Test-script
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH --qos=high
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --time=540:00
#SBATCH -o ./test_script_output.txt
#SBATCH -e ./test_script_output.txt

#WRITE SCRIPT CODE HERE
echo "Hello World!"
