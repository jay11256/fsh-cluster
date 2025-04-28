#!/bin/sh

#Test script for slurm on FSH page
#BDS 4/28/25

#SBATCH --job-name=Test-script
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --qos=high
#SBATCH --partition=tron
#SBATCH --time=540:00
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT
#SBATCH --mail-user=bds062@terpmail.umd.edu
#SBATCH -o ./test_script_output.txt
#SBATCH -e ./test_script_output.txt

#WRITE SCRIPT CODE HERE
echo "Hello World!"
