#!/bin/bash
#SBATCH --job-name=af_spanlen
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/spanlen_%j.out
#SBATCH --error=logs/spanlen_%j.out
set -e
/fs/vulcan-projects/fsh_track/envs/trokens/bin/python measure_span_len.py
