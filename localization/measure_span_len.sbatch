#!/bin/bash
#SBATCH --job-name=af_spanlen
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/logs/spanlen_%j.out
#SBATCH --error=/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/logs/spanlen_%j.out
set -e
/fs/vulcan-projects/fsh_track/envs/trokens/bin/python /fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/measure_span_len.py
