#!/bin/bash
#SBATCH --job-name=confmatml
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/logs/confmatml_%j.out
#SBATCH --error=/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/logs/confmatml_%j.out
set -e
cd /fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal
PY=/fs/vulcan-projects/fsh_track/envs/trokens/bin/python
$PY confusion_matrix_fishformer.py --multilabel
