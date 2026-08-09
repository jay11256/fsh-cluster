#!/bin/bash
#SBATCH --job-name=confmatml
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/confmatml_%j.out
#SBATCH --error=logs/confmatml_%j.out
set -e
cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"   # repo localization/ root
PY=/fs/vulcan-projects/fsh_track/envs/trokens/bin/python
$PY analysis/confusion_matrix_fishformer.py --multilabel
