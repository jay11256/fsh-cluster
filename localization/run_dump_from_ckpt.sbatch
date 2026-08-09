#!/bin/bash
#SBATCH --job-name=dumpFF
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/logs/dumpFF_%j.out
#SBATCH --error=/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/logs/dumpFF_%j.out
set -e
cd /fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal
PY=/fs/vulcan-projects/fsh_track/envs/trokens/bin/python
$PY dump_former_spans_from_ckpt.py
