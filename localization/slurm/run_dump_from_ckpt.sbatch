#!/bin/bash
#SBATCH --job-name=dumpFF
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dumpFF_%j.out
#SBATCH --error=logs/dumpFF_%j.out
set -e
cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"   # repo localization/ root
PY=/fs/vulcan-projects/fsh_track/envs/trokens/bin/python
$PY analysis/dump_former_spans_from_ckpt.py
