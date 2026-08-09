#!/bin/bash
#SBATCH --job-name=fishpoint
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/point-%j.out
#SBATCH --error=/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/point-%j.out
set -e
cd /fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal
echo "=== node $(hostname) label=${FT_LABEL} args=${FT_ARGS} ==="
/fs/vulcan-projects/fsh_track/envs/trokens/bin/python train_point.py --fold all --label "${FT_LABEL}" ${FT_ARGS}
echo "=== FISHPOINT DONE ==="
