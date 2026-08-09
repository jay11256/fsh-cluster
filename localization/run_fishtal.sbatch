#!/bin/bash
#SBATCH --job-name=fishtal
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/fishtal-%j.out
#SBATCH --error=/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/fishtal-%j.out

set -e
cd /fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal
PY=/fs/vulcan-projects/fsh_track/envs/trokens/bin/python

echo "=== node $(hostname) label=${FT_LABEL} ==="
echo "=== args: ${FT_ARGS} ==="
$PY train.py --fold all --label "${FT_LABEL}" ${FT_ARGS}
echo "=== FISHTAL DONE ==="
