#!/bin/bash
#SBATCH --job-name=dumpspans
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=dumpspans-%j.out
#SBATCH --error=dumpspans-%j.out
set -e
cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"   # repo localization/ root
echo "=== node $(hostname) label=${FT_LABEL} args=${FT_ARGS} ==="
/fs/vulcan-projects/fsh_track/envs/trokens/bin/python dump_former_spans.py --label "${FT_LABEL}" ${FT_ARGS}
echo "=== DUMP DONE ==="
