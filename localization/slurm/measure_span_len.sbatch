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
# See run_former_5fold_ckpt.sbatch for why this uses SLURM_SUBMIT_DIR rather
# than ${BASH_SOURCE[0]}: SLURM stages the script into its own spool directory.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")}"
[ -d fishformer ] || { echo "ERROR: submit from the localization/ directory" >&2; exit 1; }
/fs/vulcan-projects/fsh_track/envs/trokens/bin/python analysis/measure_span_len.py
