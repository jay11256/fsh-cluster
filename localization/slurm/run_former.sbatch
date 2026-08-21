#!/bin/bash
#SBATCH --job-name=fishformer
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=former-%j.out
#SBATCH --error=former-%j.out
set -e
# Under SLURM, ${BASH_SOURCE[0]} names the scheduler's staged copy of this
# script in its spool dir, NOT the file in the repo, so deriving the repo root
# from it lands somewhere unwritable. SLURM_SUBMIT_DIR is where sbatch was
# invoked -- which must be localization/, the same assumption the relative
# "#SBATCH --output=logs/..." paths above already make. Falls back to the
# script's own location when run directly with bash.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")}"
[ -d fishformer ] || { echo "ERROR: submit from the localization/ directory" >&2; exit 1; }
echo "=== node $(hostname) label=${FT_LABEL} args=${FT_ARGS} ==="
/fs/vulcan-projects/fsh_track/envs/trokens/bin/python train_former.py --fold all --label "${FT_LABEL}" ${FT_ARGS}
echo "=== FISHFORMER DONE ==="
