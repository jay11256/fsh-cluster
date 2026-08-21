#!/bin/bash
#SBATCH --job-name=formerCK
#SBATCH --array=0-4
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/formerCK_%a_%j.out
#SBATCH --error=logs/formerCK_%a_%j.out
set -e
# Under SLURM, ${BASH_SOURCE[0]} names the scheduler's staged copy of this
# script in its spool dir, NOT the file in the repo, so deriving the repo root
# from it lands somewhere unwritable. SLURM_SUBMIT_DIR is where sbatch was
# invoked -- which must be localization/, the same assumption the relative
# "#SBATCH --output=logs/..." paths above already make. Falls back to the
# script's own location when run directly with bash.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")}"
[ -d fishformer ] || { echo "ERROR: submit from the localization/ directory" >&2; exit 1; }
PY=/fs/vulcan-projects/fsh_track/envs/trokens/bin/python

# The main FishFormer 5-fold result: leave-one-recording-out within each of the
# 5 backbone folds, so a fold's recordings were held out of BOTH the Trokens
# backbone and the FishFormer head.
#
# Per-fold --out. train_former.py's results write is a read-modify-write
# (`json.load` -> mutate -> `json.dump`), so pointing all 5 concurrent array
# tasks at one shared file loses whichever folds finish first. Each task writes
# its own file; merge_folds.py combines them afterwards.
F=$SLURM_ARRAY_TASK_ID
mkdir -p results
FOLD_TEST=(
  "25-05-22-Run1-Sham-Cir,25-06-26-Run1-VetBond-NoCir,25-07-21-Run1-Vetbond-Cir,25-07-21-Run2-Sham-NoCir"
  "25-06-26-Run2-VetBond-NoCir,25-07-18-Run1-Sham-Cir,25-08-07-Run1-Vetbond-Cir,25-07-24-Run2-Sham-NoCir"
  "25-07-18-Run2-Vetbond-NoCir,25-08-06-Run1-Sham-Cir,25-08-08-Run1-Vetbond-Cir"
  "25-08-13-Run1-Sham-Cir,25-08-26-Run1-Vetbond-Cir,25-08-01-Run1-Vetbond-NoCir"
  "25-05-21-Run1-Sham-NoCir,25-08-14-Run1-Sham-Cir,25-08-15-Run1-Vetbond-NoCir"
)

echo "=== fold $F  feature_mode=ds12_06_5fold_fold${F} ==="
$PY -m fishformer.train_former \
  --fold "${FOLD_TEST[$F]}" \
  --feature-mode "ds12_06_5fold_fold${F}" \
  --window-s 90.0 --span-s 4.0 --reg-bins 16 \
  --hidden 384 --depth 6 --num-heads 8 --drop 0.1 --drop-path 0.1 \
  --lamb-cls 1.0 --lamb-reg 1.0 --lamb-ctr 0.5 \
  --lr 0.0005 --wd 0.05 --clip 1.0 --epochs 100 --eval-every 5 \
  --batch-size 16 --windows-per-rec 64 --num-workers 2 --seed 0 \
  --score-thresh 0.05 --nms-iou 0.45 --max-props 400 --gt-window 4.0 \
  --save-ckpt \
  --out "results/fold${F}.json" \
  --label "Ds12_06_5fold_fold${F}_Former2"
echo "=== fold $F done ==="
