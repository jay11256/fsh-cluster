#!/bin/bash
#SBATCH --job-name=formerNP
#SBATCH --array=0-4
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/logs/formerNP_%a_%j.out
#SBATCH --error=/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/logs/formerNP_%a_%j.out
set -e
cd /fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal
PY=/fs/vulcan-projects/fsh_track/envs/trokens/bin/python

# n_levels=1 ablation: same 5 leave-out-backbone folds / same config as the
# "FishFormer (new final results)" (Ds12_06_5fold_fold*_Former2) 5-fold run,
# only --n-levels 1 changed (single-scale trunk, no FPN pyramid -- see
# former.py's level_blocks / point_model.py docstring).
F=$SLURM_ARRAY_TASK_ID
FOLD_TEST=(
  "25-05-22-Run1-Sham-Cir,25-06-26-Run1-VetBond-NoCir,25-07-21-Run1-Vetbond-Cir,25-07-21-Run2-Sham-NoCir"
  "25-06-26-Run2-VetBond-NoCir,25-07-18-Run1-Sham-Cir,25-08-07-Run1-Vetbond-Cir"
  "25-07-18-Run2-Vetbond-NoCir,25-08-06-Run1-Sham-Cir,25-08-08-Run1-Vetbond-Cir"
  "25-08-13-Run1-Sham-Cir,25-08-26-Run1-Vetbond-Cir"
  "25-05-21-Run1-Sham-NoCir,25-08-14-Run1-Sham-Cir,25-08-15-Run1-Vetbond-NoCir"
)

echo "=== fold $F  feature_mode=ds12_06_5fold_fold${F} ==="
$PY train_former.py \
  --fold "${FOLD_TEST[$F]}" \
  --feature-mode "ds12_06_5fold_fold${F}" \
  --window-s 90.0 --span-s 4.0 --n-levels 1 --reg-bins 16 \
  --hidden 384 --depth 6 --num-heads 8 --drop 0.1 --drop-path 0.1 \
  --lamb-cls 1.0 --lamb-reg 1.0 --lamb-ctr 0.5 \
  --lr 0.0005 --wd 0.05 --clip 1.0 --epochs 100 --eval-every 5 \
  --batch-size 16 --windows-per-rec 64 --num-workers 2 --seed 0 \
  --score-thresh 0.05 --nms-iou 0.45 --max-props 400 --gt-window 4.0 \
  --label "Ds12_06_5fold_fold${F}_Former2_NoPyramid"
echo "=== fold $F done ==="
