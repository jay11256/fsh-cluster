#!/bin/bash
#SBATCH --job-name=pointCK
#SBATCH --array=0-4
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/logs/pointCK_%a_%j.out
#SBATCH --error=/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/logs/pointCK_%a_%j.out
set -e
cd /fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal
PY=/fs/vulcan-projects/fsh_track/envs/trokens/bin/python

# Retrain of the main (pyramid) FishPoint 5-fold result -- SAME config/label
# as the existing Ds12_06_5fold_fold*_Point entries; --save-ckpt was already
# supported but not passed before, and per-class point-AP is now captured in
# the results json. Overwrites those entries with an equivalent run that
# also has checkpoints + per-behavior breakdown.
F=$SLURM_ARRAY_TASK_ID
FOLD_TEST=(
  "25-05-22-Run1-Sham-Cir,25-06-26-Run1-VetBond-NoCir,25-07-21-Run1-Vetbond-Cir,25-07-21-Run2-Sham-NoCir"
  "25-06-26-Run2-VetBond-NoCir,25-07-18-Run1-Sham-Cir,25-08-07-Run1-Vetbond-Cir,25-07-24-Run2-Sham-NoCir"
  "25-07-18-Run2-Vetbond-NoCir,25-08-06-Run1-Sham-Cir,25-08-08-Run1-Vetbond-Cir"
  "25-08-13-Run1-Sham-Cir,25-08-26-Run1-Vetbond-Cir,25-08-01-Run1-Vetbond-NoCir"
  "25-05-21-Run1-Sham-NoCir,25-08-14-Run1-Sham-Cir,25-08-15-Run1-Vetbond-NoCir"
)

echo "=== fold $F  feature_mode=ds12_06_5fold_fold${F} ==="
$PY train_point.py \
  --fold "${FOLD_TEST[$F]}" \
  --feature-mode "ds12_06_5fold_fold${F}" \
  --window-s 90.0 --n-levels 4 --use-offset \
  --hidden 384 --depth 6 --num-heads 8 --drop 0.1 --drop-path 0.1 \
  --cls-half 1.0 --sigma 1.0 --lamb-cls 1.0 --lamb-act 1.0 --lamb-off 1.0 \
  --lr 0.0005 --wd 0.05 --clip 1.0 --epochs 100 --eval-every 5 \
  --batch-size 16 --windows-per-rec 64 --num-workers 2 --seed 0 \
  --thresh 0.05 --min-sep-s 1.0 --tolerance 1.0 \
  --save-ckpt \
  --label "Ds12_06_5fold_fold${F}_Point"
echo "=== fold $F done ==="
