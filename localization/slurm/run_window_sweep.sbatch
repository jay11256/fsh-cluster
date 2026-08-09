#!/bin/bash
#SBATCH --job-name=winSweep
#SBATCH --array=0-14
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/winSweep_%a_%j.out
#SBATCH --error=logs/winSweep_%a_%j.out
set -e
cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"   # repo localization/ root
PY=/fs/vulcan-projects/fsh_track/envs/trokens/bin/python

# ---------------------------------------------------------------------------
# Ablate FishFormer's --window-s (the 90s context window), which was never
# swept: 83 of 85 labels in fishtal_results.json use 90.0 and the only two at
# 180.0 are an older architecture (FishTAL5/11, hidden=256/depth=4), so no
# controlled comparison exists.
#
# w=90 is NOT re-run here -- the existing Ds12_06_5fold_fold{F}_Former2
# entries are already exactly this config (windows_per_rec=64, epochs=100,
# window_s=90) and serve as the control point (40.4% avg mAP .1-.7).
#
# NORMALIZATION: windows_per_rec is scaled as round(64 * 90/w) so every
# setting sees the SAME number of seconds of video per epoch (5760 s/rec).
# Consequence, stated up front: smaller windows therefore get proportionally
# MORE gradient steps per epoch (batch_size is fixed at 16), so this
# normalization mildly favors the shorter-window settings. That asymmetry is
# deliberate -- if 90s still wins under it, the conclusion is robust; if a
# shorter window wins, the extra updates are a confound to note.
#
# Each (window, fold) writes its OWN results json to avoid the read-modify-
# write race that a shared --out would create across 15 concurrent array
# tasks; merge_window_sweep.py combines them afterwards.
# ---------------------------------------------------------------------------
WINDOWS=(30.0 45.0 180.0)
WPR=(192 128 32)

W_IDX=$((SLURM_ARRAY_TASK_ID / 5))
F=$((SLURM_ARRAY_TASK_ID % 5))
WIN=${WINDOWS[$W_IDX]}
NWPR=${WPR[$W_IDX]}
WTAG=$(printf "%.0f" $WIN)

FOLD_TEST=(
  "25-05-22-Run1-Sham-Cir,25-06-26-Run1-VetBond-NoCir,25-07-21-Run1-Vetbond-Cir,25-07-21-Run2-Sham-NoCir"
  "25-06-26-Run2-VetBond-NoCir,25-07-18-Run1-Sham-Cir,25-08-07-Run1-Vetbond-Cir,25-07-24-Run2-Sham-NoCir"
  "25-07-18-Run2-Vetbond-NoCir,25-08-06-Run1-Sham-Cir,25-08-08-Run1-Vetbond-Cir"
  "25-08-13-Run1-Sham-Cir,25-08-26-Run1-Vetbond-Cir,25-08-01-Run1-Vetbond-NoCir"
  "25-05-21-Run1-Sham-NoCir,25-08-14-Run1-Sham-Cir,25-08-15-Run1-Vetbond-NoCir"
)

echo "=== window_s=$WIN  windows_per_rec=$NWPR  fold=$F ==="
$PY -m fishformer.train_former \
  --fold "${FOLD_TEST[$F]}" \
  --feature-mode "ds12_06_5fold_fold${F}" \
  --window-s "$WIN" --span-s 4.0 --reg-bins 16 \
  --hidden 384 --depth 6 --num-heads 8 --drop 0.1 --drop-path 0.1 \
  --lamb-cls 1.0 --lamb-reg 1.0 --lamb-ctr 0.5 \
  --lr 0.0005 --wd 0.05 --clip 1.0 --epochs 100 --eval-every 5 \
  --batch-size 16 --windows-per-rec "$NWPR" --num-workers 2 --seed 0 \
  --score-thresh 0.05 --nms-iou 0.45 --max-props 400 --gt-window 4.0 \
  --out "winsweep/w${WTAG}_f${F}.json" \
  --label "Ds12_06_5fold_fold${F}_Former2_Win${WTAG}"
echo "=== done window_s=$WIN fold=$F ==="
