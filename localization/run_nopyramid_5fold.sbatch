#!/bin/bash
#SBATCH --job-name=noPyr2
#SBATCH --array=0-4
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/logs/noPyr2_%a_%j.out
#SBATCH --error=/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/logs/noPyr2_%a_%j.out
set -e
cd /fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal
PY=/fs/vulcan-projects/fsh_track/envs/trokens/bin/python

# ---------------------------------------------------------------------------
# Clean re-run of n_levels=1 (NoPyramid) on the ds12_06_5fold protocol.
#
# Motivation: levels 1-3 provably receive ZERO positive assignments under
# span_s=4.0 at stride 0.25 (verified on 64 real training windows: level 0
# got 7323 positives, levels 1/2/3 got 0), because each level's accept-range
# in seconds is [0,4]/[8,16]/[32,64]/[128,inf] while a fixed-width 4s
# pseudo-segment always yields max(left,right) in [2,4]s. The 3 dead levels
# cost 5.3M params (25% of the model) and still emit untrained candidates at
# inference that NMS must suppress.
#
# An existing NoPyramid result (42.58% vs the 4-level 40.43%) already hints
# the pyramid is harmful, but it predates --save-ckpt and was not verified
# under the current inference path, so it is not on identical footing. This
# re-run fixes that: EVERY flag below is byte-identical to
# run_former_5fold_ckpt.sbatch (the 40.43% control) except --n-levels 1, and
# --save-ckpt is on so the result can be re-decoded post-hoc the same way the
# 4-level checkpoints were (dump_former_spans_from_ckpt.py, decode_variants.py).
#
# Note on assign_targets under n_levels=1: level 0 becomes the LAST level, so
# its `hi` goes from reg_max*2^0/8 (=16 steps = 4s) to 1e9. Since a 4s span
# yields max(left,right) <= 16 steps anyway, the positive set is unchanged --
# verified separately before submitting, so this stays a clean A/B on the
# dead levels rather than a change in supervision.
#
# Per-fold --out files avoid the read-modify-write race a shared results json
# would create across 5 concurrent array tasks.
# ---------------------------------------------------------------------------
F=$SLURM_ARRAY_TASK_ID
FOLD_TEST=(
  "25-05-22-Run1-Sham-Cir,25-06-26-Run1-VetBond-NoCir,25-07-21-Run1-Vetbond-Cir,25-07-21-Run2-Sham-NoCir"
  "25-06-26-Run2-VetBond-NoCir,25-07-18-Run1-Sham-Cir,25-08-07-Run1-Vetbond-Cir,25-07-24-Run2-Sham-NoCir"
  "25-07-18-Run2-Vetbond-NoCir,25-08-06-Run1-Sham-Cir,25-08-08-Run1-Vetbond-Cir"
  "25-08-13-Run1-Sham-Cir,25-08-26-Run1-Vetbond-Cir,25-08-01-Run1-Vetbond-NoCir"
  "25-05-21-Run1-Sham-NoCir,25-08-14-Run1-Sham-Cir,25-08-15-Run1-Vetbond-NoCir"
)

echo "=== NoPyramid re-run: n_levels=1  fold=$F ==="
$PY train_former.py \
  --fold "${FOLD_TEST[$F]}" \
  --feature-mode "ds12_06_5fold_fold${F}" \
  --window-s 90.0 --span-s 4.0 --n-levels 1 --reg-bins 16 \
  --hidden 384 --depth 6 --num-heads 8 --drop 0.1 --drop-path 0.1 \
  --lamb-cls 1.0 --lamb-reg 1.0 --lamb-ctr 0.5 \
  --lr 0.0005 --wd 0.05 --clip 1.0 --epochs 100 --eval-every 5 \
  --batch-size 16 --windows-per-rec 64 --num-workers 2 --seed 0 \
  --score-thresh 0.05 --nms-iou 0.45 --max-props 400 --gt-window 4.0 \
  --save-ckpt \
  --out "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal/nopyr/f${F}.json" \
  --label "Ds12_06_5fold_fold${F}_Former2_NoPyr2"
echo "=== done fold $F ==="
