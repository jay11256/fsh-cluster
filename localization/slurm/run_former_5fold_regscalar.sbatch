#!/bin/bash
#SBATCH --job-name=formerRS
#SBATCH --array=0-4
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/formerRS_%a_%j.out
#SBATCH --error=logs/formerRS_%a_%j.out
set -e
# See run_former_5fold_ckpt.sbatch for why this uses SLURM_SUBMIT_DIR.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")}"
[ -d fishformer ] || { echo "ERROR: submit from the localization/ directory" >&2; exit 1; }
PY=/fs/vulcan-projects/fsh_track/envs/trokens/bin/python

# ABLATION: scalar boundary regression (--reg-bins 0) instead of the default
# distributional/binned one (--reg-bins 16).
#
# former.py argues the binned head is not a refinement but a match to the
# annotation: BORIS gives a timestamp, not a start/end, so the true extent is
# unknown even to the annotator and a distribution over each offset represents
# that better than a point estimate. That claim has never been tested. This is
# the control.
#
# EVERY other flag is byte-identical to run_former_5fold_ckpt.sbatch, so the
# only difference is the head's final conv (384->2 scalars vs 384->2x16 bin
# logits reduced by expectation) -- 15.776M params vs 15.787M, an 11K gap, so a
# difference in mAP is about the parameterisation of the boundary, not capacity.
# Same few-shot per-fold feature bank, same single-scale trunk, same fixed
# multi-hot / nearest-centre target assignment.
#
# Results go to results/regscalar/ so they cannot collide with the baseline in
# results/; merge with:
#   python3 analysis/merge_folds.py --results results/regscalar
#
# No --save-ckpt: this run exists to produce a number, and 17 x 63MB of weights
# for an ablation is not worth the disk.
F=$SLURM_ARRAY_TASK_ID
mkdir -p results/regscalar
FOLD_TEST=(
  "25-05-22-Run1-Sham-Cir,25-06-26-Run1-VetBond-NoCir,25-07-21-Run1-Vetbond-Cir,25-07-21-Run2-Sham-NoCir"
  "25-06-26-Run2-VetBond-NoCir,25-07-18-Run1-Sham-Cir,25-08-07-Run1-Vetbond-Cir,25-07-24-Run2-Sham-NoCir"
  "25-07-18-Run2-Vetbond-NoCir,25-08-06-Run1-Sham-Cir,25-08-08-Run1-Vetbond-Cir"
  "25-08-13-Run1-Sham-Cir,25-08-26-Run1-Vetbond-Cir,25-08-01-Run1-Vetbond-NoCir"
  "25-05-21-Run1-Sham-NoCir,25-08-14-Run1-Sham-Cir,25-08-15-Run1-Vetbond-NoCir"
)

echo "=== fold $F  reg-bins 0 (scalar)  feature_mode=ds12_06_5fold_fold${F} ==="
$PY -m fishformer.train_former \
  --fold "${FOLD_TEST[$F]}" \
  --feature-mode "ds12_06_5fold_fold${F}" \
  --window-s 90.0 --span-s 4.0 --reg-bins 0 \
  --hidden 384 --depth 6 --num-heads 8 --drop 0.1 --drop-path 0.1 \
  --lamb-cls 1.0 --lamb-reg 1.0 --lamb-ctr 0.5 \
  --lr 0.0005 --wd 0.05 --clip 1.0 --epochs 100 --eval-every 5 \
  --batch-size 16 --windows-per-rec 64 --num-workers 2 --seed 0 \
  --score-thresh 0.05 --nms-iou 0.45 --max-props 400 --gt-window 4.0 \
  --out "results/regscalar/fold${F}.json" \
  --label "Ds12_06_5fold_fold${F}_Former2_RegScalar"
echo "=== fold $F done ==="
