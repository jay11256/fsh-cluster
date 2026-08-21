#!/bin/bash
#SBATCH --job-name=winSmall
#SBATCH --array=0-9
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/winSmall_%a_%j.out
#SBATCH --error=logs/winSmall_%a_%j.out
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

# ---------------------------------------------------------------------------
# Extends run_window_sweep.sbatch DOWNWARD to the behavior timescale.
#
# The first sweep {30,45,180} was anchored around the incumbent 90s (/3, /2,
# x2), which is the wrong reference point for the actual question: behaviors
# run ~0.5-5s and every GT pseudo-segment is exactly span_s=4.0s wide, so the
# informative regime is single-digit seconds, not tens.
#
# Floor: a window must exceed the 4s span for a position to be a positive at
# all (assign_targets requires left>0 AND right>0, i.e. the span straddles
# the position). 8s = 2x the span is about as tight as is meaningful; below
# that the usable interior collapses. So this adds {8, 15}, giving a full
# sweep of {8, 15, 30, 45, 90*, 180} = 2x .. 45x the span width
# (* = existing Ds12_06_5fold_fold{F}_Former2 runs, the control).
#
# windows_per_rec is scaled as round(64 * 90/w) exactly as in the first
# sweep, so seconds-of-video per epoch is constant (5760 s/rec). A useful
# consequence: total TOKENS processed is then also constant across settings,
# so wall-clock stays roughly flat even at w=8 (many short sequences instead
# of few long ones) -- attention gets cheaper per sample as the window
# shrinks, offsetting the larger step count. The caveat carried over from the
# first sweep still applies: shorter windows get proportionally more gradient
# steps, which mildly favors them.
# ---------------------------------------------------------------------------
WINDOWS=(8.0 15.0)
WPR=(720 384)

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
