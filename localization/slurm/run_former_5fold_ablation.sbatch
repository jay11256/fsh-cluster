#!/bin/bash
#SBATCH --job-name=formerAB
#SBATCH --array=0-4
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/formerAB_%x_%a_%j.out
#SBATCH --error=logs/formerAB_%x_%a_%j.out
set -e
# See run_former_5fold_ckpt.sbatch for why this uses SLURM_SUBMIT_DIR.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")}"
[ -d fishformer ] || { echo "ERROR: submit from the localization/ directory" >&2; exit 1; }
PY=/fs/vulcan-projects/fsh_track/envs/trokens/bin/python

# The paper's ablation table: a full 2x2 FACTORIAL over the two contributions,
# backbone (few-shot vs neural) x classification target (multi-label vs single).
# All four cells, not one-factor-at-a-time, so the table reports both main
# effects AND their interaction -- e.g. whether multi-label supervision pays off
# more on one backbone than the other, which a two-row subtractive table cannot
# show and which is a real possibility here (the few-shot backbone was itself
# trained multi-label, so the two contributions may not be independent).
#
#   sbatch --export=ALL,VARIANT=fewshot_multi  ...   (== results/, already run)
#   sbatch --export=ALL,VARIANT=singlelabel    ...
#   sbatch --export=ALL,VARIANT=neural         ...
#   sbatch --export=ALL,VARIANT=neural_single  ...
#
# ONE launcher rather than one script per cell, so the ~20 shared flags below
# physically cannot drift between cells -- a difference in mAP is then
# attributable to the single line VARIANT changes, and that is the entire claim
# an ablation table makes.
#
# backbone axis:  the neural bank is the same recordings, same fold split, same
#                 backbone architecture; only the backbone's training recipe
#                 differs (FEW_SHOT.DISABLE=True).
# label axis:     single-label puts one class per timestep instead of every
#                 covering class. The winner is the nearest-centre event, i.e.
#                 the SAME event that owns reg/ctr, so reg/ctr/pos targets stay
#                 bit-identical and only the classification target changes.
#
# Cell 1 (few-shot x multi-label) is the existing results/ run from
# run_former_5fold_ckpt.sbatch; VARIANT=fewshot_multi reproduces it if the cell
# ever needs regenerating alongside the others.
: "${VARIANT:?set VARIANT, e.g. fewshot_multi | singlelabel | neural_single_none}"
F=$SLURM_ARRAY_TASK_ID

# VARIANT is COMPOSED from independent tokens rather than enumerated, so the
# three axes stay orthogonal by construction and adding an axis does not mean
# writing 2^n branches that can disagree with each other:
#   contains "neural"  -> neural backbone bank      (else few-shot)
#   contains "single"  -> --single-label-cls        (else multi-hot)
#   contains "none"    -> --none-ratio 0.6          (else uniform windows only)
# Anything not matching a known token is rejected, so a typo like
# "nueral_single" fails loudly instead of silently running the few-shot arm and
# quietly corrupting a table cell.
for tok in ${VARIANT//_/ }; do
  case "$tok" in
    fewshot|few|multi|neural|single|singlelabel|none|dinoraw|nokp) ;;
    mlp2|win30|win60|win180) ;;
    noctr|fixedbox) ;;
    span2|span6) ;;
    sam3p8|sam3p2|sam3black) ;;
    *) echo "ERROR: unrecognised token '$tok' in VARIANT='$VARIANT'" >&2; exit 1 ;;
  esac
done

# Backbone axis. dinoraw is the "no MIL at all" arm and is the one bank that is
# NOT per-fold: raw DINOv2 involves no training, so there is nothing to hold out
# at the feature level and one bank serves all 5 folds. nokp is the MIL trained
# without keypoints (models/ds12_06_5fold_none), per-fold like the others.
case "$VARIANT" in
  *dinoraw*) FEATS="dino_raw_4fps" ;;
  *sam3black*) FEATS="ds12_06_5fold_sam3black_fold${F}" ;;
  *sam3p8*)  FEATS="ds12_06_5fold_sam3p8_fold${F}" ;;
  *sam3p2*)  FEATS="ds12_06_5fold_sam3p2_fold${F}" ;;
  *nokp*)    FEATS="ds12_06_5fold_nokp_fold${F}" ;;
  *neural*)  FEATS="ds12_06_5fold_neural_fold${F}" ;;
  *)         FEATS="ds12_06_5fold_fold${F}" ;;
esac
EXTRA=""
case "$VARIANT" in *single*) EXTRA="$EXTRA --single-label-cls" ;; esac
case "$VARIANT" in *none*)   EXTRA="$EXTRA --none-ratio 0.6" ;; esac

# Capacity: the trunk MLP is ~45% of all parameters (1.18M of 1.77M per block),
# so halving its expansion is the single largest size lever in the model --
# 15.787M -> 12.244M, a bigger cut than removing the feature pyramid was.
case "$VARIANT" in *mlp2*) EXTRA="$EXTRA --mlp-ratio 2.0" ;; esac

# Component removal. Each flag ALSO forces its loss weight to 0 inside
# train_former.main() -- removing a head means both not training it and not
# consuming it at decode. Passing only one half measures untrained-head noise.
case "$VARIANT" in *noctr*)    EXTRA="$EXTRA --no-ctr-score" ;; esac
case "$VARIANT" in *fixedbox*) EXTRA="$EXTRA --fixed-box-decode" ;; esac

# Pseudo-segment width. --span-s sets the TRAINING target width only; the
# evaluation reference stays at --gt-window 4.0 above, deliberately. Sweeping
# both together would move the metric with the model and the numbers would not
# be comparable to any other row in the paper. Note --fixed-box-decode also
# reads span_s, so span+fixedbox combinations change the emitted box width too.
case "$VARIANT" in
  *span2*) EXTRA="$EXTRA --span-s 2.0" ;;
  *span6*) EXTRA="$EXTRA --span-s 6.0" ;;
esac

# Attention context. --window-s changes BOTH the training window and the
# inference sliding window, so the model is trained and evaluated at the same
# context length -- no train/test mismatch.
#
# CONFOUND, stated rather than hidden: windows_per_rec stays 64 across all
# settings, so a longer window means more supervised timesteps per epoch
# (30s->120 steps, 90s->360, 180s->720). Holding gradient steps fixed and
# holding total supervision fixed are mutually exclusive here; this holds
# gradient steps fixed, which is the more standard reading of "same training,
# different context". A context effect and a supervision-volume effect are
# therefore not separable from this sweep alone.
case "$VARIANT" in
  *win30*)  EXTRA="$EXTRA --window-s 30.0" ;;
  *win60*)  EXTRA="$EXTRA --window-s 60.0" ;;
  *win180*) EXTRA="$EXTRA --window-s 180.0" ;;
esac

OUTDIR="results/${VARIANT}"
mkdir -p "$OUTDIR"

FOLD_TEST=(
  "25-05-22-Run1-Sham-Cir,25-06-26-Run1-VetBond-NoCir,25-07-21-Run1-Vetbond-Cir,25-07-21-Run2-Sham-NoCir"
  "25-06-26-Run2-VetBond-NoCir,25-07-18-Run1-Sham-Cir,25-08-07-Run1-Vetbond-Cir,25-07-24-Run2-Sham-NoCir"
  "25-07-18-Run2-Vetbond-NoCir,25-08-06-Run1-Sham-Cir,25-08-08-Run1-Vetbond-Cir"
  "25-08-13-Run1-Sham-Cir,25-08-26-Run1-Vetbond-Cir,25-08-01-Run1-Vetbond-NoCir"
  "25-05-21-Run1-Sham-NoCir,25-08-14-Run1-Sham-Cir,25-08-15-Run1-Vetbond-NoCir"
)

echo "=== fold $F  VARIANT=$VARIANT  feats=$FEATS  extra='$EXTRA' ==="
$PY -m fishformer.train_former \
  --fold "${FOLD_TEST[$F]}" \
  --feature-mode "$FEATS" \
  --window-s 90.0 --span-s 4.0 --reg-bins 16 \
  --hidden 384 --depth 6 --num-heads 8 --drop 0.1 --drop-path 0.1 \
  --lamb-cls 1.0 --lamb-reg 1.0 --lamb-ctr 0.5 \
  --lr 0.0005 --wd 0.05 --clip 1.0 --epochs 100 --eval-every 5 \
  --batch-size 16 --windows-per-rec 64 --num-workers 2 --seed 0 \
  --score-thresh 0.05 --nms-iou 0.45 --max-props 400 --gt-window 4.0 \
  $EXTRA \
  --out "${OUTDIR}/fold${F}.json" \
  --label "Ds12_06_5fold_fold${F}_Former2_${VARIANT}"
echo "=== fold $F done ==="
