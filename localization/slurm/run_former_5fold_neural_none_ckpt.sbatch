#!/bin/bash
#SBATCH --job-name=formerNeuralNoneCK
#SBATCH --array=0-4
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/formerNeuralNoneCK_%a_%j.out
#SBATCH --error=logs/formerNeuralNoneCK_%a_%j.out
set -e
cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"   # repo localization/ root
PY=/fs/vulcan-projects/fsh_track/envs/trokens/bin/python

# NEURAL-backbone twin of run_former_5fold_none_ckpt.sbatch -- same 5 leave-
# out-backbone folds/hyperparameters as run_former_5fold_neural_ckpt.sbatch,
# --feature-mode swapped to the neural trunk (ds12_06_5fold_neural_fold${F}),
# plus --none-ratio 0.6, same as the few-shot-trunk None run. --label
# suffixed _Neural_None so it lands as its own entry in fishtal_results.json
# alongside (not overwriting) the existing _Neural and _None entries.
F=$SLURM_ARRAY_TASK_ID
FOLD_TEST=(
  "25-05-22-Run1-Sham-Cir,25-06-26-Run1-VetBond-NoCir,25-07-21-Run1-Vetbond-Cir,25-07-21-Run2-Sham-NoCir"
  "25-06-26-Run2-VetBond-NoCir,25-07-18-Run1-Sham-Cir,25-08-07-Run1-Vetbond-Cir,25-07-24-Run2-Sham-NoCir"
  "25-07-18-Run2-Vetbond-NoCir,25-08-06-Run1-Sham-Cir,25-08-08-Run1-Vetbond-Cir"
  "25-08-13-Run1-Sham-Cir,25-08-26-Run1-Vetbond-Cir,25-08-01-Run1-Vetbond-NoCir"
  "25-05-21-Run1-Sham-NoCir,25-08-14-Run1-Sham-Cir,25-08-15-Run1-Vetbond-NoCir"
)

echo "=== fold $F  feature_mode=ds12_06_5fold_neural_fold${F}  none_ratio=0.6 ==="
$PY -m fishformer.train_former \
  --fold "${FOLD_TEST[$F]}" \
  --feature-mode "ds12_06_5fold_neural_fold${F}" \
  --window-s 90.0 --span-s 4.0 --reg-bins 16 \
  --hidden 384 --depth 6 --num-heads 8 --drop 0.1 --drop-path 0.1 \
  --lamb-cls 1.0 --lamb-reg 1.0 --lamb-ctr 0.5 \
  --lr 0.0005 --wd 0.05 --clip 1.0 --epochs 100 --eval-every 5 \
  --batch-size 16 --windows-per-rec 64 --num-workers 2 --seed 0 \
  --none-ratio 0.6 \
  --score-thresh 0.05 --nms-iou 0.45 --max-props 400 --gt-window 4.0 \
  --save-ckpt --ckpt-dir checkpoints \
  --label "Ds12_06_5fold_fold${F}_Former2_Neural_None"
echo "=== fold $F done ==="
