#!/bin/sh
#SBATCH --job-name=ds12_06_5fold_neural
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=../trial_run_outputs/ds12_06_5fold_neural.out
#SBATCH --error=../trial_run_outputs/ds12_06_5fold_neural.out
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT

# ''' USAGE
# Sweeps the 5 dataset12_06 folds, EACH FOLD AS ITS OWN SEPARATE sbatch job --
# the "neural" (FEW_SHOT.DISABLE True) counterpart to trokens_folds.sh's
# few-shot 5-fold run. Same dataset12_06 fold CSVs / same architecture flags
# (dino, motion modules, sam3 points) as trokens_folds.sh; the only real
# difference is FEW_SHOT.DISABLE True, which swaps the episodic few-shot
# sampler + Q2S loss for standard classification training over all classes
# (see custom_config.py's FEW_SHOT.DISABLE docstring).
#
# Run this directly from the scripts folder -- do NOT `sbatch` it yourself:
#   sh trokens_sweep.sh
# It submits one sbatch job per fold and exits immediately. The #SBATCH
# header above is only used by the jobs it submits: each of those jobs
# re-runs this same file, which detects SWEEP_WORKER_VALUE is set and trains
# exactly one fold instead of looping/re-submitting.
#
# Fold CSVs (dataset12_06_fold0.csv .. fold4.csv) must exist first -- build
# them with `python tools/make_fold_csvs.py` from the trokens/ directory (see
# trokens_folds.sh) if not already present.
# '''

# ---------------------------------------------------------------------------
# FIXED HYPERPARAMETERS -- edit here for different defaults; no CLI args.
# N_WAY/K_SHOT are kept only for exp-name continuity with the few-shot run;
# FEW_SHOT.DISABLE True below makes them inert (no episode sampler).
# ---------------------------------------------------------------------------
N_WAY=5
K_SHOT=3
PT_DATA=sam3
MODE=both
CACHE_MODE=cache

# ---------------------------------------------------------------------------
# SWEEP CONFIG — one training run (= one sbatch job) per fold ID.
# Each job gets its own OUTPUT_DIR and WANDB_ID so results never collide
# across folds; all 5 share the same frame cache (clip content is identical
# across folds -- only the train/test split changes, see trokens_folds.sh).
# ---------------------------------------------------------------------------
SWEEP_PARAM="FOLD"
SWEEP_VALUES=(0 1 2 3 4)
SWEEP_PARAM_SLUG=$(echo "$SWEEP_PARAM" | tr '[:upper:].' '[:lower:]_')

# ===========================================================================
# LAUNCHER -- runs when this script is invoked directly (SWEEP_WORKER_VALUE
# unset). Submits one independent sbatch job per fold, then exits; does no
# training itself.
# ===========================================================================
if [ -z "$SWEEP_WORKER_VALUE" ]; then
	LOG_DIR=../trial_run_outputs
	mkdir -p "$LOG_DIR"
	for SWEEP_VALUE in "${SWEEP_VALUES[@]}"; do
		SUFFIX="${SWEEP_PARAM_SLUG}${SWEEP_VALUE}"
		echo "Submitting sweep job: $SWEEP_PARAM=$SWEEP_VALUE"
		export SWEEP_WORKER_VALUE=$SWEEP_VALUE
		sbatch \
			--job-name="ds12_06_5fold_neural_${SUFFIX}" \
			--output="$LOG_DIR/ds12_06_5fold_neural_${SUFFIX}_%j.out" \
			--error="$LOG_DIR/ds12_06_5fold_neural_${SUFFIX}_%j.out" \
			"$0"
	done
	exit 0
fi

# ===========================================================================
# WORKER -- everything below only runs inside a submitted job, training
# exactly one fold (SWEEP_WORKER_VALUE, set by the launcher above).
# ===========================================================================
SWEEP_VALUE=$SWEEP_WORKER_VALUE
SUFFIX="${SWEEP_PARAM_SLUG}${SWEEP_VALUE}"
FOLD=$SWEEP_VALUE

FOLD_CSV_DIR="/fs/vulcan-projects/fsh_track/bhargav/fsh-cluster/trokens/data_splits/dataset12_06_folds"
FOLD_CSV="$FOLD_CSV_DIR/dataset12_06_fold${FOLD}.csv"

if [ ! -f "$FOLD_CSV" ]; then
	echo "Error: fold CSV not found at $FOLD_CSV"
	echo "Generate it first: cd .. && python tools/make_fold_csvs.py"
	exit 1
fi

export FRAME_CACHE_ENABLE=True

POINT_INFO_ENABLE=True
TROKENS_PT_DATA="/fs/vulcan-projects/fsh_track/processed_data/sam3pklds12_06"
export NUM_POINTS_TO_SAMPLE=18
# Same frame cache as trokens_folds.sh's few-shot 5-fold run -- decoded frames
# are identical regardless of few-shot vs. neural training, so it's shared
# rather than re-decoded per mode.
export FRAME_CACHE_DIR=${FRAME_CACHE_DIR:-/fs/vulcan-projects/fsh_track/processed_data/frame_cache/ds12_06_5x3}


source ~/miniconda3/bin/activate
conda init

conda config --add envs_dirs /fs/vulcan-projects/fsh_track/envs/
conda activate trokens

export CONFIG_TO_USE=fshdata
export EXP_NAME=ds12_06_5fold_neural
export SECONDARY_EXP_NAME="fold${FOLD}_neural-${PT_DATA}-${MODE}"
export TORCH_HOME=/fs/vulcan-projects/fsh_track/programs/trokens_workspace/trokens/torch_home
export DATA_DIR=/fs/vulcan-projects/fsh_track/processed_data/dataset12_06
export BASE_OUTPUT_DIR=/fs/vulcan-projects/fsh_track/models/
export OUTPUT_DIR=$BASE_OUTPUT_DIR/$EXP_NAME/$SECONDARY_EXP_NAME
export NUM_CLASSES=7
export FILTER_ONE=True

TRAIN_ENABLE=True
TEST_ENABLE=True
export NUM_GPUS=1
export NUM_WORKERS=4
export MASTER_PORT=$(( ( RANDOM % 64511 ) + 1024 ))
export POINT_INFO_NAME="cotracker3_bip_fr_32"
#set wandb id to random 8 character string
export WANDB_ID="${EXP_NAME}_${SECONDARY_EXP_NAME}_"$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)

#export CHECKPOINT_FILE=/fs/vulcan-projects/fsh_track/models/ds6/5_way-3_shot-sam3-both/checkpoints/checkpoint_best.pyth

mkdir -p $OUTPUT_DIR

cd ..

echo "=== Sweep worker: fold=$FOLD (neural, FEW_SHOT.DISABLE=True)  csv=$FOLD_CSV ==="

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
	tools/run_net.py --init_method env:// --new_dist_init \
	--cfg configs/trokens/$CONFIG_TO_USE.yaml \
	WANDB.ID $WANDB_ID \
	WANDB.EXP_NAME "${EXP_NAME}_${SECONDARY_EXP_NAME}" \
	MASTER_PORT $MASTER_PORT \
	OUTPUT_DIR $OUTPUT_DIR \
	NUM_GPUS $NUM_GPUS \
	DATA_LOADER.NUM_WORKERS $NUM_WORKERS \
	DATA_LOADER.DATA_CSV_PATH $FOLD_CSV \
	DATA.USE_RAND_AUGMENT True \
	DATA.PATH_TO_DATA_DIR $DATA_DIR \
	DATA.PATH_TO_TROKEN_PT_DATA $TROKENS_PT_DATA \
	DATA.FRAME_CACHE_DIR $FRAME_CACHE_DIR \
	DATA.FRAME_CACHE_ENABLE $FRAME_CACHE_ENABLE \
	FEW_SHOT.K_SHOT $K_SHOT \
	FEW_SHOT.TRAIN_QUERY_PER_CLASS 6 \
	FEW_SHOT.N_WAY $N_WAY \
	POINT_INFO.ENABLE $POINT_INFO_ENABLE \
	POINT_INFO.NAME $POINT_INFO_NAME \
	POINT_INFO.NUM_POINTS_TO_SAMPLE $NUM_POINTS_TO_SAMPLE \
	MODEL.FEAT_EXTRACTOR dino \
	MODEL.DINO_CONFIG dinov2_vitb14 \
	MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE True \
	MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE True \
    TRAIN.ENABLE $TRAIN_ENABLE \
    TEST.ENABLE $TEST_ENABLE \
	DATA_LOADER.FILTER_ONE $FILTER_ONE \
	MODEL.NUM_CLASSES $NUM_CLASSES \
	FEW_SHOT.DISABLE True
	#TEST.CHECKPOINT_FILE_PATH $CHECKPOINT_FILE
