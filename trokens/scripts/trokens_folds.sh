#!/bin/sh
#SBATCH --job-name=ds12_06_5fold
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --qos=high
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=../trial_run_outputs/ds12_06_5fold_%a_%j.out
#SBATCH --error=../trial_run_outputs/ds12_06_5fold_%a_%j.out
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT

# ''' USAGE
# 5-fold cross-validation on dataset12_06, leave-videos-out per fold.
# Fixed hyperparameters below are equivalent to:
#   sbatch trokens_exp.sh 5 3 sam3 both
# but instead of the dataset's baked-in clip-level 80/20 split, each of the
# 5 folds holds out a distinct group of ~3-4 of the 17 source videos
# entirely (never seen in training), rotating so every video is tested on
# exactly once across the 5 array tasks.
#
# Run this directly from the scripts folder:
#   sbatch trokens_folds.sh
# This submits ONE array job with 5 tasks (SLURM_ARRAY_TASK_ID 0..4), each
# training/testing its own model on its own fold IN PARALLEL as a separate
# GPU allocation.
#
# Fold CSVs (dataset12_06_fold0.csv .. fold4.csv) must exist first -- build
# them with:
#   python tools/make_fold_csvs.py
# from the trokens/ directory. That script partitions the 17 videos into 5
# folds stratified by treatment x circling condition and writes the CSVs to
# trokens/data_splits/dataset12_06_folds/.
# '''

FOLD_CSV_DIR="/fs/vulcan-projects/fsh_track/bhargav/fsh-cluster/trokens/data_splits/dataset12_06_folds"
FOLD=$SLURM_ARRAY_TASK_ID
FOLD_CSV="$FOLD_CSV_DIR/dataset12_06_fold${FOLD}.csv"

if [ ! -f "$FOLD_CSV" ]; then
	echo "Error: fold CSV not found at $FOLD_CSV"
	echo "Generate it first: cd .. && python tools/make_fold_csvs.py"
	exit 1
fi

# ---------------------------------------------------------------------------
# FIXED HYPERPARAMETERS -- equivalent to: sbatch trokens_exp.sh 5 3 sam3 both
# ---------------------------------------------------------------------------
N_WAY=5
K_SHOT=3
PT_DATA=sam3
MODE=both

export FRAME_CACHE_ENABLE=True

POINT_INFO_ENABLE=True
TROKENS_PT_DATA="/fs/vulcan-projects/fsh_track/processed_data/sam3pklds12_06"
export NUM_POINTS_TO_SAMPLE=18
# Frame cache is keyed by clip content, which is identical across folds (only
# the train/test label per video changes) -- shared across all 5 fold jobs.
export FRAME_CACHE_DIR=${FRAME_CACHE_DIR:-/fs/vulcan-projects/fsh_track/processed_data/frame_cache/ds12_06_5x3}


source ~/miniconda3/bin/activate
conda init

conda config --add envs_dirs /fs/vulcan-projects/fsh_track/envs/
conda activate trokens

export CONFIG_TO_USE=fshdata
export EXP_NAME=ds12_06_5fold
export SECONDARY_EXP_NAME="fold${FOLD}_${N_WAY}_way-${K_SHOT}_shot-${PT_DATA}-${MODE}"
export TORCH_HOME=/fs/vulcan-projects/fsh_track/programs/trokens_workspace/trokens/torch_home
export DATA_DIR=/fs/vulcan-projects/fsh_track/processed_data/dataset12_06
export BASE_OUTPUT_DIR=/fs/vulcan-projects/fsh_track/models
export OUTPUT_DIR=$BASE_OUTPUT_DIR/$EXP_NAME/$SECONDARY_EXP_NAME
export NUM_CLASSES=7
export FILTER_ONE=True

TRAIN_ENABLE=True
TEST_ENABLE=True
export NUM_GPUS=1
export NUM_WORKERS=12
export MASTER_PORT=$(( ( RANDOM % 64511 ) + 1024 ))
export POINT_INFO_NAME="cotracker3_bip_fr_32"
#set wandb id to random 8 character string
export WANDB_ID="${EXP_NAME}_${SECONDARY_EXP_NAME}_"$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)

mkdir -p $OUTPUT_DIR

cd ..

echo "=== Fold worker: fold=$FOLD  csv=$FOLD_CSV ==="

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
	MODEL.NUM_CLASSES $NUM_CLASSES
