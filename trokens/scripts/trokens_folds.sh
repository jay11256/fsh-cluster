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
# FIXED HYPERPARAMETERS -- equivalent to: sbatch trokens_exp.sh 5 3 <PT_DATA> both
#
# PT_DATA is the one free argument (default sam3, i.e. exactly what this script
# did before it took an argument, so existing OUTPUT_DIRs and AUTO_RESUME
# checkpoints keep resolving unchanged):
#
#   sbatch trokens_folds.sh          # SAM3 keypoints  -> models/ds12_06_5fold
#   sbatch trokens_folds.sh none     # NO keypoints    -> models/ds12_06_5fold_none
#
# "none" is the keypoint ablation: POINT_INFO.ENABLE=False makes the backbone a
# plain uniform DINO patch grid with no trajectory tokens, so `patch_x` becomes
# (B, T, 256, D) instead of (B, T, 18, D). Everything downstream pools over that
# axis and is count-agnostic, so the dumped feature dimension stays 768 and the
# stride stays 0.25s -- the resulting bank is drop-in for FishFormer.
# ---------------------------------------------------------------------------
N_WAY=5
K_SHOT=3
PT_DATA=${1:-sam3}
MODE=both

case "$PT_DATA" in
	none|trokens|sam3|sam3p8|sam3p2|sam3black) ;;
	*) echo "Error: Invalid PT_DATA '$PT_DATA'. Must be none|trokens|sam3|sam3p8|sam3p2|sam3black"; exit 1 ;;
esac

# Cache is READ but never written when FRAME_CACHE_READONLY=1. The cache key is
# (clip, total_frames, frame indices) only -- independent of point data -- so
# these runs hit the existing ds12_06_5x3 entries and add nothing to a
# filesystem already at 92%. Export FRAME_CACHE_READONLY=1 when submitting.
export FRAME_CACHE_ENABLE=True

case $PT_DATA in
	"none")
		POINT_INFO_ENABLE=False
		# NOT trokens_exp.sh's ds6 cotracker path: that dump holds 2,515 pkls
		# all named 080225_spawn_B1-5_clipNNN.pkl, not one ds12_06 clip, and
		# base_ds.py:160 pickle.loads a pkl for EVERY clip regardless of
		# POINT_INFO.ENABLE -- so it is an immediate FileNotFoundError here.
		# Its points are also 192, which would not match the 256-patch grid
		# that ENABLE=False produces (pointformer.py:492-499 adds the
		# point-derived motion features to the patch-grid features).
		#
		# tools/make_grid_pkls.py writes the substitute: a static 16x16=256
		# uniform grid over the 1280x720 frame, constant across frames. Points
		# exist and count 256 so both constraints are satisfied, the model
		# never grid-samples at them (ENABLE=False -> pointformer.py:423
		# else-branch), and a grid that does not move gives the HOD and
		# cross-motion modules zero displacement. No trajectory information
		# reaches the model from any path.
		TROKENS_PT_DATA="/fs/vulcan-projects/fsh_track/processed_data/gridpklds12_06"
		export NUM_POINTS_TO_SAMPLE=256
		;;
	"trokens")
		POINT_INFO_ENABLE=True
		TROKENS_PT_DATA="/fs/vulcan-projects/fsh_track/processed_data/cotrackpklds6/cotracker3_bip_fr_32_fps_10/fshdata/feat_dump/"
		export NUM_POINTS_TO_SAMPLE=256
		;;
	"sam3")
		POINT_INFO_ENABLE=True
		TROKENS_PT_DATA="/fs/vulcan-projects/fsh_track/processed_data/sam3pklds12_06"
		export NUM_POINTS_TO_SAMPLE=18
		;;
	"sam3black")
		# APPEARANCE ablation: identical to "sam3" -- the real 18-point SAM3
		# pkls, POINT_INFO enabled -- but BLACK_FRAMES=1 zeroes the pixels in
		# the dataloader. DINO on a uniform frame is spatially constant, so no
		# appearance survives; the model keeps point positions (via the
		# positional embedding, added before grid_sample) and the HOD /
		# cross-motion features computed from pred_tracks. This is the exact
		# complement of "none", which keeps appearance and destroys trajectory.
		POINT_INFO_ENABLE=True
		TROKENS_PT_DATA="/fs/vulcan-projects/fsh_track/processed_data/sam3pklds12_06"
		export NUM_POINTS_TO_SAMPLE=18
		export BLACK_FRAMES=1
		;;
	"sam3p8"|"sam3p2")
		# Point-density ablation. Same SAM3 masks, fewer sampled points:
		# tools/subsample_sam3_points.py keeps the 3x3 grid corners (8 total)
		# or the grid centre (2 total) per fish, object-major ordering intact.
		# The model factorizes the count as [grid, n/grid] with grid the middle
		# divisor, so 8 -> 4x2 and 2 -> 2x1 are both valid layouts.
		POINT_INFO_ENABLE=True
		NPTS=${PT_DATA#sam3p}
		TROKENS_PT_DATA="/fs/vulcan-projects/fsh_track/processed_data/sam3pklds12_06_p${NPTS}"
		export NUM_POINTS_TO_SAMPLE=$NPTS
		;;
esac

# Frame cache is keyed by clip content, which is identical across folds (only
# the train/test label per video changes) -- shared across all 5 fold jobs.
# Deliberately shared across PT_DATA settings too: _frame_cache_path keys on
# (clip stem, total_frames, frame indices) only -- see trokens/datasets/utils.py
# -- so it is independent of whether point info is enabled and of how many
# points are sampled. Reusing it saves re-decoding every clip for the ablation.
export FRAME_CACHE_DIR=${FRAME_CACHE_DIR:-/fs/vulcan-projects/fsh_track/processed_data/frame_cache/ds12_06_5x3}


source ~/miniconda3/bin/activate
conda init

conda config --add envs_dirs /fs/vulcan-projects/fsh_track/envs/
conda activate trokens

export CONFIG_TO_USE=fshdata
# sam3 keeps the historical EXP_NAME so the existing 5 fold checkpoints in
# models/ds12_06_5fold/ stay exactly where every downstream dump expects them;
# other PT_DATA settings get their own tree.
if [ "$PT_DATA" = "sam3" ]; then
	export EXP_NAME=ds12_06_5fold
else
	export EXP_NAME=ds12_06_5fold_${PT_DATA}
fi
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
