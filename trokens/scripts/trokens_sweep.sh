#!/bin/sh
#SBATCH --job-name=ds12_04_leave3_sweep_frames
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --qos=high
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=../trial_run_outputs/ds12_04_leave3_sweep_frames.out
#SBATCH --error=../trial_run_outputs/ds12_04_leave3_sweep_frames.out
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT

# ''' USAGE
# Sweeps one config parameter across several values, EACH VALUE AS ITS OWN
# SEPARATE sbatch job. Hyperparameters are fixed below (5-way, 5-shot, sam3
# points, mode=both, cache enabled) -- no CLI arguments needed.
#
# Run this directly from the scripts folder -- do NOT `sbatch` it yourself:
#   sh trokens_sweep.sh
# It submits one sbatch job per sweep value and exits immediately. The
# #SBATCH header above is only used by the jobs it submits: each of those
# jobs re-runs this same file, which detects SWEEP_WORKER_VALUE is set and
# trains exactly one sweep value instead of looping/re-submitting.
# '''

# ---------------------------------------------------------------------------
# FIXED HYPERPARAMETERS -- edit here for different defaults; no CLI args.
# ---------------------------------------------------------------------------
N_WAY=5
K_SHOT=5
PT_DATA=sam3
MODE=both
CACHE_MODE=cache

# ---------------------------------------------------------------------------
# SWEEP CONFIG — edit these two lines to sweep a different parameter.
# SWEEP_PARAM is any dotted yacs config key (see trokens/config/defaults.py);
# SWEEP_VALUES is the list of values to try, one training run (= one sbatch
# job) per value. Each job gets its own FRAME_CACHE_DIR, OUTPUT_DIR, and
# WANDB_ID so results never collide across sweep values.
# ---------------------------------------------------------------------------
SWEEP_PARAM="DATA.NUM_FRAMES"
SWEEP_VALUES=(8 16 32)
# Filesystem/name-safe tag for SWEEP_PARAM, e.g. "DATA.NUM_FRAMES" -> "data_num_frames"
SWEEP_PARAM_SLUG=$(echo "$SWEEP_PARAM" | tr '[:upper:].' '[:lower:]_')

# ===========================================================================
# LAUNCHER -- runs when this script is invoked directly (SWEEP_WORKER_VALUE
# unset). Submits one independent sbatch job per sweep value, then exits;
# does no training itself.
# ===========================================================================
if [ -z "$SWEEP_WORKER_VALUE" ]; then
	LOG_DIR=../trial_run_outputs
	mkdir -p "$LOG_DIR"
	for SWEEP_VALUE in "${SWEEP_VALUES[@]}"; do
		SUFFIX="${SWEEP_PARAM_SLUG}${SWEEP_VALUE}"
		echo "Submitting sweep job: $SWEEP_PARAM=$SWEEP_VALUE"
		export SWEEP_WORKER_VALUE=$SWEEP_VALUE
		sbatch \
			--job-name="ds12_04_leave3_sweep_frames_${SUFFIX}" \
			--output="$LOG_DIR/ds12_04_leave3_sweep_frames_${SUFFIX}_%j.out" \
			--error="$LOG_DIR/ds12_04_leave3_sweep_frames_${SUFFIX}_%j.out" \
			"$0"
	done
	exit 0
fi

# ===========================================================================
# WORKER -- everything below only runs inside a submitted job, training
# exactly one sweep value (SWEEP_WORKER_VALUE, set by the launcher above).
# ===========================================================================
SWEEP_VALUE=$SWEEP_WORKER_VALUE
SUFFIX="${SWEEP_PARAM_SLUG}${SWEEP_VALUE}"

export FRAME_CACHE_ENABLE=True

POINT_INFO_ENABLE=True
TROKENS_PT_DATA="/fs/vulcan-projects/fsh_track/processed_data/sam3pklds12_04_leave3"
export NUM_POINTS_TO_SAMPLE=18
# Decoded-frame cache; filled lazily on first epoch, or ahead of time with
# tools/dump_frame_cache.py. Each sweep value gets its own suffixed dir
# (below) so cached frames from different DATA.NUM_FRAMES values never mix.
export FRAME_CACHE_DIR="/fs/vulcan-projects/fsh_track/processed_data/frame_cache/ds12_4sam3_leave3_${SUFFIX}"


source ~/miniconda3/bin/activate
conda init

conda config --add envs_dirs /fs/vulcan-projects/fsh_track/envs/
conda activate trokens

export CONFIG_TO_USE=fshdata
export EXP_NAME=ds12_04_leave3_sweep_frames
export SECONDARY_EXP_NAME="${N_WAY}_way-${K_SHOT}_shot-${PT_DATA}-${MODE}-${SUFFIX}"
export TORCH_HOME=/fs/vulcan-projects/fsh_track/programs/trokens_workspace/trokens/torch_home
export DATA_DIR=/fs/vulcan-projects/fsh_track/processed_data/dataset12_04_leave3
export BASE_OUTPUT_DIR=/fs/vulcan-projects/fsh_track/models/
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
export WANDB_ID="${EXP_NAME}_${N_WAY}_way-${K_SHOT}_shot-${PT_DATA}-${MODE}-${SUFFIX}_"$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)

#export CHECKPOINT_FILE=/fs/vulcan-projects/fsh_track/models/ds6/5_way-3_shot-sam3-both/checkpoints/checkpoint_best.pyth

mkdir -p $OUTPUT_DIR

cd ..

echo "=== Sweep worker: $SWEEP_PARAM=$SWEEP_VALUE  (cache: $FRAME_CACHE_DIR) ==="

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
	tools/run_net.py --init_method env:// --new_dist_init \
	--cfg configs/trokens/$CONFIG_TO_USE.yaml \
	WANDB.ID $WANDB_ID \
	WANDB.EXP_NAME "${EXP_NAME}_${SECONDARY_EXP_NAME}" \
	MASTER_PORT $MASTER_PORT \
	OUTPUT_DIR $OUTPUT_DIR \
	NUM_GPUS $NUM_GPUS \
	DATA_LOADER.NUM_WORKERS $NUM_WORKERS \
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
	$SWEEP_PARAM $SWEEP_VALUE
	#TEST.CHECKPOINT_FILE_PATH $CHECKPOINT_FILE
