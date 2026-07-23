#!/bin/sh
#SBATCH --job-name=ds12_04_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --qos=high
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=../trial_run_outputs/ds12_04_sweep.out
#SBATCH --error=../trial_run_outputs/ds12_04_sweep.out
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT

# ''' USAGE
# run this file from the scripts folder
# sbatch trokens_exp.sh <N_WAY> <K_SHOT> <PT_DATA> <MODE> [CACHE_MODE]
# CACHE_MODE options: 'cache' (default, use the decoded-frame cache) or
#   'nocache' (always decode from mp4; for ablations comparing the two)
# If you are just testing, set the checkpoint file and uncomment the last line in script
# '''

#command line arguments
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
	echo "Error: Missing required parameters. Usage: $0 <N_WAY> <K_SHOT> <PT_DATA> <MODE> [CACHE_MODE]"
	echo "PT_DATA options: 'none', 'trokens', 'sam3'"
	echo "MODE options: 'train', 'test', 'both'"
	echo "CACHE_MODE options: 'cache' (default), 'nocache'"
	exit 1
fi

N_WAY=$1
K_SHOT=$2
PT_DATA=$3
MODE=$4
CACHE_MODE=${5:-cache}

if [[ "$CACHE_MODE" != "cache" && "$CACHE_MODE" != "nocache" ]]; then
    echo "Error: Invalid CACHE_MODE option. Must be 'cache' or 'nocache'"
    exit 1
fi
if [[ "$CACHE_MODE" == "cache" ]]; then
    export FRAME_CACHE_ENABLE=True
else
    export FRAME_CACHE_ENABLE=False
fi

if [[ "$PT_DATA" != "none" && "$PT_DATA" != "trokens" && "$PT_DATA" != "sam3" ]]; then
    echo "Error: Invalid PT_DATA option. Must be 'none', 'trokens', or 'sam3'"
    exit 1
fi
case $PT_DATA in
    "none")
		POINT_INFO_ENABLE=False 
        TROKENS_PT_DATA="/fs/vulcan-projects/fsh_track/processed_data/cotrackpklds6/cotracker3_bip_fr_32_fps_10/fshdata/feat_dump/"
		export NUM_POINTS_TO_SAMPLE=256
		# Decoded-frame cache shared by all runs on this dataset; filled lazily on
		# first epoch, or ahead of time with tools/dump_frame_cache.py.
		# Respects a pre-exported FRAME_CACHE_DIR (e.g. from an ablation
		# script) instead of always overwriting it.
		export FRAME_CACHE_DIR=${FRAME_CACHE_DIR:-/fs/vulcan-projects/fsh_track/processed_data/frame_cache/ds12_02none}
        ;;
    "trokens")
		POINT_INFO_ENABLE=True
        TROKENS_PT_DATA="/fs/vulcan-projects/fsh_track/processed_data/cotrackpklds6/cotracker3_bip_fr_32_fps_10/fshdata/feat_dump/"
		export NUM_POINTS_TO_SAMPLE=256
		export FRAME_CACHE_DIR=${FRAME_CACHE_DIR:-/fs/vulcan-projects/fsh_track/processed_data/frame_cache/ds12_2trokens}
        ;;
    "sam3")
		POINT_INFO_ENABLE=True 
        TROKENS_PT_DATA="/fs/vulcan-projects/fsh_track/processed_data/sam3pklds12_04"
		export NUM_POINTS_TO_SAMPLE=18
		export FRAME_CACHE_DIR=${FRAME_CACHE_DIR:-/fs/vulcan-projects/fsh_track/processed_data/frame_cache/ds12_4sam3}
        ;;
esac


source ~/miniconda3/bin/activate 
conda init 

conda config --add envs_dirs /fs/vulcan-projects/fsh_track/envs/
conda activate trokens

export CONFIG_TO_USE=fshdata
export EXP_NAME=ds12_04_sweep
# export EXP_NAME=${EXP_NAME:-ds11_bsri}
# Default naming is unchanged from before (no CACHE_MODE suffix) so existing
# AUTO_RESUME checkpoints keep resolving to the same OUTPUT_DIR. Callers that
# want cache/nocache runs kept separate (e.g. the cache ablation) should
# export SECONDARY_EXP_NAME themselves before invoking this script.
export SECONDARY_EXP_NAME=${SECONDARY_EXP_NAME:-"${N_WAY}_way-${K_SHOT}_shot-${PT_DATA}-${MODE}"}
export TORCH_HOME=/fs/vulcan-projects/fsh_track/programs/trokens_workspace/trokens/torch_home
export DATA_DIR=/fs/vulcan-projects/fsh_track/processed_data/dataset12_02
export BASE_OUTPUT_DIR=/fs/vulcan-projects/fsh_track/models/
export OUTPUT_DIR=$BASE_OUTPUT_DIR/$EXP_NAME/$SECONDARY_EXP_NAME
export NUM_CLASSES=7
export FILTER_ONE=True

case $MODE in
	"train")
		TRAIN_ENABLE=True
		TEST_ENABLE=False
		;;
	"test")
		TRAIN_ENABLE=False
		TEST_ENABLE=True
		;;
	"both")
		TRAIN_ENABLE=True
		TEST_ENABLE=True
		;;
	*)
		echo "Error: Invalid MODE option. Must be 'train', 'test', or 'both'"
		exit 1
		;;
esac
export NUM_GPUS=1
export NUM_WORKERS=12
# export MASTER_PORT=$(cat /dev/urandom | tr -dc '0-9' | fold -w 4 | head -n 1) 
export MASTER_PORT=$(( ( RANDOM % 64511 ) + 1024 ))
export POINT_INFO_NAME="cotracker3_bip_fr_32"
#set wandb id to random 8 character string
export WANDB_ID="${EXP_NAME}_${N_WAY}_way-${K_SHOT}_shot-${PT_DATA}-${MODE}_"$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)


#export CHECKPOINT_FILE=/fs/vulcan-projects/fsh_track/models/ds6/5_way-3_shot-sam3-both/checkpoints/checkpoint_best.pyth

mkdir -p $OUTPUT_DIR

cd ..

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
	NUM_FRAMES: 8 \
	#TEST.CHECKPOINT_FILE_PATH $CHECKPOINT_FILE

	