#!/bin/sh
#SBATCH --job-name=ds6_trokens
#SBATCH --ntasks=4
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --qos=default
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=../trial_run_outputs/trokens_ds6_%j.out
#SBATCH --error=../trial_run_outputs/trokens_ds6_%j.out
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT

# ''' USAGE 
# run this file from the scripts folder 
# sbatch trokens_exp.sh <N_WAY> <K_SHOT> <PT_DATA> <MODE>
# If you are just testing, set the checkpoint file and uncomment the last line in script
# '''

#command line arguments
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
	echo "Error: Missing required parameters. Usage: $0 <N_WAY> <K_SHOT> <PT_DATA> <MODE>"
	echo "PT_DATA options: 'none', 'trokens', 'sam3'"
	echo "MODE options: 'train', 'test', 'both'"
	exit 1
fi

N_WAY=$1
K_SHOT=$2
PT_DATA=$3
MODE=$4
if [[ "$PT_DATA" != "none" && "$PT_DATA" != "trokens" && "$PT_DATA" != "sam3" ]]; then
    echo "Error: Invalid PT_DATA option. Must be 'none', 'trokens', or 'sam3'"
    exit 1
fi
case $PT_DATA in
    "none")
		POINT_INFO_ENABLE=False 
        TROKENS_PT_DATA="/fs/vulcan-projects/fsh_track/processed_data/cotrackpklds6/cotracker3_bip_fr_32_fps_10/fshdata/feat_dump/"
		export NUM_POINTS_TO_SAMPLE=256
        ;;
    "trokens")
		POINT_INFO_ENABLE=True 
        TROKENS_PT_DATA="/fs/vulcan-projects/fsh_track/processed_data/cotrackpklds6/cotracker3_bip_fr_32_fps_10/fshdata/feat_dump/"
		export NUM_POINTS_TO_SAMPLE=256
        ;;
    "sam3")
		POINT_INFO_ENABLE=True 
        TROKENS_PT_DATA="/fs/vulcan-projects/fsh_track/processed_data/sam3pklds6/"
		export NUM_POINTS_TO_SAMPLE=18
        ;;
esac


source ~/miniconda3/bin/activate 
conda init 

conda config --add envs_dirs /fs/vulcan-projects/fsh_track/envs/
conda activate trokens

export CONFIG_TO_USE=fshdata
export EXP_NAME=ds6
export SECONDARY_EXP_NAME="BCEL-${N_WAY}_way-${K_SHOT}_shot-${PT_DATA}-${MODE}"
export TORCH_HOME=/fs/vulcan-projects/fsh_track/programs/trokens_workspace/trokens/torch_home
export DATA_DIR=/fs/vulcan-projects/fsh_track/processed_data/dataset6
export BASE_OUTPUT_DIR=/fs/vulcan-projects/fsh_track/models
export OUTPUT_DIR=$BASE_OUTPUT_DIR/$EXP_NAME/$SECONDARY_EXP_NAME
export NUM_CLASSES=6
export CUT_SMALLS=False
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
export NUM_WORKERS=4
export MASTER_PORT=$(cat /dev/urandom | tr -dc '0-9' | fold -w 4 | head -n 1) 
export POINT_INFO_NAME="cotracker3_bip_fr_32"
#set wandb id to random 8 character string
export WANDB_ID="${EXP_NAME}_${N_WAY}_way-${K_SHOT}_shot-${PT_DATA}-${MODE}_"$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)


#export CHECKPOINT_FILE=/fs/vulcan-projects/fsh_track/models/fshdata/ds3trained/5_way-5_shot-trokens/checkpoints/checkpoint_best.pyth
#export TRAIN_EVAL_PERIOD=1

mkdir -p $OUTPUT_DIR

cd ..

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
	tools/run_net.py --init_method env:// --new_dist_init \
	--cfg configs/trokens/$CONFIG_TO_USE.yaml \
	WANDB.ID $WANDB_ID \
	WANDB.EXP_NAME $SECONDARY_EXP_NAME \
	MASTER_PORT $MASTER_PORT \
	OUTPUT_DIR $OUTPUT_DIR \
	NUM_GPUS $NUM_GPUS \
	DATA_LOADER.NUM_WORKERS $NUM_WORKERS \
	DATA.USE_RAND_AUGMENT True \
	DATA.PATH_TO_DATA_DIR $DATA_DIR \
	DATA.PATH_TO_TROKEN_PT_DATA $TROKENS_PT_DATA \
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
    TRAIN.CHECKPOINT_EPOCH_RESET False \
    TRAIN.AUTO_RESUME True \
	DATA_LOADER.CUT_SMALLS $CUT_SMALLS \
	DATA_LOADER.FILTER_ONE $FILTER_ONE \
	MODEL.NUM_CLASSES $NUM_CLASSES
# 	TEST.CHECKPOINT_FILE_PATH $CHECKPOINT_FILE
#    TRAIN.EVAL_PERIOD $TRAIN_EVAL_PERIOD

	