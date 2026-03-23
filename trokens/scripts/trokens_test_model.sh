#!/bin/sh
#SBATCH --job-name=test5x5_dataset3
#SBATCH --ntasks=4
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --qos=default
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=../trial_run_outputs/test5x5_dataset3_%j.txt
#SBATCH --error=../trial_run_outputs/test5x5_dataset3_%j.txt
#SBATCH --mail-user=wlamousi@terpmail.umd.edu
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT

# navigate to programs/trokens_workspace/trokens

# these two lines are only relevant if ur doing sbatch
source ~/miniconda3/bin/activate 
# ^change this depending on where your miniconda is
conda init 

conda config --add envs_dirs /fs/vulcan-projects/fsh_track/envs/
conda activate trokens

export CONFIG_TO_USE=fshdata
export DATASET=fshdata
export EXP_NAME=trokens_release
export SECONDAY_EXP_NAME=sample_exp

# Path to store PyTorch models and weights
export TORCH_HOME=/fs/vulcan-projects/fsh_track/programs/trokens_workspace/trokens/torch_home

# Path to dataset directory containing videos
export DATA_DIR=/fs/vulcan-projects/fsh_track/processed_data/dataset3

# Path to pre-computed Trokens point tracking data and few shot info from huggingface.
export TROKENS_PT_DATA=/fs/vulcan-projects/fsh_track/will/ptattempts/att8/cotracker3_bip_fr_32_fps_10/fshdata/feat_dump/

# Base output directory for experiments

export BASE_OUTPUT_DIR=/fs/vulcan-projects/fsh_track/will/dataset3/N5K5dataset3

export TRAIN.ENABLE=False
export TEST.ENABLE=True

#CHANGE WAND B CONFIG ACCORDING TO YOUR USER
#trokens/config/custom_config.py

# After making sure the base output directory you want to put stuff in is empty, you can run the sample script

export NUM_GPUS=1
export NUM_WORKERS=4
# export MASTER_PORT=$(cat /dev/urandom | tr -dc '0-9' | fold -w 4 | head -n 1) 
export MASTER_PORT=29501
export N_WAY=5 # changed from 5
export K_SHOT=5
export NUM_POINTS_TO_SAMPLE=256
export POINT_INFO_NAME="cotracker3_bip_fr_32"
#set wandb id to random 8 character string
export WANDB_ID="test5x5_$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)"

export DATASET=$CONFIG_TO_USE

export OUTPUT_DIR=$BASE_OUTPUT_DIR/$CONFIG_TO_USE/$EXP_NAME/$SECONDAY_EXP_NAME

export CHECKPOINT_FILE=/fs/vulcan-projects/fsh_track/will/dataset3/N5K5dataset3/fshdata/trokens_release/sample_exp/checkpoints/checkpoint_best.pyth

mkdir -p $OUTPUT_DIR

cd ..

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
	tools/run_net.py --init_method env:// --new_dist_init \
	--cfg configs/trokens/$CONFIG_TO_USE.yaml \
	WANDB.ID $WANDB_ID \
	WANDB.EXP_NAME $EXP_NAME \
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
	POINT_INFO.NAME $POINT_INFO_NAME \
	POINT_INFO.SAMPLING_TYPE cluster_sample \
	POINT_INFO.NUM_POINTS_TO_SAMPLE $NUM_POINTS_TO_SAMPLE \
	MODEL.FEAT_EXTRACTOR dino \
	MODEL.DINO_CONFIG dinov2_vitb14 \
	MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE True \
	MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE True \
    TRAIN.ENABLE False \
    TEST.ENABLE True \
    TRAIN.CHECKPOINT_EPOCH_RESET False \
    TRAIN.AUTO_RESUME True \
    TEST.CHECKPOINT_FILE_PATH $CHECKPOINT_FILE
	