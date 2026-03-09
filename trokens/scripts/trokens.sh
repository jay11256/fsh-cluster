export CONFIG_TO_USE=$1
export NUM_GPUS=1
export NUM_WORKERS=32
# export MASTER_PORT=$(cat /dev/urandom | tr -dc '0-9' | fold -w 4 | head -n 1) 
export MASTER_PORT=29500
export N_WAY=5 # changed from 5
export K_SHOT=5
export NUM_POINTS_TO_SAMPLE=256
export POINT_INFO_NAME="cotracker3_bip_fr_32"
#set wandb id to random 8 character string
export WANDB_ID="trokens_N=5+K=5_$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)"

# if config to use has ssv2, set DATASET as ssv2 else set it to the name of the dataset
if [[ $CONFIG_TO_USE == *"ssv2"* ]]; then
	export DATASET="ssv2"
# else if [[ $CONFIG_TO_USE == *"fsh"* ]]; then
	# export DATASET="fsh_ds"
else
	export DATASET=$CONFIG_TO_USE
fi
export OUTPUT_DIR=$BASE_OUTPUT_DIR/$CONFIG_TO_USE/$EXP_NAME/$SECONDAY_EXP_NAME
#ssv2
# export DATA_DIR=/fs/vulcan-datasets/SomethingV2/
#fsh data:
#export DATA_DIR=/fs/vulcan-projects/fsh_track/processed_data/test2/


mkdir -p $OUTPUT_DIR

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
	MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE True
	