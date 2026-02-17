#!/bin/bash
#SBATCH --job-name=sam3_array
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-99
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT

#SBATCH --output=$CLUSTER_LOGS
#SBATCH --error=$CLUSTER_LOGS
#SBATCH --mail-user=$CLUSTER_MAIL
VIDEO_DIR=$SAM3_INPUT
SCRIPT=$SAM3_SCRIPT
# ------------------------

# Total number of splits = number of array jobs
TOTAL_SPLITS=$SLURM_ARRAY_TASK_COUNT

module load cuda
module load ffmpeg
source ~/.bashrc
conda activate sam3   # or your env

# Build sorted list of all videos
VIDEOS=($(ls ${VIDEO_DIR}/*.mp4 | sort))
NUM_VIDEOS=${#VIDEOS[@]}

# Determine which subset of videos this task should process
START_INDEX=$(( SLURM_ARRAY_TASK_ID * NUM_VIDEOS / TOTAL_SPLITS ))
END_INDEX=$(( (SLURM_ARRAY_TASK_ID + 1) * NUM_VIDEOS / TOTAL_SPLITS ))

echo "========== TASK $SLURM_ARRAY_TASK_ID =========="
echo "Host: $(hostname)"
echo "Processing videos $START_INDEX to $((END_INDEX-1)) out of $NUM_VIDEOS"
echo "Start time: $(date)"
echo "=============================================="
echo

# Loop over assigned videos
for (( i=START_INDEX; i<END_INDEX; i++ )); do
    VIDEO=${VIDEOS[$i]}
    echo "Processing video [$i]: $VIDEO"
    python "$SCRIPT" "$VIDEO"
    echo "Finished video [$i]: $VIDEO at $(date)"
    echo
done

echo "========== TASK $SLURM_ARRAY_TASK_ID COMPLETE =========="
echo "End time: $(date)"
echo "=============================================="