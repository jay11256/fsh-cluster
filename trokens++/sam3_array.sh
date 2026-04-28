#!/bin/bash
#SBATCH --job-name=sam3pklds6
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-99
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT
#SBATCH --output=/fs/vulcan-projects/fsh_track/bhargav/logs/sam3pklds6_%A_%a.txt
#SBATCH --error=/fs/vulcan-projects/fsh_track/bhargav/logs/sam3pklds6%_%A_%a.txt

# --mail-user=jliu1230@terpmail.umd.edu

module load cuda
module load ffmpeg

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam3


VIDEO_DIR=../../processed_data/dataset6
SCRIPT=./run_sam3.py 
OUT_DIR=../../processed_data/sam3pklds6
# ------------------------

# Total number of splits = number of array jobs
TOTAL_SPLITS=$SLURM_ARRAY_TASK_COUNT


# echo $CLUSTER_MAIL
# echo $CLUSTER_LOGS

# Build sorted list of all videos
# VIDEOS=($(ls ${VIDEO_DIR}/*.mp4 | sort))
# NUM_VIDEOS=${#VIDEOS[@]}
# Build sorted list of all videos (space-safe)
mapfile -d '' VIDEOS < <(
    find "$VIDEO_DIR" -maxdepth 1 -type f -name "*.mp4" -print0 | sort -z
)
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
    python "$SCRIPT" --output_dir "$OUT_DIR" "$VIDEO"
    echo "Finished video [$i]: $VIDEO at $(date)"
    echo
done

echo "========== TASK $SLURM_ARRAY_TASK_ID COMPLETE =========="
echo "End time: $(date)"
# echo "=============================================="