#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --mem=50G
#SBATCH --time=10:00:00
#SBATCH --job-name=fsh_pt_data_generation
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

source ~/miniconda3/bin/activate
conda config --add envs_dirs /fs/vulcan-projects/fsh_track/envs/trokens_env
conda activate trokens
conda init

python new_point_tracking.py --fps 10 --csv_path /fs/vulcan-projects/fsh_track/processed_data/dataset2/formatted_summary.csv --base_feat_path /fs/vulcan-projects/fsh_track/will/ptattempts/att8