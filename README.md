# fsh-cluster

hello 🐟!🫧

Modules required: Python3, ffmpeg

Commands:
cd /fs/vulcan-projects/fsh_track
srun --pty --ntasks=4 --gres gpu:rtxa4000:1 --qos scavenger --account scavenger --partition scavenger --mem 20G --time 48:00:00 bash
jupyter notebook --no-browser --port=8889 --ip=0.0.0.0
