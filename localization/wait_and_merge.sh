#!/bin/bash
# Wait for the 5-fold array job, then aggregate. A script rather than an inline
# loop so it survives session teardown.
cd /fs/vulcan-projects/fsh_track/bhargav/fsh-cluster/localization
export PYTHONNOUSERSITE=1
while squeue -j "$1" -h 2>/dev/null | grep -q .; do sleep 120; done
echo "=== array $1 finished $(date) ==="
echo "exit states:"; sacct -j "$1" --format=JobID%20,State,Elapsed -n 2>/dev/null | grep -v "\.ba\|\.ex" | head
echo; echo "per-fold result files: $(ls results/fold*.json 2>/dev/null | wc -l)/5"
/fs/vulcan-projects/fsh_track/envs/trokens/bin/python analysis/merge_folds.py --json results/summary.json
