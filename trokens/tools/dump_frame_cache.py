#!/usr/bin/env python3
"""Pre-populate the frame cache used by BaseDataset.__getitem__.

The dataloader samples a deterministic np.linspace of NUM_FRAMES frames per
video (see base_ds.py), so the decoded frames can be extracted once and
reused by every epoch/episode. This script decodes those frames for every
video in the dataset CSV and writes them as JPEG bytes into the cache dir
that trokens.datasets.utils.read_video reads from (DATA.FRAME_CACHE_DIR).

Training also lazily fills the same cache on first access, so running this
script is optional — it just moves the one-time decode cost out of epoch 1.

Example:
    python tools/dump_frame_cache.py \
        --data-dir /fs/vulcan-projects/fsh_track/processed_data/dataset8 \
        --pkl-dir /fs/vulcan-projects/fsh_track/processed_data/sam3pklds8 \
        --cache-dir /fs/vulcan-projects/fsh_track/processed_data/frame_cache/dataset8 \
        --num-frames 8 --workers 16
"""
import argparse
import glob
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trokens.datasets.utils import read_video, _frame_cache_path  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="Dataset dir containing the videos and one CSV "
                             "(DATA.PATH_TO_DATA_DIR)")
    parser.add_argument("--pkl-dir", required=True,
                        help="Dir of per-video point-track pkls "
                             "(DATA.PATH_TO_TROKEN_PT_DATA); pred_tracks "
                             "length defines the frame indices")
    parser.add_argument("--cache-dir", required=True,
                        help="Cache dir to populate (DATA.FRAME_CACHE_DIR)")
    parser.add_argument("--num-frames", type=int, default=8,
                        help="DATA.NUM_FRAMES used at train time")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process the first N videos (for testing)")
    return parser.parse_args()


def process_one(job):
    video_path, pkl_path, cache_dir, num_frames = job
    try:
        with open(pkl_path, "rb") as f:
            pt_dict = pickle.load(f)
        total_frames = pt_dict["pred_tracks"].squeeze(0).shape[0]
        index_select = np.linspace(0, total_frames - 1, num_frames).astype(int)
        cache_path = _frame_cache_path(cache_dir, video_path, total_frames,
                                       index_select)
        if os.path.exists(cache_path):
            return "cached"
        read_video(video_path, total_frames=total_frames,
                   indices_to_take=index_select, cache_dir=cache_dir)
        return "done"
    except Exception as e:  # keep going; training falls back to decoding
        print(f"FAILED {video_path}: {e}", file=sys.stderr)
        return "failed"


def main():
    args = parse_args()

    csv_files = glob.glob(os.path.join(args.data_dir, "*.csv"))
    if len(csv_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one CSV in {args.data_dir}, found {len(csv_files)}")
    df = pd.read_csv(csv_files[0])

    jobs = []
    missing_pkl = 0
    for p in df["video_path"]:
        video_path = p if os.path.isabs(p) else os.path.join(args.data_dir, p)
        stem = os.path.splitext(os.path.basename(video_path))[0]
        pkl_path = os.path.join(args.pkl_dir, stem + ".pkl")
        if not os.path.exists(pkl_path):
            missing_pkl += 1
            continue
        jobs.append((video_path, pkl_path, args.cache_dir, args.num_frames))
    if args.limit:
        jobs = jobs[:args.limit]

    print(f"{len(jobs)} videos to process ({missing_pkl} skipped, no pkl)")
    os.makedirs(args.cache_dir, exist_ok=True)

    counts = {"done": 0, "cached": 0, "failed": 0}
    with Pool(args.workers) as pool:
        for i, status in enumerate(pool.imap_unordered(process_one, jobs, chunksize=8)):
            counts[status] += 1
            if (i + 1) % 500 == 0 or i + 1 == len(jobs):
                print(f"[{i + 1}/{len(jobs)}] {counts}", flush=True)

    print(f"Finished: {counts}")


if __name__ == "__main__":
    main()
