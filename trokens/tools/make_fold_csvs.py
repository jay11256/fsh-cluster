#!/usr/bin/env python3
"""
Generate leave-videos-out fold CSVs for 5-fold cross-validation on dataset12_06.

The source CSV (dataset12_06/dataset12.csv) has a clip-level 'split' column
(train/test) stratified per-behavior within each video -- every video
contributes to both train and test. For proper k-fold CV we instead want
each fold's test set to be entirely unseen videos.

This script partitions the 17 source videos (stratified by treatment x
circling condition, parsed from the master_video name) into 5 folds, then
writes 5 CSVs where 'split' is overwritten: a row is 'test' if its
master_video falls in that fold's held-out group, else 'train'. All other
columns are copied unchanged, and video files are NOT duplicated -- every
fold CSV still points at the shared dataset12_06/*.mp4 clips.
"""
import argparse
import os
import re

import pandas as pd

SOURCE_CSV = "/fs/vulcan-projects/fsh_track/processed_data/dataset12_06/dataset12.csv"
# dataset12_06/ itself is not group-writable, so fold CSVs (which just
# reference the shared mp4s via video_path) live in this repo instead.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data_splits", "dataset12_06_folds")
NUM_FOLDS = 5


def parse_condition(master_video):
    treatment = "Vetbond" if re.search(r"vetbond", master_video, re.IGNORECASE) else "Sham"
    circling = "NoCir" if re.search(r"nocir", master_video, re.IGNORECASE) else "Cir"
    return (treatment, circling)


def assign_folds(videos, num_folds=NUM_FOLDS):
    """
    Assign each video to a fold, spreading each condition group across folds
    while keeping total fold sizes as balanced as possible: process condition
    groups largest-first, and for each video (in deterministic sorted order)
    place it in whichever fold currently has the fewest videos (ties broken
    by lowest fold index).
    """
    by_condition = {}
    for v in sorted(videos):
        by_condition.setdefault(parse_condition(v), []).append(v)

    fold_of = {}
    fold_counts = [0] * num_folds
    for condition, vids in sorted(by_condition.items(), key=lambda kv: -len(kv[1])):
        for v in vids:
            fold = min(range(num_folds), key=lambda f: (fold_counts[f], f))
            fold_of[v] = fold
            fold_counts[fold] += 1
    return fold_of


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", default=SOURCE_CSV)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--num-folds", type=int, default=NUM_FOLDS)
    args = parser.parse_args()

    df = pd.read_csv(args.source_csv)
    fold_of = assign_folds(df["master_video"].unique().tolist(), args.num_folds)

    print("Video -> fold assignment:")
    for v in sorted(fold_of, key=lambda v: (fold_of[v], v)):
        print(f"  fold {fold_of[v]}  {parse_condition(v)}  {v}")

    os.makedirs(args.output_dir, exist_ok=True)
    for fold in range(args.num_folds):
        fold_df = df.copy()
        is_test = fold_df["master_video"].map(fold_of) == fold
        fold_df["split"] = is_test.map({True: "test", False: "train"})

        out_path = os.path.join(args.output_dir, f"dataset12_06_fold{fold}.csv")
        fold_df.to_csv(out_path, index=False)

        held_out = sorted(v for v, f in fold_of.items() if f == fold)
        print(
            f"\nfold {fold}: {len(held_out)} held-out videos {held_out} -> "
            f"{is_test.sum()} test rows / {(~is_test).sum()} train rows -> {out_path}"
        )


if __name__ == "__main__":
    main()
