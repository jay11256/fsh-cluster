#!/usr/bin/env python3
"""Aggregate the per-fold result JSONs written by run_former_5fold_ckpt.sbatch.

The launcher gives every array task its own `results/fold{F}.json` because
train_former.py's results write is a read-modify-write and concurrent tasks
would clobber each other. This recombines them.

Reports two aggregations, because they answer different questions and the paper
uses the per-recording one:

  per-RECORDING  every held-out recording weighted equally (17 of them). This
                 is what `evaluate` produces per fold and what the table rows
                 elsewhere in this project use.
  per-FOLD       each fold's mean weighted equally (5 of them), which is what a
                 paired per-fold significance test would consume. Folds hold 3
                 or 4 recordings, so the two differ slightly.

    python3 analysis/merge_folds.py [--results results] [--json out.json]
"""
import os
import sys
import json
import glob
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

IOUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--json", default=None, help="also write the summary here")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, "fold*.json")))
    if not files:
        sys.exit(f"no fold*.json in {args.results}")

    per_rec, per_fold, folds_seen = {}, [], []
    for f in files:
        blob = json.load(open(f))
        for label, s in blob.items():
            folds_seen.append(label)
            curves = []
            for rec, r in s["folds"].items():
                per_rec[rec] = r["per_iou"]
                curves.append(r["per_iou"])
            per_fold.append(np.nanmean(np.array(curves), axis=0))

    R = np.array([per_rec[k] for k in sorted(per_rec)])
    F = np.array(per_fold)

    print(f"folds found: {len(files)}   recordings: {len(per_rec)}\n")
    print("per-recording mAP@tIoU (%)  [n={}]".format(len(per_rec)))
    print("   tIoU  " + "".join(f"{i:>7.1f}" for i in IOUS))
    print("   mean  " + "".join(f"{100*v:>7.1f}" for v in np.nanmean(R, axis=0)))
    print("   std   " + "".join(f"{100*v:>7.1f}" for v in np.nanstd(R, axis=0)))

    m_rec = np.nanmean(R, axis=0)
    m_fold = np.nanmean(F, axis=0)
    print(f"\n   avg mAP .1-.7 : {100*np.nanmean(m_rec[:7]):6.2f}%  (per-recording)"
          f"   {100*np.nanmean(m_fold[:7]):6.2f}%  (per-fold)")
    print(f"   avg mAP .1-.9 : {100*np.nanmean(m_rec):6.2f}%  (per-recording)"
          f"   {100*np.nanmean(m_fold):6.2f}%  (per-fold)")

    print("\nper-fold avg mAP .1-.7 (%):")
    for lab, c in zip(folds_seen, F):
        print(f"   {lab:38} {100*np.nanmean(c[:7]):6.2f}")

    print("\nper-recording avg mAP .1-.7 (%):")
    for k in sorted(per_rec):
        print(f"   {k:34} {100*np.nanmean(np.array(per_rec[k])[:7]):6.2f}")

    if args.json:
        json.dump({"ious": list(IOUS), "n_recordings": len(per_rec),
                   "per_recording": per_rec,
                   "per_iou_mean_recording": m_rec.tolist(),
                   "per_iou_mean_fold": m_fold.tolist(),
                   "avg_map_1_7_recording": float(np.nanmean(m_rec[:7])),
                   "avg_map_1_7_fold": float(np.nanmean(m_fold[:7]))},
                  open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
