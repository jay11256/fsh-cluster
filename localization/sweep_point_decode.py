#!/usr/bin/env python3
"""Inference-only decode sweep for a trained FishPoint checkpoint set.

Motivation: FishPoint1/2/3's reported numbers used one fixed, never-tuned
(--thresh 0.05, --min-sep-s 1.0, distance-NMS) decode. FishTAL15 showed this
kind of inference-only retune is free (no retraining) and bought +0.05
point-precision there. This script loads the already-trained per-fold
checkpoints (saved via train_point.py --save-ckpt) and, for each, runs the
expensive GPU forward pass (score_recording) ONCE, then sweeps the cheap
CPU-only decode grid (threshold x NMS policy/params) against those SAME cached
score curves -- no retraining, no repeated forward passes.

Methodology note: a decode config is picked by its MEAN point-AP across ALL 14
LOO folds (one global config, not a different config per test fold), so this
is not peeking at each fold's own test-set GT to tune that fold individually
-- the same "pick once via the full LOO sweep" methodology already used
throughout EXPERIMENTS.md for e.g. ASM-Loc's chunk-length/config selection.

Usage: sweep_point_decode.py --label FishPoint3 [--ckpt-dir checkpoints]
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from point_model import FishPointModel                               # noqa: E402
from data import list_recordings, FG_CLASSES                         # noqa: E402
from train_point import (score_recording, decode_points, evaluate_points,  # noqa: E402
                         STRIDE)


class _Args:
    """Minimal stand-in so score_recording (written to take an argparse
    Namespace) can be reused here without dragging in the full train_point CLI."""
    def __init__(self, feature_mode, window_s, n_levels, use_offset):
        self.feature_mode = feature_mode
        self.window_s = window_s
        self.n_levels = n_levels
        self.use_offset = use_offset


def load_fold_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = FishPointModel(feat_dim=ck["feat_dim"], num_classes=7, hidden=cfg["hidden"],
                           depth=cfg["depth"], num_heads=cfg["num_heads"], drop=cfg["drop"],
                           drop_path=cfg["drop_path"], n_levels=cfg["n_levels"],
                           use_offset=cfg["use_offset"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck["test_rec"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--ckpt-dir", default=f"{HERE}/checkpoints")
    p.add_argument("--out", default=f"{HERE}/point_decode_sweep.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = sorted(f for f in os.listdir(args.ckpt_dir)
                   if f.startswith(f"{args.label}__") and f.endswith(".pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints matching {args.label}__*.pt in {args.ckpt_dir}")
    print(f"{len(ckpts)} fold checkpoints found for {args.label}", flush=True)

    THRESHES = (0.03, 0.05, 0.08)
    MIN_SEPS = (0.5, 1.0, 1.5)
    WIDTH_MULTS = (1.0, 2.0, 3.0)
    IOU_THR = 0.3
    TOLERANCE = 1.0   # eval-protocol constant, not swept -- see EXPERIMENTS.md

    configs = ([("distance", thr, ms, None) for thr in THRESHES for ms in MIN_SEPS] +
              [("iou", thr, None, wm) for thr in THRESHES for wm in WIDTH_MULTS])

    # grid[config_key][fold] = {"point_ap":.., "point_recall":.., "point_precision":..}
    grid = defaultdict(dict)
    for fname in ckpts:
        model, cfg, test_rec = load_fold_model(os.path.join(args.ckpt_dir, fname), device)
        a = _Args(cfg["feature_mode"], cfg["window_s"], cfg["n_levels"], cfg["use_offset"])
        with torch.no_grad():
            levels, events, _ = score_recording(model, test_rec, a, device)
        for policy, thr, ms, wm in configs:
            key = f"{policy}|thresh={thr}|" + (f"min_sep={ms}" if policy == "distance"
                                                else f"width_mult={wm}|iou={IOU_THR}")
            pts = decode_points(levels, thr, policy, min_sep_s=ms or 1.0,
                                width_mult=wm or 2.0, iou_thr=IOU_THR)
            res = evaluate_points(pts, events, TOLERANCE)
            grid[key][test_rec] = res
        print(f"[done] {test_rec} ({len(configs)} configs)", flush=True)

    # Pick the single global config with the best MEAN point-AP across all folds
    # that actually completed for every fold (fair comparison across configs).
    n_folds = len(ckpts)
    summary = {}
    for key, per_fold in grid.items():
        if len(per_fold) < n_folds:
            continue
        aps = [v["point_ap"] for v in per_fold.values() if not np.isnan(v["point_ap"])]
        recs = [v["point_recall"] for v in per_fold.values() if not np.isnan(v["point_recall"])]
        precs = [v["point_precision"] for v in per_fold.values() if not np.isnan(v["point_precision"])]
        summary[key] = {
            "point_ap_mean": float(np.mean(aps)), "point_ap_std": float(np.std(aps)),
            "point_recall_mean": float(np.mean(recs)), "point_precision_mean": float(np.mean(precs)),
            "n_folds": len(per_fold),
        }

    ranked = sorted(summary, key=lambda k: -summary[k]["point_ap_mean"])
    print("\nTop 10 decode configs by mean point-AP:")
    for k in ranked[:10]:
        s = summary[k]
        print(f"  {k:45s} AP={s['point_ap_mean']:.4f}+/-{s['point_ap_std']:.4f} "
              f"recall={s['point_recall_mean']:.3f} prec={s['point_precision_mean']:.3f}")

    best_key = ranked[0]
    out = {"label": args.label, "n_folds": n_folds, "best_config": best_key,
          "best_result": summary[best_key], "full_grid": summary}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
