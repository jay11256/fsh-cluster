"""Sweep FishTAL's inference-only knobs against saved checkpoints -- no retraining.

Threshold list, NMS IoU and minimum span length only affect how the two heads'
score curves get turned into spans, so retraining 11 folds to change them is pure
waste (FishTAL14/15 did exactly that before checkpoints existed). This loads the
per-fold weights saved by `train.py --save-ckpt` and re-scores.

Aimed at FishTAL's one clear weakness vs MIL post-proc: point-precision ~0.17 vs
0.25. Recall and mAP are already ahead, so the question is whether stricter
span-formation buys precision without giving those back.

Usage: sweep_inference.py --label FishTAL6 [--out sweep_inference.json]
"""
import os
import sys
import json
import copy
import argparse
import itertools

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import FishTAL                                     # noqa: E402
from data import NUM_CLASSES                                  # noqa: E402
from train import evaluate                                    # noqa: E402


class Args:
    """Minimal stand-in for the argparse namespace evaluate() expects."""
    def __init__(self, d):
        self.__dict__.update(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="FishTAL6")
    p.add_argument("--ckpt-dir", default=f"{HERE}/checkpoints")
    p.add_argument("--out", default=f"{HERE}/sweep_inference.json")
    p.add_argument("--nms", type=float, nargs="+", default=[0.30, 0.45, 0.60])
    p.add_argument("--min-len", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    cli = p.parse_args()

    thresh_sets = {
        "low":    [0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
        "mid":    [0.10, 0.20, 0.30, 0.40, 0.50],
        "high":   [0.20, 0.30, 0.40, 0.50, 0.60],
        "single": [0.30],
    }

    ckpts = sorted(f for f in os.listdir(cli.ckpt_dir)
                   if f.startswith(f"{cli.label}__") and f.endswith(".pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints for {cli.label} in {cli.ckpt_dir} "
                         f"(train with --save-ckpt)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(ckpts)} folds for {cli.label} | device={device}", flush=True)

    loaded = []
    for f in ckpts:
        ck = torch.load(os.path.join(cli.ckpt_dir, f), map_location=device,
                        weights_only=False)
        cfg = ck["config"]
        model = FishTAL(feat_dim=ck["feat_dim"], num_classes=NUM_CLASSES,
                        hidden=cfg["hidden"], depth=cfg["depth"],
                        num_heads=cfg["num_heads"], drop=cfg["drop"],
                        attn_drop=cfg["drop"], drop_path=cfg["drop_path"],
                        use_conv=not cfg["no_conv"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        loaded.append((ck["test_rec"], model, cfg))

    results = {}
    for tname, nms, mlen in itertools.product(thresh_sets, cli.nms, cli.min_len):
        key = f"thr={tname} nms={nms} minlen={mlen}"
        maps, recs, precs = [], [], []
        for test_rec, model, cfg in loaded:
            a = Args({**cfg, "thresholds": thresh_sets[tname],
                      "nms_iou": nms, "min_len_s": mlen})
            r = evaluate(model, test_rec, a, device)
            maps.append(r["avg_map"])
            if not np.isnan(r["point_recall"]):
                recs.append(r["point_recall"])
            if not np.isnan(r["point_precision"]):
                precs.append(r["point_precision"])
        results[key] = {"avg_map": float(np.mean(maps)),
                        "avg_map_std": float(np.std(maps)),
                        "point_recall": float(np.mean(recs)) if recs else float("nan"),
                        "point_precision": float(np.mean(precs)) if precs else float("nan")}
        print(f"{key:42s} mAP={results[key]['avg_map']:.4f} "
              f"pr={results[key]['point_recall']:.3f} "
              f"pp={results[key]['point_precision']:.3f}", flush=True)

    best = max(results, key=lambda k: results[k]["avg_map"])
    print(f"\nBEST by mAP: {best} -> {results[best]}")
    allr = json.load(open(cli.out)) if os.path.isfile(cli.out) else {}
    allr[cli.label] = {"results": results, "best_key": best}
    json.dump(allr, open(cli.out, "w"), indent=1)
    print("wrote", cli.out)


if __name__ == "__main__":
    main()
