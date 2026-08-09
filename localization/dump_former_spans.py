"""Retrain FishFormer for an already-reported config and, at each fold's best
epoch, dump the predicted spans for that held-out recording to disk.

train_former.py never checkpoints a model or saves predictions -- only the
aggregate metrics land in fishtal_results.json -- so there is nothing to load
for a box-timeline visualization. This script mirrors train_fold()'s loop
exactly (same hyperparameters, same seed) so the reproduced metrics match
what's already reported; the only addition is capturing predict_spans()
output at the epoch that set a new best avg_map, matching how train_former.py
itself decides "best".

Usage: same flags as train_former.py, e.g.
    python dump_former_spans.py --feature-mode leave3_nf16 \
        --fold 25-05-22-Run1-Sham-Cir,25-06-26-Run2-VetBond-NoCir,25-08-13-Run1-Sham-Cir \
        --epochs 100 --hidden 384 --depth 6 --lr 0.0005 --span-s 4.0 \
        --label Leave3_NF16_Former2
Writes span_dumps/<label>__<recording>.json per fold held out.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from former import FishFormer, FishFormerLoss                        # noqa: E402
from data import list_recordings, FG_CLASSES, BG_INDEX, NUM_CLASSES  # noqa: E402
import train_former                                                  # noqa: E402
from train_former import FormerWindows, predict_spans, evaluate      # noqa: E402

OUT_DIR = f"{HERE}/span_dumps"
os.makedirs(OUT_DIR, exist_ok=True)


def train_fold_and_dump(test_rec, all_recs, args, device, label):
    train_recs = [r for r in all_recs if r != test_rec]
    ds = FormerWindows(train_recs, args.feature_mode, args.window_s, args.span_s,
                       args.n_levels, windows_per_rec=args.windows_per_rec, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True)
    sample_feats = ds.recs[0]["feats"]
    feat_dim = sample_feats.shape[-1]
    spatial_pool = sample_feats.ndim == 3
    model = FishFormer(feat_dim=feat_dim, num_classes=NUM_CLASSES, hidden=args.hidden,
                       depth=args.depth, num_heads=args.num_heads, drop=args.drop,
                       drop_path=args.drop_path, n_levels=args.n_levels,
                       reg_bins=args.reg_bins, spatial_pool=spatial_pool,
                       use_motion=args.use_motion).to(device)
    crit = FishFormerLoss(num_classes=NUM_CLASSES, bg_index=BG_INDEX,
                          lamb_cls=args.lamb_cls, lamb_reg=args.lamb_reg,
                          lamb_ctr=args.lamb_ctr).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = {"avg_map": -1.0}
    best_dump = None
    for ep in range(args.epochs):
        model.train()
        for batch in dl:
            feats = batch["feats"].to(device)
            tgts = [{"cls": batch[f"cls{l}"].to(device), "reg": batch[f"reg{l}"].to(device),
                     "ctr": batch[f"ctr{l}"].to(device), "pos": batch[f"pos{l}"].to(device)}
                    for l in range(args.n_levels)]
            loss, _ = crit(model(feats), tgts)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip); opt.step()
        sched.step()
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            res = evaluate(model, test_rec, args, device)
            print(f"[{test_rec}] ep{ep+1}/{args.epochs} avg_mAP={res['avg_map']:.4f} "
                  f"pr={res['point_recall']:.3f} pp={res['point_precision']:.3f}", flush=True)
            if res["avg_map"] > best["avg_map"]:
                best = res
                props, events, stride, n = predict_spans(model, test_rec, args, device)
                best_dump = {
                    "label": label, "recording": test_rec, "epoch": ep + 1,
                    "stride": stride, "n_steps": n, "duration": n * stride,
                    "gt_events": [[float(t), FG_CLASSES[c]] for t, c in events],
                    "spans": {FG_CLASSES[c]: [[float(s), float(e), float(sc)] for s, e, sc in v]
                              for c, v in props.items()},
                    "metrics": {k: v for k, v in res.items() if k != "per_iou"},
                }
    out_path = f"{OUT_DIR}/{label}__{test_rec}.json"
    json.dump(best_dump, open(out_path, "w"), indent=1)
    print(f"=== fold done: {test_rec} -> avg_mAP={best['avg_map']:.4f}, wrote {out_path}", flush=True)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", default="all")
    p.add_argument("--exclude-recs", default="")
    p.add_argument("--feature-mode", default="patchx")
    p.add_argument("--window-s", type=float, default=90.0)
    p.add_argument("--span-s", type=float, default=2.0)
    p.add_argument("--n-levels", type=int, default=4)
    p.add_argument("--reg-bins", type=int, default=16)
    p.add_argument("--use-motion", action="store_true")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--drop", type=float, default=0.1)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--lamb-cls", type=float, default=1.0)
    p.add_argument("--lamb-reg", type=float, default=1.0)
    p.add_argument("--lamb-ctr", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--windows-per-rec", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--score-thresh", type=float, default=0.05)
    p.add_argument("--nms-iou", type=float, default=0.45)
    p.add_argument("--max-props", type=int, default=400)
    p.add_argument("--gt-window", type=float, default=4.0)
    p.add_argument("--label", default="FishFormer1")
    args = p.parse_args()

    train_former.GT_DURATION = args.gt_window
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_recs = list_recordings(args.feature_mode)
    if args.exclude_recs:
        excluded = set(args.exclude_recs.split(","))
        all_recs = [r for r in all_recs if r not in excluded]
    print(f"{len(all_recs)} recordings | device={device} | label={args.label}", flush=True)

    folds = all_recs if args.fold == "all" else args.fold.split(",")
    for rec in folds:
        print(f"\n=== FOLD: held out {rec} ===", flush=True)
        train_fold_and_dump(rec, all_recs, args, device, args.label)


if __name__ == "__main__":
    main()
