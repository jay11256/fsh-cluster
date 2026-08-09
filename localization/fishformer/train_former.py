"""FishFormer LOO training + evaluation.

Same harness and same +/-2s proxy-box evaluation as FishTAL and every ASM-Loc
sweep, so numbers are directly comparable.

The one modelling decision worth stating plainly: BORIS gives a *timestamp*, not
a duration, so there is no true extent to regress toward. We supervise the
regression head with a nominal span of `--span-s` seconds centred on each point
(default 2s, i.e. narrower than the 4s eval box) and let the distributional
regression head absorb the uncertainty. `--span-s` is therefore a real
hyperparameter of the method, not a fact about the data -- it is swept.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
from .former import FishFormer, FishFormerLoss                       # noqa: E402
from .data import (list_recordings, load_recording, gap_none_windows,  # noqa: E402
                   FG_CLASSES, BG_INDEX, NUM_CLASSES, STRIDE)
from .nms import _nms                                                # noqa: E402

# mAP helpers live in the sibling `pipeline/` package of this repo; resolved
# relative to this file so a clone works from any checkout path.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(HERE)), "pipeline"))
from visualize_matrix import _make_gt_spans, _match_detections, _compute_ap  # noqa: E402

IOUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
GT_DURATION = 4.0          # eval-only proxy box; overridden by --gt-window below


def assign_targets(events, win_start, win_len, stride, span_s):
    """Anchor-free assignment of point events to timesteps, single scale.

    A position is positive for an event when it lies strictly inside that
    event's `span_s` box; it then regresses the distances to the box's two
    edges. Returned as a one-element list to match the model's single-element
    output list.

    There is no per-level accept range any more. The pyramid version gated each
    level on whether max(left, right) fell in that level's [lo, hi) band, which
    only does useful work when ground-truth segments vary in duration. Here
    every segment is exactly `span_s` wide by construction -- it is a BORIS
    point expanded to a fixed box -- so max(left, right) is always in
    [span_s/2, span_s] and only the finest level ever qualified. See former.py's
    module docstring for the measurement.
    """
    half = span_s / 2.0
    t_len = max(1, win_len)
    cls_t = np.full(t_len, BG_INDEX, np.int64)
    reg_t = np.zeros((t_len, 2), np.float32)
    ctr_t = np.zeros(t_len, np.float32)
    pos = np.zeros(t_len, bool)

    for t_abs, c in events:
        t_rel = t_abs - win_start * stride          # seconds into window
        s, e = t_rel - half, t_rel + half
        idx = np.arange(t_len, dtype=np.float32) * stride
        left = (idx - s) / stride                    # distance in steps
        right = (e - idx) / stride
        fits = (left > 0) & (right > 0)
        if not fits.any():
            continue
        cls_t[fits] = c
        reg_t[fits, 0] = left[fits]
        reg_t[fits, 1] = right[fits]
        # centerness: 1 at the event centre, decaying toward the edges
        ctr_t[fits] = np.sqrt(
            np.minimum(left[fits], right[fits]) /
            np.maximum(np.maximum(left[fits], right[fits]), 1e-6))
        pos |= fits
    return [{"cls": cls_t, "reg": reg_t, "ctr": ctr_t, "pos": pos}]


class FormerWindows(Dataset):
    """`none_ratio` adds curated hard-negative windows on top of the usual
    windows_per_rec uniform-random ones. Each recording's None-clip anchors
    (data11make's own gap-sampled, pre-annotation 4s hard negatives -- see
    gap_none_windows) get an extra `windows_per_rec * none_ratio /
    (1-none_ratio)` window draws, jittered to a random offset around the
    anchor each access, so the ratio of None-anchored : uniform windows in
    the enlarged pool matches none_ratio the same way data11make's own
    none_count/total_count target does. Purely additive: uniform draws are
    untouched, none_ratio=0.0 (default) reproduces the old behavior exactly.
    """

    def __init__(self, recordings, feature_mode, window_s, span_s,
                 train=True, windows_per_rec=64, seed=0, none_ratio=0.0):
        self.stride = STRIDE[feature_mode]
        self.win = int(round(window_s / self.stride))
        self.span_s = span_s
        self.train = train
        self.rng = np.random.RandomState(seed)
        self.recs = []
        self.none_anchors = []
        for name in recordings:
            feats, events = load_recording(name, feature_mode)
            self.recs.append({"name": name, "feats": feats, "events": events})
            centers = []
            if none_ratio > 0:
                duration = feats.shape[0] * self.stride
                windows = gap_none_windows(name, duration, none_ratio=none_ratio, seed=seed)
                centers = [((s + e) / 2.0) / self.stride for s, e in windows]
            self.none_anchors.append(centers)

        self.index = [(ri, None) for ri in range(len(self.recs))
                      for _ in range(windows_per_rec)]
        if none_ratio > 0:
            n_none = int(round(windows_per_rec * none_ratio / (1.0 - none_ratio)))
            for ri, centers in enumerate(self.none_anchors):
                if centers:
                    self.index += [(ri, "none")] * n_none

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ri, kind = self.index[i]
        r = self.recs[ri]
        n = r["feats"].shape[0]
        if kind == "none":
            centers = self.none_anchors[ri]
            c = centers[self.rng.randint(0, len(centers))]
            start = int(round(c - self.rng.uniform(0.1, 0.9) * self.win))
            start = int(np.clip(start, 0, max(0, n - self.win)))
        else:
            start = self.rng.randint(0, max(1, n - self.win + 1))
        end = min(start + self.win, n)
        feats = r["feats"][start:end]
        if feats.shape[0] < self.win:
            feats = np.concatenate(
                [feats, np.zeros((self.win - feats.shape[0], *feats.shape[1:]), np.float32)])
        t0, t1 = start * self.stride, (start + self.win) * self.stride
        ev = [(t, c) for t, c in r["events"] if t0 <= t < t1]
        tg = assign_targets(ev, start, self.win, self.stride, self.span_s)
        out = {"feats": torch.from_numpy(feats)}
        for lvl, t in enumerate(tg):
            out[f"cls{lvl}"] = torch.from_numpy(t["cls"])
            out[f"reg{lvl}"] = torch.from_numpy(t["reg"])
            out[f"ctr{lvl}"] = torch.from_numpy(t["ctr"])
            out[f"pos{lvl}"] = torch.from_numpy(t["pos"])
        return out


@torch.no_grad()
def predict_spans(model, name, args, device):
    """Decode boundary-regressed spans over a full recording."""
    model.eval()
    feats, events = load_recording(name, args.feature_mode)
    stride = STRIDE[args.feature_mode]
    n = feats.shape[0]
    win = int(round(args.window_s / stride))
    hop = win // 2
    starts = list(range(0, max(1, n - win + 1), hop))
    if n > win and starts[-1] != n - win:
        starts.append(n - win)

    raw = {c: [] for c in range(len(FG_CLASSES))}
    for s in starts:
        e = min(s + win, n)
        chunk = feats[s:e]
        if chunk.shape[0] < win:
            chunk = np.concatenate(
                [chunk, np.zeros((win - chunk.shape[0], *chunk.shape[1:]), np.float32)])
        outs = model(torch.from_numpy(chunk).unsqueeze(0).to(device))
        for out in outs:
            lvl_stride = out["stride"]
            scores = torch.sigmoid(out["cls"])[0] * torch.sigmoid(out["ctr"])[0].unsqueeze(-1)
            reg = out["reg"][0]
            sc = scores.cpu().numpy()
            rg = reg.cpu().numpy()
            t_idx = np.arange(sc.shape[0]) * lvl_stride + s          # in base steps
            for c in range(len(FG_CLASSES)):
                keep = np.flatnonzero(sc[:, c] >= args.score_thresh)
                for k in keep:
                    centre = t_idx[k] * stride
                    st = centre - rg[k, 0] * lvl_stride * stride
                    en = centre + rg[k, 1] * lvl_stride * stride
                    if en > st:
                        raw[c].append((float(st), float(en), float(sc[k, c])))
    return {c: _nms(v, args.nms_iou)[:args.max_props] for c, v in raw.items()}, events, stride, n


def evaluate(model, name, args, device):
    props, events, stride, n = predict_spans(model, name, args, device)
    duration = n * stride
    gt_times = np.array([t for t, _ in events], dtype=float)
    gt_beh = np.array([FG_CLASSES[c] for _, c in events], dtype=object)

    ap = {i: [] for i in IOUS}
    per_class = {}
    rec_l, prec_l = [], []
    for ci, label in enumerate(FG_CLASSES):
        spans = props[ci]
        gt_spans = _make_gt_spans(gt_times, gt_beh, label, gt_duration=GT_DURATION,
                                  min_time=0.0, max_time=duration)
        pts = gt_times[gt_beh == label]
        cls_recall = cls_precision = float("nan")
        if pts.size:
            cls_recall = float(np.mean([any(s <= t <= e for s, e, _ in spans) for t in pts]))
            rec_l.append(cls_recall)
        if spans:
            cls_precision = float(np.mean([any(s <= t <= e for t in pts) for s, e, _ in spans]))
            prec_l.append(cls_precision)
        cls_per_iou = {}
        if gt_spans:
            for iou_t in IOUS:
                tp, fp, _ = _match_detections(gt_spans, spans, iou_t)
                tpc, fpc = np.cumsum(tp), np.cumsum(fp)
                a = _compute_ap(tpc / len(gt_spans), tpc / np.maximum(tpc + fpc, 1e-8))
                ap[iou_t].append(a)
                cls_per_iou[iou_t] = float(a)
        per_class[label] = {
            "per_iou": cls_per_iou,
            "avg_map": float(np.mean(list(cls_per_iou.values()))) if cls_per_iou else float("nan"),
            "point_recall": cls_recall, "point_precision": cls_precision,
            "n_gt": int(pts.size),
        }
    per_iou = [float(np.mean(ap[i])) if ap[i] else float("nan") for i in IOUS]
    return {"per_iou": per_iou, "avg_map": float(np.nanmean(per_iou)),
            "point_recall": float(np.mean(rec_l)) if rec_l else float("nan"),
            "point_precision": float(np.mean(prec_l)) if prec_l else float("nan"),
            "per_class": per_class}


def train_fold(test_rec, all_recs, args, device):
    train_recs = [r for r in all_recs if r != test_rec]
    ds = FormerWindows(train_recs, args.feature_mode, args.window_s, args.span_s,
                       windows_per_rec=args.windows_per_rec, seed=args.seed,
                       none_ratio=args.none_ratio)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True)
    sample_feats = ds.recs[0]["feats"]
    feat_dim = sample_feats.shape[-1]
    spatial_pool = sample_feats.ndim == 3   # (T,P,D) coarse DINO grid vs (T,D) pooled
    model = FishFormer(feat_dim=feat_dim, num_classes=NUM_CLASSES, hidden=args.hidden,
                       depth=args.depth, num_heads=args.num_heads, drop=args.drop,
                       drop_path=args.drop_path,
                       reg_bins=args.reg_bins, spatial_pool=spatial_pool,
                       use_motion=args.use_motion).to(device)
    crit = FishFormerLoss(num_classes=NUM_CLASSES, bg_index=BG_INDEX,
                          lamb_cls=args.lamb_cls, lamb_reg=args.lamb_reg,
                          lamb_ctr=args.lamb_ctr).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = {"avg_map": -1.0}
    best_state = None
    for ep in range(args.epochs):
        model.train(); agg = {}
        for batch in dl:
            feats = batch["feats"].to(device)
            tgts = [{"cls": batch[f"cls{l}"].to(device), "reg": batch[f"reg{l}"].to(device),
                     "ctr": batch[f"ctr{l}"].to(device), "pos": batch[f"pos{l}"].to(device)}
                    for l in range(1)]          # single output level
            loss, parts = crit(model(feats), tgts)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip); opt.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
        sched.step()
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            res = evaluate(model, test_rec, args, device)
            msg = " ".join(f"{k}:{v/max(1,len(dl)):.4f}" for k, v in agg.items())
            print(f"[{test_rec}] ep{ep+1}/{args.epochs} {msg} | test avg_mAP={res['avg_map']:.4f} "
                  f"pr={res['point_recall']:.3f} pp={res['point_precision']:.3f}", flush=True)
            if res["avg_map"] > best["avg_map"]:
                best = res
                if args.save_ckpt:
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if args.save_ckpt and best_state is not None:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        torch.save({"model": best_state, "config": vars(args), "feat_dim": feat_dim,
                   "test_rec": test_rec, "result": best},
                  os.path.join(args.ckpt_dir, f"{args.label}__{test_rec}.pt"))
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", default="all")
    p.add_argument("--exclude-recs", default="",
                   help="comma-separated recordings to drop from the training pool "
                        "entirely (e.g. other held-out-from-MIL recordings that must "
                        "never appear in training even for a different fold's run)")
    p.add_argument("--feature-mode", default="patchx")
    p.add_argument("--window-s", type=float, default=90.0)
    p.add_argument("--span-s", type=float, default=2.0)
    p.add_argument("--reg-bins", type=int, default=16)
    p.add_argument("--use-motion", action="store_true",
                   help="add CrossMotionLite, a learned temporal-difference block, before the trunk")
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
    p.add_argument("--none-ratio", type=float, default=0.0,
                   help="add curated hard-negative (pre-annotation gap) training windows on "
                        "top of windows_per_rec, at this target None:total ratio in the "
                        "enlarged pool (0.0 = off, exactly reproduces old behavior). Reuses "
                        "data11make.py's own gap-sampling algorithm -- see gap_none_windows.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--score-thresh", type=float, default=0.05)
    p.add_argument("--nms-iou", type=float, default=0.45)
    p.add_argument("--max-props", type=int, default=400)
    p.add_argument("--gt-window", type=float, default=4.0,
                   help="eval-only proxy box width (s): 4.0 = +/-2s (default), 2.0 = +/-1s (gtw1)")
    p.add_argument("--label", default="FishFormer1")
    p.add_argument("--out", default=None,
                   help="default: fishtal_results.json (gt-window=4.0) or fishtal_results_gtw1.json (2.0)")
    p.add_argument("--save-ckpt", action="store_true",
                   help="save the best-epoch model weights per fold (train_former.py never did "
                        "this before -- needed for any post-hoc re-evaluation, e.g. per-class)")
    p.add_argument("--ckpt-dir", default=f"{HERE}/checkpoints")
    args = p.parse_args()
    if args.out is None:
        suffix = "" if args.gt_window == 4.0 else "_gtw1"
        args.out = f"{HERE}/fishtal_results{suffix}.json"

    global GT_DURATION
    GT_DURATION = args.gt_window
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_recs = list_recordings(args.feature_mode)
    if args.exclude_recs:
        excluded = set(args.exclude_recs.split(","))
        all_recs = [r for r in all_recs if r not in excluded]
    print(f"{len(all_recs)} recordings | device={device} | label={args.label}", flush=True)

    # --fold also accepts a comma-separated list (e.g. a fixed 3-recording test
    # set for the leave-3 MIL backbone) so multiple folds aggregate into one
    # summary/label within a single process, instead of racing on the shared
    # results json if run as separate invocations under the same label.
    folds = all_recs if args.fold == "all" else args.fold.split(",")
    results = {}
    for rec in folds:
        print(f"\n=== FOLD: held out {rec} ===", flush=True)
        results[rec] = train_fold(rec, all_recs, args, device)
        print(f"=== fold done: {rec} -> avg_mAP={results[rec]['avg_map']:.4f}", flush=True)

    per_iou = np.array([r["per_iou"] for r in results.values()], dtype=float)
    avgs = [r["avg_map"] for r in results.values()]
    rp = [r["point_recall"] for r in results.values() if not np.isnan(r["point_recall"])]
    pp = [r["point_precision"] for r in results.values() if not np.isnan(r["point_precision"])]
    # Per-class avg_map, averaged across folds (each fold's own per_class dict
    # comes straight out of evaluate() -- see per_class in that function).
    class_avg = {c: [] for c in FG_CLASSES}
    for r in results.values():
        for c in FG_CLASSES:
            v = r.get("per_class", {}).get(c, {}).get("avg_map")
            if v is not None and not np.isnan(v):
                class_avg[c].append(v)
    per_class_avg_map = {c: (float(np.mean(v)) if v else float("nan")) for c, v in class_avg.items()}
    summary = {"label": args.label, "config": vars(args), "folds": results,
               "n_folds_complete": len(results),
               "per_iou_mean": np.nanmean(per_iou, axis=0).tolist(),
               "per_iou_std": np.nanstd(per_iou, axis=0).tolist(),
               "avg_mean": float(np.mean(avgs)), "avg_std": float(np.std(avgs)),
               "point_recall_mean": float(np.mean(rp)) if rp else float("nan"),
               "point_recall_std": float(np.std(rp)) if rp else float("nan"),
               "point_precision_mean": float(np.mean(pp)) if pp else float("nan"),
               "point_precision_std": float(np.std(pp)) if pp else float("nan"),
               "per_class_avg_map": per_class_avg_map,
               "traj_mean": [], "traj_std": []}
    print(f"\n=== {args.label}: avg mAP = {summary['avg_mean']:.4f} +/- {summary['avg_std']:.4f}"
          f" | point-recall {summary['point_recall_mean']:.3f} precision "
          f"{summary['point_precision_mean']:.3f} ({len(results)} folds) ===", flush=True)
    allr = json.load(open(args.out)) if os.path.isfile(args.out) else {}
    allr[args.label] = summary
    json.dump(allr, open(args.out, "w"), indent=1)
    print("wrote", args.out, sorted(allr.keys()), flush=True)


if __name__ == "__main__":
    main()
