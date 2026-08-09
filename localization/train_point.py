"""FishPoint LOO training + evaluation -- pure point localization, no boxes.

Primary metric is point-AP (point_ap.py, confidence-ranked point-containment
AP), not tIoU-mAP: this model never regresses a span, so there is nothing
honest to compare against a proxy box with. Point-recall/precision (at the
inference operating point) are reported alongside for continuity with every
other model in EXPERIMENTS.md, but point-AP is the number to cite.

Usage: train_point.py --fold <recording|all> [--n-levels 1] [--use-offset] ...
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from point_model import FishPointModel, FishPointLoss                # noqa: E402
from data import (list_recordings, load_recording, class_weights_from,  # noqa: E402
                  FG_CLASSES, BG_INDEX, NUM_CLASSES, STRIDE)

sys.path.insert(0, "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training")
from point_ap import point_ap                                        # noqa: E402


def build_point_targets(events, n_steps, stride, cls_half=1.0, sigma=1.0):
    """Dense cls/act targets (same shape as FishTAL's build_targets) plus a
    sub-stride offset target, assigned ONLY at the single feature step nearest
    each event -- never a window, so it can't become a proxy box."""
    cls_t = np.full(n_steps, BG_INDEX, dtype=np.int64)
    act_t = np.zeros(n_steps, dtype=np.float32)
    off_t = np.zeros(n_steps, dtype=np.float32)
    off_mask = np.zeros(n_steps, dtype=bool)
    idx = np.arange(n_steps, dtype=np.float32) * stride
    for t, c in events:
        d = idx - t
        act_t = np.maximum(act_t, np.exp(-(d ** 2) / (2 * sigma ** 2)))
        near = np.abs(d) <= cls_half
        cls_t[near] = c
        nearest = int(np.argmin(np.abs(d)))
        off_t[nearest] = (t - idx[nearest]) / stride
        off_mask[nearest] = True
    return cls_t, act_t, off_t, off_mask


def targets_for_levels(events, win_start, win_len, stride, n_levels, cls_half, sigma):
    targets = []
    for lvl in range(n_levels):
        lvl_stride = 2 ** lvl
        t_len = max(1, win_len // lvl_stride)
        shifted = [(t - win_start * stride, c) for t, c in events]
        cls_t, act_t, off_t, off_m = build_point_targets(
            shifted, t_len, stride * lvl_stride, cls_half, sigma)
        targets.append({"cls": cls_t, "act": act_t, "offset": off_t, "offset_mask": off_m})
    return targets


class PointWindows(Dataset):
    def __init__(self, recordings, feature_mode, window_s, n_levels, cls_half, sigma,
                train=True, windows_per_rec=64, seed=0):
        self.stride = STRIDE[feature_mode]
        self.win = int(round(window_s / self.stride))
        self.n_levels, self.cls_half, self.sigma = n_levels, cls_half, sigma
        self.rng = np.random.RandomState(seed)
        self.recs = []
        for name in recordings:
            feats, events = load_recording(name, feature_mode)
            self.recs.append({"name": name, "feats": feats, "events": events})
        self.index = [(ri, None) for ri in range(len(self.recs)) for _ in range(windows_per_rec)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ri, _ = self.index[i]
        r = self.recs[ri]
        n = r["feats"].shape[0]
        start = self.rng.randint(0, max(1, n - self.win + 1))
        end = min(start + self.win, n)
        feats = r["feats"][start:end]
        if feats.shape[0] < self.win:
            feats = np.concatenate(
                [feats, np.zeros((self.win - feats.shape[0], *feats.shape[1:]), np.float32)])
        t0, t1 = start * self.stride, (start + self.win) * self.stride
        ev = [(t, c) for t, c in r["events"] if t0 <= t < t1]
        tg = targets_for_levels(ev, start, self.win, self.stride, self.n_levels,
                                self.cls_half, self.sigma)
        out = {"feats": torch.from_numpy(feats)}
        for lvl, t in enumerate(tg):
            out[f"cls{lvl}"] = torch.from_numpy(t["cls"])
            out[f"act{lvl}"] = torch.from_numpy(t["act"])
            out[f"offset{lvl}"] = torch.from_numpy(t["offset"])
            out[f"offset_mask{lvl}"] = torch.from_numpy(t["offset_mask"])
        return out


def _nms_by_distance(points, min_sep_s):
    """1-D analogue of IoU-NMS: drop a lower-score point if a kept point of the
    same class already lies within min_sep_s of it. points: [(t, score), ...]."""
    if not points:
        return []
    points = sorted(points, key=lambda x: -x[1])
    kept = []
    for t, sc in points:
        if all(abs(t - kt) >= min_sep_s for kt, _ in kept):
            kept.append((t, sc))
    return kept


@torch.no_grad()
def score_recording(model, name, args, device):
    """Stitch windowed predictions back into full-recording score curves, per
    level (level 0 = finest / native stride)."""
    model.eval()
    feats, events = load_recording(name, args.feature_mode)
    stride = STRIDE[args.feature_mode]
    n = feats.shape[0]
    win = int(round(args.window_s / stride))
    hop = win // 2
    starts = list(range(0, max(1, n - win + 1), hop))
    if n > win and starts[-1] != n - win:
        starts.append(n - win)

    n_levels = args.n_levels
    sums = [{"cls": np.zeros((n, NUM_CLASSES), np.float32), "act": np.zeros(n, np.float32),
            "offset": np.zeros(n, np.float32), "hits": np.zeros(n, np.float32)}
           for _ in range(n_levels)]

    for s in starts:
        e = min(s + win, n)
        chunk = feats[s:e]
        if chunk.shape[0] < win:
            chunk = np.concatenate(
                [chunk, np.zeros((win - chunk.shape[0], *chunk.shape[1:]), np.float32)])
        x = torch.from_numpy(chunk).unsqueeze(0).to(device)
        outs = model(x)
        for lvl, out in enumerate(outs):
            lvl_stride = out["stride"]
            p_cls = torch.softmax(out["cls"], dim=-1)[0].cpu().numpy()
            p_act = torch.sigmoid(out["act"])[0].cpu().numpy()
            p_off = out["offset"][0].cpu().numpy() if "offset" in out else None
            # round (not floor) the window's start to the nearest coarse-level
            # index -- windows start at multiples of hop=win//2, which isn't
            # always an exact multiple of lvl_stride for lvl>0, so floor
            # division would systematically bias coarse levels' stitching.
            base = int(round(s / lvl_stride))
            m = min(p_cls.shape[0], sums[lvl]["cls"].shape[0] - base)
            sl = slice(base, base + m)
            sums[lvl]["cls"][sl] += p_cls[:m]
            sums[lvl]["act"][sl] += p_act[:m]
            if p_off is not None:
                sums[lvl]["offset"][sl] += p_off[:m]
            sums[lvl]["hits"][sl] += 1.0

    out_levels = []
    for lvl in range(n_levels):
        hits = np.maximum(sums[lvl]["hits"], 1.0)
        out_levels.append({
            "cls": sums[lvl]["cls"] / hits[:, None], "act": sums[lvl]["act"] / hits,
            "offset": sums[lvl]["offset"] / hits if args.use_offset else None,
            "stride": stride * (2 ** lvl),
        })
    return out_levels, events, stride


def _nms_by_iou(cands, width_mult, iou_thr):
    """IoU-NMS the point analogue of FishFormer's span NMS: each candidate gets a
    NOMINAL box of width `width_mult * its own level stride` (coarser levels ->
    wider suppression radius, same intuition FCOS-style pyramids use for
    duration -- but here it is purely a decode-time NMS aid, never trained,
    never scored against GT, never output). Returns points, not boxes."""
    if not cands:
        return []
    boxed = sorted(((t - (width_mult * ls) / 2, t + (width_mult * ls) / 2, sc, t)
                    for t, sc, ls in cands), key=lambda x: -x[2])
    keep = []
    for s, e, sc, t in boxed:
        ok = True
        for ks, ke, _, _ in keep:
            inter = max(0.0, min(e, ke) - max(s, ks))
            union = (e - s) + (ke - ks) - inter
            if union > 0 and inter / union > iou_thr:
                ok = False
                break
        if ok:
            keep.append((s, e, sc, t))
    return [(t, sc) for _, _, sc, t in keep]


def decode_points(levels, thresh, nms_policy="distance", min_sep_s=1.0,
                  width_mult=2.0, iou_thr=0.3):
    """The cheap, CPU-only, sweepable half of inference -- everything after the
    (expensive, GPU) score_recording forward pass. Kept separate so a decode
    hyperparameter grid can be swept without re-running the model."""
    raw = {c: [] for c in range(len(FG_CLASSES))}
    for lvl in levels:
        s_curve = lvl["cls"][:, :len(FG_CLASSES)] * lvl["act"][:, None]
        lvl_stride = lvl["stride"]
        for c in range(len(FG_CLASSES)):
            sc = s_curve[:, c]
            idx = np.flatnonzero(sc >= thresh)
            for i in idx:
                off = float(lvl["offset"][i]) if lvl["offset"] is not None else 0.0
                t = (i + off) * lvl_stride
                raw[c].append((float(t), float(sc[i]), lvl_stride))
    if nms_policy == "distance":
        return {c: _nms_by_distance([(t, sc) for t, sc, _ in v], min_sep_s)
                for c, v in raw.items()}
    return {c: _nms_by_iou(v, width_mult, iou_thr) for c, v in raw.items()}


def evaluate_points(points, events, tolerance):
    gt_times = np.array([t for t, _ in events], dtype=float)
    gt_beh = np.array([FG_CLASSES[c] for _, c in events], dtype=object)

    aps, recs, precs = [], [], []
    per_class = {}
    for ci, label in enumerate(FG_CLASSES):
        pts = gt_times[gt_beh == label]
        if pts.size == 0:
            continue
        spans = [(t, t, sc) for t, sc in points[ci]]      # zero-width -- true points
        ap, rec, prec, _ = point_ap(spans, pts, tolerance=tolerance)
        if not np.isnan(ap):
            aps.append(ap)
            recs.append(rec)
            precs.append(prec)
            per_class[label] = {"point_ap": float(ap), "point_recall": float(rec),
                                "point_precision": float(prec), "n_gt": int(pts.size)}
    return {"point_ap": float(np.mean(aps)) if aps else float("nan"),
            "point_recall": float(np.mean(recs)) if recs else float("nan"),
            "point_precision": float(np.mean(precs)) if precs else float("nan"),
            "per_class": per_class}


def evaluate(model, name, args, device):
    levels, events, _ = score_recording(model, name, args, device)
    points = decode_points(levels, args.thresh, "distance", min_sep_s=args.min_sep_s)
    return evaluate_points(points, events, args.tolerance)


def train_fold(test_rec, all_recs, args, device):
    train_recs = [r for r in all_recs if r != test_rec]
    ds = PointWindows(train_recs, args.feature_mode, args.window_s, args.n_levels,
                      args.cls_half, args.sigma, windows_per_rec=args.windows_per_rec,
                      seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True)
    feat_dim = ds.recs[0]["feats"].shape[-1]
    model = FishPointModel(feat_dim=feat_dim, num_classes=NUM_CLASSES, hidden=args.hidden,
                           depth=args.depth, num_heads=args.num_heads, drop=args.drop,
                           drop_path=args.drop_path, n_levels=args.n_levels,
                           use_offset=args.use_offset).to(device)
    cw = class_weights_from(train_recs, args.feature_mode, args.cls_half) if args.class_weight else None
    crit = FishPointLoss(num_classes=NUM_CLASSES, class_weights=cw, lamb_cls=args.lamb_cls,
                         lamb_act=args.lamb_act, lamb_off=args.lamb_off).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = {"point_ap": -1.0, "point_recall": float("nan"), "point_precision": float("nan")}
    traj = []
    for ep in range(args.epochs):
        model.train(); agg = {}
        for batch in dl:
            feats = batch["feats"].to(device)
            tgts = [{"cls": batch[f"cls{l}"].to(device), "act": batch[f"act{l}"].to(device),
                    "offset": batch[f"offset{l}"].to(device),
                    "offset_mask": batch[f"offset_mask{l}"].to(device)}
                   for l in range(args.n_levels)]
            loss, parts = crit(model(feats), tgts)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip); opt.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
        sched.step()
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            res = evaluate(model, test_rec, args, device)
            msg = " ".join(f"{k}:{v/max(1,len(dl)):.4f}" for k, v in agg.items())
            print(f"[{test_rec}] ep{ep+1}/{args.epochs} {msg} | test point_ap={res['point_ap']:.4f} "
                  f"pr={res['point_recall']:.3f} pp={res['point_precision']:.3f}", flush=True)
            traj.append((ep + 1, res["point_ap"]))
            if res["point_ap"] > best["point_ap"]:
                best = res
                if args.save_ckpt:
                    os.makedirs(args.ckpt_dir, exist_ok=True)
                    torch.save({"model": model.state_dict(), "config": vars(args),
                               "feat_dim": feat_dim, "test_rec": test_rec, "result": res},
                              os.path.join(args.ckpt_dir, f"{args.label}__{test_rec}.pt"))
    return best, traj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", default="all")
    p.add_argument("--exclude-recs", default="",
                   help="comma-separated recordings to drop from the training pool "
                        "entirely (e.g. other held-out-from-MIL recordings that must "
                        "never appear in training even for a different fold's run)")
    p.add_argument("--feature-mode", default="patchx")
    p.add_argument("--window-s", type=float, default=90.0)
    p.add_argument("--n-levels", type=int, default=1)
    p.add_argument("--use-offset", action="store_true")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--drop", type=float, default=0.1)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--cls-half", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--lamb-cls", type=float, default=1.0)
    p.add_argument("--lamb-act", type=float, default=1.0)
    p.add_argument("--lamb-off", type=float, default=1.0)
    p.add_argument("--class-weight", action="store_true", default=True)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--windows-per-rec", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thresh", type=float, default=0.05)
    p.add_argument("--min-sep-s", type=float, default=1.0,
                   help="min time between two kept points of the same class (point-NMS)")
    p.add_argument("--tolerance", type=float, default=1.0,
                   help="eval slack (s): a predicted point counts as matching a GT "
                        "timestamp within this many seconds")
    p.add_argument("--label", default="FishPoint1")
    p.add_argument("--out", default=None)
    p.add_argument("--save-ckpt", action="store_true")
    p.add_argument("--ckpt-dir", default=f"{HERE}/checkpoints")
    args = p.parse_args()
    if args.out is None:
        args.out = f"{HERE}/point_results.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_recs = list_recordings(args.feature_mode)
    if args.exclude_recs:
        excluded = set(args.exclude_recs.split(","))
        all_recs = [r for r in all_recs if r not in excluded]
    # --fold also accepts a comma-separated list (e.g. a fixed 3-recording test
    # set for the leave-3 MIL backbone) so multiple folds aggregate into one
    # summary/label within a single process, instead of racing on the shared
    # results json if run as separate invocations under the same label.
    folds = all_recs if args.fold == "all" else args.fold.split(",")
    print(f"{len(all_recs)} recordings | device={device} | label={args.label}", flush=True)

    per_fold = {}
    for test_rec in folds:
        print(f"\n=== FOLD: held out {test_rec} ===", flush=True)
        best, traj = train_fold(test_rec, all_recs, args, device)
        per_fold[test_rec] = {"point_ap": best["point_ap"], "point_recall": best["point_recall"],
                              "point_precision": best["point_precision"], "traj": traj,
                              "per_class": best.get("per_class", {})}
        print(f"=== fold done: {test_rec} -> point_ap={best['point_ap']:.4f}", flush=True)

    aps = [v["point_ap"] for v in per_fold.values() if not np.isnan(v["point_ap"])]
    recs_ = [v["point_recall"] for v in per_fold.values() if not np.isnan(v["point_recall"])]
    precs_ = [v["point_precision"] for v in per_fold.values() if not np.isnan(v["point_precision"])]
    # Per-class point-AP, averaged across folds.
    class_ap = {c: [] for c in FG_CLASSES}
    for v in per_fold.values():
        for c in FG_CLASSES:
            pc = v.get("per_class", {}).get(c, {}).get("point_ap")
            if pc is not None and not np.isnan(pc):
                class_ap[c].append(pc)
    per_class_point_ap = {c: (float(np.mean(v)) if v else float("nan")) for c, v in class_ap.items()}
    summary = {
        "label": args.label, "config": vars(args), "folds": per_fold,
        "n_folds_complete": len(aps),
        "point_ap_mean": float(np.mean(aps)) if aps else float("nan"),
        "point_ap_std": float(np.std(aps)) if aps else float("nan"),
        "point_recall_mean": float(np.mean(recs_)) if recs_ else float("nan"),
        "point_precision_mean": float(np.mean(precs_)) if precs_ else float("nan"),
        "per_class_point_ap": per_class_point_ap,
    }
    print(f"\n=== {args.label}: point-AP = {summary['point_ap_mean']:.4f} +/- "
          f"{summary['point_ap_std']:.4f} | point-recall {summary['point_recall_mean']:.3f} "
          f"precision {summary['point_precision_mean']:.3f} ({summary['n_folds_complete']} folds) ===")

    out = json.load(open(args.out)) if os.path.isfile(args.out) else {}
    out[args.label] = summary
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out} {list(out.keys())}")


if __name__ == "__main__":
    main()
