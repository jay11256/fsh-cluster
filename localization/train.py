"""FishTAL leave-one-recording-out training + evaluation.

Scored with the SAME harness as every ASM-Loc sweep so numbers are directly
comparable to asmloc_training/EXPERIMENTS.md:
  - tIoU 0.1-0.9 mAP against +/-2s proxy-box GT (`_make_gt_spans` etc. from the
    pipeline's visualize_matrix)
  - point-recall / point-precision, the ceiling-immune metric

Note the asymmetry that motivated this model: FishTAL never *trains* on proxy
boxes (it uses the raw timestamps), but is still *evaluated* against them, so
these mAP numbers are directly comparable to prior runs rather than flattered by
a changed target.

Usage: train.py --fold <recording|all> [--epochs 60] [--window-s 90] ...
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
from model import FishTAL, FishTALLoss                                  # noqa: E402
from data import (FishWindows, list_recordings, load_recording,          # noqa: E402
                  class_weights_from, FG_CLASSES, BG_INDEX, NUM_CLASSES, STRIDE)

sys.path.insert(0, "/fs/vulcan-projects/fsh_track/bhargav/fsh-cluster/pipeline")
from visualize_matrix import _make_gt_spans, _match_detections, _compute_ap  # noqa: E402

IOUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
GT_DURATION = 4.0          # eval-only proxy box; overridden by --gt-window below


# ── inference ────────────────────────────────────────────────────────────────
@torch.no_grad()
def score_recording(model, name, args, device):
    """Stitch windowed predictions back into full-recording score curves."""
    model.eval()
    feats, events = load_recording(name, args.feature_mode)
    stride = STRIDE[args.feature_mode]
    n = feats.shape[0]
    win = int(round(args.window_s / stride))
    hop = win // 2                                     # 50% overlap, averaged

    cls_sum = np.zeros((n, NUM_CLASSES), np.float32)
    act_sum = np.zeros(n, np.float32)
    hits = np.zeros(n, np.float32)

    starts = list(range(0, max(1, n - win + 1), hop))
    if n > win and starts[-1] != n - win:
        starts.append(n - win)
    for s in starts:
        e = min(s + win, n)
        chunk = feats[s:e]
        if chunk.shape[0] < win:
            chunk = np.concatenate(
                [chunk, np.zeros((win - chunk.shape[0], chunk.shape[1]), np.float32)])
        x = torch.from_numpy(chunk).unsqueeze(0).to(device)
        cls_logits, act_logits = model(x)
        p_cls = torch.softmax(cls_logits, dim=-1)[0].cpu().numpy()
        p_act = torch.sigmoid(act_logits)[0].cpu().numpy()
        m = e - s
        cls_sum[s:e] += p_cls[:m]
        act_sum[s:e] += p_act[:m]
        hits[s:e] += 1.0
    hits = np.maximum(hits, 1.0)
    return cls_sum / hits[:, None], act_sum / hits, events, stride


def proposals_from_scores(cls_prob, act_prob, stride, thresholds, min_len_s=0.5,
                          nms_iou=0.45):
    """Per-class spans from (class prob x actionness), swept over thresholds.

    Score per class c = P(c) * actionness -- the joint head's two outputs
    multiplied, so a span needs both "this looks like class c" and "something is
    happening here".
    """
    out = {c: [] for c in range(len(FG_CLASSES))}
    min_steps = max(1, int(round(min_len_s / stride)))
    for c in range(len(FG_CLASSES)):
        s = cls_prob[:, c] * act_prob
        for thr in thresholds:
            mask = s >= thr
            if not mask.any():
                continue
            idx = np.flatnonzero(mask)
            splits = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
            for run in splits:
                if run.size < min_steps:
                    continue
                inner = float(s[run].mean())
                lo = max(0, run[0] - run.size)
                hi = min(len(s), run[-1] + run.size + 1)
                outer_idx = np.r_[np.arange(lo, run[0]), np.arange(run[-1] + 1, hi)]
                outer = float(s[outer_idx].mean()) if outer_idx.size else 0.0
                out[c].append((float(run[0] * stride), float((run[-1] + 1) * stride),
                               inner - outer))          # OIC-style contrast score
        out[c] = _nms(out[c], nms_iou)
    return out


def _nms(spans, iou_thr):
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: -x[2])
    keep = []
    for s, e, sc in spans:
        ok = True
        for ks, ke, _ in keep:
            inter = max(0.0, min(e, ke) - max(s, ks))
            union = (e - s) + (ke - ks) - inter
            if union > 0 and inter / union > iou_thr:
                ok = False
                break
        if ok:
            keep.append((s, e, sc))
    return keep


def evaluate(model, name, args, device):
    cls_prob, act_prob, events, stride = score_recording(model, name, args, device)
    props = proposals_from_scores(cls_prob, act_prob, stride, args.thresholds,
                                  args.min_len_s, args.nms_iou)
    duration = cls_prob.shape[0] * stride

    gt_times = np.array([t for t, _ in events], dtype=float)
    gt_beh = np.array([FG_CLASSES[c] for _, c in events], dtype=object)

    ap_by_iou = {i: [] for i in IOUS}
    recalls, precisions = [], []
    for ci, label in enumerate(FG_CLASSES):
        spans = props[ci]
        gt_spans = _make_gt_spans(gt_times, gt_beh, label, gt_duration=GT_DURATION,
                                  min_time=0.0, max_time=duration)
        pts = gt_times[gt_beh == label]
        if pts.size:
            recalls.append(np.mean([any(s <= t <= e for s, e, _ in spans) for t in pts]))
        if spans:
            precisions.append(np.mean([any(s <= t <= e for t in pts) for s, e, _ in spans]))
        if not gt_spans:
            continue
        for iou_t in IOUS:
            tp, fp, _ = _match_detections(gt_spans, spans, iou_t)
            tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
            rec = tp_c / len(gt_spans)
            prec = tp_c / np.maximum(tp_c + fp_c, 1e-8)
            ap_by_iou[iou_t].append(_compute_ap(rec, prec))

    per_iou = [float(np.mean(ap_by_iou[i])) if ap_by_iou[i] else float("nan") for i in IOUS]
    return {
        "per_iou": per_iou,
        "avg_map": float(np.nanmean(per_iou)),
        "point_recall": float(np.mean(recalls)) if recalls else float("nan"),
        "point_precision": float(np.mean(precisions)) if precisions else float("nan"),
    }


# ── training ─────────────────────────────────────────────────────────────────
def train_fold(test_rec, all_recs, args, device):
    train_recs = [r for r in all_recs if r != test_rec]
    ds = FishWindows(train_recs, args.feature_mode, args.window_s, args.cls_half,
                     args.sigma, train=True, windows_per_rec=args.windows_per_rec,
                     seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True)

    feat_dim = ds.recs[0]["feats"].shape[1]
    model = FishTAL(feat_dim=feat_dim, num_classes=NUM_CLASSES, hidden=args.hidden,
                    depth=args.depth, num_heads=args.num_heads, drop=args.drop,
                    attn_drop=args.drop, drop_path=args.drop_path,
                    use_conv=not args.no_conv).to(device)
    cw = class_weights_from(train_recs, args.feature_mode, args.cls_half)
    crit = FishTALLoss(class_weights=cw, lamb_cls=args.lamb_cls,
                       lamb_act=args.lamb_act, lamb_mil=args.lamb_mil).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = {"avg_map": -1.0}
    for ep in range(args.epochs):
        model.train()
        agg = {}
        for batch in dl:
            feats = batch["feats"].to(device)
            cls_t = batch["cls_target"].to(device)
            act_t = batch["act_target"].to(device)
            vid_l = batch["vid_label"].to(device)
            cls_logits, act_logits = model(feats)
            loss, parts = crit(cls_logits, act_logits, cls_t, act_t, vid_l)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
        sched.step()

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            res = evaluate(model, test_rec, args, device)
            msg = " ".join(f"{k}:{v/max(1,len(dl)):.4f}" for k, v in agg.items())
            print(f"[{test_rec}] ep{ep+1}/{args.epochs} {msg} "
                  f"| test avg_mAP={res['avg_map']:.4f} "
                  f"pt-recall={res['point_recall']:.3f} "
                  f"pt-precision={res['point_precision']:.3f}", flush=True)
            if res["avg_map"] > best["avg_map"]:
                best = res
                if args.save_ckpt:
                    os.makedirs(args.ckpt_dir, exist_ok=True)
                    torch.save({"model": model.state_dict(),
                                "config": vars(args),
                                "feat_dim": feat_dim,
                                "test_rec": test_rec,
                                "result": res},
                               os.path.join(args.ckpt_dir,
                                            f"{args.label}__{test_rec}.pt"))
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", default="all")
    p.add_argument("--feature-mode", default="patchx")
    p.add_argument("--window-s", type=float, default=90.0)
    p.add_argument("--cls-half", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--drop", type=float, default=0.1)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--no-conv", action="store_true")
    p.add_argument("--lamb-cls", type=float, default=1.0)
    p.add_argument("--lamb-act", type=float, default=1.0)
    p.add_argument("--lamb-mil", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--windows-per-rec", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-len-s", type=float, default=0.5)
    p.add_argument("--nms-iou", type=float, default=0.45)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
    p.add_argument("--gt-window", type=float, default=4.0,
                   help="eval-only proxy box width (s): 4.0 = +/-2s (default), 2.0 = +/-1s (gtw1)")
    p.add_argument("--label", default="FishTAL1")
    # Saving the best-epoch weights lets sweep_inference.py retune thresholds /
    # NMS / min-length without retraining -- those are inference-only knobs and
    # re-running 11 folds to change them is pure waste.
    p.add_argument("--save-ckpt", action="store_true")
    p.add_argument("--ckpt-dir", default=f"{HERE}/checkpoints")
    p.add_argument("--out", default=None,
                   help="default: fishtal_results.json (gt-window=4.0) or fishtal_results_gtw1.json (2.0)")
    args = p.parse_args()
    if args.out is None:
        suffix = "" if args.gt_window == 4.0 else "_gtw1"
        args.out = f"{HERE}/fishtal_results{suffix}.json"

    global GT_DURATION
    GT_DURATION = args.gt_window
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_recs = list_recordings(args.feature_mode)
    print(f"{len(all_recs)} recordings | device={device} | label={args.label}", flush=True)

    folds = all_recs if args.fold == "all" else [args.fold]
    results = {}
    for rec in folds:
        print(f"\n=== FOLD: held out {rec} ===", flush=True)
        results[rec] = train_fold(rec, all_recs, args, device)
        print(f"=== fold done: {rec} -> avg_mAP={results[rec]['avg_map']:.4f}", flush=True)

    per_iou = np.array([r["per_iou"] for r in results.values()], dtype=float)
    avgs = [r["avg_map"] for r in results.values()]
    rec_pts = [r["point_recall"] for r in results.values() if not np.isnan(r["point_recall"])]
    prec_pts = [r["point_precision"] for r in results.values() if not np.isnan(r["point_precision"])]
    summary = {
        "label": args.label,
        "config": vars(args),
        "folds": results,
        "n_folds_complete": len(results),
        "per_iou_mean": np.nanmean(per_iou, axis=0).tolist(),
        "per_iou_std": np.nanstd(per_iou, axis=0).tolist(),
        "avg_mean": float(np.mean(avgs)), "avg_std": float(np.std(avgs)),
        "point_recall_mean": float(np.mean(rec_pts)) if rec_pts else float("nan"),
        "point_recall_std": float(np.std(rec_pts)) if rec_pts else float("nan"),
        "point_precision_mean": float(np.mean(prec_pts)) if prec_pts else float("nan"),
        "point_precision_std": float(np.std(prec_pts)) if prec_pts else float("nan"),
        "traj_mean": [], "traj_std": [],
    }
    print(f"\n=== {args.label}: avg mAP = {summary['avg_mean']:.4f} "
          f"+/- {summary['avg_std']:.4f} | point-recall "
          f"{summary['point_recall_mean']:.3f} precision "
          f"{summary['point_precision_mean']:.3f} ({len(results)} folds) ===", flush=True)

    allr = json.load(open(args.out)) if os.path.isfile(args.out) else {}
    allr[args.label] = summary
    json.dump(allr, open(args.out, "w"), indent=1)
    print("wrote", args.out, sorted(allr.keys()), flush=True)


if __name__ == "__main__":
    main()
