"""Sweep FishFormer's CANDIDATE-GENERATION threshold (--score-thresh, default
0.05) to test whether 0.05 is leaving mAP on the table.

The point of this sweep: score_thresh plays two different roles that the
current setup conflates.
  1. Candidate generation (pre-NMS): a permissive filter whose only job is
     to not throw away detections the ranking metric could still use. For a
     rank-based metric like mAP, LOWERING it can only add lower-ranked
     detections to the tail of the PR curve -- it should be monotonically
     non-harmful until `max_props` (400/class) starts truncating.
  2. A reported operating point: what Table 1's recall/precision/F1 are read
     off. That one wants to be *tuned* (nested CV picked 0.18), and is a
     completely different quantity.
This script measures (1) only: for each candidate threshold it reports mAP
on the full resulting list, plus how often the max_props cap binds (which is
what would make "lower" stop helping).

Inference-only on the existing per-recording fold checkpoints -- nothing is
retrained, so any delta isolates the decode threshold.
"""
import os
import sys
import glob
import json
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repo root: data dirs + fishformer package
sys.path.insert(0, ROOT)
from fishformer.former import FishFormer                                        # noqa: E402
from fishformer.data import NUM_CLASSES, FG_CLASSES, load_recording, STRIDE     # noqa: E402
from fishformer.nms import _nms                                               # noqa: E402
import fishformer.train_former as train_former                                                  # noqa: E402
from fishformer.train_former import IOUS                                        # noqa: E402
from visualize_matrix import _make_gt_spans, _match_detections, _compute_ap  # noqa: E402

CKPT_DIR = f"{ROOT}/checkpoints"
OUT_PATH = f"{ROOT}/cand_thresh_sweep_results.json"

CAND_THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]


@torch.no_grad()
def predict_at_thresholds(model, name, args, device, thresholds):
    """One forward pass; bucket candidates for every candidate threshold at once."""
    model.eval()
    feats, events = load_recording(name, args.feature_mode)
    stride = STRIDE[args.feature_mode]
    n = feats.shape[0]
    win = int(round(args.window_s / stride))
    hop = win // 2
    starts = list(range(0, max(1, n - win + 1), hop))
    if n > win and starts[-1] != n - win:
        starts.append(n - win)

    lo = min(thresholds)
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
            sc = (torch.sigmoid(out["cls"])[0]
                  * torch.sigmoid(out["ctr"])[0].unsqueeze(-1)).cpu().numpy()
            rg = out["reg"][0].cpu().numpy()
            t_idx = np.arange(sc.shape[0]) * lvl_stride + s
            for c in range(len(FG_CLASSES)):
                for k in np.flatnonzero(sc[:, c] >= lo):
                    centre = t_idx[k] * stride
                    st = centre - rg[k, 0] * lvl_stride * stride
                    en = centre + rg[k, 1] * lvl_stride * stride
                    if en > st:
                        raw[c].append((float(st), float(en), float(sc[k, c])))

    per_thresh = {}
    for t in thresholds:
        props, capped = {}, 0
        for c, v in raw.items():
            kept = _nms([x for x in v if x[2] >= t], args.nms_iou)
            if len(kept) > args.max_props:
                capped += 1
            props[c] = kept[:args.max_props]
        per_thresh[t] = (props, capped)
    return per_thresh, events, stride, n


def score_props(props, events, stride, n, gt_duration):
    duration = n * stride
    gt_times = np.array([t for t, _ in events], dtype=float)
    gt_beh = np.array([FG_CLASSES[c] for _, c in events], dtype=object)
    ap = {i: [] for i in IOUS}
    for ci, label in enumerate(FG_CLASSES):
        spans = props[ci]
        gt_spans = _make_gt_spans(gt_times, gt_beh, label, gt_duration=gt_duration,
                                  min_time=0.0, max_time=duration)
        if gt_spans:
            for iou_t in IOUS:
                tp, fp, _ = _match_detections(gt_spans, spans, iou_t)
                tpc, fpc = np.cumsum(tp), np.cumsum(fp)
                ap[iou_t].append(_compute_ap(tpc / len(gt_spans),
                                             tpc / np.maximum(tpc + fpc, 1e-8)))
    per_iou = [float(np.mean(ap[i])) if ap[i] else float("nan") for i in IOUS]
    return per_iou


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(glob.glob(f"{CKPT_DIR}/Ds12_06_5fold_fold*_Former2__*.pt"))
    ckpts = [c for c in ckpts
             if "_Neural" not in c and "_None" not in c and "_NoPyramid" not in c]
    print(f"{len(ckpts)} checkpoints | device={device}", flush=True)

    acc = {t: [] for t in CAND_THRESHOLDS}
    n_props = {t: [] for t in CAND_THRESHOLDS}
    n_capped = {t: 0 for t in CAND_THRESHOLDS}
    for i, cpath in enumerate(ckpts):
        ckpt = torch.load(cpath, map_location="cpu", weights_only=False)
        args = SimpleNamespace(**ckpt["config"])
        train_former.GT_DURATION = args.gt_window
        rec = ckpt["test_rec"]
        feats, _ = load_recording(rec, args.feature_mode)
        model = FishFormer(feat_dim=ckpt["feat_dim"], num_classes=NUM_CLASSES,
                           hidden=args.hidden, depth=args.depth, num_heads=args.num_heads,
                           drop=args.drop, drop_path=args.drop_path,
                           reg_bins=args.reg_bins, spatial_pool=feats.ndim == 3,
                           use_motion=args.use_motion).to(device)
        model.load_state_dict(ckpt["model"])

        per_thresh, events, stride, n = predict_at_thresholds(
            model, rec, args, device, CAND_THRESHOLDS)
        for t in CAND_THRESHOLDS:
            props, capped = per_thresh[t]
            acc[t].append(score_props(props, events, stride, n, args.gt_window))
            n_props[t].append(sum(len(v) for v in props.values()))
            n_capped[t] += capped
        print(f"[{i+1}/{len(ckpts)}] {rec} done", flush=True)

    out = {}
    print(f"\n{'cand_thr':>9s} {'mAP.1-.7':>9s} {'mAP.1-.9':>9s} {'props/rec':>10s} "
          f"{'classes@cap':>12s}")
    for t in CAND_THRESHOLDS:
        per_iou = np.nanmean(np.array(acc[t]), axis=0)
        out[str(t)] = {"avg_map_1_7": float(np.nanmean(per_iou[:7])),
                       "avg_map_1_9": float(np.nanmean(per_iou)),
                       "per_iou": per_iou.tolist(),
                       "mean_props_per_rec": float(np.mean(n_props[t])),
                       "class_slots_at_cap": n_capped[t]}
        d = out[str(t)]
        print(f"{t:9.3f} {100*d['avg_map_1_7']:8.1f}% {100*d['avg_map_1_9']:8.1f}% "
              f"{d['mean_props_per_rec']:10.0f} {d['class_slots_at_cap']:12d}")
    json.dump({"ious": list(IOUS), "n_recordings": len(ckpts),
               "max_props": 400, "results": out}, open(OUT_PATH, "w"), indent=1)
    print(f"\nwrote {OUT_PATH}")
    print("(class_slots_at_cap = how many recording x class slots hit the 400 cap,"
          f" out of {len(ckpts)*len(FG_CLASSES)})")


if __name__ == "__main__":
    main()
