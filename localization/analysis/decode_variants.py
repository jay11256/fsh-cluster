"""Decode-time ablations on the already-trained Ds12_06_5fold_Former2
checkpoints: class-agnostic vs class-specific NMS, and multi-label vs
single-class (argmax) prediction.

Nothing is retrained. Each recording's own fold checkpoint (the exact
weights behind the paper's reported numbers -- see
dump_former_spans_from_ckpt.py) is loaded and only the DECODING of its
outputs changes, so any metric delta here isolates the decode rule, not
training noise.

The four variants (2x2):
  cls_nms  x  multilabel : the current/reported behavior (baseline)
  agn_nms  x  multilabel : NMS pooled across all 6 classes jointly
  cls_nms  x  argmax     : only the top-scoring class per position emits
  agn_nms  x  argmax     : both

Candidates are always generated at the permissive score_thresh from each
checkpoint's own config (0.05), then NMS'd; the point-metric THRESHOLD is
applied post-hoc afterwards, so one inference pass supports the whole
threshold sweep (mAP is rank-based and reported on the full 0.05 list,
which is the convention every reported number already uses).

Writes decode_variants_results.json (per-recording + aggregate) so the
sweep can be re-analyzed without a GPU.
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
OUT_PATH = f"{ROOT}/decode_variants_results.json"

THRESHOLDS = [0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
VARIANTS = ["cls_nms__multilabel", "agn_nms__multilabel",
            "cls_nms__argmax", "agn_nms__argmax"]


def _nms_agnostic(per_class, iou_thr, max_props):
    """NMS across ALL classes jointly, then re-split by class.

    _nms() in train.py suppresses within one class's list only; here every
    class's candidates compete in one pool, so an overlapping Bite and
    Chase/Charge box can suppress each other (they cannot in the baseline).
    """
    pooled = [(s, e, sc, c) for c, v in per_class.items() for s, e, sc in v]
    pooled.sort(key=lambda x: -x[2])
    keep = []
    for s, e, sc, c in pooled:
        ok = True
        for ks, ke, _, _ in keep:
            inter = max(0.0, min(e, ke) - max(s, ks))
            union = (e - s) + (ke - ks) - inter
            if union > 0 and inter / union > iou_thr:
                ok = False
                break
        if ok:
            keep.append((s, e, sc, c))
    keep = keep[:max_props]
    out = {c: [] for c in range(len(FG_CLASSES))}
    for s, e, sc, c in keep:
        out[c].append((s, e, sc))
    return out


@torch.no_grad()
def predict_all_variants(model, name, args, device):
    """One forward pass over the recording -> raw candidates for all 4 variants."""
    model.eval()
    feats, events = load_recording(name, args.feature_mode)
    stride = STRIDE[args.feature_mode]
    n = feats.shape[0]
    win = int(round(args.window_s / stride))
    hop = win // 2
    starts = list(range(0, max(1, n - win + 1), hop))
    if n > win and starts[-1] != n - win:
        starts.append(n - win)

    n_fg = len(FG_CLASSES)
    raw_multi = {c: [] for c in range(n_fg)}
    raw_argmax = {c: [] for c in range(n_fg)}
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
            sc = scores.cpu().numpy()
            rg = out["reg"][0].cpu().numpy()
            t_idx = np.arange(sc.shape[0]) * lvl_stride + s
            # argmax over FOREGROUND channels only (BG_INDEX is excluded from
            # the reported class set exactly as the baseline loop does).
            top_c = np.argmax(sc[:, :n_fg], axis=1)
            for c in range(n_fg):
                keep = np.flatnonzero(sc[:, c] >= args.score_thresh)
                for k in keep:
                    centre = t_idx[k] * stride
                    st = centre - rg[k, 0] * lvl_stride * stride
                    en = centre + rg[k, 1] * lvl_stride * stride
                    if en > st:
                        cand = (float(st), float(en), float(sc[k, c]))
                        raw_multi[c].append(cand)
                        if top_c[k] == c:
                            raw_argmax[c].append(cand)

    variants = {}
    for tag, raw in (("multilabel", raw_multi), ("argmax", raw_argmax)):
        variants[f"cls_nms__{tag}"] = {
            c: _nms(v, args.nms_iou)[:args.max_props] for c, v in raw.items()}
        variants[f"agn_nms__{tag}"] = _nms_agnostic(raw, args.nms_iou, args.max_props)
    return variants, events, stride, n


def score_props(props, events, stride, n, gt_duration, thresh):
    """train_former.evaluate()'s metric body, but on a supplied props dict and
    with a post-hoc score threshold applied first."""
    duration = n * stride
    gt_times = np.array([t for t, _ in events], dtype=float)
    gt_beh = np.array([FG_CLASSES[c] for _, c in events], dtype=object)

    ap = {i: [] for i in IOUS}
    rec_l, prec_l = [], []
    for ci, label in enumerate(FG_CLASSES):
        spans = [(s, e, sc) for s, e, sc in props[ci] if sc >= thresh]
        gt_spans = _make_gt_spans(gt_times, gt_beh, label, gt_duration=gt_duration,
                                  min_time=0.0, max_time=duration)
        pts = gt_times[gt_beh == label]
        if pts.size:
            rec_l.append(float(np.mean([any(s <= t <= e for s, e, _ in spans) for t in pts])))
        if spans:
            prec_l.append(float(np.mean([any(s <= t <= e for t in pts) for s, e, _ in spans])))
        if gt_spans:
            for iou_t in IOUS:
                tp, fp, _ = _match_detections(gt_spans, spans, iou_t)
                tpc, fpc = np.cumsum(tp), np.cumsum(fp)
                ap[iou_t].append(_compute_ap(tpc / len(gt_spans),
                                             tpc / np.maximum(tpc + fpc, 1e-8)))
    per_iou = [float(np.mean(ap[i])) if ap[i] else float("nan") for i in IOUS]
    return {"per_iou": per_iou, "avg_map": float(np.nanmean(per_iou)),
            "point_recall": float(np.mean(rec_l)) if rec_l else float("nan"),
            "point_precision": float(np.mean(prec_l)) if prec_l else float("nan")}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(glob.glob(f"{CKPT_DIR}/Ds12_06_5fold_fold*_Former2__*.pt"))
    ckpts = [c for c in ckpts
             if "_Neural" not in c and "_None" not in c and "_NoPyramid" not in c]
    print(f"{len(ckpts)} checkpoints | device={device}", flush=True)

    results = {v: {t: [] for t in THRESHOLDS} for v in VARIANTS}
    n_spans = {v: [] for v in VARIANTS}
    for ci, cpath in enumerate(ckpts):
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

        variants, events, stride, n = predict_all_variants(model, rec, args, device)
        for v in VARIANTS:
            n_spans[v].append(sum(len(x) for x in variants[v].values()))
            for t in THRESHOLDS:
                results[v][t].append(score_props(variants[v], events, stride, n,
                                                 args.gt_window, t))
        base = results["cls_nms__multilabel"][0.05][-1]["avg_map"]
        print(f"[{ci+1}/{len(ckpts)}] {rec}: baseline avg_mAP={base:.4f} "
              f"(ckpt stored {ckpt['result']['avg_map']:.4f})", flush=True)

    summary = {}
    for v in VARIANTS:
        summary[v] = {"mean_spans_per_rec": float(np.mean(n_spans[v])), "by_thresh": {}}
        for t in THRESHOLDS:
            rows = results[v][t]
            r = float(np.nanmean([x["point_recall"] for x in rows]))
            p = float(np.nanmean([x["point_precision"] for x in rows]))
            per_iou = np.nanmean(np.array([x["per_iou"] for x in rows]), axis=0)
            summary[v]["by_thresh"][str(t)] = {
                "avg_map_1_9": float(np.nanmean(per_iou)),
                "avg_map_1_7": float(np.nanmean(per_iou[:7])),
                "per_iou": per_iou.tolist(),
                "point_recall": r, "point_precision": p,
                "f1": float(2 * r * p / (r + p)) if (r + p) > 0 else 0.0,
            }
    json.dump({"ious": list(IOUS), "thresholds": THRESHOLDS,
               "n_recordings": len(ckpts), "summary": summary},
              open(OUT_PATH, "w"), indent=1)

    print(f"\n{'variant':26s} {'thr':>5s} {'mAP.1-.7':>9s} {'recall':>8s} {'prec':>8s} {'F1':>7s}")
    for v in VARIANTS:
        for t in (0.05, 0.18, 0.50):
            d = summary[v]["by_thresh"][str(t)]
            print(f"{v:26s} {t:5.2f} {100*d['avg_map_1_7']:8.1f}% {100*d['point_recall']:7.1f}% "
                  f"{100*d['point_precision']:7.1f}% {100*d['f1']:6.1f}%")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
