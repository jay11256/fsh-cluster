#!/usr/bin/env python3
"""FishFormer behavior-confusion matrix, out-of-fold.

FishFormer scores each behavior independently (6 sigmoid heads, not one
softmax), so there's no native "predicted class" the way a single-label
classifier has one. The closest honest analog: for every true BORIS point
event, look at every class's predicted spans at that instant and take the
highest-scoring one as "what the model called it" -- if no span from any
class covers the point, it counts as a miss ("None").

Uses the checkpoints saved during the recent 5-fold retrain (train_former.py
--save-ckpt) -- one model per held-out recording, so every recording's
events are scored by a model that never trained on it (true out-of-fold).
"""
import os
import sys
import glob
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from former import FishFormer                                # noqa: E402
from data import FG_CLASSES, NUM_CLASSES, load_recording      # noqa: E402
from train_former import predict_spans                        # noqa: E402

CKPT_DIR = f"{HERE}/checkpoints"
CKPT_GLOB = "Ds12_06_5fold_fold*_Former2__*.pt"
NONE_LABEL = "None (missed)"
LABELS = FG_CLASSES + [NONE_LABEL]


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    feats, _ = load_recording(ck["test_rec"], cfg["feature_mode"])
    spatial_pool = feats.ndim == 3
    model = FishFormer(feat_dim=ck["feat_dim"], num_classes=NUM_CLASSES, hidden=cfg["hidden"],
                       depth=cfg["depth"], num_heads=cfg["num_heads"], drop=cfg["drop"],
                       drop_path=cfg["drop_path"], n_levels=cfg["n_levels"],
                       reg_bins=cfg["reg_bins"], spatial_pool=spatial_pool,
                       use_motion=cfg.get("use_motion", False)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    args = argparse.Namespace(**cfg)
    return model, args, ck["test_rec"]


def confuse_one(ckpt_path, device):
    """Return list of (true_label, pred_label) string pairs for every GT
    point event in this checkpoint's held-out recording -- pred_label is
    NONE_LABEL if no class's span covers the point."""
    model, args, test_rec = load_model(ckpt_path, device)
    props, events, stride, n = predict_spans(model, test_rec, args, device)

    out = []
    for t, true_c in events:
        best_c, best_score = None, -1.0
        for c in range(len(FG_CLASSES)):
            for s, e, score in props[c]:
                if s <= t <= e and score > best_score:
                    best_c, best_score = c, score
        pred_label = FG_CLASSES[best_c] if best_c is not None else NONE_LABEL
        out.append((FG_CLASSES[true_c], pred_label))
    return out


def confuse_one_multilabel(ckpt_path, device):
    """Return (y_true, y_pred) multi-hot arrays, one ROW per unique true
    timestamp in this recording (collapsing multiple same-instant true
    labels -- e.g. Bite+Chase/Charge logged at the same second -- into one
    multi-hot row instead of splitting them, since the model IS allowed to
    call both at once). Columns = FG_CLASSES.

    y_pred[i, c] = 1 iff any of class c's predicted spans (already
    score-thresholded + NMS'd by predict_spans) covers that timestamp --
    each class checked independently, no argmax competition between classes.
    This is the same "does any span cover the point" rule evaluate() already
    uses for point_recall/point_precision, just assembled per-class instead
    of collapsed to one winner."""
    model, args, test_rec = load_model(ckpt_path, device)
    props, events, stride, n = predict_spans(model, test_rec, args, device)

    by_time = defaultdict(set)
    for t, c in events:
        by_time[t].add(c)

    n_classes = len(FG_CLASSES)
    y_true, y_pred = [], []
    for t, true_cs in by_time.items():
        tv = np.zeros(n_classes, dtype=int)
        for c in true_cs:
            tv[c] = 1
        pv = np.array([1 if any(s <= t <= e for s, e, _ in props[c]) else 0
                       for c in range(n_classes)], dtype=int)
        y_true.append(tv)
        y_pred.append(pv)
    return np.array(y_true), np.array(y_pred)


def main_multilabel():
    """Per-class binary confusion matrices (multilabel_confusion_matrix),
    the honest fit for a model with independent per-class heads and ground
    truth that itself allows co-occurring behaviors -- no forced single
    winner anywhere in this path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, CKPT_GLOB)))
    print(f"{len(ckpts)} checkpoints, device={device}", flush=True)

    Y_true, Y_pred = [], []
    for ckpt_path in ckpts:
        rec = os.path.basename(ckpt_path).split("__")[1][:-3]
        yt, yp = confuse_one_multilabel(ckpt_path, device)
        Y_true.append(yt)
        Y_pred.append(yp)
        print(f"[ok] {os.path.basename(ckpt_path)} ({rec}): {len(yt)} timestamps", flush=True)

    Y_true = np.concatenate(Y_true, axis=0)
    Y_pred = np.concatenate(Y_pred, axis=0)
    mcm = multilabel_confusion_matrix(Y_true, Y_pred)   # (n_classes, 2, 2): [[TN,FP],[FN,TP]]

    out = {
        "classes": FG_CLASSES,
        "mcm": mcm.tolist(),
        "n_timestamps": int(Y_true.shape[0]),
        "n_checkpoints": len(ckpts),
    }
    out_path = f"{HERE}/fishformer_confusion_multilabel.json"
    json.dump(out, open(out_path, "w"), indent=2)

    print("\nPer-class binary confusion (TN, FP / FN, TP):")
    for c, m in zip(FG_CLASSES, mcm):
        tn, fp, fn, tp = m.ravel()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        print(f"  {c:14s} TN={tn:5d} FP={fp:5d} FN={fn:5d} TP={tp:5d}  "
              f"precision={prec:.1%} recall={rec:.1%}")
    print("\nwrote", out_path)


def plot_multilabel(json_path=None, out_path=None, cmap="Blues"):
    """Grid of per-class 2x2 binary confusion matrices (sklearn's
    ConfusionMatrixDisplay again, row-normalized to percentages), one panel
    per behavior -- the multi-label-correct alternative to the single NxN
    matrix, since it never forces one class to 'win' over another at a
    shared timestamp."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    json_path = json_path or f"{HERE}/fishformer_confusion_multilabel.json"
    out_path = out_path or f"{HERE}/fishformer_confusion_multilabel.png"
    d = json.load(open(json_path))
    classes, mcm = d["classes"], np.array(d["mcm"], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, c, m in zip(axes.ravel(), classes, mcm):
        row_totals = m.sum(axis=1, keepdims=True)
        frac = np.divide(m, row_totals, out=np.zeros_like(m), where=row_totals > 0)
        disp = ConfusionMatrixDisplay(confusion_matrix=frac, display_labels=[f"not {c}", c])
        disp.plot(ax=ax, cmap=cmap, values_format=".1%", colorbar=False)
        ax.set_title(c, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("wrote", out_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, CKPT_GLOB)))
    print(f"{len(ckpts)} checkpoints, device={device}", flush=True)

    y_true, y_pred = [], []
    for ckpt_path in ckpts:
        rec = os.path.basename(ckpt_path).split("__")[1][:-3]
        pairs = confuse_one(ckpt_path, device)
        y_true += [t for t, _ in pairs]
        y_pred += [p for _, p in pairs]
        print(f"[ok] {os.path.basename(ckpt_path)} ({rec}): {len(pairs)} events", flush=True)

    # sklearn's confusion_matrix: rows = true labels, cols = predicted labels,
    # in the order given by `labels=`. NONE_LABEL never appears as a true
    # label (it's a valid prediction only), so its row is all zeros -- we
    # only report the FG_CLASSES rows below.
    counts = confusion_matrix(y_true, y_pred, labels=LABELS)[:len(FG_CLASSES)]
    row_totals = counts.sum(axis=1, keepdims=True)
    frac = np.divide(counts, row_totals, out=np.zeros_like(counts, dtype=float), where=row_totals > 0)

    out = {
        "classes": FG_CLASSES,
        "columns": LABELS,
        "counts": counts.tolist(),
        "frac": frac.tolist(),
        "n_checkpoints": len(ckpts),
    }
    out_path = f"{HERE}/fishformer_confusion.json"
    json.dump(out, open(out_path, "w"), indent=2)

    print("\nConfusion (row = true, col = predicted; % of that true class's events):")
    hdr = "".join(f"{c:>14s}" for c in out["columns"])
    print(" " * 14 + hdr)
    for i, c in enumerate(FG_CLASSES):
        print(f"{c:14s}" + "".join(f"{frac[i, j]:14.1%}" for j in range(len(LABELS))))
    print("\nwrote", out_path)


def plot_sklearn_default(json_path=None, out_path=None):
    """sklearn's own default look (ConfusionMatrixDisplay, cmap='viridis',
    colorbar on) -- no house styling, so it's directly comparable to any
    other sklearn confusion matrix. Runs off the already-saved JSON, no GPU/
    checkpoints needed. NONE_LABEL's row is reconstructed as all-zero (it
    never occurs as a true label -- see main()'s comment) so the matrix is
    square, matching what ConfusionMatrixDisplay expects."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    json_path = json_path or f"{HERE}/fishformer_confusion.json"
    out_path = out_path or f"{HERE}/fishformer_confusion_sklearn.png"
    d = json.load(open(json_path))
    labels = d["columns"]                          # FG_CLASSES + [NONE_LABEL]
    counts = np.array(d["counts"])                  # (6, 7)
    square = np.zeros((len(labels), len(labels)), dtype=int)
    square[:counts.shape[0]] = counts

    disp = ConfusionMatrixDisplay(confusion_matrix=square, display_labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp.plot(ax=ax, xticks_rotation=45)            # default cmap="viridis"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("wrote", out_path)


def plot_sklearn_blues_normalized(json_path=None, out_path=None):
    """Classic 'Blues' confusion-matrix look (cmap="Blues", the other common
    sklearn-tutorial default), row-normalized to relative percentages (each
    true class's own row sums to 100%) instead of raw counts."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    json_path = json_path or f"{HERE}/fishformer_confusion.json"
    out_path = out_path or f"{HERE}/fishformer_confusion_sklearn_blues.png"
    d = json.load(open(json_path))
    labels = d["columns"]
    counts = np.array(d["counts"])
    square = np.zeros((len(labels), len(labels)), dtype=int)
    square[:counts.shape[0]] = counts

    row_totals = square.sum(axis=1, keepdims=True)
    frac = np.divide(square, row_totals, out=np.zeros_like(square, dtype=float), where=row_totals > 0)

    disp = ConfusionMatrixDisplay(confusion_matrix=frac, display_labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp.plot(ax=ax, cmap="Blues", values_format=".1%", xticks_rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("wrote", out_path)


if __name__ == "__main__":
    if "--multilabel" in sys.argv:
        main_multilabel()
        plot_multilabel()
    elif "--plot-only" in sys.argv:
        plot_sklearn_default()
        plot_sklearn_blues_normalized()
    else:
        main()
        plot_sklearn_default()
        plot_sklearn_blues_normalized()
