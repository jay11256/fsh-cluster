"""
visualize_matrix.py
-------------------
Visualise model behavior-detection predictions against ground truth labels
on a per-behavior timeline.

Usage
-----
    from visualize_matrix import visualize_matrix
    fig = visualize_matrix(
        ground_truth_path="labels.tsv",
        pred_matrix=logits,          # shape: (num_behaviors, num_clips)
        threshold=0.5,
        window_len=8,
        overlap_len=4,
        save_path="output_dir",      # saves visualize_predictions.png + metrics
    )
"""

import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from typing import List, Optional, Tuple


# Canonical behavior-name aliases (lower-case key → display name).
BEHAVIOR_MAP = {
    "peck":                "Peck",
    "peck (specify sex)":  "Peck",
    "quiver-m":            "Quiver",
    "quiver (male)":       "Quiver",
    "male quiver":         "Quiver",
    "bite (male)":         "Bite",
    "tilt (specify sex)":  "Tilt",
    "lead (male)":         "Lead",
    "lead-m":              "Lead",
    "male lead":           "Lead",
    "pot entry":           "Enter Pot",
    "chase/charge (male)": "Chase/Charge",
    "run/flee (female)":   "Run/Flee",
    "exit plot":           "Exit Plot",
    "egg retrieval":       "Egg Retrieval",
    "circling":            "Circling",
    "female follow":       "Follow",
    "follow-f":            "Follow",
    "follow (female)":     "Follow",
    "spawning-f":          "Spawning",
}

# Default row order (alphabetical) when behavior_names is not supplied.
#DEFAULT_LABELS = ["Bite", "Lead", "Peck", "Quiver", "Run/Flee", "Tilt"]
#DEFAULT_LABELS = ["Bite", "Chase/Charge", "Lead", "Peck", "Quiver", "Tilt"]
DEFAULT_LABELS = ["Bite", "Chase/Charge", "Lead", "NoBehavior", "Peck", "Quiver", "Tilt"]

# Used when visualize_matrix is called without an explicit video_window.
VIDEO_WINDOW = (0.0, float("inf"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _merge_spans(starts: np.ndarray, ends: np.ndarray):
    """Merge overlapping / adjacent (start, end) intervals."""
    if len(starts) == 0:
        return [], []
    pairs = sorted(zip(starts, ends))
    merged = [list(pairs[0])]
    for s, e in pairs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    ms, me = zip(*merged)
    return list(ms), list(me)

def _temporal_iou(pred_span, gt_span):
    """IoU between two temporal intervals."""
    ps, pe = pred_span
    gs, ge = gt_span

    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)

    return inter / union if union > 0 else 0.0

def _compute_ap(recalls, precisions):
    """
    Compute interpolated AP (VOC style).
    """
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    idx = np.where(recalls[1:] != recalls[:-1])[0]

    return np.sum(
        (recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]
    )

def _safe_divide(numerator, denominator):
    """Return nan for undefined rates instead of silently reporting zero."""
    return numerator / denominator if denominator else np.nan


def _as_numpy_array(values):
    """Accept NumPy arrays or Torch tensors without importing torch here."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def _make_gt_spans(
    gt_times,
    gt_behaviors,
    label,
    gt_duration,
    min_time=0.0,
    max_time=None,
):
    """Build clipped temporal GT spans for one behavior label."""
    half_duration = gt_duration / 2
    gt_spans = []

    for t in gt_times[gt_behaviors == label]:
        start = t - half_duration
        end = t + half_duration

        if max_time is not None:
            start = max(start, min_time)
            end = min(end, max_time)

        if end > start:
            gt_spans.append((float(start), float(end)))

    return gt_spans


def _build_pred_spans(scores, clip_starts, clip_ends, threshold):
    """
    Convert thresholded per-clip scores into merged temporal detections.
    Each merged detection is scored by the max clip score inside it.
    """
    positive = np.where(scores >= threshold)[0]
    if len(positive) == 0:
        return []

    pred_spans = []
    start = float(clip_starts[positive[0]])
    end = float(clip_ends[positive[0]])
    best_score = float(scores[positive[0]])

    for clip_idx in positive[1:]:
        clip_start = float(clip_starts[clip_idx])
        clip_end = float(clip_ends[clip_idx])
        clip_score = float(scores[clip_idx])

        if clip_start <= end:
            end = max(end, clip_end)
            best_score = max(best_score, clip_score)
        else:
            pred_spans.append((start, end, best_score))
            start = clip_start
            end = clip_end
            best_score = clip_score

    pred_spans.append((start, end, best_score))
    pred_spans.sort(key=lambda x: x[2], reverse=True)
    return pred_spans


def _match_detections(gt_spans, pred_spans, iou_thresh):
    """Greedy one-to-one temporal detection matching."""
    matched = np.zeros(len(gt_spans), dtype=bool)
    tp = np.zeros(len(pred_spans), dtype=float)
    fp = np.zeros(len(pred_spans), dtype=float)

    for i, (ps, pe, _score) in enumerate(pred_spans):
        best_iou = 0.0
        best_gt = -1

        for j, gt in enumerate(gt_spans):
            if matched[j]:
                continue

            iou = _temporal_iou((ps, pe), gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if best_iou >= iou_thresh and best_gt >= 0:
            tp[i] = 1
            matched[best_gt] = True
        else:
            fp[i] = 1

    return tp, fp, matched


def _clip_ground_truth_mask(gt_spans, clip_starts, clip_ends):
    """Mark clips that overlap any GT span."""
    mask = np.zeros(len(clip_starts), dtype=bool)

    for gt_span in gt_spans:
        for i, clip_span in enumerate(zip(clip_starts, clip_ends)):
            if _temporal_iou(clip_span, gt_span) > 0:
                mask[i] = True

    return mask


def _clip_eval_inputs(pred_matrix, clip_starts, clip_ends, eval_start, eval_end):
    """Keep only prediction clips that overlap the evaluation window."""
    if eval_end is None:
        eval_end = float(clip_ends[-1]) if len(clip_ends) else eval_start

    eval_mask = (clip_starts < eval_end) & (clip_ends > eval_start)

    return (
        pred_matrix[:, eval_mask],
        np.maximum(clip_starts[eval_mask], eval_start),
        np.minimum(clip_ends[eval_mask], eval_end),
    )


def compute_detection_report(
    gt_times,
    gt_behaviors,
    pred_matrix,
    labels,
    clip_starts,
    clip_ends,
    threshold=0.5,
    thresholds=None,
    iou_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    gt_duration=4.0,
    eval_start=0.0,
    eval_end=None,
):
    """
    Compute temporal mAP plus per-class clip precision/recall.

    thresholds : list[float] | None
        Per-class decision thresholds (one per row of pred_matrix).
        When provided and non-empty, overrides the scalar ``threshold``.
    """
    gt_times = _as_numpy_array(gt_times)
    gt_behaviors = _as_numpy_array(gt_behaviors)
    pred_matrix = _as_numpy_array(pred_matrix)
    clip_starts = _as_numpy_array(clip_starts)
    clip_ends = _as_numpy_array(clip_ends)
    eval_start = float(eval_start)
    eval_end = float(eval_end) if eval_end is not None else None

    # Resolve per-class threshold array.
    num_behaviors = pred_matrix.shape[0]
    if thresholds is not None and len(thresholds) > 0:
        if len(thresholds) != num_behaviors:
            raise ValueError(
                f"thresholds has {len(thresholds)} entries but pred_matrix "
                f"has {num_behaviors} rows."
            )
        thresh_per_class = list(thresholds)
    else:
        thresh_per_class = [threshold] * num_behaviors

    pred_matrix, clip_starts, clip_ends = _clip_eval_inputs(
        pred_matrix=pred_matrix,
        clip_starts=clip_starts,
        clip_ends=clip_ends,
        eval_start=eval_start,
        eval_end=eval_end,
    )

    timeline_end = eval_end if eval_end is not None else (
        float(clip_ends[-1]) if len(clip_ends) else None
    )
    aps_by_iou = {iou: [] for iou in iou_thresholds}
    class_metrics = {}

    for behavior_idx, label in enumerate(labels):
        thr = thresh_per_class[behavior_idx]

        gt_spans = _make_gt_spans(
            gt_times=gt_times,
            gt_behaviors=gt_behaviors,
            label=label,
            gt_duration=gt_duration,
            min_time=eval_start,
            max_time=timeline_end,
        )
        n_gt = len(gt_spans)
        scores = pred_matrix[behavior_idx]
        pred_spans = _build_pred_spans(scores, clip_starts, clip_ends, thr)

        gt_clip_mask = _clip_ground_truth_mask(gt_spans, clip_starts, clip_ends)
        pred_clip_mask = scores >= thr

        clip_tp = int(np.sum(pred_clip_mask & gt_clip_mask))
        clip_fp = int(np.sum(pred_clip_mask & ~gt_clip_mask))
        clip_tn = int(np.sum(~pred_clip_mask & ~gt_clip_mask))
        clip_fn = int(np.sum(~pred_clip_mask & gt_clip_mask))

        per_iou_ap = {}
        for iou_thresh in iou_thresholds:
            if n_gt == 0:
                per_iou_ap[iou_thresh] = np.nan
                continue

            tp, fp, _matched = _match_detections(gt_spans, pred_spans, iou_thresh)
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)

            recalls = tp_cum / n_gt
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

            ap = _compute_ap(recalls, precisions)

            per_iou_ap[iou_thresh] = ap
            aps_by_iou[iou_thresh].append(ap)

        precision = _safe_divide(clip_tp, clip_tp + clip_fp)
        recall = _safe_divide(clip_tp, clip_tp + clip_fn)

        if np.isnan(precision) or np.isnan(recall):
            f1 = np.nan
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        class_metrics[label] = {
            "num_gt_events": n_gt,
            "num_pred_events": len(pred_spans),
            "num_eval_clips": len(clip_starts),
            "num_gt_clips": int(np.sum(gt_clip_mask)),
            "num_pred_clips": int(np.sum(pred_clip_mask)),
            "clip_tp": clip_tp,
            "clip_fp": clip_fp,
            "clip_tn": clip_tn,
            "clip_fn": clip_fn,
            "precision": _safe_divide(clip_tp, clip_tp + clip_fp),
            "recall": _safe_divide(clip_tp, clip_tp + clip_fn),
            "f1": f1,
            "ap_by_iou": per_iou_ap,
            "threshold": thr,
        }

    return {
        "map_by_iou": {
            iou: np.mean(vals) if len(vals) else np.nan
            for iou, vals in aps_by_iou.items()
        },
        "class_metrics": class_metrics,
    }


def compute_map(
    gt_times,
    gt_behaviors,
    pred_matrix,
    labels,
    clip_starts,
    clip_ends,
    threshold=0.5,
    thresholds=None,
    iou_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    gt_duration=4.0,
    eval_start=0.0,
    eval_end=None,
):
    """Backward-compatible helper returning only mAP@IoU values."""
    report = compute_detection_report(
        gt_times=gt_times,
        gt_behaviors=gt_behaviors,
        pred_matrix=pred_matrix,
        labels=labels,
        clip_starts=clip_starts,
        clip_ends=clip_ends,
        threshold=threshold,
        thresholds=thresholds,
        iou_thresholds=iou_thresholds,
        gt_duration=gt_duration,
        eval_start=eval_start,
        eval_end=eval_end,
    )
    return report["map_by_iou"]


def _format_metric(value):
    return "nan" if np.isnan(value) else f"{value:.4f}"


def _format_detection_report(report, labels):
    lines = []

    lines.append("mAP@IoU results:")
    lines.append("-" * 40)
    for iou, value in report["map_by_iou"].items():
        lines.append(f"mAP@IoU={iou:.2f}: {_format_metric(value)}")

    lines.append("")
    lines.append("Per-class clip metrics:")
    lines.append("-" * 120)
    header = (
        f"{'Class':<15}"
        f"{'Threshold':>10}"
        f"{'GT ev':>6}"
        f"{'Pred ev':>8}"
        f"{'Eval clips':>11}"
        f"{'GT clips':>9}"
        f"{'Pred clips':>11}"
        f"{'TP':>6}"
        f"{'FP':>6}"
        f"{'FN':>6}"
        f"{'Precision':>11}"
        f"{'Recall':>11}"
        f"{'F1':>11}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for label in labels:
        metrics = report["class_metrics"][label]
        lines.append(
            f"{label:<15}"
            f"{metrics['threshold']:>10.4f}"
            f"{metrics['num_gt_events']:>6}"
            f"{metrics['num_pred_events']:>8}"
            f"{metrics['num_eval_clips']:>11}"
            f"{metrics['num_gt_clips']:>9}"
            f"{metrics['num_pred_clips']:>11}"
            f"{metrics['clip_tp']:>6}"
            f"{metrics['clip_fp']:>6}"
            f"{metrics['clip_fn']:>6}"
            f"{_format_metric(metrics['precision']):>11}"
            f"{_format_metric(metrics['recall']):>11}"
            f"{_format_metric(metrics['f1']):>11}"
        )
    precisions = []
    recalls = []
    f1s = []

    for label in labels:
        metrics = report["class_metrics"][label]

        if not np.isnan(metrics["precision"]):
            precisions.append(metrics["precision"])

        if not np.isnan(metrics["recall"]):
            recalls.append(metrics["recall"])

        if not np.isnan(metrics["f1"]):
            f1s.append(metrics["f1"])

    lines.append("-" * len(header))

    lines.append(
        f"{'Mean':<15}"
        f"{'':>10}"
        f"{'':>6}"
        f"{'':>8}"
        f"{'':>11}"
        f"{'':>9}"
        f"{'':>11}"
        f"{'':>6}"
        f"{'':>6}"
        f"{'':>6}"
        f"{_format_metric(np.mean(precisions)):>11}"
        f"{_format_metric(np.mean(recalls)):>11}"
        f"{_format_metric(np.mean(f1s)):>11}"
    )

    return "\n".join(lines) + "\n"


def _find_data_start(path: str) -> int:
    """Return the row index of the real header line."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if "Time" in line and "Behavior" in line:
                return i
    raise ValueError("Could not find a header row containing 'Time' and 'Behavior'.")


def _load_ground_truth(path: str, video_window: tuple):
    """
    Read a CSV or TSV with 'Time' and 'Behavior'/'Behavior type' columns.
    Handles BORIS-style TSV exports with metadata preamble rows.
    Filters to video_window, re-indexes times to 0 (window-relative),
    and applies BEHAVIOR_MAP.
    Returns (times_array, behaviors_array).
    """
    start, end = video_window

    # Detect separator from extension, fall back to sniffing
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        sep = ","
    elif ext in (".tsv", ".txt"):
        sep = "\t"
    else:
        with open(path, newline="", encoding="utf-8-sig") as f:
            sample = f.read(2048)
        sep = "\t" if sample.count("\t") > sample.count(",") else ","

    # Skip metadata preamble if present (e.g. BORIS exports)
    try:
        header_row = _find_data_start(path)
    except ValueError:
        header_row = 0

    gt = pd.read_csv(path, sep=sep, skiprows=header_row)
    gt.columns = gt.columns.str.strip()

    col_map  = {c.lower(): c for c in gt.columns}
    time_col = col_map.get("time") or next(
        (c for c in gt.columns if "time" in c.lower()), None
    )
    # 'Behavior' must win over 'Behavior type': in BORIS's aggregated-events
    # export both columns exist, and 'Behavior type' is the event-type flag
    # (always "POINT"), not the label. Preferring it silently parsed those files
    # to zero real events -- three ~1hr recordings (1771 annotations) looked
    # empty and were excluded from every experiment because of it. Files that
    # only have 'Behavior' are unaffected by the reordering.
    beh_col  = (
        col_map.get("behavior")
        or col_map.get("behavior type")
        or next((c for c in gt.columns if "behavior" in c.lower()), None)
    )
    if time_col is None or beh_col is None:
        raise ValueError(
            f"Could not locate 'Time' and 'Behavior' columns. "
            f"Columns found: {list(gt.columns)}"
        )

    gt = gt[gt[time_col].between(start, end)].copy()
    gt[time_col] = gt[time_col] - start          # re-index to window start = 0

    times = gt[time_col].values.astype(float)
    raw   = gt[beh_col].astype(str).str.strip().values
    behaviors = np.array(
        [BEHAVIOR_MAP.get(b.lower(), b) for b in raw], dtype=str
    )
    return times, behaviors


def _build_preds_debug_figure(
    gt_times,
    gt_behaviors,
    labels,
    colors,
    pred_matrix,
    clip_centers,
    duration,
    display_offset,
    window_len,
    overlap_len,
):
    """Single-panel figure: score curves with ground-truth events overlaid."""
    fig_w = max(14.0, duration / 8.0 + 2.0)
    fig, ax = plt.subplots(
        1, 1,
        figsize=(fig_w, 4.5),
        gridspec_kw=dict(top=0.88, bottom=0.12, left=0.09, right=0.97),
    )

    ax.set_xlim(0.0, duration)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score", fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9, labelpad=4)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x + display_offset:.4g}")
    )

    grid_step = 30.0 if duration > 200 else 20.0 if duration > 80 else 10.0
    for xs in np.arange(grid_step, duration, grid_step):
        ax.axvline(xs, color="#e0e0e0", lw=0.6, zorder=0)

    for label, color, probs in zip(labels, colors, pred_matrix):
        ax.plot(clip_centers, probs, color=color, lw=1.5, label=label)

    tick_ymin, tick_ymax = 0.88, 0.98
    events_by_time = defaultdict(list)
    for t, beh in zip(gt_times, gt_behaviors):
        events_by_time[float(t)].append(beh)

    x_step = min(0.35, max(0.12, duration * 0.008))
    label_y_step = 0.055

    for t, behs in sorted(events_by_time.items()):
        n = len(behs)
        if n == 1:
            x_offsets = [0.0]
        else:
            x_offsets = [
                (i - (n - 1) / 2) * x_step
                for i in range(n)
            ]

        for i, (beh, x_off) in enumerate(zip(behs, x_offsets)):
            t_draw = t + x_off
            c = colors[labels.index(beh)] if beh in labels else "#777777"
            ax.vlines(t_draw, tick_ymin, tick_ymax, colors=c, lw=1.8, zorder=5)
            ax.scatter(
                [t_draw], [tick_ymax], s=18, color=c, marker="v",
                zorder=6, clip_on=False,
            )
            blended = mtransforms.blended_transform_factory(
                ax.transData, ax.transAxes
            )
            label_y = tick_ymax + 0.04 + (i * label_y_step if n > 1 else 0.0)
            ax.text(
                t_draw, label_y, beh,
                transform=blended,
                ha="center", va="bottom",
                fontsize=6.5, color=c,
                rotation=90, clip_on=True,
                zorder=7,
            )

    gt_handle = Line2D(
        [0], [0], color="#555555", lw=1.8, marker="v", markersize=5,
        label="Ground-truth event",
    )
    curve_handles, curve_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=curve_handles + [gt_handle],
        labels=curve_labels + [gt_handle.get_label()],
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    fig.suptitle(
        f"Behavior scores vs ground truth"
        f"  |  window = {window_len} s,  overlap = {overlap_len} s",
        fontsize=10, y=0.96,
    )
    return fig


# ── Main function ─────────────────────────────────────────────────────────────

def visualize_matrix(
    ground_truth_path: str,
    pred_matrix: np.ndarray,
    threshold: float = 0.5,
    thresholds: Optional[List[float]] = None,
    window_len: float = 8.0,
    overlap_len: float = 4.0,
    behavior_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    video_window: Optional[Tuple[float, float]] = None,
    **kwargs,                         # absorb unknown keyword args gracefully
) -> "plt.Figure":
    """
    Visualise model predictions against ground truth on a per-behavior timeline.

    Layout
    ------
    - Top row    : all ground-truth events as colour-coded vertical ticks.
    - Second row : per-class score curves with threshold line(s).
    - Lower rows : one row per behavior — solid spans where the model fires.

    Parameters
    ----------
    ground_truth_path : str
        TSV with 'Time' (seconds) and 'Behavior type' columns.
    pred_matrix : np.ndarray  (num_behaviors, num_clips)
        Raw logit / probability scores.
    threshold : float
        Scalar decision threshold used when ``thresholds`` is empty/None.
    thresholds : list[float] | None
        Per-class decision thresholds (one entry per behavior row).
        When provided and non-empty, overrides ``threshold`` for every class.
    window_len : float
        Clip duration in seconds (default 8).
    overlap_len : float
        Overlap between consecutive clips in seconds (default 4).
    behavior_names : list[str] | None
        Row-index → behavior name mapping.  Defaults to DEFAULT_LABELS.
    save_path : str | None
        Output directory. When given, saves ``visualize_predictions.png``,
        ``preds_debug.png``, and a ``metrics`` text file with detection-report
        statistics.
    video_window : tuple[float, float] | None
        (start, end) recording time in seconds.
    **kwargs
        Extra keyword arguments are silently ignored (e.g. legacy params).

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Resolve window: parameter takes precedence over the module-level global.
    _window = video_window if video_window is not None else VIDEO_WINDOW

    # ── 1. Ground truth ───────────────────────────────────────────────────────
    gt_times, gt_behaviors = _load_ground_truth(ground_truth_path, _window)
    pred_matrix = _as_numpy_array(pred_matrix)

    # ── 2. Behavior labels ────────────────────────────────────────────────────
    num_behaviors, num_clips = pred_matrix.shape

    if behavior_names is not None:
        if len(behavior_names) != num_behaviors:
            raise ValueError(
                f"behavior_names has {len(behavior_names)} entries but "
                f"pred_matrix has {num_behaviors} rows."
            )
        labels = list(behavior_names)
    else:
        labels = DEFAULT_LABELS
        if len(labels) != num_behaviors:
            raise ValueError(
                f"DEFAULT_LABELS has {len(labels)} entries but pred_matrix "
                f"has {num_behaviors} rows.  Pass behavior_names= explicitly."
            )

    # ── 3. Resolve per-class thresholds ──────────────────────────────────────
    if thresholds is not None and len(thresholds) > 0:
        if len(thresholds) != num_behaviors:
            raise ValueError(
                f"thresholds has {len(thresholds)} entries but pred_matrix "
                f"has {num_behaviors} rows."
            )
        thresh_per_class = list(thresholds)
    else:
        thresh_per_class = [threshold] * num_behaviors

    using_per_class = thresholds is not None and len(thresholds) > 0

    # ── 4. Clip timeline (0-based, window-relative) ───────────────────────────
    stride      = window_len - overlap_len
    clip_starts = np.arange(num_clips) * stride
    clip_ends   = clip_starts + window_len
    if _window is not None and np.isfinite(_window[1]):
        duration = float(_window[1] - _window[0])
    else:
        duration = float(clip_ends[-1])

    # Build binary mask using per-class thresholds: shape (num_behaviors, num_clips)
    pred_binary = np.stack(
        [pred_matrix[i] >= thresh_per_class[i] for i in range(num_behaviors)]
    )

    detection_report = compute_detection_report(
        gt_times=gt_times,
        gt_behaviors=gt_behaviors,
        pred_matrix=pred_matrix,
        labels=labels,
        clip_starts=clip_starts,
        clip_ends=clip_ends,
        threshold=threshold,          # scalar fallback (ignored when thresholds used)
        thresholds=thresh_per_class,  # always pass resolved list
        iou_thresholds=(.1, .2, .3, .4, .5, .6, .7, .8, .9),
        gt_duration=window_len,
        eval_start=0.0,
        eval_end=duration,
    )

    # ── 5. Colour palette ─────────────────────────────────────────────────────
    try:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(num_behaviors)]

    # ── 6. Figure layout ──────────────────────────────────────────────────────
    GT_HEIGHT   = 3
    PROB_HEIGHT = 2
    PRED_HEIGHT = 1

    height_ratios = [GT_HEIGHT, PROB_HEIGHT] + [PRED_HEIGHT] * num_behaviors

    fig_w = max(14.0, duration / 8.0 + 2.0)
    fig_h = max(5.0, (GT_HEIGHT + num_behaviors * PRED_HEIGHT) * 0.55 + 1.5)

    fig, axes = plt.subplots(
        num_behaviors + 2, 1,
        figsize=(fig_w, fig_h),
        gridspec_kw=dict(
            height_ratios=height_ratios,
            hspace=0.06,
            top=0.88, bottom=0.08, left=0.09, right=0.97,
        ),
    )
    ax_gt   = axes[0]
    ax_prob = axes[1]
    ax_pred = axes[2:]

    # ── 7. Axis styling helper ────────────────────────────────────────────────
    display_offset = _window[0]

    def _style_ax(ax, is_bottom: bool, label: str, color=None):
        ax.set_xlim(0.0, duration)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])

        ax.set_ylabel(
            label,
            rotation=0, ha="right", va="center",
            labelpad=6, fontsize=9, fontweight="bold",
            color=color if color is not None else "#444444",
        )

        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#888888" if is_bottom else "#cccccc")

        if is_bottom:
            ax.tick_params(bottom=True, labelbottom=True, labelsize=8)
            ax.set_xlabel("Time (s)", fontsize=9, labelpad=4)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x + display_offset:.4g}")
            )
        else:
            ax.tick_params(bottom=False, labelbottom=False)

        grid_step = 30.0 if duration > 200 else 20.0 if duration > 80 else 10.0
        for xs in np.arange(grid_step, duration, grid_step):
            ax.axvline(xs, color="#e0e0e0", lw=0.6, zorder=0)

        if color is not None:
            r, g, b = color[:3]
            ax.set_facecolor((r, g, b, 0.06))
        else:
            ax.set_facecolor("#f5f5f5")

    def _style_prob_ax(ax):
        ax.set_xlim(0.0, duration)
        ax.set_ylim(0.0, 1.0)

        ax.set_ylabel(
            "Probabilities",
            rotation=0,
            ha="right",
            va="center",
            labelpad=6,
            fontsize=9,
            fontweight="bold",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(bottom=False, labelbottom=False)

        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(["0", "0.5", "1"])

        grid_step = (
            30.0 if duration > 200
            else 20.0 if duration > 80
            else 10.0
        )

        for xs in np.arange(grid_step, duration, grid_step):
            ax.axvline(xs, color="#e0e0e0", lw=0.6, zorder=0)

    # ── 8. Ground-truth row ───────────────────────────────────────────────────
    _style_ax(ax_gt, is_bottom=False, label="Ground\nTruth")

    TICK_YMIN, TICK_YMAX = 0.05, 0.55

    for t, beh in zip(gt_times, gt_behaviors):
        c = colors[labels.index(beh)] if beh in labels else "#777777"

        ax_gt.vlines(t, TICK_YMIN, TICK_YMAX, colors=c, lw=1.8, zorder=5)
        ax_gt.scatter([t], [TICK_YMAX], s=18, color=c, marker="v",
                      zorder=6, clip_on=False)

        blended = mtransforms.blended_transform_factory(
            ax_gt.transData, ax_gt.transAxes
        )
        ax_gt.text(
            t, TICK_YMAX + 0.04, beh,
            transform=blended,
            ha="center", va="bottom",
            fontsize=6.5, color=c,
            rotation=90, clip_on=True,
            zorder=7,
        )

    # ── 9. Probability curves row ─────────────────────────────────────────────
    _style_prob_ax(ax_prob)

    clip_centers = (clip_starts + clip_ends) / 2

    for label, color, probs in zip(labels, colors, pred_matrix):
        ax_prob.plot(clip_centers, probs, color=color, lw=1.5, label=label)

    # Draw threshold line(s): per-class color when thresholds differ by class.
    unique_thresholds = sorted(set(thresh_per_class))
    if using_per_class:
        for label, color, thr_val in zip(labels, colors, thresh_per_class):
            ax_prob.axhline(
                y=thr_val,
                color=color,
                linestyle=":",
                linewidth=1.5,
                alpha=0.45,
                label=f"Threshold ({thr_val:.2f}) – {label}",
            )
    else:
        for thr_val in unique_thresholds:
            ax_prob.axhline(
                y=thr_val,
                color="red",
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
                label=f"Threshold ({thr_val:.2f})",
            )

    # ── 10. Prediction rows ───────────────────────────────────────────────────
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax      = ax_pred[i]
        is_last = i == num_behaviors - 1
        _style_ax(ax, is_bottom=is_last, label=label, color=color)

        if not using_per_class:
            ax.axhline(
                thresh_per_class[i],
                color=color, ls=":", lw=0.8, alpha=0.5, zorder=1,
            )

        pos_clips = np.where(pred_binary[i])[0]
        if len(pos_clips):
            m_starts, m_ends = _merge_spans(
                clip_starts[pos_clips], clip_ends[pos_clips]
            )
            for x0, x1 in zip(m_starts, m_ends):
                ax.axvspan(x0, x1, ymin=0.1, ymax=0.9,
                           facecolor=color, alpha=0.85, zorder=2, lw=0)

    # ── 11. Legend ────────────────────────────────────────────────────────────
    # Build threshold legend entries.
    thresh_handles = []
    if using_per_class:
        for label, color, thr_val in zip(labels, colors, thresh_per_class):
            thresh_handles.append(
                Line2D(
                    [0], [0], color=color, ls=":", lw=1.5, alpha=0.45,
                    label=f"Threshold = {thr_val:.2f} ({label})",
                )
            )
    else:
        for thr_val in unique_thresholds:
            thresh_handles.append(
                Line2D(
                    [0], [0], color="red", ls=":", lw=1.5, alpha=0.8,
                    label=f"Threshold = {thr_val:.2f}",
                )
            )

    fig.legend(
        handles=[
            mpatches.Patch(facecolor="#888888", alpha=0.85,
                           label="Predicted positive"),
            Line2D([0], [0], color="#555555", lw=1.8, marker="v", markersize=5,
                   label="Ground-truth event"),
            *thresh_handles,
        ],
        loc="upper right", bbox_to_anchor=(0.97, 0.975),
        fontsize=8, framealpha=0.9, edgecolor="#cccccc",
    )

    # ── 12. Title ─────────────────────────────────────────────────────────────
    thresh_str = (
        "per-class thresholds"
        if using_per_class
        else f"threshold = {threshold}"
    )
    fig.suptitle(
        f"Behavior detections vs ground truth"
        f"  |  window = {window_len} s,  overlap = {overlap_len} s,  {thresh_str}",
        fontsize=10, y=0.93,
    )

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig_path = os.path.join(save_path, "visualize_predictions.png")
        debug_path = os.path.join(save_path, "preds_debug.png")
        metrics_path = os.path.join(save_path, "metrics")
        fig.savefig(fig_path, dpi=400, bbox_inches="tight")

        debug_fig = _build_preds_debug_figure(
            gt_times=gt_times,
            gt_behaviors=gt_behaviors,
            labels=labels,
            colors=colors,
            pred_matrix=pred_matrix,
            clip_centers=clip_centers,
            duration=duration,
            display_offset=display_offset,
            window_len=window_len,
            overlap_len=overlap_len,
        )
        debug_fig.savefig(debug_path, dpi=400, bbox_inches="tight")
        plt.close(debug_fig)

        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(_format_detection_report(detection_report, labels))

    return fig


# ── Demo / smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import shutil
    import tempfile

    gt_tsv = (
        "Time\tBehavior type\n"
        "505.0\tGrooming\n"
        "518.0\tFeeding\n"
        "522.0\tFeeding\n"
        "535.0\tGrooming\n"
        "548.0\tResting\n"
        "560.0\tFeeding\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        f.write(gt_tsv)
        gt_path = f.name

    rng    = np.random.default_rng(42)
    logits = rng.normal(loc=-0.5, scale=1.2, size=(3, 15))
    logits[0, [3, 4, 13, 14]] = 1.8   # Feeding
    logits[1, [1, 2, 8]]      = 1.5   # Grooming
    logits[2, [9, 10]]        = 1.2   # Resting

    out_dir = tempfile.mkdtemp()
    try:
        # Demo with per-class thresholds
        visualize_matrix(
            ground_truth_path=gt_path,
            pred_matrix=logits,
            threshold=0.5,                         # fallback (not used here)
            thresholds=[0.4, 0.5, 0.6],            # per-class override
            window_len=8,
            overlap_len=4,
            behavior_names=["Feeding", "Grooming", "Resting"],
            save_path=out_dir,
        )
        print(f"Demo saved → {out_dir}")
    finally:
        os.unlink(gt_path)
        shutil.rmtree(out_dir, ignore_errors=True)