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
    )
    fig.savefig("output.png", bbox_inches="tight", dpi=150)
"""

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
DEFAULT_LABELS = ["Bite", "Lead", "Peck", "Quiver", "Run/Flee", "Tilt"]

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
    iou_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    gt_duration=4.0,
    eval_start=0.0,
    eval_end=None,
):
    """
    Compute temporal mAP plus per-class clip precision/recall.
    """
    gt_times = _as_numpy_array(gt_times)
    gt_behaviors = _as_numpy_array(gt_behaviors)
    pred_matrix = _as_numpy_array(pred_matrix)
    clip_starts = _as_numpy_array(clip_starts)
    clip_ends = _as_numpy_array(clip_ends)
    eval_start = float(eval_start)
    eval_end = float(eval_end) if eval_end is not None else None

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
        pred_spans = _build_pred_spans(scores, clip_starts, clip_ends, threshold)

        gt_clip_mask = _clip_ground_truth_mask(gt_spans, clip_starts, clip_ends)
        pred_clip_mask = scores >= threshold

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
            "ap_by_iou": per_iou_ap,
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
        iou_thresholds=iou_thresholds,
        gt_duration=gt_duration,
        eval_start=eval_start,
        eval_end=eval_end,
    )
    return report["map_by_iou"]


def _format_metric(value):
    return "nan" if np.isnan(value) else f"{value:.4f}"


def _print_detection_report(report, labels):
    print("\nmAP@IoU results:")
    print("-" * 40)
    for iou, value in report["map_by_iou"].items():
        print(f"mAP@IoU={iou:.2f}: {_format_metric(value)}")

    print("\nPer-class clip metrics:")
    print("-" * 96)
    header = (
        f"{'Class':<14} {'GT ev':>5} {'Pred ev':>7} {'Eval clips':>10} "
        f"{'GT clips':>8} {'Pred clips':>10} {'TP':>5} {'FP':>5} {'FN':>5} "
        f"{'Precision':>10} {'Recall':>10}"
    )
    print(header)
    print("-" * 96)

    for label in labels:
        metrics = report["class_metrics"][label]
        print(
            f"{label:<14} "
            f"{metrics['num_gt_events']:>5} "
            f"{metrics['num_pred_events']:>7} "
            f"{metrics['num_eval_clips']:>10} "
            f"{metrics['num_gt_clips']:>8} "
            f"{metrics['num_pred_clips']:>10} "
            f"{metrics['clip_tp']:>5} "
            f"{metrics['clip_fp']:>5} "
            f"{metrics['clip_fn']:>5} "
            f"{_format_metric(metrics['precision']):>10} "
            f"{_format_metric(metrics['recall']):>10}"
        )

def _load_ground_truth(path: str, video_window: tuple):
    """
    Read a TSV with 'Time' and 'Behavior type' columns.
    Filters to video_window, re-indexes times to 0 (window-relative),
    and applies BEHAVIOR_MAP.
    Returns (times_array, behaviors_array).
    """
    start, end = video_window

    gt = pd.read_csv(path, sep="\t")
    gt.columns = gt.columns.str.strip()

    col_map  = {c.lower(): c for c in gt.columns}
    time_col = col_map.get("time") or next(
        (c for c in gt.columns if "time" in c.lower()), None
    )
    beh_col  = col_map.get("behavior type") or next(
        (c for c in gt.columns if "behavior" in c.lower()), None
    )
    if time_col is None or beh_col is None:
        raise ValueError(
            f"Could not locate 'Time' and 'Behavior type' columns. "
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


# ── Main function ─────────────────────────────────────────────────────────────

def visualize_matrix(
    ground_truth_path: str,
    pred_matrix: np.ndarray,
    threshold: float,
    window_len: float = 8.0,
    overlap_len: float = 4.0,
    behavior_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    video_window: Optional[Tuple[float, float]] = None,
) -> "plt.Figure":
    """
    Visualise model predictions against ground truth on a per-behavior timeline.

    Layout
    ------
    - Top row    : all ground-truth events as colour-coded vertical ticks.
    - Lower rows : one row per behavior — solid spans where the model fires.

    Parameters
    ----------
    ground_truth_path : str
        TSV with 'Time' (seconds) and 'Behavior type' columns.
    pred_matrix : np.ndarray  (num_behaviors, num_clips)
        Raw logit / probability scores.  Clip j is predicted positive for
        behavior i when pred_matrix[i, j] >= threshold.
    threshold : float
        Decision threshold.
    window_len : float
        Clip duration in seconds (default 8).
    overlap_len : float
        Overlap between consecutive clips in seconds (default 4).
    behavior_names : list[str] | None
        Row-index → behavior name mapping.  Defaults to DEFAULT_LABELS.
    save_path : str | None
        If given, save the figure here (PNG / PDF / SVG).
    video_window : tuple[float, float] | None
        (start, end) recording time in seconds.  GT events outside this range
        are ignored and x-axis labels are offset by start so they read as
        recording time.  Defaults to the module-level VIDEO_WINDOW global.

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

    # ── 3. Clip timeline (0-based, window-relative) ───────────────────────────
    stride      = window_len - overlap_len
    clip_starts = np.arange(num_clips) * stride
    clip_ends   = clip_starts + window_len
    if _window is not None and np.isfinite(_window[1]):
        duration = float(_window[1] - _window[0])
    else:
        duration = float(clip_ends[-1])          # total span in seconds
    pred_binary = pred_matrix >= threshold       # (num_behaviors, num_clips)

    detection_report = compute_detection_report(
        gt_times=gt_times,
        gt_behaviors=gt_behaviors,
        pred_matrix=pred_matrix,
        labels=labels,
        clip_starts=clip_starts,
        clip_ends=clip_ends,
        threshold=threshold,
        iou_thresholds=(.1, .2, .3, .4, .5, .6, .7, .8, .9),
        gt_duration=window_len,
        eval_start=0.0,
        eval_end=duration,
    )

    # ── 4. Colour palette ─────────────────────────────────────────────────────
    try:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(num_behaviors)]

    # ── 5. Figure layout ──────────────────────────────────────────────────────
    GT_HEIGHT     = 3
    PRED_HEIGHT   = 1
    height_ratios = [GT_HEIGHT] + [PRED_HEIGHT] * num_behaviors

    fig_w = max(14.0, duration / 8.0 + 2.0)
    fig_h = max(5.0, (GT_HEIGHT + num_behaviors * PRED_HEIGHT) * 0.55 + 1.5)

    fig, axes = plt.subplots(
        num_behaviors + 1, 1,
        figsize=(fig_w, fig_h),
        gridspec_kw=dict(
            height_ratios=height_ratios,
            hspace=0.06,
            top=0.88, bottom=0.08, left=0.09, right=0.97,
        ),
    )
    ax_gt   = axes[0]
    ax_pred = axes[1:]

    # ── 6. Axis styling helper ────────────────────────────────────────────────
    # X-axis data coordinates are 0-based (window-relative).
    # Tick labels are offset by VIDEO_WINDOW[0] so they show recording time.
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

    # ── 7. Ground-truth row ───────────────────────────────────────────────────
    _style_ax(ax_gt, is_bottom=False, label="Ground\nTruth")

    TICK_YMIN, TICK_YMAX = 0.05, 0.55

    for t, beh in zip(gt_times, gt_behaviors):
        c = colors[labels.index(beh)] if beh in labels else "#777777"

        ax_gt.vlines(t, TICK_YMIN, TICK_YMAX, colors=c, lw=1.8, zorder=5)
        ax_gt.scatter([t], [TICK_YMAX], s=18, color=c, marker="v",
                      zorder=6, clip_on=False)

        # Blended transform: x tracks data position, y stays inside the axes
        # so bbox_inches="tight" doesn't balloon the figure width.
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

    # ── 8. Prediction rows ────────────────────────────────────────────────────
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax      = ax_pred[i]
        is_last = i == num_behaviors - 1
        _style_ax(ax, is_bottom=is_last, label=label, color=color)

        pos_clips = np.where(pred_binary[i])[0]
        if len(pos_clips):
            m_starts, m_ends = _merge_spans(
                clip_starts[pos_clips], clip_ends[pos_clips]
            )
            for x0, x1 in zip(m_starts, m_ends):
                ax.axvspan(x0, x1, ymin=0.1, ymax=0.9,
                           facecolor=color, alpha=0.85, zorder=2, lw=0)

    # ── 9. Legend ─────────────────────────────────────────────────────────────
    fig.legend(
        handles=[
            mpatches.Patch(facecolor="#888888", alpha=0.85,
                           label=f"Predicted positive (threshold = {threshold})"),
            Line2D([0], [0], color="#555555", lw=1.8, marker="v", markersize=5,
                   label="Ground-truth event"),
        ],
        loc="upper right", bbox_to_anchor=(0.97, 0.975),
        fontsize=8, framealpha=0.9, edgecolor="#cccccc",
    )

    # ── 10. Title ─────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Behavior detections vs ground truth"
        f"  |  window = {window_len} s,  overlap = {overlap_len} s,"
        f"  threshold = {threshold}",
        fontsize=10, y=0.93,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    _print_detection_report(detection_report, labels)

    return fig


# ── Demo / smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os

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

    out_path = "/mnt/user-data/outputs/demo_visualization.png"
    try:
        visualize_matrix(
            ground_truth_path=gt_path,
            pred_matrix=logits,
            threshold=0.5,
            window_len=8,
            overlap_len=4,
            behavior_names=["Feeding", "Grooming", "Resting"],
            save_path=out_path,
        )
        print(f"Demo saved → {out_path}")
    finally:
        os.unlink(gt_path)
