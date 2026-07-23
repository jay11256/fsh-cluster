"""
asmloc_postprocess.py
---------------------
ASM-Loc-style temporal post-processing for per-clip behavior scores.

This is a drop-in replacement for the threshold-merge localization used in
``visualize_matrix`` (`_build_pred_spans`). Instead of "threshold -> merge
adjacent positive clips", it converts a per-clip score sequence into temporal
detections with ASM-Loc's proposal generator (He et al., CVPR 2022):

    1. temporally up-sample the per-clip score curve (finer boundary resolution),
    2. sweep MANY thresholds (not one) to get candidate segments,
    3. score each candidate with the Outer-Inner-Contrast (OIC) score
       = mean(inside) - mean(margin just outside)  [ + gamma * recording-level score ],
    4. pool candidates across thresholds and de-duplicate with temporal NMS.

Metrics are computed with the SAME VOC AP matching, GT spans, and clip timeline
as ``visualize_matrix.compute_detection_report`` (those helpers are imported,
not re-implemented), so the ONLY thing that changes vs. the baseline is the
span-building step -- the numbers are therefore directly comparable.

Span convention: a detected run of high scores maps to
``[center_first - window/2, center_last + window/2]``, i.e. exactly the clip
extent the baseline uses, so both methods ground spans in time identically.
"""

import os
import numpy as np

# Reuse the pipeline's metric + IO helpers as the single source of truth.
from visualize_matrix import (
    _load_ground_truth,
    _make_gt_spans,
    _match_detections,
    _compute_ap,
    _clip_eval_inputs,
    _clip_ground_truth_mask,
    _safe_divide,
    _as_numpy_array,
    _temporal_iou,
    _format_detection_report,
    _build_preds_debug_figure,
    DEFAULT_LABELS,
)

# Default multi-threshold sweep for sigmoid probabilities in [0, 1].
DEFAULT_CAS_THRESHOLDS = np.round(np.arange(0.10, 0.85, 0.05), 3)


# ── OIC proposal generation ─────────────────────────────────────────────────

def _contiguous_runs(idxs):
    """Split a sorted 1-D index array into lists of consecutive indices."""
    if len(idxs) == 0:
        return []
    return np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)


def _temporal_nms(spans, iou_thresh):
    """Greedy temporal NMS. spans = list of (start, end, score). Highest score wins."""
    spans = sorted(spans, key=lambda x: x[2], reverse=True)
    keep = []
    while spans:
        best = spans.pop(0)
        keep.append(best)
        spans = [
            s for s in spans
            if _temporal_iou((best[0], best[1]), (s[0], s[1])) <= iou_thresh
        ]
    return keep


def asmloc_build_pred_spans(
    scores,
    clip_starts,
    clip_ends,
    upscale=20,
    cas_thresh_list=None,
    oic_lambda=0.2,
    oic_gamma=0.0,
    nms_iou=0.45,
):
    """
    ASM-Loc OIC proposal generation for one behavior's per-clip score sequence.

    Returns a list of (t_start, t_end, score) sorted by score descending -- the
    SAME format as ``visualize_matrix._build_pred_spans`` so it is drop-in.
    """
    scores = np.asarray(scores, dtype=float)
    clip_starts = np.asarray(clip_starts, dtype=float)
    clip_ends = np.asarray(clip_ends, dtype=float)
    n = len(scores)
    if n == 0:
        return []
    if cas_thresh_list is None:
        cas_thresh_list = DEFAULT_CAS_THRESHOLDS

    clip_centers = (clip_starts + clip_ends) / 2.0
    half = float(np.mean(clip_ends - clip_starts)) / 2.0     # window/2
    eval_start = float(clip_starts.min())
    eval_end = float(clip_ends.max())

    if n == 1:
        return [(eval_start, eval_end, float(scores[0]))]

    # Up-sample the score curve onto a fine time grid for sub-clip boundaries.
    T = max(2, n * int(upscale))
    t_grid = np.linspace(clip_centers[0], clip_centers[-1], T)
    fine = np.interp(t_grid, clip_centers, scores)
    recording_score = float(scores.max())   # video-level proxy (only used if gamma>0)

    proposals = []
    for thr in cas_thresh_list:
        above = np.where(fine >= thr)[0]
        if above.size == 0:
            continue
        for run in _contiguous_runs(above):
            if len(run) < 2:
                continue
            L = len(run)
            inner = float(fine[run].mean())
            outer_s = max(0, int(run[0] - oic_lambda * L))
            outer_e = min(T - 1, int(run[-1] + oic_lambda * L))
            outer_idx = list(range(outer_s, int(run[0]))) + \
                list(range(int(run[-1]) + 1, outer_e + 1))
            outer = float(fine[outer_idx].mean()) if outer_idx else 0.0

            score = inner - outer + oic_gamma * recording_score
            t_start = max(eval_start, float(t_grid[run[0]]) - half)
            t_end = min(eval_end, float(t_grid[run[-1]]) + half)
            if t_end > t_start:
                proposals.append((t_start, t_end, score))

    if not proposals:
        return []
    return _temporal_nms(proposals, nms_iou)


# ── Metrics (mirror of visualize_matrix.compute_detection_report) ───────────

def compute_detection_report_asmloc(
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
    asmloc_kwargs=None,
):
    """
    Same as ``visualize_matrix.compute_detection_report`` but temporal
    detections come from ``asmloc_build_pred_spans`` instead of threshold-merge.
    Clip-level precision/recall/F1 are still reported at the per-class
    ``threshold`` (unchanged) so the metrics file lines up column-for-column
    with the baseline; only ``ap_by_iou`` / ``map_by_iou`` reflect ASM-Loc.
    """
    asmloc_kwargs = dict(asmloc_kwargs or {})

    gt_times = _as_numpy_array(gt_times)
    gt_behaviors = _as_numpy_array(gt_behaviors)
    pred_matrix = _as_numpy_array(pred_matrix)
    clip_starts = _as_numpy_array(clip_starts)
    clip_ends = _as_numpy_array(clip_ends)
    eval_start = float(eval_start)
    eval_end = float(eval_end) if eval_end is not None else None

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
        scores = pred_matrix[behavior_idx]

        gt_spans = _make_gt_spans(
            gt_times=gt_times,
            gt_behaviors=gt_behaviors,
            label=label,
            gt_duration=gt_duration,
            min_time=eval_start,
            max_time=timeline_end,
        )
        n_gt = len(gt_spans)

        # >>> the only change vs. the baseline: ASM-Loc OIC spans <<<
        pred_spans = asmloc_build_pred_spans(
            scores, clip_starts, clip_ends, **asmloc_kwargs
        )

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
            "precision": precision,
            "recall": recall,
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


# ── Visualization + save (parallels visualize_matrix.visualize_matrix) ──────

def visualize_matrix_asmloc(
    ground_truth_path,
    pred_matrix,
    threshold=0.5,
    thresholds=None,
    window_len=8.0,
    overlap_len=4.0,
    behavior_names=None,
    save_path=None,
    video_window=None,
    upscale=20,
    cas_thresh_list=None,
    oic_lambda=0.2,
    oic_gamma=0.0,
    nms_iou=0.45,
    **kwargs,
):
    """
    ASM-Loc post-processing counterpart to ``visualize_matrix.visualize_matrix``.

    Produces, under ``save_path``:
      - ``metrics``                  : same-format report (mAP now from ASM-Loc),
      - ``visualize_predictions.png``: GT ticks, score curves, and the ASM-Loc
                                       predicted spans (one row per behavior),
      - ``preds_debug.png``          : score curves vs GT (reused, span-agnostic).

    Returns the detection report dict.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
    import matplotlib.transforms as mtransforms
    from matplotlib.lines import Line2D

    _window = video_window if video_window is not None else (0.0, float("inf"))
    gt_times, gt_behaviors = _load_ground_truth(ground_truth_path, _window)
    pred_matrix = _as_numpy_array(pred_matrix)
    num_behaviors, num_clips = pred_matrix.shape

    if behavior_names is not None:
        labels = list(behavior_names)
    else:
        labels = DEFAULT_LABELS
    if len(labels) != num_behaviors:
        raise ValueError(
            f"labels has {len(labels)} entries but pred_matrix has "
            f"{num_behaviors} rows. Pass behavior_names= explicitly."
        )

    if thresholds is not None and len(thresholds) > 0:
        thresh_per_class = list(thresholds)
    else:
        thresh_per_class = [threshold] * num_behaviors

    stride = window_len - overlap_len
    clip_starts = np.arange(num_clips) * stride
    clip_ends = clip_starts + window_len
    if _window is not None and np.isfinite(_window[1]):
        duration = float(_window[1] - _window[0])
    else:
        duration = float(clip_ends[-1])

    asmloc_kwargs = dict(
        upscale=upscale,
        cas_thresh_list=cas_thresh_list,
        oic_lambda=oic_lambda,
        oic_gamma=oic_gamma,
        nms_iou=nms_iou,
    )

    report = compute_detection_report_asmloc(
        gt_times=gt_times,
        gt_behaviors=gt_behaviors,
        pred_matrix=pred_matrix,
        labels=labels,
        clip_starts=clip_starts,
        clip_ends=clip_ends,
        threshold=threshold,
        thresholds=thresh_per_class,
        iou_thresholds=(.1, .2, .3, .4, .5, .6, .7, .8, .9),
        gt_duration=window_len,
        eval_start=0.0,
        eval_end=duration,
        asmloc_kwargs=asmloc_kwargs,
    )

    if save_path is None:
        return report

    os.makedirs(save_path, exist_ok=True)

    # ── Figure: GT row + score-curve row + per-class ASM-Loc span rows ───────
    try:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(num_behaviors)]
    clip_centers = (clip_starts + clip_ends) / 2.0
    display_offset = _window[0] if np.isfinite(_window[0]) else 0.0

    height_ratios = [3, 2] + [1] * num_behaviors
    fig_w = max(14.0, duration / 8.0 + 2.0)
    fig_h = max(5.0, (3 + num_behaviors) * 0.55 + 1.5)
    fig, axes = plt.subplots(
        num_behaviors + 2, 1, figsize=(fig_w, fig_h),
        gridspec_kw=dict(height_ratios=height_ratios, hspace=0.06,
                         top=0.88, bottom=0.08, left=0.09, right=0.97),
    )
    ax_gt, ax_prob, ax_pred = axes[0], axes[1], axes[2:]

    def _grid(ax):
        step = 30.0 if duration > 200 else 20.0 if duration > 80 else 10.0
        for xs in np.arange(step, duration, step):
            ax.axvline(xs, color="#e0e0e0", lw=0.6, zorder=0)

    # GT row
    ax_gt.set_xlim(0.0, duration); ax_gt.set_ylim(0.0, 1.0); ax_gt.set_yticks([])
    ax_gt.set_ylabel("Ground\nTruth", rotation=0, ha="right", va="center",
                     fontsize=9, fontweight="bold")
    ax_gt.spines[["top", "right", "left"]].set_visible(False)
    ax_gt.tick_params(bottom=False, labelbottom=False)
    _grid(ax_gt)
    for t, beh in zip(gt_times, gt_behaviors):
        c = colors[labels.index(beh)] if beh in labels else "#777777"
        ax_gt.vlines(t, 0.05, 0.55, colors=c, lw=1.6, zorder=5)
        ax_gt.scatter([t], [0.55], s=14, color=c, marker="v", zorder=6, clip_on=False)

    # Score-curve row
    ax_prob.set_xlim(0.0, duration); ax_prob.set_ylim(0.0, 1.0)
    ax_prob.set_ylabel("Scores", rotation=0, ha="right", va="center",
                       fontsize=9, fontweight="bold")
    ax_prob.set_yticks([0.0, 0.5, 1.0]); ax_prob.set_yticklabels(["0", "0.5", "1"])
    ax_prob.spines[["top", "right"]].set_visible(False)
    ax_prob.tick_params(bottom=False, labelbottom=False)
    _grid(ax_prob)
    for label, color, probs in zip(labels, colors, pred_matrix):
        ax_prob.plot(clip_centers, probs, color=color, lw=1.3, label=label)

    # Per-class ASM-Loc span rows
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax = ax_pred[i]
        is_last = i == num_behaviors - 1
        ax.set_xlim(0.0, duration); ax.set_ylim(0.0, 1.0); ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center",
                      fontsize=9, fontweight="bold", color=color)
        ax.spines[["top", "right", "left"]].set_visible(False)
        r, g, b = color[:3]; ax.set_facecolor((r, g, b, 0.06))
        _grid(ax)
        if is_last:
            ax.tick_params(bottom=True, labelbottom=True, labelsize=8)
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x + display_offset:.4g}"))
        else:
            ax.tick_params(bottom=False, labelbottom=False)

        spans = asmloc_build_pred_spans(
            pred_matrix[i], clip_starts, clip_ends, **asmloc_kwargs)
        for (x0, x1, _sc) in spans:
            ax.axvspan(x0, x1, ymin=0.1, ymax=0.9, facecolor=color,
                       alpha=0.85, zorder=2, lw=0)

    fig.legend(
        handles=[
            mpatches.Patch(facecolor="#888888", alpha=0.85, label="ASM-Loc detection"),
            Line2D([0], [0], color="#555555", lw=1.8, marker="v", markersize=5,
                   label="Ground-truth event"),
        ],
        loc="upper right", bbox_to_anchor=(0.97, 0.975),
        fontsize=8, framealpha=0.9, edgecolor="#cccccc",
    )
    fig.suptitle(
        f"ASM-Loc post-processed detections vs ground truth"
        f"  |  window = {window_len} s, overlap = {overlap_len} s",
        fontsize=10, y=0.93,
    )
    fig.savefig(os.path.join(save_path, "visualize_predictions.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Reuse the (span-agnostic) score-vs-GT debug figure for parity.
    debug_fig = _build_preds_debug_figure(
        gt_times=gt_times, gt_behaviors=gt_behaviors, labels=labels, colors=colors,
        pred_matrix=pred_matrix, clip_centers=clip_centers, duration=duration,
        display_offset=display_offset, window_len=window_len, overlap_len=overlap_len,
    )
    debug_fig.savefig(os.path.join(save_path, "preds_debug.png"),
                      dpi=300, bbox_inches="tight")
    plt.close(debug_fig)

    with open(os.path.join(save_path, "metrics"), "w", encoding="utf-8") as f:
        f.write(_format_detection_report(report, labels))

    return report
