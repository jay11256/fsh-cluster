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
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


# ── Global offset ──────────────────────────────────────────────────────────────
# The time (in seconds) at which your clip sequence begins relative to the
# ground-truth TSV timestamps.  e.g. if the TSV starts at 0 s but your clips
# start at 500 s into the recording, set VIDEO_START_OFFSET = 500.0.
VIDEO_START_OFFSET = 500.0


# Canonical behavior-name aliases (lower-case key → display name).
BEHAVIOR_MAP = {
    "peck":               "Peck",
    "peck (specify sex)": "Peck",
    "quiver-m":           "Quiver",
    "quiver (male)":      "Quiver",
    "male quiver":        "Quiver",
    "bite (male)":        "Bite",
    "tilt (specify sex)": "Tilt",
    "lead (male)":        "Lead",
    "lead-m":             "Lead",
    "male lead":          "Lead",
    "pot entry":          "Enter Pot",
    "chase/charge (male)":"Chase/Charge",
    "run/flee (female)":  "Run/Flee",
    "exit plot":          "Exit Plot",
    "egg retrieval":      "Egg Retrieval",
    "circling":           "Circling",
    "female follow":      "Follow",
    "follow-f":           "Follow",
    "follow (female)":    "Follow",
    "spawning-f":         "Spawning",
}

# Default behavior labels (alphabetical) when behavior_names is not supplied.
DEFAULT_LABELS = ["Bite", "Lead", "Peck", "Quiver", "Run/Flee", "Tilt"]


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


def _load_ground_truth(path: str):
    """
    Read a TSV with at least 'Time' and 'Behavior type' columns.
    Returns (times_array, behaviors_array) after applying BEHAVIOR_MAP and
    subtracting VIDEO_START_OFFSET from every timestamp.
    """
    gt = pd.read_csv(path, sep="\t")
    gt.columns = gt.columns.str.strip()
    col_map = {c.lower(): c for c in gt.columns}

    time_col = col_map.get("time") or next(
        (c for c in gt.columns if "time" in c.lower()), None
    )
    beh_col = col_map.get("behavior type") or next(
        (c for c in gt.columns if "behavior" in c.lower()), None
    )
    if time_col is None or beh_col is None:
        raise ValueError(
            f"Could not locate 'Time' and 'Behavior type' columns. "
            f"Columns found: {list(gt.columns)}"
        )

    times = gt[time_col].values.astype(float) - VIDEO_START_OFFSET
    raw   = gt[beh_col].astype(str).str.strip().values
    behaviors = np.array(
        [BEHAVIOR_MAP.get(b.lower(), b) for b in raw], dtype=str
    )
    return times, behaviors


def visualize_matrix(
    ground_truth_path: str,
    pred_matrix: np.ndarray,
    threshold: float,
    window_len: float = 8.0,
    overlap_len: float = 4.0,
    behavior_names: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
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
        Raw logit scores.  Clip j is predicted positive for behavior i when
        pred_matrix[i, j] >= threshold.
    threshold : float
        Decision threshold applied to logits.
    window_len : float
        Clip duration in seconds (default 8).
    overlap_len : float
        Overlap between consecutive clips in seconds (default 4).
    behavior_names : list[str] | None
        Row-index → behavior name mapping.  Defaults to DEFAULT_LABELS.
    save_path : str | None
        If given, save the figure here (PNG / PDF / SVG).

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── 1. Ground truth ───────────────────────────────────────────────────────
    gt_times, gt_behaviors = _load_ground_truth(ground_truth_path)

    # ── 2. Labels ─────────────────────────────────────────────────────────────
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

    # ── 3. Clip timeline ──────────────────────────────────────────────────────
    stride         = window_len - overlap_len
    clip_starts    = np.arange(num_clips) * stride
    clip_ends      = clip_starts + window_len
    video_duration = float(clip_ends[-1])
    pred_binary    = pred_matrix >= threshold        # (num_behaviors, num_clips)

    # ── 4. Colour palette ─────────────────────────────────────────────────────
    try:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(num_behaviors)]

    # ── 5. Figure / GridSpec ──────────────────────────────────────────────────
    # Single column of axes; GT row is taller to accommodate rotated tick labels.
    GT_HEIGHT   = 3      # relative height units for the GT row
    PRED_HEIGHT = 1      # relative height units for each prediction row
    height_ratios = [GT_HEIGHT] + [PRED_HEIGHT] * num_behaviors

    fig_w = max(14.0, video_duration / 8.0 + 2.0)
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

    # ── 6. Shared axis helpers ────────────────────────────────────────────────
    def _style_ax(ax, is_bottom: bool, label: str, color=None):
        """Apply common styling; write the row label on the y-axis."""
        ax.set_xlim(0.0, video_duration)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])

        # y-axis label (row name)
        label_color = color if color is not None else "#444444"
        ax.set_ylabel(
            label,
            rotation=0,
            ha="right",
            va="center",
            labelpad=6,
            fontsize=9,
            fontweight="bold",
            color=label_color,
        )

        # Spines
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(
            "#888888" if is_bottom else "#cccccc"
        )

        # X ticks only on the bottom row
        if is_bottom:
            ax.tick_params(bottom=True, labelbottom=True, labelsize=8)
            ax.set_xlabel("Time (s)", fontsize=9, labelpad=4)
        else:
            ax.tick_params(bottom=False, labelbottom=False)

        # Light vertical grid
        grid_step = 30.0 if video_duration > 200 else 20.0 if video_duration > 80 else 10.0
        for xs in np.arange(grid_step, video_duration, grid_step):
            ax.axvline(xs, color="#e0e0e0", lw=0.6, zorder=0)

        # Tinted row background
        if color is not None:
            r, g, b = color[:3]
            ax.set_facecolor((r, g, b, 0.06))
        else:
            ax.set_facecolor("#f5f5f5")

    # ── 7. Ground-truth row ───────────────────────────────────────────────────
    _style_ax(ax_gt, is_bottom=False, label="Ground\nTruth")

    # Draw ticks in the lower portion of the GT row; labels float above in
    # axes-fraction space so they never overlap the prediction rows below.
    TICK_YMIN, TICK_YMAX = 0.05, 0.55   # tick line extent (data coords 0–1)

    for t, beh in zip(gt_times, gt_behaviors):
        c = colors[labels.index(beh)] if beh in labels else "#777777"
        # Vertical tick line
        ax_gt.vlines(t, TICK_YMIN, TICK_YMAX, colors=c, lw=1.8, zorder=5)
        # Small triangle marker at top of tick
        ax_gt.scatter([t], [TICK_YMAX], s=18, color=c, marker="v",
                      zorder=6, clip_on=False)
        # Label above the tick, in axes-fraction coords so it sits in the
        # generous GT row height and never spills into adjacent cells.
        ax_gt.annotate(
            beh,
            xy=(t, TICK_YMAX),
            xycoords=("data", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=6.5, color=c,
            rotation=90,
            clip_on=False,           # never clipped by axis boundary
            zorder=7,
        )

    # ── 8. Prediction rows ────────────────────────────────────────────────────
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax = ax_pred[i]
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
    legend_handles = [
        mpatches.Patch(facecolor="#888888", alpha=0.85,
                       label=f"Predicted positive (threshold = {threshold})"),
        Line2D([0], [0], color="#555555", lw=1.8, marker="v", markersize=5,
               label="Ground-truth event"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.975),
        fontsize=8,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    # ── 10. Title ─────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Behavior detections vs ground truth"
        f"  |  window = {window_len} s,  overlap = {overlap_len} s,"
        f"  threshold = {threshold}",
        fontsize=10,
        y=0.93,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

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