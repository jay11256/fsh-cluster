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


def _merge_spans(starts, ends):
    """Merge a list of (start, end) intervals into non-overlapping spans."""
    if len(starts) == 0:
        return [], []
    pairs = sorted(zip(starts, ends))
    merged = [list(pairs[0])]
    for s, e in pairs[1:]:
        if s <= merged[-1][1]:          # overlaps or adjacent
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    ms, me = zip(*merged)
    return list(ms), list(me)


def visualize_matrix(
    ground_truth_path: str,
    pred_matrix: np.ndarray,
    threshold: float,
    window_len: float = 8,
    overlap_len: float = 4,
    behavior_names: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualise model predictions against ground truth on a per-behavior timeline.

    Layout
    ------
    - Top row  : ground-truth events for every behavior, drawn as labelled
                 vertical ticks on a single shared timeline.
    - Lower rows: one row per behavior showing solid predicted spans.

    Parameters
    ----------
    ground_truth_path : str
        Path to a TSV file with (at minimum) the columns:
            - "Time"          : float, time in seconds of each event
            - "Behavior type" : str, name of the behavior
        Column names are matched case-insensitively.
    pred_matrix : np.ndarray, shape (num_behaviors, num_clips)
        Matrix of raw logit scores.  Entry [i, j] is the model's logit for
        behavior i in clip j.  A clip is predicted positive if logit >= threshold.
        Row i corresponds to behavior_names[i] (see below).
    threshold : float
        Decision threshold applied to the logits.
    window_len : float
        Duration of each clip in seconds (default 8).
    overlap_len : float
        Overlap between consecutive clips in seconds (default 4).
    behavior_names : list[str] or None
        Explicit mapping from pred_matrix row index → behavior name.
        If None, the sorted unique behavior names from the TSV are used
        (row 0 → alphabetically first behavior, etc.).
    save_path : str or None
        If given, the figure is saved to this path (PNG/PDF/SVG accepted).

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── 1. Parse ground truth ─────────────────────────────────────────────────
    gt = pd.read_csv(ground_truth_path, sep="\t")
    gt.columns = gt.columns.str.strip()

    col_map  = {c.lower(): c for c in gt.columns}
    time_col = col_map.get(
        "time", next((c for c in gt.columns if "time" in c.lower()), None)
    )
    beh_col = col_map.get(
        "behavior type",
        next((c for c in gt.columns if "behavior" in c.lower()), None),
    )

    if time_col is None or beh_col is None:
        raise ValueError(
            f"Could not locate 'Time' and 'Behavior type' columns. "
            f"Columns found: {list(gt.columns)}"
        )

    gt_times     = gt[time_col].values.astype(float) - 500.0
    gt_behaviors = gt[beh_col].astype(str).str.strip().values


    # Mapping/dict from Will's data6make

    behavior_map = {
        "peck": "Peck",
        "quiver-m": "Quiver",
        "quiver (male)": "Quiver",
        "bite (male)": "Bite",
        "tilt (specify sex)": "Tilt",
        "peck (specify sex)": "Peck",
        "lead (male)": "Lead",
        "lead-m": "Lead",
        "pot entry": "Enter Pot",
        "chase/charge (male)": "Chase/Charge",
        "run/flee (female)": "Run/Flee",
        "male quiver": "Quiver",
        "exit plot": "Exit Plot",
        "male lead": "Lead",
        "egg retrieval": "Egg Retrieval",
        "circling": "Circling",
        "female follow": "Follow",
        "follow-f": "Follow",
        "spawning-f": "Spawning",
        "follow (female)": "Follow",
    }
    gt_behaviors = np.array(
        [behavior_map.get(beh.lower(), beh) for beh in gt_behaviors],
        dtype=str,
    )

    # ── 2. Behavior label mapping ─────────────────────────────────────────────
    num_behaviors, num_clips = pred_matrix.shape

    if behavior_names is not None:
        if len(behavior_names) != num_behaviors:
            raise ValueError(
                f"behavior_names has {len(behavior_names)} entries but "
                f"pred_matrix has {num_behaviors} rows."
            )
        labels = list(behavior_names)
    else:
        # Default: sorted unique names from the TSV, in alphabetical order.
        # Row i of pred_matrix is assumed to correspond to the i-th name.
        labels = ["Bite", "Lead", "Peck", "Quiver", "Run/Flee", "Tilt"]
        if len(labels) != num_behaviors:
            raise ValueError(
                f"Found {len(labels)} unique behaviors in the TSV but "
                f"pred_matrix has {num_behaviors} rows. "
                f"Pass behavior_names= explicitly to resolve the mapping.\n"
                f"TSV behaviors: {labels}"
            )

    # ── 3. Clip timeline ──────────────────────────────────────────────────────
    stride         = window_len - overlap_len
    clip_starts    = np.arange(num_clips) * stride
    clip_ends      = clip_starts + window_len
    video_duration = float(clip_ends[-1])

    pred_binary = pred_matrix >= threshold   # (num_behaviors, num_clips)

    # ── 4. Colour palette ─────────────────────────────────────────────────────
    try:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(num_behaviors)]

    # ── 5. Figure layout ──────────────────────────────────────────────────────
    # Rows: 1 GT row + num_behaviors prediction rows
    total_rows = num_behaviors + 1
    timeline_w = max(10.0, video_duration / 5.0)
    fig_height = max(4.0, total_rows * 0.9 + 2.5)

    fig = plt.figure(figsize=(2.5 + timeline_w, fig_height))
    gs  = GridSpec(
        total_rows, 2,
        figure=fig,
        width_ratios=[1, 5],
        wspace=0.02,
        top=0.88, bottom=0.10, left=0.01, right=0.99,
        hspace=0.08,
    )

    # ── helper: shared timeline axis setup ───────────────────────────────────
    def _setup_timeline(ax, is_last):
        ax.set_xlim(0, video_duration)
        ax.set_ylim(0, 1)
        ax.set_facecolor("#f8f8f8")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.spines["bottom"].set_visible(True)
        if not is_last:
            ax.spines["bottom"].set_linestyle("dotted")
            ax.spines["bottom"].set_alpha(0.4)
        ax.tick_params(
            left=False, labelleft=False,
            bottom=is_last, labelbottom=is_last,
            labelsize=8,
        )
        grid_step = 30.0 if video_duration > 120 else max(window_len * 2, 10.0)
        for xs in np.arange(grid_step, video_duration, grid_step):
            ax.axvline(xs, color="#dedede", lw=0.5, zorder=0)

    def _setup_label(ax, text, color=None):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        fc = (*color[:3], 0.12) if color is not None else (0.95, 0.95, 0.95, 1.0)
        ax.patch.set_facecolor(fc)
        ax.patch.set_visible(True)
        ax.text(0.5, 0.5, text, transform=ax.transAxes,
                ha="center", va="center",
                fontsize=9, fontweight="bold", color="#222222")

    # ── 6. Ground-truth row (row 0) ───────────────────────────────────────────
    ax_gt_lbl  = fig.add_subplot(gs[0, 0])
    ax_gt_time = fig.add_subplot(gs[0, 1])

    _setup_label(ax_gt_lbl, "Ground truth")
    _setup_timeline(ax_gt_time, is_last=False)

    for t, beh in zip(gt_times, gt_behaviors):
        # find the colour for this behavior
        if beh in labels:
            c = colors[labels.index(beh)]
        else:
            c = (0.4, 0.4, 0.4, 1.0)
        ax_gt_time.vlines(t, 0.05, 0.88, color=c, lw=2.2, zorder=5)
        ax_gt_time.scatter([t], [0.88], s=22, color=c,
                           marker="v", zorder=6, clip_on=False)
        ax_gt_time.text(t, 0.95, beh, ha="center", va="bottom",
                        fontsize=6.5, color=c,
                        rotation=45, rotation_mode="anchor",
                        clip_on=True)

    # ── 7. Prediction rows (rows 1 … num_behaviors) ───────────────────────────
    for row_idx, (label, color) in enumerate(zip(labels, colors)):
        gs_row  = row_idx + 1          # offset by the GT row
        is_last = row_idx == num_behaviors - 1

        ax_lbl  = fig.add_subplot(gs[gs_row, 0])
        ax_time = fig.add_subplot(gs[gs_row, 1])

        _setup_label(ax_lbl, label, color)
        _setup_timeline(ax_time, is_last)

        # Merge overlapping predicted clips → solid contiguous blocks
        pred_clip_indices = np.where(pred_binary[row_idx])[0]
        if len(pred_clip_indices) > 0:
            raw_starts = clip_starts[pred_clip_indices]
            raw_ends   = clip_ends[pred_clip_indices]
            m_starts, m_ends = _merge_spans(raw_starts, raw_ends)
            for x0, x1 in zip(m_starts, m_ends):
                ax_time.axvspan(x0, x1, ymin=0.08, ymax=0.92,
                                facecolor=color, zorder=2, lw=0)

        if is_last:
            ax_time.set_xlabel("Time (s)", fontsize=9, labelpad=4)

    # ── 8. Legend + title ─────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=(0.4, 0.4, 0.4, 0.8),
                       label=f"Predicted (logit ≥ {threshold})"),
        Line2D([0], [0], color="#555555", lw=2.2,
               marker="v", markersize=5, label="Ground-truth event"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.98),
        ncol=1, fontsize=8, framealpha=0.9, edgecolor="#cccccc",
    )
    fig.suptitle(
        f"Behavior predictions vs ground truth"
        f"  (window={window_len}s, overlap={overlap_len}s, threshold={threshold})",
        fontsize=10,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── Demo / smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os

    gt_tsv = (
        "Time\tBehavior type\n"
        "5.0\tGrooming\n"
        "18.0\tFeeding\n"
        "22.0\tFeeding\n"
        "35.0\tGrooming\n"
        "48.0\tResting\n"
        "60.0\tFeeding\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        f.write(gt_tsv)
        gt_path = f.name

    rng    = np.random.default_rng(42)
    logits = rng.normal(loc=-0.5, scale=1.2, size=(3, 15))
    # pred_matrix rows are in sorted label order: Feeding=0, Grooming=1, Resting=2
    logits[0, [3, 4, 13, 14]] = 1.8   # Feeding  (clips 3-4 are adjacent → merge)
    logits[1, [1, 2, 8]]      = 1.5   # Grooming
    logits[2, [9, 10]]        = 1.2   # Resting

    try:
        out_path = "/mnt/user-data/outputs/demo_visualization.png"
        visualize_matrix(
            ground_truth_path=gt_path,
            pred_matrix=logits,
            threshold=0.5,
            window_len=8,
            overlap_len=4,
            save_path=out_path,
        )
        print(f"Demo saved to {out_path}")
    finally:
        os.unlink(gt_path)