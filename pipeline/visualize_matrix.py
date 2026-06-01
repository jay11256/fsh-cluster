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
 
def visualize_matrix(
    ground_truth_path: str,
    pred_matrix: np.ndarray,
    threshold: float,
    window_len: float = 8,
    overlap_len: float = 4,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualise model predictions against ground truth on a per-behavior timeline.
 
    Each behavior gets its own horizontal row.  Predicted clips are drawn as
    shaded spans; ground-truth events are shown as vertical tick marks with
    a downward-pointing triangle at the top.
 
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
    threshold : float
        Decision threshold applied to the logits.
    window_len : float
        Duration of each clip in seconds (default 8).
    overlap_len : float
        Overlap between consecutive clips in seconds (default 4).
    save_path : str or None
        If given, the figure is saved to this path (PNG/PDF/SVG accepted).
 
    Returns
    -------
    matplotlib.figure.Figure
        The completed figure object (can be further customised or saved).
 
    Notes
    -----
    - Behavior labels are inferred from the unique values in the ground-truth
      TSV.  If the number of unique GT behaviors equals ``num_behaviors``, the
      sorted unique names are used as row labels; otherwise rows are labelled
      "Behavior 0", "Behavior 1", … .
    - Row order in the figure matches the sorted order of unique GT behavior
      names (or ascending integer order for anonymous rows).
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
 
    gt_times     = gt[time_col].values.astype(float)
    gt_behaviors = gt[beh_col].values.astype(str)
 
    # ── 2. Clip timeline ──────────────────────────────────────────────────────
    stride         = window_len - overlap_len
    num_behaviors, num_clips = pred_matrix.shape
    clip_starts    = np.arange(num_clips) * stride
    clip_ends      = clip_starts + window_len
    video_duration = float(clip_ends[-1])
 
    all_behaviors   = sorted(set(gt_behaviors))
    behavior_labels = (
        all_behaviors
        if len(all_behaviors) == num_behaviors
        else [f"Behavior {i}" for i in range(num_behaviors)]
    )
 
    pred_binary = pred_matrix >= threshold   # (num_behaviors, num_clips)
 
    # ── 3. Figure layout ──────────────────────────────────────────────────────
    timeline_w = max(10.0, video_duration / 5.0)
    fig_height = max(4.0, num_behaviors * 0.9 + 2.5)
 
    fig = plt.figure(figsize=(2.5 + timeline_w, fig_height))
    gs  = GridSpec(
        num_behaviors, 2,
        figure=fig,
        width_ratios=[1, 5],
        wspace=0.02,
        top=0.88, bottom=0.12, left=0.01, right=0.99,
        hspace=0.08,
    )
 
    # Colour palette – one distinct colour per behavior
    try:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    except AttributeError:          # matplotlib < 3.7
        cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(num_behaviors)]
 
    # ── 4. Draw each behavior row ─────────────────────────────────────────────
    for row_idx, (label, color) in enumerate(zip(behavior_labels, colors)):
        ax_lbl  = fig.add_subplot(gs[row_idx, 0])
        ax_time = fig.add_subplot(gs[row_idx, 1])
        is_last = row_idx == num_behaviors - 1
 
        # Label panel
        ax_lbl.set_xlim(0, 1); ax_lbl.set_ylim(0, 1)
        ax_lbl.axis("off")
        ax_lbl.patch.set_facecolor((*color[:3], 0.12))
        ax_lbl.patch.set_visible(True)
        ax_lbl.text(
            0.5, 0.5, label,
            transform=ax_lbl.transAxes,
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="#222222",
        )
 
        # Timeline panel
        ax_time.set_xlim(0, video_duration); ax_time.set_ylim(0, 1)
        ax_time.set_facecolor("#f8f8f8")
        ax_time.spines[["top", "right", "left"]].set_visible(False)
        ax_time.spines["bottom"].set_color("#bbbbbb")
        ax_time.spines["bottom"].set_visible(True)
        if not is_last:
            ax_time.spines["bottom"].set_linestyle("dotted")
            ax_time.spines["bottom"].set_alpha(0.4)
        ax_time.tick_params(
            left=False, labelleft=False,
            bottom=is_last, labelbottom=is_last,
            labelsize=8,
        )
 
        # Vertical grid lines
        grid_step = 30.0 if video_duration > 120 else max(window_len * 2, 10.0)
        for xs in np.arange(grid_step, video_duration, grid_step):
            ax_time.axvline(xs, color="#dedede", lw=0.5, zorder=0)
 
        # Predicted clips (shaded spans with edge lines)
        for clip_i in np.where(pred_binary[row_idx])[0]:
            x0 = clip_starts[clip_i]
            x1 = clip_ends[clip_i]
            ax_time.axvspan(x0, x1, ymin=0.08, ymax=0.92,
                            facecolor=(*color[:3], 0.28), zorder=2, lw=0)
            for xv in (x0, x1):
                ax_time.axvline(xv, ymin=0.08, ymax=0.92,
                                color=(*color[:3], 0.55), lw=0.8, zorder=3)
 
        # Ground-truth event markers
        for t in gt_times[gt_behaviors == label]:
            ax_time.vlines(t, 0.05, 0.95, color=color, lw=2.2, zorder=5)
            ax_time.scatter([t], [0.95], s=25, color=color,
                            marker="v", zorder=6, clip_on=False)
 
        if is_last:
            ax_time.set_xlabel("Time (s)", fontsize=9, labelpad=4)
 
    # ── 5. Legend + title ─────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(
            facecolor=(0.4, 0.4, 0.4, 0.30),
            label=f"Predicted clip  (logit ≥ {threshold})",
        ),
        Line2D(
            [0], [0], color="#555555", lw=2.2,
            marker="v", markersize=5, label="Ground-truth event",
        ),
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
    logits[0, [1, 2, 8]]  = 1.5   # Grooming
    logits[1, [3, 4, 14]] = 1.8   # Feeding
    logits[2, [9, 10]]    = 1.2   # Resting
 
    try:
        out_path = "demo_visualization.png"
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