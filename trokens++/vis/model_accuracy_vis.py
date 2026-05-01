"""
plot_from_wandb.py
==================
Pulls test_acc1 from 9 wandb runs and produces a bar chart.

HOW TO USE
----------
Paste the full run ID for each of the 9 (shot, method) combinations into
RUN_IDS below.  The full run ID is the long string you see in wandb, e.g.:
    ds6_5_way-1_shot-sam3-both_b9MYdeB8

That is what goes in the dict.  Nothing else needs to change.
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt

#   Format: (n_shot, method) -> "full_wandb_run_id"
#
#   n_shot  : 1, 3, or 5
#   method  : "none", "trokens", or "sam3"
#   run id  : the full ID string from wandb, e.g. ds6_5_way-1_shot-none-both_XXXXXXXX
#
FIG_TITLE = "DS6_Big6 Test Accuracies"
RUN_IDS = {
    (1, "none"):    "ds6_5_way-1_shot-none-both_jyBVQb5W",   
    (1, "trokens"): "ds6_5_way-1_shot-trokens-both_Xf4K4k8B",  
    (1, "sam3"):    "ds6_5_way-1_shot-sam3-both_b9MYdeB8",   

    (3, "none"):    "ds6_5_way-3_shot-none-both_ac1HVfkR",   
    (3, "trokens"): "ds6_5_way-3_shot-trokens-both_Z1acLKKz",
    (3, "sam3"):    "ds6_5_way-3_shot-sam3-both_UCYsyAq8",

    (5, "none"):    "ds7_5_way-5_shot-none-both_CBxD48Wo",
    (5, "trokens"): "ds7_big6_5_way-5_shot-trokens-both_QSVXZgwp",
    (5, "sam3"):    "ds7_big6_5_way-5_shot-sam3-both_UDLfMCut",
}

# ── Wandb config ──────────────────────────────────────────────────────────────

ENTITY  = "fsh-gems"
PROJECT = "trokens"
METRIC  = "test_acc1"

SHOTS   = [1, 3, 5]
METHODS = ["none", "trokens", "sam3"]

# ── Pull accuracies ───────────────────────────────────────────────────────────

api = wandb.Api()

def fetch(run_id: str) -> float:
    if run_id == "REPLACE_ME":
        raise ValueError(
            "RUN_IDS still contains a placeholder. "
            "Please fill in all 9 run IDs before running the script."
        )
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    val = run.summary.get(METRIC)
    if val is None:
        raise KeyError(
            f"'{METRIC}' not found in run '{run_id}'.\n"
            f"Available summary keys: {list(run.summary.keys())}"
        )
    return float(val)

print("Pulling test_acc1 from wandb …\n")

# accuracies[shot_idx][method_idx]
accuracies = []
for k in SHOTS:
    row = []
    for method in METHODS:
        val = fetch(RUN_IDS[(k, method)])
        print(f"  5×{k}  {method:8s}  →  {val:.2f}%")
        row.append(val)
    accuracies.append(row)

print()

# ── Plot ──────────────────────────────────────────────────────────────────────

series_labels = ["None", "Trokens", "Trokens++"]
colors        = ["grey", "#6FB9E3", "#DC7D2D"]
x_labels      = ["5x1", "5x3", "5x5"]

x        = np.arange(len(x_labels))
n_series = len(series_labels)
width    = 0.25

FS_AXIS   = 16
FS_TICK   = 16
FS_BAR    = 16
FS_LEGEND = 16

fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for i, (label, color) in enumerate(zip(series_labels, colors)):
    offsets = (i - (n_series - 1) / 2) * width
    heights = [accuracies[g][i] for g in range(len(x_labels))]
    bars = ax.bar(
        x + offsets, heights, width,
        label=label, color=color,
        edgecolor="white", linewidth=0.5,
    )
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 1,
            f"{yval:.1f}%",
            ha="center", va="bottom",
            fontsize=FS_BAR,
        )

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=FS_TICK)
ax.set_ylim(0, 100)
ax.set_xlabel("Few Shot Configuration", fontsize=FS_AXIS)
ax.set_ylabel("Model Accuracy (%)", fontsize=FS_AXIS)
ax.legend(
    title="Model",
    frameon=True,
    facecolor="white",
    loc="lower right",
    fontsize=FS_LEGEND,
    title_fontsize=FS_LEGEND,
)
ax.grid(axis="y", color="grey", linestyle="-", linewidth=0.8, alpha=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="x", which="both", length=0, labelsize=FS_TICK)
ax.tick_params(axis="y", which="major", length=4, labelsize=FS_TICK)
ax.set_title(f"{FIG_TITLE}")

for spine in ax.spines.values():
    spine.set_visible(True)

plt.tight_layout()

output_path = "./model_accuracy_wandb.png"
plt.savefig(output_path, facecolor="white", edgecolor="none", dpi=300)
plt.close()
print(f"Bar chart saved to '{output_path}'.")