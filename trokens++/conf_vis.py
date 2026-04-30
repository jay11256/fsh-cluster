import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os

# === CONFIG ===
OUTPUT_DIR = "."

K_VALUES = ["1", "3"]
POINTS_VALUES = ["none", "sam3", "trokens"]
CLASS_NAMES = ["Bite", "Lead", "Peck", "Quiver", "Run/Flee", "Tilt"]

AVG_FIGURE_TITLE = "DS6 Confusion Matrix (Averaged)"

os.makedirs(OUTPUT_DIR, exist_ok=True)
cm_accumulator = []

for K in K_VALUES:
    for points in POINTS_VALUES:
        test = f"5_way-{K}_shot-{points}-both"
        csv_path = f"/fs/vulcan-projects/fsh_track/models/ds6/{test}/confusion_matrix.csv"

        if not os.path.exists(csv_path):
            print(f"Skipping {test}: file not found at {csv_path}")
            continue

        df = pd.read_csv(csv_path, index_col=0)
        df = df.dropna(how="all")

        col_mapping = {f"pred_{i}": CLASS_NAMES[i] for i in range(len(CLASS_NAMES))}
        row_mapping = {f"true_{i}": CLASS_NAMES[i] for i in range(len(CLASS_NAMES))}
        df = df.rename(columns=col_mapping, index=row_mapping)

        cm = df.values.astype(float)
        cm = cm / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

        cm_accumulator.append(cm)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=df.columns,
        )

        disp.plot(cmap="Blues", values_format=".2f")
        plt.title(f"Confusion Matrix (5-way {K}-shot, {points})")

        out_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{K}shot_{points}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")

# === AVERAGED CONFUSION MATRIX ===
if cm_accumulator:
    cm_avg = np.mean(np.stack(cm_accumulator), axis=0)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_avg,
        display_labels=CLASS_NAMES,
    )

    disp.plot(cmap="Blues", values_format=".2f")
    plt.title(AVG_FIGURE_TITLE)

    out_path = os.path.join(OUTPUT_DIR, "confusion_matrix_avg.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
else:
    print("No confusion matrices were generated; skipping average.")

print("Done.")