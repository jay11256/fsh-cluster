import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repo root: data dirs + fishformer package
sys.path.insert(0, ROOT)
from fishformer.former import FishFormer
from fishformer.data import list_recordings, load_recording, FG_CLASSES, NUM_CLASSES, STRIDE
from fishformer.train_former import predict_spans

CKPT_DIR = f"{ROOT}/checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Args: pass

recs_to_test = [
    ("Ds12_06_5fold_fold0_Former2", "25-05-22-Run1-Sham-Cir"),
    ("Ds12_06_5fold_fold0_Former2", "25-06-26-Run1-VetBond-NoCir"),
]

all_lengths = []
per_rec_summary = []
for label, rec in recs_to_test:
    ckpt = torch.load(f"{CKPT_DIR}/{label}__{rec}.pt", map_location=device)
    cfg = ckpt["config"]
    model = FishFormer(feat_dim=ckpt["feat_dim"], num_classes=NUM_CLASSES, hidden=cfg["hidden"],
                       depth=cfg["depth"], num_heads=cfg["num_heads"], drop=0.0, drop_path=0.0, reg_bins=cfg["reg_bins"],
                       spatial_pool=False, use_motion=cfg["use_motion"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    args = Args()
    args.feature_mode = cfg["feature_mode"]
    args.window_s = cfg["window_s"]
    args.score_thresh = cfg["score_thresh"]
    args.nms_iou = cfg["nms_iou"]
    args.max_props = cfg["max_props"]

    props, events, stride, n = predict_spans(model, rec, args, device)
    rec_lengths = []
    for c, spans in props.items():
        for s, e, score in spans:
            rec_lengths.append(e - s)
    all_lengths.extend(rec_lengths)
    per_rec_summary.append((rec, len(rec_lengths), float(np.mean(rec_lengths)) if rec_lengths else float("nan"),
                            float(np.median(rec_lengths)) if rec_lengths else float("nan")))
    print(f"{rec}: n_preds={len(rec_lengths)} mean_len={np.mean(rec_lengths):.3f}s "
          f"median_len={np.median(rec_lengths):.3f}s min={min(rec_lengths):.3f}s max={max(rec_lengths):.3f}s")

print(f"\n=== OVERALL ({len(all_lengths)} predictions across {len(recs_to_test)} recordings) ===")
print(f"mean length = {np.mean(all_lengths):.3f}s")
print(f"median length = {np.median(all_lengths):.3f}s")
print(f"std = {np.std(all_lengths):.3f}s")
print(f"min/max = {min(all_lengths):.3f}s / {max(all_lengths):.3f}s")
print(f"(for reference: training span_s=4.0 -> nominal box width = 4.0s during training)")
