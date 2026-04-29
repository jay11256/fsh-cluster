import argparse
import sys
import os

# Ensure the trokens project is in sys.path so we can import its modules
script_dir = os.path.dirname(os.path.abspath(__file__))
trokens_dir = os.path.join(script_dir, 'trokens')
if trokens_dir not in sys.path:
    sys.path.insert(0, trokens_dir)

import torch
import numpy as np
from einops import rearrange

from trokens.config.defaults import get_cfg, assert_and_infer_cfg
from trokens.models import build_model
from trokens.utils import checkpoint as cu
from trokens.datasets import utils as data_utils

def parse_args():
    parser = argparse.ArgumentParser(description="Test model on a single video input")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the config file (e.g., configs/trokens/fshdata.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pyth)")
    parser.add_argument(
        "opts",
        help="See trokens/config/defaults.py for all options (e.g., MODEL.NUM_CLASSES 6)",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()

def main():
    if 'TORCH_HOME' not in os.environ:
        os.environ['TORCH_HOME'] = '/fs/vulcan-projects/fsh_track/programs/trokens_workspace/trokens/torch_home'

    args = parse_args()
    
    # 1. Setup Configuration
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    
    # Enable USE_CORRELATION so that Pointformer computes tracks internally 
    # instead of relying on precomputed point tracking data.
    cfg.POINT_INFO.USE_CORRELATION = True
    
    # In inference mode on a single clip, turn off training
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    
    cfg = assert_and_infer_cfg(cfg)
    
    # 2. Build Model
    print("Building model...")
    model = build_model(cfg)
    
    # 3. Load Checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
    cu.load_test_checkpoint(cfg, model)
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        
    # 4. Process Video
    print(f"Reading video: {args.video}")
    frames = data_utils.read_video(args.video, cfg.DATA.NUM_FRAMES)
    
    from torchvision import transforms
    print("Applying spatial transform and normalization...")
    
    # Convert from (T, H, W, C) numpy array to (T, C, H, W) tensor and scale to [0, 1]
    frames_tensor = torch.from_numpy(frames).float()
    frames_tensor = frames_tensor.permute(0, 3, 1, 2) / 255.0
    
    # Apply Resize and Normalize
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    ])
    frames_tensor = data_transform(frames_tensor)
    
    # Add batch dimension -> [B, T, C, H, W]
    frames_tensor = frames_tensor.unsqueeze(0)
    
    # 5. Prepare Metadata and Input Dictionary
    meta = {
        'sample_type': np.array(['query']),
        'batch_label': torch.tensor([0]),
    }
    
    input_dict = {
        'video': frames_tensor,
        'metadata': meta
    }
    
    if torch.cuda.is_available():
        input_dict['video'] = input_dict['video'].cuda()
        
    # 6. Run Inference
    print("Running inference...")
    with torch.no_grad():
        preds, _ = model(input_dict)
        
    preds_np = preds.cpu().numpy()
    
    # 7. Print Results
    print("\n--------------------------------------------------")
    print("Prediction Results")
    print("--------------------------------------------------")
    print(f"Raw Logits: {preds_np[0]}")
    
    predicted_class = np.argmax(preds_np[0])
    
    # Assuming logits are returned, we can compute probabilities
    probs = torch.sigmoid(preds)[0].cpu().numpy()
    confidence = probs[predicted_class]
    
    print(f"Predicted Class Index: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("Probabilities per class:")
    for i, prob in enumerate(probs):
        print(f"  Class {i}: {prob:.4f}")

if __name__ == "__main__":
    main()
