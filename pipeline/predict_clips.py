"""Clip prediction script"""

import os
import sys
import glob

import torch
import numpy as np
from torchvision import transforms

# Ensure the trokens project is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
trokens_dir = os.path.join(script_dir, '..', 'trokens')
if trokens_dir not in sys.path:
    sys.path.insert(0, trokens_dir)

from trokens.config.defaults import get_cfg, assert_and_infer_cfg
from trokens.models import build_model
from trokens.utils import checkpoint as cu
from trokens.datasets import utils as data_utils


def _load_model(model_path, cfg_path=None):
    """Builds, loads, and returns an eval-mode model along with its config."""
    if 'TORCH_HOME' not in os.environ:
        os.environ['TORCH_HOME'] = (
            '/fs/vulcan-projects/fsh_track/programs/trokens_workspace/trokens/torch_home'
        )

    if cfg_path is None:
        cfg_path = os.path.join(script_dir, 'configs', 'trokens', 'fshdata.yaml')

    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.POINT_INFO.USE_CORRELATION = True
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    cfg.NUM_GPUS = torch.cuda.device_count()
    cfg = assert_and_infer_cfg(cfg)

    model = build_model(cfg)
    cfg.TEST.CHECKPOINT_FILE_PATH = model_path
    cu.load_test_checkpoint(cfg, model)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, cfg, device


def _preprocess_video(video_path, cfg, device):
    """Reads a video file and returns a preprocessed (1, T, C, H, W) tensor."""
    frames = data_utils.read_video(video_path, cfg.DATA.NUM_FRAMES)

    frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
    ])
    frames_tensor = data_transform(frames_tensor)

    return frames_tensor.unsqueeze(0).to(device)


def predict_clip(video_path, model_path, cfg_path="/fs/vulcan-projects/fsh_track/jason/fsh-cluster/trokens/configs/trokens/fshdata.yaml"):
    """
    Generates a prediction vector given a clip.

    Args:
        video_path (str): absolute path to the video
        model_path (str): absolute path to the model being run
        cfg_path (str): absolute path to the config file

    Returns:
        prediction_vector (torch.Tensor): (6,) tensor of prediction values
    """
    model, cfg, device = _load_model(model_path, cfg_path)
    frames_tensor = _preprocess_video(video_path, cfg, device)

    input_dict = {
        'video': frames_tensor,
        'metadata': {
            'sample_type': np.array(['query']),
            'batch_label': torch.tensor([0]),
        },
    }

    with torch.no_grad():
        preds, _ = model(input_dict)

    return preds[0].cpu()


def predict_clips(clips_dir, model_path, cfg_path="/fs/vulcan-projects/fsh_track/jason/fsh-cluster/trokens/configs/trokens/fshdata.yaml"):
    """
    Runs predictions on every video clip in a directory and returns a matrix
    of all prediction vectors stacked row-wise.

    Args:
        clips_dir (str): absolute path to directory containing video clips
        model_path (str): absolute path to the model being run
        cfg_path (str): absolute path to the config file

    Returns:
        predictions (torch.Tensor): (N, 6) tensor where N is the number of clips
        clip_paths (list[str]): ordered list of clip paths matching each row
    """
    VIDEO_EXTENSIONS = ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm')

    clip_paths = sorted([
        path
        for ext in VIDEO_EXTENSIONS
        for path in glob.glob(os.path.join(clips_dir, ext))
    ])

    if not clip_paths:
        raise ValueError(f"No video files found in directory: {clips_dir}")

    # Load model once and reuse across all clips
    model, cfg, device = _load_model(model_path, cfg_path)

    prediction_vectors = []
    for clip_path in clip_paths:
        frames_tensor = _preprocess_video(clip_path, cfg, device)

        input_dict = {
            'video': frames_tensor,
            'metadata': {
                'sample_type': np.array(['query']),
                'batch_label': torch.tensor([0]),
            },
        }

        with torch.no_grad():
            preds, _ = model(input_dict)

        prediction_vectors.append(preds[0].cpu())

    predictions = torch.stack(prediction_vectors, dim=0)  # (N, 6)
    return predictions, clip_paths

def main():
    clips = "/fs/vulcan-projects/fsh_track/jason/fsh-cluster/pipeline/clips"
    model = "/fs/vulcan-projects/fsh_track/models/ds6/5_way-3_shot-none-both/checkpoints/checkpoint_best.pyth"
    preds, order = predict_clips(clips, model)
    print(preds)
    print(order)
main()