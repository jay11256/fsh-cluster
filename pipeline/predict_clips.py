"""Clip prediction script"""

import os
import sys
import glob
import pickle

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
from trokens.datasets.point_sampler import point_sampler, get_point_query_mask

import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

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
    cfg.TEST.ENABLE = False
    cfg.NUM_GPUS = torch.cuda.device_count()
    cfg = assert_and_infer_cfg(cfg)

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True

    model = build_model(cfg)
    cfg.TEST.CHECKPOINT_FILE_PATH = model_path
    cu.load_test_checkpoint(cfg, model)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, cfg, device


def _preprocess_video(video_path, cfg, device):
    """Reads a video file and returns a preprocessed (1, T, C, H, W) tensor
    along with the original (max_x, max_y) spatial dims for point normalization."""
    frames = data_utils.read_video(video_path, cfg.DATA.NUM_FRAMES)

    frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0  # [T, C, H, W]

    max_y, max_x = frames_tensor.shape[-2], frames_tensor.shape[-1]

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
    ])
    frames_tensor = data_transform(frames_tensor)

    return frames_tensor.unsqueeze(0).to(device), max_x, max_y


def _load_pkl_metadata(pkl_path, cfg, max_x, max_y, index_select, index_seed=0):
    """
    Loads a pkl point file and returns a metadata dict matching what
    base_ds.__getitem__ would produce.

    Args:
        pkl_path (str): path to the .pkl file
        cfg: config node
        max_x (int): original video width (before resize), for normalization
        max_y (int): original video height (before resize), for normalization
        index_select (np.ndarray): frame indices selected for this clip
        index_seed (int): passed to point_sampler for deterministic test sampling

    Returns:
        dict with pred_tracks, pred_visibility, pred_query_mask
    """
    pt_dict = pickle.load(open(pkl_path, 'rb'))
    pred_tracks = pt_dict['pred_tracks'].squeeze(0)        # [T, N, 2]
    pred_visibility = pt_dict['pred_visibility'].squeeze(0) # [T, N]

    if 'per_point_queries' in pt_dict:
        per_point_queries = pt_dict['per_point_queries']
    else:
        per_point_queries = torch.zeros(pred_tracks.shape[1], dtype=torch.int64)

    num_points = pred_tracks.shape[1]

    # --- Point sampling (mirrors base_ds.__getitem__) ---
    if ((num_points > cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE) or
        cfg.POINT_INFO.PT_FIX_SAMPLING_TRAIN or
        cfg.POINT_INFO.PT_FIX_SAMPLING_TEST):

        assert cfg.POINT_INFO.SAMPLING_TYPE != 'None'
        filtered_points, _ = point_sampler(
            cfg, pt_dict,
            pred_tracks.clone(),
            pred_visibility.clone(),
            points_to_sample=cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE,
            sampling_type=cfg.POINT_INFO.SAMPLING_TYPE,
            index_select=index_select,
            split='test',
            index_seed=index_seed,
        )
        pred_tracks_to_take = pred_tracks[:, filtered_points]
        pred_visibility_to_take = pred_visibility[:, filtered_points]
        per_point_queries_to_take = per_point_queries[filtered_points]
    else:
        pred_tracks_to_take = pred_tracks
        pred_visibility_to_take = pred_visibility
        per_point_queries_to_take = per_point_queries
        filtered_points = np.ones(num_points, dtype=bool)

    # Pad up to NUM_POINTS_TO_SAMPLE if still short
    if pred_tracks_to_take.shape[1] < cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE:
        all_indices = np.argwhere(filtered_points)[:, 0]
        points_missing = cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE - pred_tracks_to_take.shape[1]
        try:
            random_indices = np.random.choice(all_indices, points_missing, replace=False)
        except ValueError:
            random_indices = np.random.choice(all_indices, points_missing, replace=True)
        pred_tracks_to_take = torch.cat([pred_tracks_to_take, pred_tracks[:, random_indices]], dim=1)
        pred_visibility_to_take = torch.cat([pred_visibility_to_take, pred_visibility[:, random_indices]], dim=1)
        per_point_queries_to_take = np.concatenate([per_point_queries_to_take, per_point_queries[random_indices]])

    pred_tracks = pred_tracks_to_take
    pred_visibility = pred_visibility_to_take
    per_point_queries = per_point_queries_to_take

    # --- Point query mask ---
    pt_query_mask = torch.ones_like(pred_visibility, dtype=torch.bool)
    if cfg.POINT_INFO.USE_PT_QUERY_MASK:
        pt_query_mask = get_point_query_mask(per_point_queries, pt_query_mask)

    # --- Normalize coordinates to [-1, 1] ---
    div_factor = torch.tensor([max_x, max_y]).view(1, 1, 2)
    pred_tracks = pred_tracks / div_factor
    pred_tracks = (pred_tracks - 0.5) / 0.5

    # --- Select frames ---
    pt_to_take = pred_tracks[index_select].float()
    pred_visibility = pred_visibility[index_select]
    pt_query_mask = pt_query_mask[index_select]

    return {
        'pred_tracks': pt_to_take,
        'pred_visibility': pred_visibility,
        'pred_query_mask': pt_query_mask,
    }


def predict_clip(video_path, model_path, pkl_path=None,
                 cfg_path="/fs/vulcan-projects/fsh_track/jason/fsh-cluster/trokens/configs/trokens/fshdata.yaml"):
    """
    Generates a prediction vector given a clip, optionally with point tracks.

    Args:
        video_path (str): absolute path to the video
        model_path (str): absolute path to the model being run
        pkl_path (str | None): absolute path to the corresponding .pkl point file,
                               or None to run without point data
        cfg_path (str): absolute path to the config file

    Returns:
        prediction_vector (torch.Tensor): (6,) tensor of prediction values
    """
    model, cfg, device = _load_model(model_path, cfg_path)
    frames_tensor, max_x, max_y = _preprocess_video(video_path, cfg, device)

    index_select = np.linspace(0, cfg.DATA.NUM_FRAMES - 1, cfg.DATA.NUM_FRAMES).astype(int)

    metadata = {
        'sample_type': np.array(['query']),
        'batch_label': torch.tensor([0]),
    }

    if pkl_path is not None:
        point_meta = _load_pkl_metadata(pkl_path, cfg, max_x, max_y, index_select)
        metadata.update(point_meta)

    input_dict = {'video': frames_tensor, 'metadata': metadata}

    with torch.no_grad():
        preds, _ = model(input_dict)

    return preds[0].cpu()


def predict_clips(clips_dir, model_path, pkl_dir=None,
                  cfg_path="/fs/vulcan-projects/fsh_track/jason/fsh-cluster/trokens/configs/trokens/fshdata.yaml"):
    """
    Runs predictions on every video clip in a directory.

    Args:
        clips_dir (str): absolute path to directory containing video clips
        model_path (str): absolute path to the model being run
        pkl_dir (str | None): directory containing .pkl files named to match
                              video stems (e.g. clip01.mp4 -> clip01.pkl),
                              or None to run without point data
        cfg_path (str): absolute path to the config file

    Returns:
        predictions (torch.Tensor): (N, 6) tensor
        clip_paths (list[str]): ordered clip paths matching each row
    """
    VIDEO_EXTENSIONS = ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm')

    clip_paths = sorted([
        path
        for ext in VIDEO_EXTENSIONS
        for path in glob.glob(os.path.join(clips_dir, ext))
    ])

    if not clip_paths:
        raise ValueError(f"No video files found in directory: {clips_dir}")

    model, cfg, device = _load_model(model_path, cfg_path)

    prediction_vectors = []
    for clip_path in clip_paths:
        frames_tensor, max_x, max_y = _preprocess_video(clip_path, cfg, device)

        index_select = np.linspace(0, cfg.DATA.NUM_FRAMES - 1, cfg.DATA.NUM_FRAMES).astype(int)

        metadata = {
            'sample_type': np.array(['query']),
            'batch_label': torch.tensor([0]),
        }

        if pkl_dir is not None:
            stem = os.path.splitext(os.path.basename(clip_path))[0]
            pkl_path = os.path.join(pkl_dir, stem + '.pkl')
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(
                    f"Expected pkl file not found: {pkl_path}\n"
                    f"Ensure every clip in {clips_dir} has a matching .pkl in {pkl_dir}"
                )
            point_meta = _load_pkl_metadata(pkl_path, cfg, max_x, max_y, index_select)
            metadata.update(point_meta)

        input_dict = {'video': frames_tensor, 'metadata': metadata}

        with torch.no_grad():
            preds, _ = model(input_dict)

        prediction_vectors.append(preds[0].cpu())

    predictions = torch.stack(prediction_vectors, dim=0)
    return predictions, clip_paths


def main():
    clips = "/fs/vulcan-projects/fsh_track/jason/fsh-cluster/pipeline/clips"
    pkls  = "/fs/vulcan-projects/fsh_track/jason/fsh-cluster/pipeline/pkls"  # set to None to skip points
    model = "/fs/vulcan-projects/fsh_track/models/ds6/5_way-3_shot-sam3-both/checkpoints/checkpoint_best.pyth"
    preds, order = predict_clips(clips, model, pkl_dir=pkls)
    preds = torch.softmax(preds, dim=1)

    one_hot = torch.zeros_like(preds)
    preds = one_hot.scatter_(1, preds.argmax(dim=1, keepdim=True), 1)

    print(preds)
    # print(order)

    from visualize_matrix import visualize_matrix
    visualize_matrix(
        ground_truth_path = "/fs/vulcan-projects/fsh_track/raw_data/box/AR_natural_spawns_JB/080225_spawn_B1-5_ARdoublehet/080225_spawn_B1-5_ARdoublehet.tsv",
        pred_matrix = preds.T,
        threshold = 0.5,
        window_len = 14,
        overlap_len = 4,
        save_path = "test_vis.png"
    )

    # /fs/vulcan-projects/fsh_track/raw_data/box/AR_natural_spawns_JB/080225_spawn_B1-5_ARdoublehet/080225_spawn_B1-5_ARdoublehet.tsv

main()