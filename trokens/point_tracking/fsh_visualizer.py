#!/usr/bin/env python3

"""
Script to visualize tracking results using the exact --make_vis procedure
from new_point_tracking.py with your own pkl and mp4 files.
"""
import os
import pickle
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from einops import rearrange

# Import the visualization functions from new_point_tracking
from omni_vis import vis_trail
from utils import save_video
from new_video_loader import load_video_pyvideo_reader


def visualize_with_make_vis_procedure(pkl_file_path, video_path, output_gif_path,
                                      custom_fps=None, device='cpu'):
    """
    Visualize tracking results using the exact --make_vis procedure from new_point_tracking.py
    
    Args:
        pkl_path (str): Path to pkl file with tracking results
        video_path (str): Path to mp4 video
        output_gif_path (str): Path where output gif will be saved
        custom_fps (int): Custom FPS (if video duration > 90s)
        device (str): Device to use ('cpu' or 'cuda')
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Load pkl file
    print(f"Loading pkl file: {pkl_file_path}")
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pkl file: {e}")
        return False
    
    # Extract tracking data
    try:
        pred_tracks = data['pred_tracks']
        pred_visibility = data['pred_visibility']
        obj_ids = data['obj_ids']
        point_queries = data.get('point_queries', None)  # Optional, may not be used in visualization
        
        # Convert to numpy if needed
        if isinstance(pred_tracks, torch.Tensor):
            pred_tracks = pred_tracks.cpu().numpy()
        if isinstance(pred_visibility, torch.Tensor):
            pred_visibility = pred_visibility.cpu().numpy()
        if isinstance(obj_ids, torch.Tensor):
            obj_ids = obj_ids.cpu().numpy()
        if point_queries is not None and isinstance(point_queries, torch.Tensor):
            point_queries = point_queries.cpu().numpy()
        
        print(f"Loaded tracking data:")
        print(f"  pred_tracks shape: {pred_tracks.shape}")
        print(f"  pred_visibility shape: {pred_visibility.shape}")
        print(f"  obj_ids shape: {obj_ids.shape}")
        print(f"  point_queries shape: {point_queries.shape if point_queries is not None else 'N/A'}")
    except KeyError as e:
        print(f"Error: Missing key in pkl file: {e}")
        return False
    
    # Load video using the SAME procedure as new_point_tracking.py
    print(f"Loading video: {video_path}")
    try:
        video_loaded, video, _ = load_video_pyvideo_reader(
            video_path,
            return_tensor=True,
            use_float=True,  # Important: use float for visualization
            device=device,
            sample_all_frames=True,  # Get all frames, not just clustering frames
            fps=custom_fps
        )
        
        if not video_loaded:
            print(f"Failed to load video: {video_path}")
            return False
    except Exception as e:
        print(f"Error loading video: {e}")
        return False
    
    # Convert video to numpy and rearrange
    # From (B, T, C, H, W) to (T, H, W, C)
    try:
        video = video.cpu().squeeze(0).numpy()
        video = rearrange(video, 't c h w -> t h w c')
        print(f"Video shape after rearrange: {video.shape}")
    except Exception as e:
        print(f"Error processing video: {e}")
        return False
    
    # Generate visualization frames using vis_trail (EXACT procedure from new_point_tracking.py)
    try:
        print("Generating visualization frames using vis_trail...")
        frames = vis_trail(video, pred_tracks, pred_visibility,
                          cluster_ids=obj_ids)
        print(f"Generated {len(frames)} visualization frames")
    except Exception as e:
        print(f"Error generating visualization frames: {e}")
        return False
    
    # Save video using save_video (EXACT procedure from new_point_tracking.py)
    try:
        os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
        save_video(frames, output_gif_path)
        print(f"Saved visualization: {output_gif_path}")
        return True
    except Exception as e:
        print(f"Error saving video: {e}")
        return False


def process_batch_from_csv(csv_path, pkl_base_path, output_base_path, dump_name,
                          custom_fps=None, device='cpu'):
    """
    Process multiple videos from CSV using the exact --make_vis procedure.
    
    Args:
        csv_path (str): Path to CSV file with columns: video_path, dataset
        pkl_base_path (str): Base path where pkl files are located
        output_base_path (str): Base path where output gifs will be saved
        dump_name (str): Name of the dump folder (e.g., 'cotracker3_bip_fr_32')
        custom_fps (int): Custom FPS for videos > 90s
        device (str): Device to use ('cpu' or 'cuda')
    """
    
    df = pd.read_csv(csv_path)
    
    if 'video_path' not in df.columns or 'dataset' not in df.columns:
        raise ValueError("CSV must have 'video_path' and 'dataset' columns")
    
    success_count = 0
    fail_count = 0
    
    for idx, row in df.iterrows():
        video_path = row['video_path']
        dataset = row['dataset']
        video_name = os.path.basename(video_path)
        vid_id = video_name.split('.')[0]
        
        # Construct pkl path (same structure as new_point_tracking.py)
        pkl_path = os.path.join(pkl_base_path, dump_name, dataset, 'feat_dump', 
                               f'{vid_id}.pkl')
        
        # Construct output gif path (same structure as new_point_tracking.py)
        output_gif_path = os.path.join(output_base_path, dump_name, dataset, 'gif_dump',
                                      f'{vid_id}.gif')
        
        # Check if pkl file exists
        if not os.path.exists(pkl_path):
            print(f"\n[{idx+1}/{len(df)}] PKL file not found: {pkl_path}")
            fail_count += 1
            continue
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"\n[{idx+1}/{len(df)}] Video file not found: {video_path}")
            fail_count += 1
            continue
        
        print(f"\n[{idx+1}/{len(df)}] Processing {vid_id}...")
        
        # Determine custom_fps if duration available in CSV
        fps_to_use = custom_fps
        if 'duration' in row and row['duration'] > 90:
            fps_to_use = 1
        
        # Create visualization using exact procedure
        success = visualize_with_make_vis_procedure(pkl_path, video_path, output_gif_path,
                                                   custom_fps=fps_to_use, device=device)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete: {success_count} succeeded, {fail_count} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize tracking results using the exact --make_vis procedure from new_point_tracking.py"
    )
    
    # Single file arguments
    parser.add_argument("--pkl_path", type=str,
                       help="Path to pkl file")
    parser.add_argument("--video_path", type=str,
                       help="Path to mp4 video")
    parser.add_argument("--output_gif_path", type=str,
                       help="Output path for visualization gif")
    
    # Batch processing arguments
    parser.add_argument("--csv_path", type=str,
                       help="Path to CSV file with video information")
    parser.add_argument("--pkl_base_path", type=str,
                       help="Base path where pkl files are located")
    parser.add_argument("--output_base_path", type=str,
                       default='/fs/cfar-projects/actionloc/camera_ready/tats_v2/dumps',
                       help="Base path for output gifs (same structure as new_point_tracking.py)")
    parser.add_argument("--dump_name", type=str, default='cotracker3_bip_fr_32',
                       help="Name of the dump folder (e.g., 'cotracker3_bip_fr_32')")
    parser.add_argument("--fps", type=int, default=None,
                       help="FPS for videos > 90s")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Single file processing
    if args.pkl_path and args.video_path and args.output_gif_path:
        os.makedirs(os.path.dirname(args.output_gif_path), exist_ok=True)
        visualize_with_make_vis_procedure(args.pkl_path, args.video_path, args.output_gif_path,
                                         custom_fps=args.fps, device=args.device)
    
    # Batch processing from CSV
    elif args.csv_path and args.pkl_base_path:
        process_batch_from_csv(args.csv_path, args.pkl_base_path, args.output_base_path,
                              args.dump_name, custom_fps=args.fps, device=args.device)
    
    else:
        parser.print_help()