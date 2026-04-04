#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""FSH dataset."""
import os
import pandas as pd
import numpy as np

import trokens.utils.logging as logging

from .build import DATASET_REGISTRY

from .base_ds import BaseDataset


logger = logging.get_logger(__name__)

# When True, remove behaviors that have fewer than 15 occurrences in total
CUT_SMALLS = True


@DATASET_REGISTRY.register()
class Fshdata(BaseDataset):
    """FSH dataset."""
    def __init__(self, cfg, mode):
        super(Fshdata, self).__init__(cfg, mode)
    
    def _construct_loader(self):
        """
        Load FSH data (frame paths, labels, etc. )
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
        """
        self.data_root = self.cfg.DATA.PATH_TO_DATA_DIR
        
        # Read CSV file
        csv_path = "/fs/vulcan-projects/fsh_track/processed_data/dataset5/dataset5.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        
        self.dataset_df = pd.read_csv(csv_path)

        # Validate required columns (expect `video_path` column)
        required_columns = ['behavior', 'start_time', 'end_time', 'video_path']
        missing_columns = [col for col in required_columns if col not in self.dataset_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")

        # Normalize `video_path`: if relative, join with data_root; ensure string
        def _normalize_video_path(p):
            if not isinstance(p, str):
                return str(p)
            p = p.strip()
            if p == '':
                return p
            return p if os.path.isabs(p) else os.path.join(self.data_root, p)

        self.dataset_df['video_path'] = self.dataset_df['video_path'].apply(_normalize_video_path)

        # Create vid_id from video_path (basename without extension)
        self.dataset_df['vid_id'] = self.dataset_df['video_path'].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0] if isinstance(x, str) and x != '' else str(x))
        
        # Map behavior strings to numeric label_id
        # Normalize behavior column: convert to str, strip whitespace, and replace missing/empty with 'unknown'
        self.dataset_df['behavior'] = self.dataset_df['behavior'].fillna('unknown').apply(lambda x: str(x).strip())
        self.dataset_df.loc[self.dataset_df['behavior'] == '', 'behavior'] = 'unknown'

        # Optionally remove behaviors with too few occurrences
        if CUT_SMALLS:
            counts = self.dataset_df['behavior'].value_counts()
            small_behaviors = counts[counts < 15].index.tolist()
            if len(small_behaviors) > 0:
                logger.info(f"Removing behaviors with <15 occurrences: {small_behaviors}")
                self.dataset_df = self.dataset_df[~self.dataset_df['behavior'].isin(small_behaviors)].reset_index(drop=True)
            else:
                logger.info("No behaviors to remove based on CUT_SMALLS threshold.")

        unique_behaviors = sorted(self.dataset_df['behavior'].unique())
        behavior_to_label = {behavior: idx for idx, behavior in enumerate(unique_behaviors)}
        self.dataset_df['label_id'] = self.dataset_df['behavior'].map(behavior_to_label).astype(int)
        
        # Create video_name (basename without extension)
        self.dataset_df['video_name'] = self.dataset_df['video_path'].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0])
        
        # Create feat_base_name
        self.dataset_df['feat_base_name'] = self.dataset_df['video_name'].apply(
            lambda x: x + '.pkl')
        
        # Handle splits
        # If CSV has a 'split' or 'new_split' column, use it; otherwise create automatic split
        if 'split' in self.dataset_df.columns:
            self.dataset_df['new_split'] = self.dataset_df['split']
        elif 'new_split' in self.dataset_df.columns:
            pass  # Already exists
        else:
            # If no split column, create automatic train/test split
            logger.info("No split column found in CSV. Creating automatic train/test split.")
            self._create_automatic_split()
        
        # Filter by mode
        self.split_df = self.dataset_df[
            self.dataset_df['new_split'] == self.mode].reset_index(drop=True)
        
        # Create feat_path
        self.base_feature_path =  self.cfg.DATA.PATH_TO_TROKEN_PT_DATA
        self.split_df['feat_path'] = self.split_df['feat_base_name'].apply(
            lambda x: os.path.join(self.base_feature_path, x))
        
        # Filter out rows where feature files don't exist
        original_len = len(self.split_df)
        self.split_df = self.split_df[
            self.split_df['feat_path'].apply(os.path.exists)].reset_index(drop=True)
        new_len = len(self.split_df)
        
        if new_len == 0:
            raise ValueError(f"No feature files found for {self.mode} mode. Check feature path: {self.base_feature_path}")
        
        if new_len < 0.95 * original_len:
            logger.warning(f"Some features are missing. Expected {original_len}, found {new_len}")
        
        self._make_final_lists()
    
    def _create_automatic_split(self):
        """
        Create automatic train/test split with stratification by behavior.
        Uses configurable split ratio from cfg.DATA.TRAIN_SPLIT_RATIO (default 0.8).
        """
        # Get split ratio from config, default to 0.8 (80% train, 20% test)
        train_ratio = getattr(self.cfg.DATA, 'TRAIN_SPLIT_RATIO', 0.8)
        split_seed = getattr(self.cfg.DATA, 'SPLIT_SEED', 42)
        
        # Set random seed for reproducibility
        np.random.seed(split_seed)
        
        # Create stratified split based on behavior labels
        # Group by behavior to ensure each behavior is represented in both splits
        self.dataset_df['new_split'] = 'test'  # Initialize all as test
        
        # For each behavior, randomly assign train/test
        for behavior in self.dataset_df['behavior'].unique():
            behavior_mask = self.dataset_df['behavior'] == behavior
            behavior_indices = self.dataset_df[behavior_mask].index
            
            # Shuffle indices
            shuffled_indices = np.random.permutation(behavior_indices)
            
            # Calculate split point
            n_train = int(len(shuffled_indices) * train_ratio)
            
            # Assign train/test
            train_indices = shuffled_indices[:n_train]
            self.dataset_df.loc[train_indices, 'new_split'] = 'train'
        
        # Log split statistics
        train_count = (self.dataset_df['new_split'] == 'train').sum()
        test_count = (self.dataset_df['new_split'] == 'test').sum()
        logger.info(f"Automatic split created: {train_count} train samples ({train_count/len(self.dataset_df)*100:.1f}%), "
                   f"{test_count} test samples ({test_count/len(self.dataset_df)*100:.1f}%)")
        
        # Log behavior distribution in each split
        split_counts = self.dataset_df.groupby(['behavior', 'new_split']).size().unstack(fill_value=0)
        logger.info(f"Split distribution by behavior:\n{split_counts}")
