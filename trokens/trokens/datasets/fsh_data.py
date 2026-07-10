#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""FSH dataset."""
import os
import glob
import pandas as pd
import numpy as np

import trokens.utils.logging as logging

from .build import DATASET_REGISTRY

from .base_ds import BaseDataset


logger = logging.get_logger(__name__)


def _parse_behavior_labels(value):
    """Parse a CSV behavior cell into a list of label strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ["unknown"]
    text = str(value).strip()
    if text == "":
        return ["unknown"]
    return text.split()


def _behavior_key(labels):
    """Stable hashable key for grouping / stratifying multi-label rows."""
    return tuple(sorted(labels))

FILTER_ONE_BEHAVIORS = ["Peck","Quiver","Lead","Bite","Tilt","Chase/Charge","NoBehavior"]
FILTER_TWO_BEHAVIORS = ["Peck","Quiver","Lead","Bite","Tilt","Chase/Charge"]

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
        CUT_SMALLS = self.cfg.DATA_LOADER.CUT_SMALLS
        FILTER_ONE = self.cfg.DATA_LOADER.FILTER_ONE
        FILTER_TWO = self.cfg.DATA_LOADER.FILTER_TWO
        CSV_FILE = self.cfg.DATA_LOADER.DATA_CSV_PATH
        
        # Read CSV file
        if CSV_FILE:
            csv_path = CSV_FILE
        else:
            #csv_path = "/fs/vulcan-projects/fsh_track/processed_data/dataset6/dataset6.csv"
            csv_files = glob.glob(os.path.join(self.data_root, "*.csv"))

            if len(csv_files) != 1:
                raise FileNotFoundError(f"Expected exactly one CSV in {self.data_root}, found {len(csv_files)}")

            csv_path = csv_files[0]

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        
        self.dataset_df = pd.read_csv(csv_path)

        # Validate required columns (expect `video_path` column)
        required_columns = ['behavior', 'video_path']
        missing_columns = [col for col in required_columns if col not in self.dataset_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")

        # Parse behavior column: space-separated labels -> list of strings
        self.dataset_df['behavior'] = self.dataset_df['behavior'].apply(_parse_behavior_labels)

        # Optionally remove rows containing any behavior with too few occurrences
        if CUT_SMALLS:
            label_counts = self.dataset_df['behavior'].explode().value_counts()
            small_behaviors = set(label_counts[label_counts < 15].index.tolist())
            if len(small_behaviors) > 0:
                logger.info(f"Removing rows with behaviors that have <15 occurrences: {sorted(small_behaviors)}")
                keep_mask = ~self.dataset_df['behavior'].apply(
                    lambda labels: any(label in small_behaviors for label in labels)
                )
                self.dataset_df = self.dataset_df[keep_mask].reset_index(drop=True)
            else:
                logger.info("No behaviors to remove based on CUT_SMALLS threshold.")
        
        # Optionally filter to only the requested example/behavior names.
        def filter_and_reduce_behaviors(df, requested, filter_name):
            if len(requested) == 0:
                logger.warning(f"{filter_name} enabled but behavior list is empty; no filtering applied.")
                return df
            before = len(df)
            # Remove any behavior not in requested for each row, and drop rows with no remaining behaviors
            def keep_only_requested(labels):
                return [label for label in labels if label in requested]
            df = df.copy()
            df['behavior'] = df['behavior'].apply(keep_only_requested)
            # Only keep rows that still have at least one label
            keep_mask = df['behavior'].apply(lambda labels: len(labels) > 0)
            df = df[keep_mask].reset_index(drop=True)
            after = len(df)
            logger.info(f"{filter_name} enabled: kept {after}/{before} rows by behavior filter. Unwanted behaviors removed from multi-labeled rows.")
            return df

        if FILTER_ONE:
            requested = set([str(x).strip() for x in FILTER_ONE_BEHAVIORS if str(x).strip() != ""])
            self.dataset_df = filter_and_reduce_behaviors(self.dataset_df, requested, "FILTER_ONE")

        if FILTER_TWO:
            requested = set([str(x).strip() for x in FILTER_TWO_BEHAVIORS if str(x).strip() != ""])
            self.dataset_df = filter_and_reduce_behaviors(self.dataset_df, requested, "FILTER_TWO")

        # Map each behavior string to a numeric id; store label_id as arrays per row
        unique_behaviors = sorted(self.dataset_df['behavior'].explode().unique())
        behavior_to_label = {behavior: idx for idx, behavior in enumerate(unique_behaviors)}
        self.behavior_to_label = behavior_to_label
        self.dataset_df['label_id'] = self.dataset_df['behavior'].apply(
            lambda labels: np.array([behavior_to_label[label] for label in labels], dtype=np.int64)
        )

        # Create video_name (basename without extension)
        self.dataset_df['video_name'] = self.dataset_df['video_path'].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0])

        # Create feat_base_name on the full dataframe before splitting
        self.dataset_df['feat_base_name'] = self.dataset_df['video_name'].apply(
            lambda x: x + '.pkl')

        # Create vid_id from video_path (basename without extension)
        self.dataset_df['vid_id'] = self.dataset_df['video_path'].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0] if isinstance(x, str) and x != '' else str(x))

        # Handle splits
        # If CSV has a 'split' or 'new_split' column, use it; otherwise create automatic split
        if 'split' in self.dataset_df.columns:
            self.dataset_df['new_split'] = self.dataset_df['split']
            # Log behavior distribution in each split
            behavior_for_log = self.dataset_df['behavior'].apply(lambda labels: " ".join(sorted(labels)))
            split_counts = (
                self.dataset_df.assign(behavior=behavior_for_log)
                .groupby(['behavior', 'new_split'])
                .size()
                .unstack(fill_value=0)
            )
            logger.info(f"Split distribution by behavior:\n{split_counts}")
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
        self.base_feature_path = self.cfg.DATA.PATH_TO_TROKEN_PT_DATA
        self.split_df['feat_path'] = self.split_df['feat_base_name'].apply(
            lambda x: os.path.join(self.base_feature_path, x))

        # Filter out rows where feature files don't exist
        if self.cfg.POINT_INFO.ENABLE:
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
        
        # Create stratified split based on multi-label behavior combinations
        self.dataset_df['new_split'] = 'test'  # Initialize all as test
        behavior_groups = self.dataset_df['behavior'].apply(_behavior_key)

        for behavior_key in behavior_groups.unique():
            behavior_mask = behavior_groups == behavior_key
            behavior_indices = self.dataset_df[behavior_mask].index

            shuffled_indices = np.random.permutation(behavior_indices)
            n_train = int(len(shuffled_indices) * train_ratio)
            train_indices = shuffled_indices[:n_train]
            self.dataset_df.loc[train_indices, 'new_split'] = 'train'

        # Log split statistics
        train_count = (self.dataset_df['new_split'] == 'train').sum()
        test_count = (self.dataset_df['new_split'] == 'test').sum()
        logger.info(f"Automatic split created: {train_count} train samples ({train_count/len(self.dataset_df)*100:.1f}%), "
                   f"{test_count} test samples ({test_count/len(self.dataset_df)*100:.1f}%)")

        # Log behavior distribution in each split
        behavior_for_log = self.dataset_df['behavior'].apply(lambda labels: " ".join(sorted(labels)))
        split_counts = (
            self.dataset_df.assign(behavior=behavior_for_log)
            .groupby(['behavior', 'new_split'])
            .size()
            .unstack(fill_value=0)
        )
        logger.info(f"Split distribution by behavior:\n{split_counts}")
