import random
import torch
import numpy as np
from torch.utils.data import Sampler


def _as_label_set(label):
    """Normalize a dataset label entry to a set of int class ids."""
    if label is None:
        return set()
    if isinstance(label, (set, frozenset)):
        return {int(x) for x in label}
    if isinstance(label, (list, tuple, np.ndarray)):
        arr = np.asarray(label).reshape(-1)
        return {int(x) for x in arr.tolist()}
    return {int(label)}


class FewShotEpisodeSampler(Sampler):
    """Original single-label N-way K-shot episode sampler."""

    def __init__(self, dataset, cfg, mode, less_iters=False):
        self.cfg = cfg
        random.seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)

        self.mode = mode
        labels = dataset._labels
        self.class_ids = list(np.unique(labels))
        self.num_way = cfg.FEW_SHOT.N_WAY
        self.num_support = cfg.FEW_SHOT.K_SHOT
        self.num_queries = (cfg.FEW_SHOT.TRAIN_QUERY_PER_CLASS if mode == 'train'
                                            else cfg.FEW_SHOT.TEST_QUERY_PER_CLASS)
        self.samples_per_class = self.num_support + self.num_queries
        self.batch_size = (self.num_way * self.samples_per_class)

        # Create a list of indices for each class
        self.class_indices = {class_label: [idx for idx, (label) in enumerate(labels) if label == class_label]
                              for class_label in self.class_ids}
        self.less_iters = less_iters

    def __iter__(self):
        while True:
            selected_classes = random.sample(self.class_ids, self.num_way)

            batch_indices = []
            sample_types = []
            batch_label = []

            sample_type = (['support'] * self.num_support +
                                            ['query'] * self.num_queries)
            for idx, class_label in enumerate(selected_classes):
                # Sample 'samples_per_class' indices from each selected class
                class_indices = random.sample(self.class_indices[class_label],
                                                        self.samples_per_class)
                batch_indices.extend(class_indices)
                sample_types.extend(sample_type)
                batch_label.extend([idx] * self.samples_per_class)
            batch_indices = np.array(batch_indices)
            sample_types = np.array(sample_types)
            batch_label = np.array(batch_label)
            indices = list(range(len(batch_indices)))

            # Shuffle the batch indices to mix the classes
            random.shuffle(indices)
            batch_indices = batch_indices[indices]
            sample_types = sample_types[indices]
            batch_label = batch_label[indices]
            # episode_classes kept for API parity with the multilabel sampler
            episode_classes = np.asarray(selected_classes, dtype=np.int64)
            index_and_sample_info = list(
                zip(batch_indices, batch_label, sample_types,
                    [episode_classes] * len(batch_indices))
            )

            # Yield batches of size 'batch_size'
            for i in range(0, len(batch_indices), self.batch_size):
                yield index_and_sample_info[i:i+self.batch_size]

    def __len__(self):
        div_factor = self.cfg.NUM_GPUS if not self.cfg.FEW_SHOT.TRAIN_OG_EPISODES else 1
        if self.mode == 'train':
            return self.cfg.FEW_SHOT.TRAIN_EPISODES // div_factor
        else:
            if self.less_iters:
                return self.cfg.FEW_SHOT.TEST_EPISODES // div_factor // 5
            return self.cfg.FEW_SHOT.TEST_EPISODES // self.cfg.NUM_GPUS


class MultilabelFewShotEpisodeSampler(Sampler):
    """
    Multilabel N-way K-shot episode sampler.

    Episode construction rules:
      1. Sample N global class ids.
      2. Only include examples whose positive labels are a subset of those N
         classes (no out-of-episode labels).
      3. For each episode class, sample K support + Q query examples that are
         positive for that class from the closed pool, without replacement
         across the episode.
      4. Each yielded item is
         (dataset_index, episode_slot, sample_type, episode_class_ids)
         so supports can later contribute to every episode class they are
         positive for.
    """

    def __init__(self, dataset, cfg, mode, less_iters=False, max_retries=200):
        self.cfg = cfg
        random.seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)

        self.mode = mode
        self.less_iters = less_iters
        self.max_retries = max_retries

        self.label_sets = [_as_label_set(label) for label in dataset._labels]
        all_classes = set()
        for label_set in self.label_sets:
            all_classes.update(label_set)
        self.class_ids = sorted(all_classes)

        self.num_way = cfg.FEW_SHOT.N_WAY
        self.num_support = cfg.FEW_SHOT.K_SHOT
        self.num_queries = (
            cfg.FEW_SHOT.TRAIN_QUERY_PER_CLASS if mode == 'train'
            else cfg.FEW_SHOT.TEST_QUERY_PER_CLASS
        )
        self.samples_per_class = self.num_support + self.num_queries
        self.batch_size = self.num_way * self.samples_per_class

        self.class_indices = {
            class_id: [
                idx for idx, label_set in enumerate(self.label_sets)
                if class_id in label_set
            ]
            for class_id in self.class_ids
        }

        if len(self.class_ids) < self.num_way:
            raise ValueError(
                f"Need at least N_WAY={self.num_way} classes, found {len(self.class_ids)}"
            )

    def _closed_pool(self, class_id, selected_set):
        """Indices positive for class_id with no labels outside selected_set."""
        return [
            idx for idx in self.class_indices[class_id]
            if self.label_sets[idx] and self.label_sets[idx].issubset(selected_set)
        ]

    def _sample_episode(self):
        selected_classes = random.sample(self.class_ids, self.num_way)
        selected_set = set(selected_classes)
        episode_classes = np.asarray(selected_classes, dtype=np.int64)

        pools = {}
        for class_id in selected_classes:
            pool = self._closed_pool(class_id, selected_set)
            if len(pool) < self.samples_per_class:
                return None
            pools[class_id] = pool

        used = set()
        batch_indices = []
        sample_types = []
        batch_labels = []

        for slot, class_id in enumerate(selected_classes):
            available = [idx for idx in pools[class_id] if idx not in used]
            if len(available) < self.samples_per_class:
                return None
            chosen = random.sample(available, self.samples_per_class)
            used.update(chosen)
            for j, idx in enumerate(chosen):
                batch_indices.append(idx)
                sample_types.append(
                    'support' if j < self.num_support else 'query'
                )
                batch_labels.append(slot)

        order = list(range(len(batch_indices)))
        random.shuffle(order)
        batch_indices = [batch_indices[i] for i in order]
        sample_types = [sample_types[i] for i in order]
        batch_labels = [batch_labels[i] for i in order]

        return list(
            zip(
                batch_indices,
                batch_labels,
                sample_types,
                [episode_classes] * len(batch_indices),
            )
        )

    def __iter__(self):
        while True:
            episode = None
            for _ in range(self.max_retries):
                episode = self._sample_episode()
                if episode is not None:
                    break
            if episode is None:
                raise RuntimeError(
                    "Failed to sample a closed multilabel few-shot episode. "
                    "Try reducing N_WAY / K_SHOT / queries, or check label coverage."
                )
            for i in range(0, len(episode), self.batch_size):
                yield episode[i:i + self.batch_size]

    def __len__(self):
        div_factor = self.cfg.NUM_GPUS if not self.cfg.FEW_SHOT.TRAIN_OG_EPISODES else 1
        if self.mode == 'train':
            return self.cfg.FEW_SHOT.TRAIN_EPISODES // div_factor
        if self.less_iters:
            return self.cfg.FEW_SHOT.TEST_EPISODES // div_factor // 5
        return self.cfg.FEW_SHOT.TEST_EPISODES // max(self.cfg.NUM_GPUS, 1)
