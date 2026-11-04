from torch.utils.data.sampler import Sampler
import random
from collections import defaultdict


class BatchSampler(Sampler):
    r"""Yield a mini-batch of indices.

    Args:
        data: Dataset for building sampling logic.
        batch_size: Size of mini-batch.
    """

    def __init__(self, dataset, shuffle=True):
        # build data for sampling here
        self.shuffle = shuffle
        self.batch_to_indices = self.batch_years(loc_year_pairs=dataset.training_pairs)
        print(f"Built BatchSampler with {len(self.batch_to_indices)} batches.")
        self.batch_ids = list(self.batch_to_indices.keys())

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batch_ids)
        for batch_id in self.batch_ids:
            yield self.batch_to_indices[batch_id]

    def __len__(self):
        return len(self.batch_ids)

    def batch_years(self, loc_year_pairs):
        batch_to_indices = defaultdict(list)
        for idx, (location, year) in enumerate(loc_year_pairs):
            batch_to_indices[location].append(idx)
        return batch_to_indices
