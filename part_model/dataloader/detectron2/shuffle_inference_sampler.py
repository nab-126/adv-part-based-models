"""InferenceSampler but shuffles the indices. Used for visualization."""

from __future__ import annotations

import random

from detectron2.data.samplers import InferenceSampler
from detectron2.utils import comm
from torch.utils.data.sampler import Sampler


class ShuffleInferenceSampler(InferenceSampler):
    """InferenceSampler but shuffles the indices. Used for visualization."""

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        # Original: return range(begin, end)
        indices = list(range(total_size))
        random.shuffle(indices)
        return indices[begin:end]
