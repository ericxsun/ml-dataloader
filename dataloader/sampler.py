#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import math
from typing import Generic
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sized
from typing import TypeVar

import numpy as np

T_co = TypeVar('T_co', covariant=True)


class Sampler(Generic[T_co]):
    """Base class for all Samplers"""

    def __init__(self, data_source: Optional[Sized]):
        self.data_source = data_source
        if data_source is not None:
            self.n_data_source = len(data_source)

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError


class SequentialSampler(Sampler[int]):
    """Samples elements sequentially, always in the same order

    Args:
        data_source: dataset to sample from

    Examples:
        >>> list(SequentialSampler(range(10)))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    data_source: Sized

    def __init__(self, data_source):
        super().__init__(data_source)

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    """Sample elements randomly

    If without replacement, then sample from a shuffled dataset
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if True, default=False
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument is supposed to be
                           specified only when `replacement` is ``True``

    Examples:
        >>> rs = RandomSampler(range(10), False)
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement=False, num_samples=None) -> None:
        super().__init__(data_source)

        self.replacement = replacement
        self._num_samples = num_samples
        self._default_rng = np.random.default_rng()

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)

        return self._num_samples

    def __iter__(self):
        if self.replacement:
            yield from self._default_rng.choice(self.n_data_source, self.num_samples, self.replacement).tolist()
        else:
            yield from self._default_rng.permutation(self.n_data_source).tolist()

    def __len__(self):
        return self.num_samples


class InfiniteConstantSampler(Sampler):
    """nothing to do. 主要用于 Iterator 类型的数据, 其仅依赖于 iter, 而不需要 index, 故不用 sampler"""
    def __init__(self):
        super().__init__(None)

    def __iter__(self):
        while True:
            yield None


class DistributedSampler(Sampler[List[int]]):
    """

    ref: torch.utils.data.distributed.DistributedSampler
    """
    def __init__(self, horovod, dataset_size, shuffle=True, seed=0, drop_last=False):
        super().__init__(None)

        if horovod is None:
            raise ValueError('horovod should not be None in DistributedSampler')

        self.dataset_size = dataset_size

        self.num_replicas = horovod.size()
        self.rank = horovod.rank()
        self.epoch = 0
        self.drop_last = drop_last

        if drop_last and dataset_size % self.num_replicas != 0:
            self.num_samples = math.ceil((dataset_size - self.num_replicas) * 1.0 / self.num_replicas)
        else:
            self.num_samples = math.ceil(dataset_size * 1.0 / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        if self.shuffle:
            indices = np.random.RandomState(seed=self.seed + self.epoch).permutation(self.dataset_size).tolist()
        else:
            indices = list(range(self.dataset_size))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices)

    def set_epoch(self, epoch):
        """Sets the epoch for this sampler

        Notes:
            when `shuffle=True`, this ensures all replicas use a different random ordering for each epoch,
            otherwise, the next iteration of this sampler will yield the same ordering.
        """
        self.epoch = epoch


class BatchSampler(Sampler[List[int]]):
    """sampler to yield a mini-batch of indices

    Args:
        sampler: (Sampler or Iterable) base sampler
        batch_size: (int) size of mini-batch
        drop_last: (bool) drop the last batch if its size is less than `batch_size`

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool) -> None:
        super().__init__(None)

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []

        for idx in self.sampler:
            batch.append(idx)

            if len(batch) == self.batch_size:
                yield batch

                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        sz = len(self.sampler) if self.drop_last else len(self.sampler) + self.batch_size - 1

        return sz // self.batch_size
