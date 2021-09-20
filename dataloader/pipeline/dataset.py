#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from copy import copy

from dataloader.pipeline.datapipe import RandomDataPipe
from dataloader.raw_dataset import RawDataset
from dataloader.util import get_rng
from dataloader.util.data_kind import DataKind

__all__ = ['Dataset', 'shuffle_dataset']


class Dataset(RandomDataPipe, RawDataset):
    def __init__(self, data, kind=DataKind.MEM_SEQ):
        """

        Args:
            data:
              - list of things: if kind=DataKind.MEM_SEQ
              - filename: if kind=DataKind.FILE or kind=DataKind.MMAP_FILE
            kind: @see DataKind
        """
        super().__init__(data, kind)

        self._local_rng = get_rng(self)

        self.indices = list(range(self.n_data))
        self._iter_idx = iter(self.indices)

        self._idx = -1

    def shuffle(self):
        indices = list(range(self.n_data))
        self._local_rng.shuffle(indices)
        self._iter_idx = iter(indices)

        return self

    def __iter__(self):
        return self

    def __next__(self):
        try:
            idx = next(self._iter_idx)
            return self[idx]
        except IndexError:
            raise StopIteration()


def shuffle_dataset(dataset, shuffle):
    if not shuffle:
        return dataset

    dataset = copy(dataset)
    dataset.shuffle()
    return dataset
