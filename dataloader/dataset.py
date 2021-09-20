#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from abc import ABC
import bisect
from typing import Callable
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import TypeVar

from dataloader.raw_dataset import RawDataset
from dataloader.transform import apply_transform
from dataloader.util.data_kind import DataKind

__all__ = ['Dataset', 'IterableDataset', 'BaseDataset', 'ChainDataset', 'ConcatDataset']

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class BaseDataset(Generic[T_co]):
    """an abstract class representing a Dataset"""

    def __getitem__(self, index: int) -> T_co:
        raise NotImplementedError

    def __add__(self, other: 'BaseDataset[T_co]') -> 'ConcatDataset[T_co]':
        return ConcatDataset([self, other])


class _BaseIterableDataset(BaseDataset[T_co]):
    """an iterable Dataset"""
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    def __add__(self, other: BaseDataset[T_co]):
        return ChainDataset([self, other])

    def __getitem__(self, index: int):
        pass


class ConcatDataset(BaseDataset[T_co]):
    """assemble different existing datasets"""
    datasets: List[BaseDataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            n = len(e)
            r.append(n + s)
            s += n

        return r

    def __init__(self, datasets: Iterable[BaseDataset]):
        super().__init__()
        self.datasets = list(datasets)

        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError('absolute value of index should not exceed dataset length')
            idx = len(self) + idx

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][sample_idx]


class ChainDataset(_BaseIterableDataset):
    """assemble different existing dataset streams on-the-fly"""
    def __init__(self, datasets: Iterable[BaseDataset]):
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        for d in self.datasets:
            for x in d:
                yield x

    def __len__(self):
        ns = [len(d) for d in self.datasets]
        return sum(ns)


class IterableDataset(_BaseIterableDataset):
    def __init__(self, data: Iterable, transform: Optional[Callable] = None):
        self.data = data
        self.transform = transform

    def __iter__(self):
        data = iter(self.data)

        for d in data:
            d = apply_transform(self.transform, d)
            yield d


class Dataset(BaseDataset, RawDataset, ABC):
    def __init__(self, data, kind=DataKind.MEM_SEQ, transform: Optional[Callable] = None):
        """

        Args:
            data:
              - list of things: if kind=DataKind.MEM_SEQ
              - filename: if kind=DataKind.FILE or kind=DataKind.MMAP_FILE
            kind: @see DataKind
            transform: func for mapping raw data into feature
        """
        super().__init__(data, kind, transform)

    def __getitem__(self, index):
        return self.get(index)
