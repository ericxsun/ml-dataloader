#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import bisect
import os
from typing import Callable
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import TypeVar

import multiprocess as multiprocessing

__all__ = ['Dataset', 'IterableDataset', 'BaseDataset', 'ChainDataset', 'ConcatDataset']

from dataloader import logger
from dataloader.transform import apply_transform
from dataloader.util.data_kind import DataKind
from dataloader.util.misc import bytes_to_str

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


class Dataset(BaseDataset):
    """

    Notes:
        1. the following ways can be used in separately or combined to load huge training data
          - shuffle and split it first, and then make many Dataset, chained with ChainDataset
          - do not set transform in Dataset, but in DataLoader (i.e., transform data in every batch)
        2. for huge data, offsets of all lines are stored, but real content is loaded when needed with threading Lock,
           this could lead to poor speed (sometimes stuck)
    """

    data_kinds = {DataKind.FILE, DataKind.MEM_SEQ}

    def __init__(self, data, kind=DataKind.MEM_SEQ, transform: Optional[Callable] = None):
        """

        Args:
            data:
            kind: @see DataKind
            transform:
        """
        self.kind = kind
        if kind not in self.data_kinds:
            raise ValueError(f'not supported data kind: {kind}, choose one from [{",".join(self.data_kinds)}]')

        self.meta = {'offset': []}
        self.n_data = 0

        if self.kind == DataKind.MEM_SEQ:
            if not isinstance(data, list):
                raise ValueError(f'if kind is DataKind.MEM_SEQ, data should be a list. type(data)={type(data)}')

            self.data = data
            self.n_data = len(data)
        elif self.kind == DataKind.FILE:
            if not os.path.exists(data):
                raise ValueError(f'filename does not exist: {data}')

            logger.debug(f'loading offset from {data}')
            with open(data, 'rb') as fd:
                offset = [0]
                while fd.readline():
                    offset.append(fd.tell())
                self.meta['offset'] = offset[:-1]
            logger.debug(f'loading offset done: n_offset={len(self.meta["offset"])}')

            self.n_data = len(self.meta['offset'])

            self.filename = data
            self.fd = open(data, 'rb', buffering=0)
            self.lock = multiprocessing.Lock()
            logger.warning('with multiprocessing.Lock lead to poor speed')

        self.transform = transform

    def __len__(self) -> int:
        return self.n_data

    def __getitem__(self, index: int):
        data = None

        if index < 0:
            index = self.n_data + index

        if self.kind == DataKind.MEM_SEQ:
            data = self.data[index]
        elif self.kind == DataKind.FILE:
            with self.lock:
                self.fd.seek(self.meta['offset'][index])
                line = self.fd.readline()

            try:
                data = bytes_to_str(line).strip('\n')
            except Exception as e:
                logger.error(f'decode failed: index={index}, offset={self.meta["offset"][index]}, line={line}')
                raise e

        data = apply_transform(self.transform, data)

        return data
