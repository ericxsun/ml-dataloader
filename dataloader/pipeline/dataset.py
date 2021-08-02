#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from copy import copy
import multiprocessing
import os

from dataloader import logger
from dataloader.pipeline.datapipe import RandomDataPipe
from dataloader.util import get_rng
from dataloader.util.data_kind import DataKind
from dataloader.util.misc import bytes_to_str

__all__ = ['Dataset', 'shuffle_dataset']


class Dataset(RandomDataPipe):
    data_kinds = {DataKind.FILE, DataKind.MEM_SEQ}

    def __init__(self, data, kind=DataKind.MEM_SEQ):
        """

        Args:
            data: list of data if kind = DataKind.MEM_SEQ, else the filename storing data
            kind: @see DataKind
        """
        self.kind = kind
        if kind not in self.data_kinds:
            raise ValueError(f'not supported data kind: {kind}, choose one from [{",".join(self.data_kinds)}]')

        self.meta = {'offset': []}
        self.n_data = 0

        self._local_rng = get_rng(self)

        if self.kind == DataKind.MEM_SEQ:
            if not isinstance(data, list):
                raise ValueError(f'if kind is DataKind.MEM_SEQ, data should be a list. type(data)={type(data)}')

            self.n_data = len(data)

            self.data = data
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

        self.indices = list(range(self.n_data))
        self._iter_idx = iter(self.indices)

        self._idx = -1

    def shuffle(self):
        indices = list(range(self.n_data))
        self._local_rng.shuffle(indices)
        self._iter_idx = iter(indices)

        return self

    def __len__(self) -> int:
        return self.n_data

    def __iter__(self):
        return self

    def __next__(self):
        try:
            idx = next(self._iter_idx)
            return self[idx]
        except IndexError:
            raise StopIteration()

    def __getitem__(self, idx: int):
        data = None

        if idx < 0:
            idx = self.n_data + idx

        if self.kind == DataKind.MEM_SEQ:
            data = self.data[idx]
        elif self.kind == DataKind.FILE:
            with self.lock:
                self.fd.seek(self.meta['offset'][idx])
                line = self.fd.readline()

            try:
                data = bytes_to_str(line).strip('\n')
            except Exception as e:
                logger.error(f'decode failed: index={idx}, offset={self.meta["offset"][idx]}, line={line}')
                raise e

        return data


def shuffle_dataset(dataset, shuffle):
    if not shuffle:
        return dataset

    dataset = copy(dataset)
    dataset.shuffle()
    return dataset
