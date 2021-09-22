#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from collections import Callable
import mmap
import os
from typing import Optional

import multiprocess as multiprocessing

from dataloader import logger
from dataloader.transform import apply_transform
from dataloader.util.data_kind import DataKind
from dataloader.util.misc import bytes_to_str
from dataloader.util.misc import get_offset


class RawDataset:
    data_kinds = {DataKind.FILE, DataKind.MMAP_FILE, DataKind.MEM_SEQ}

    def __init__(self, data, kind=DataKind.MEM_SEQ, transform: Optional[Callable] = None):
        """

        Args:
            data:
              - list of things: if kind=DataKind.MEM_SEQ
              - filename: if kind=DataKind.FILE or kind=DataKind.MMAP_FILE
            kind: @see DataKind
            transform: func for mapping raw data into feature

        >>> import os, mmap
        >>> fp = open('filename', 'rb', os.O_RDONLY)
        >>> mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        >>> fp.close()
        >>> dataset = RawDataset(mm, DataKind.MMAP_FILE, filename='filename')

        Notes:
            - speed: DataKind.MEM_SEQ > DataKind.MMAP_FILE >> DataKind.FILE
        """

        self.kind = kind
        if kind not in self.data_kinds:
            raise ValueError(f'not supported data kind: {kind}, choose one from [{",".join(self.data_kinds)}]')

        self.offset = []
        self.n_data = 0

        if self.kind == DataKind.MEM_SEQ:
            if not isinstance(data, list):
                raise ValueError(f'if kind is DataKind.MEM_SEQ, data should be a list. type(data)={type(data)}')

            self.data = data
            self.n_data = len(data)
        elif self.kind == DataKind.FILE or self.kind == DataKind.MMAP_FILE:
            if not os.path.exists(data):
                raise ValueError(f'filename does not exist: {data}')
            self.filename = data

            self.offset, self.n_data = get_offset(self.filename)

            if self.kind == DataKind.FILE:
                self.fd = open(data, 'rb', buffering=0)
                self.lock = multiprocessing.Lock()
                logger.warning('with multiprocessing.Lock lead to poor speed')
            elif self.kind == DataKind.MMAP_FILE:
                fp = os.open(self.filename, os.O_RDONLY)
                self.data = mmap.mmap(fp, 0, access=mmap.ACCESS_READ)

                logger.warning('with mmap lead to huge virtual memory cost')

        self.transform = transform

    def __len__(self) -> int:
        return self.n_data

    def get(self, index: int):
        data = None

        if index < 0:
            index += self.n_data

        if self.kind == DataKind.MEM_SEQ:
            data = self.data[index]
        elif self.kind == DataKind.FILE:
            with self.lock:
                self.fd.seek(self.offset[index])
                line = self.fd.readline()

            try:
                data = bytes_to_str(line).strip('\n')
            except Exception as e:
                logger.error(f'decode failed: index={index}, offset={self.offset[index]}, line={line}')
                raise e
        elif self.kind == DataKind.MMAP_FILE:
            start = self.offset[index]
            if index + 1 < self.n_data:
                line = self.data[start:self.offset[index + 1]]
            else:
                line = self.data[start:]

            try:
                data = bytes_to_str(line).split('\n')[0].strip('\n')
            except Exception as e:
                logger.error(f'decode failed: index={index}, offset={self.offset[index]}, line={line}')
                raise e

        data = apply_transform(self.transform, data)

        return data

    def __getitem__(self, index: int):
        return self.get(index)
