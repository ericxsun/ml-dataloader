#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from dataloader import logger
from dataloader.pipeline.datapipe import ProxyDataPipe
from dataloader.pipeline.to_tensor import to_tf_tensor
from dataloader.util import get_rng


class RepeatInBatch:
    No = 'no'  # without repeat
    Append = 'append'  # append at the end, same as [x0, x1, x2] * times
    NextTo = 'next_to'  # e.g., when times = 1,  [x0, x0, x1, x1, ...]
    Shuffle = 'shuffle'  # repeat and shuffle

    kinds = {No, Append, NextTo, Shuffle}

    def __init__(self, kind, times=0):
        """

        Args
            kind: repeat kind, in ['no', 'append', 'next_to', 'shuffle']
            times: repeat times, repeat times when times > 0, otherwise no repeat
        """
        if kind not in self.kinds:
            raise ValueError(
                f'dataset repeat kind does not support: kind={kind}, supported: [{",".join(self.kinds)}]'
            )
        self.kind = kind

        if times < 0:
            raise ValueError(f'repeat times should not be less than 0. times={times}')
        self.times = times + 1

        logger.info(f'repeat data in batch with kind={self.kind}, times={times}')

        self._local_rng = get_rng(self)

    def repeat(self, batch):
        if self.kind == self.No or self.times <= 1:
            return batch

        if self.kind == self.Append:
            return batch * self.times

        if self.kind == self.NextTo:
            batch = [[data] * self.times for data in batch]
            batch = [data for mini_batch in batch for data in mini_batch]
            return batch

        if self.kind == self.Shuffle:
            batch = batch * self.times
            self._local_rng.shuffle(batch)

            return batch


class Batch(ProxyDataPipe):
    def __init__(self, datapipe, batch_size, drop_last, repeat_in_batch=None, to_tensor_func=to_tf_tensor):
        super().__init__(datapipe)

        self.batch_size = batch_size
        self.drop_last = drop_last

        self.repeat_in_batch = repeat_in_batch
        if repeat_in_batch is None or not isinstance(repeat_in_batch, RepeatInBatch):
            self.repeat_in_batch = RepeatInBatch(kind='no')

        self._to_tensor_func = to_tensor_func
        if self._to_tensor_func is None:
            self._to_tensor_func = to_tf_tensor

    def __len__(self):
        sz = len(self.datapipe) if self.drop_last else len(self.datapipe) + self.batch_size - 1
        return sz // self.batch_size

    def __iter__(self):
        batch = []

        for data in self.datapipe:
            batch.append(data)

            if len(batch) == self.batch_size:
                batch = self.repeat_in_batch.repeat(batch)
                yield self._to_tensor_func(batch)

                del batch[:]

        if len(batch) == self.batch_size:
            batch = self.repeat_in_batch.repeat(batch)
            yield self._to_tensor_func(batch)

        if len(batch) > 0 and not self.drop_last:
            batch = self.repeat_in_batch.repeat(batch)
            yield self._to_tensor_func(batch)

