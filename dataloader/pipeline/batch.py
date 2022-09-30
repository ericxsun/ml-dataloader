#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from dataloader.pipeline.datapipe import ProxyDataPipe
from dataloader.util.batch import RepeatInBatch


class Batch(ProxyDataPipe):
    def __init__(self, datapipe, batch_size, drop_last, fn_to_tensor, repeat_in_batch=None):
        super().__init__(datapipe)

        self.batch_size = batch_size
        self.drop_last = drop_last

        self.repeat_in_batch = repeat_in_batch
        if repeat_in_batch is None or not isinstance(repeat_in_batch, RepeatInBatch):
            self.repeat_in_batch = RepeatInBatch(kind='no')

        self._fn_to_tensor = fn_to_tensor

        if self._fn_to_tensor is None:
            raise ValueError('to_tensor_func should not be None')

    def __len__(self):
        sz = len(self.datapipe) if self.drop_last else len(self.datapipe) + self.batch_size - 1
        return sz // self.batch_size

    def __iter__(self):
        batch = []

        for data in self.datapipe:
            batch.append(data)

            if len(batch) == self.batch_size:
                batch = self.repeat_in_batch.repeat(batch)
                yield self._fn_to_tensor(batch)

                del batch[:]

        if len(batch) == self.batch_size:
            batch = self.repeat_in_batch.repeat(batch)
            yield self._fn_to_tensor(batch)

        if len(batch) > 0 and not self.drop_last:
            batch = self.repeat_in_batch.repeat(batch)
            yield self._fn_to_tensor(batch)

