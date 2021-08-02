#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from dataloader import logger
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
