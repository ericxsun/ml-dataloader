#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import threading

from dataloader.util import get_rng

__all__ = ['DataPipe', 'ProxyDataPipe', 'RandomDataPipe', 'DataPipeTerminated', 'DataPipeReEntrantGuard']


class DataPipeTerminated(BaseException):
    pass


class DataPipeReEntrantGuard:
    """enforce non-re-entrant"""
    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        self._succeed = self._lock.acquire(False)

        if not self._succeed:
            raise threading.ThreadError("this DataPipe is not re-entrant!")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()

        return False


class DataPipe:
    def __len__(self):
        raise NotImplementedError()

    def reset(self):
        pass


class ProxyDataPipe(DataPipe):
    def __init__(self, datapipe):
        self.datapipe = datapipe

    def reset(self):
        self.datapipe.reset()

    def __len__(self):
        return self.datapipe.__len__()

    def __iter__(self):
        return self.datapipe.__iter__()


class RandomDataPipe(DataPipe):
    def __len__(self):
        raise NotImplementedError()

    def reset(self):
        self.rng = get_rng(self)
