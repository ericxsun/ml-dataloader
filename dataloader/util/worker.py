#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from dataclasses import dataclass
import os
import queue
import random

from dataloader import logger
from dataloader.util import MP_STATUS_CHECK_INTERVAL
from dataloader.util.dataset_kind import DatasetKind
from dataloader.util.misc import ExceptionWrapper


class ManagerWatchdog:
    def __init__(self):
        self.manager_pid = os.getppid()
        self.manager_dead = False

    def is_alive(self):
        if not self.manager_dead:
            self.manager_dead = os.getppid() != self.manager_pid
        return not self.manager_dead


_worker_info = None


class WorkerInfo(object):
    __initialized = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__keys = tuple(kwargs.keys())
        self.__initialized = True

    def __setattr__(self, key, val):
        if self.__initialized:
            raise RuntimeError(f'Cannot assign attributes to {self.__class__.__name__} objects')

        return super(WorkerInfo, self).__setattr__(key, val)

    def __repr__(self):
        items = []
        for k in self.__keys:
            items.append(f'{k}={getattr(self, k)}')

        return '{}({})'.format(self.__class__.__name__, ', '.join(items))


def get_worker_info():
    """

    Returns:
        current iterator worker process:
        id: the current worker id
        num_workers: the total number of workers
        seed: the random seed set for the current workers, determined by main process RNG and the worker id
        dataset: the copy of the dataset object

    """
    return _worker_info


@dataclass(frozen=True)
class IterableDatasetStopIteration(object):
    """Dummy class used to signal the end of an IterableDataset"""
    worker_id: int


@dataclass(frozen=True)
class ResumeIteration(object):
    """Dummy class used to resume the fetching when worker reuse is enabled"""
    pass


def worker_loop(
    dataset_kind,
    dataset, is_batch, drop_last, fn_to_tensor, transform, repeat_in_batch,
    index_queue, data_queue, done_event,
    base_seed, init_fn, worker_id, num_workers
):
    try:
        seed = base_seed + worker_id
        random.seed(seed)

        global _worker_info
        _worker_info = WorkerInfo(id=worker_id, num_workers=num_workers, seed=seed, dataset=dataset)

        init_exception = None

        fetcher = None

        try:
            if init_fn is not None:
                init_fn(worker_id)

            fetcher = DatasetKind.create_fetcher(
                dataset_kind, dataset, is_batch, drop_last, fn_to_tensor, transform, repeat_in_batch
            )
        except Exception as e:
            logger.error(f'init_fn or create_fetcher failed: {e} for worker: {worker_id}')
            init_exception = ExceptionWrapper(
                where=f'in DataLoader worker process {worker_id}: init_fn/create_fetcher failed'
            )

        iteration_end = False

        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue

            if isinstance(r, ResumeIteration):
                # Acknowledge the main process
                data_queue.put((r, None))
                iteration_end = False
                # Recreate the fetcher for worker-reuse policy
                fetcher = DatasetKind.create_fetcher(
                    dataset_kind, dataset, is_batch, drop_last, fn_to_tensor, transform, repeat_in_batch
                )

                continue

            if r is None:
                # Received the final signal
                assert done_event.is_set() or iteration_end
                break

            if done_event.is_set() or iteration_end:
                # `done_event` is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue

            idx, index = r

            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    data = fetcher.fetch(index)
                    # data = serialize(data)
                except Exception as e:
                    if isinstance(e, StopIteration) and dataset_kind == DatasetKind.Iterable:
                        data = IterableDatasetStopIteration(worker_id)
                        # data = serialize(data)
                        # Set `iteration_end`
                        # 1. to save future `next(...)` calls, and
                        # 2. to avoid sending multiple `IterableDatasetStopIteration`s.
                        iteration_end = True
                    else:
                        logger.error(f'fetch failed index={index}, worker_id={worker_id}: {e}')
                        data = ExceptionWrapper(where="in DataLoader worker process {}".format(worker_id))

            data_queue.put((idx, data))
            del data, idx, index, r  # save memory
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass

    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()
