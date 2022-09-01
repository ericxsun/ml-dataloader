#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import errno
import itertools
import queue
import tempfile
from typing import Any
from typing import Callable
from typing import Generic
from typing import List
from typing import Optional
from typing import Sequence
from typing import TypeVar

import multiprocess as multiprocessing  # when use multiprocessing, file descriptor cannot be pickled
import numpy as np
from prefetch_generator import BackgroundGenerator
import tensorflow as tf

from dataloader import logger
from dataloader import util
from dataloader.dataset import BaseDataset
from dataloader.dataset import Dataset
from dataloader.dataset import IterableDataset
from dataloader.sampler import BatchSampler
from dataloader.sampler import InfiniteConstantSampler
from dataloader.sampler import RandomSampler
from dataloader.sampler import Sampler
from dataloader.sampler import SequentialSampler
from dataloader.util import signal_handling
from dataloader.util import to_tensor
from dataloader.util import worker
from dataloader.util.batch import RepeatInBatch
from dataloader.util.dataset_kind import DatasetKind
from dataloader.util.misc import ExceptionWrapper
from dataloader.util.misc import set_rnd

__all__ = ['DataLoader']


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

_fn_worker_init_t = Callable[[int], None]
_fn_to_tensor_t = Callable[[List[T]], Any]


class _BaseDataLoader(Generic[T_co]):
    """Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset"""

    dataset: BaseDataset[T_co]
    batch_size: Optional[int]
    num_workers: int
    drop_last: bool
    timeout: float
    sampler: Sampler
    prefetch_factor: int
    _iterator: Optional['_BaseDataLoaderIter']
    __initialized = False

    def __init__(
        self,
        dataset: BaseDataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
        num_workers: int = 0,
        fn_to_tensor: _fn_to_tensor_t = None,
        drop_last: bool = False,
        timeout: float = 0,
        fn_worker_init: _fn_worker_init_t = None,
        multiprocessing_context=None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        repeat_in_batch: RepeatInBatch = None,
        transform: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.timeout = timeout
        self.fn_worker_init = fn_worker_init
        self.multiprocessing_context = multiprocessing_context

        if isinstance(dataset, IterableDataset):
            self.dataset_kind = DatasetKind.Iterable
        else:
            self.dataset_kind = DatasetKind.Map

        if sampler is None:
            if self.dataset_kind == DatasetKind.Iterable:
                sampler = InfiniteConstantSampler()
            else:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)

        if batch_size is not None and batch_sampler is None:
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler

        if fn_to_tensor is None:
            fn_to_tensor = to_tensor.to_tf_tensor

        self.fn_to_tensor = fn_to_tensor
        self.persistent_workers = persistent_workers

        self.__initialized = True
        self._IterableDataset_len_called = None

        self._iterator = None

        self.repeat_in_batch = repeat_in_batch
        self.transform = transform

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)

        return _MultiProcessingDataLoaderIter(self)

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if isinstance(multiprocessing_context, (str, bytes)):
                    multiprocessing_context = multiprocessing.get_context(multiprocessing_context)

        self.__multiprocessing_context = multiprocessing_context

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'sampler', 'drop_last', 'dataset', 'persistent_workers'):
            raise ValueError(
                '{} attribute should not be set after {} is initialized'.format(attr, self.__class__.__name__)
            )

        super().__setattr__(attr, val)

    def __iter__(self) -> '_BaseDataLoaderIter':
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator.reset(self)

            return self._iterator

        return self._get_iterator()

    @property
    def is_batch(self):
        return self.batch_sampler is not None

    @property
    def index_sampler(self):
        if self.is_batch:
            return self.batch_sampler

        return self.sampler

    def __len__(self) -> int:
        if self.dataset_kind == DatasetKind.Iterable:
            length = self._IterableDataset_len_called = len(self.dataset)
            if self.batch_size is not None:
                from math import ceil
                if self.drop_last:
                    length = length // self.batch_size
                else:
                    length = ceil(length / self.batch_size)
            return length

        return len(self.index_sampler)


class _BaseDataLoaderIter:
    def __init__(self, loader: _BaseDataLoader) -> None:
        self._dataset = loader.dataset
        self._dataset_kind = loader.dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._is_batch = loader.is_batch
        self._drop_last = loader.drop_last
        self._index_sampler = loader.index_sampler
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._timeout = loader.timeout
        self._fn_to_tensor = loader.fn_to_tensor
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = tf.random.uniform((1, 1), 0, np.iinfo(np.int64).max, tf.int64).numpy()[0][0]
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0

        self._repeat = loader.repeat_in_batch

    def __iter__(self) -> '_BaseDataLoaderIter':
        return self

    def reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called

    def _next_index(self):
        return next(self._sampler_iter)

    def _next_data(self):
        raise NotImplementedError

    def __next__(self) -> Any:
        data = self._next_data()

        self._num_yielded += 1

        return data

    def __len__(self) -> int:
        return len(self._index_sampler)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)

        self._dataset_fetcher = DatasetKind.create_fetcher(
            self._dataset_kind,
            self._dataset, self._is_batch, self._drop_last, self._fn_to_tensor, loader.transform, loader.repeat_in_batch
        )

    def _next_data(self):
        index = self._next_index()
        data = self._dataset_fetcher.fetch(index)

        return data


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    """Iterates once over the DataLoader's dataset, as specified by the sampler"""
    def __init__(self, loader):
        super().__init__(loader)

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        # DO NOT SET METHOD, it'll stuck the creation of workers
        # forkserver stuck
        # spawn: is killed by signal: Segmentation fault
        # fork:
        # dataset 中使用了 multiprocessing.Lock() 操作, 故需要与之保持一致, 使用 multiprocess as multiprocessing
        # 若不是一致, 则会报 RuntimeError: Lock objects should only be shared between processes through inheritance
        # 同时, 没有出现「创建 worker 时的 stuck」
        #
        # set multiprocessing method explicitly with spawn or forkserver
        # if not set: RuntimeError: DataLoader worker (pid(s)) exited unexpectedly
        # ref: https://github.com/horovod/horovod/issues/2053
        #      https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn
        #      - fork: fork is fast, unsafe, and maybe bloated
        #      - spawn: spawn is safe, compact, and slower
        #      - forkserver: forkserver is fast, compact, and safe, but it's more complicated
        # try:
        #     multiprocessing_context.set_start_method(os.getenv('multiprocessing_context_method', 'forkserver'))
        # except RuntimeError as e:
        #     logger.warning(f'multiprocessing_context method has been set: {e}')
        #     pass

        self._fn_worker_init = loader.fn_worker_init
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))

        self._worker_result_queue = multiprocessing_context.Queue()
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()

        self._index_queues = []
        self._workers = []
        for worker_id in range(self._num_workers):
            index_queue = multiprocessing_context.Queue()
            index_queue.cancel_join_thread()

            w = multiprocessing_context.Process(
                target=worker.worker_loop,
                args=(
                    self._dataset_kind,
                    self._dataset, self._is_batch, self._drop_last, self._fn_to_tensor,
                    loader.transform, loader.repeat_in_batch,
                    index_queue, self._worker_result_queue, self._workers_done_event,
                    self._base_seed, self._fn_worker_init, worker_id, self._num_workers,
                )
            )

            w.daemon = True
            w.start()

            self._index_queues.append(index_queue)
            self._workers.append(w)

        self._data_queue = self._worker_result_queue

        self._worker_pids = {id(self): [w.pid for w in self._workers]}
        self._worker_pids_set = True

        signal_handling.set_sigchld_handler()

        self._send_idx = None
        self._rcvd_idx = None
        self._task_info = None
        self._tasks_outstanding = None
        self._workers_status = None

        self.reset(loader, first_iter=True)

    def reset(self, loader, first_iter=False):
        super().reset(loader, first_iter)

        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__

        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._task_info = {}
        self._tasks_outstanding = 0  # always equal to count(v for v in task_info.values() if len(v) == 1)

        self._workers_status = [True for _ in range(self._num_workers)]

        # We resume the prefetching in case it was enabled
        if not first_iter:
            for idx in range(self._num_workers):
                self._index_queues[idx].put(worker.ResumeIteration())

            resume_iteration_cnt = self._num_workers
            while resume_iteration_cnt > 0:
                data = self._get_data()
                if isinstance(data, worker.ResumeIteration):
                    resume_iteration_cnt -= 1

        # prime the prefetch loop
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()

    def _try_get_data(self, timeout=worker.MP_STATUS_CHECK_INTERVAL):
        try:
            data = self._data_queue.get(timeout=timeout)
            return True, data
        except Exception as e:
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)

            if len(failed_workers) > 0:
                pids_str = ', '.join(str(w.pid) for w in failed_workers)
                raise RuntimeError(f'DataLoader worker (pid(s) {pids_str}) exited unexpectedly: {e}')

            if isinstance(e, queue.Empty):
                return False, None

            try:
                fds_limit_margin = 10
                _ = [tempfile.NamedTemporaryFile() for _ in range(fds_limit_margin)]
            except OSError as e:
                if e.errno == errno.EMFILE:
                    raise RuntimeError(
                        """Too many open files. Communication with the workers is no longer possible. 
                        Please increase the limit using `ulimit -n` in the shell
                        """
                    )
            raise

    def _get_data(self):
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data

            raise RuntimeError(f'DataLoader timed out after {self._timeout} seconds')
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def _next_data(self):
        while True:
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                if len(info) == 2 or self._workers_status[info[0]]:  # has data or is still active
                    break

                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                if not self._persistent_workers:
                    self._shutdown_workers()

                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)

            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            if self._dataset_kind == DatasetKind.Iterable:
                if isinstance(data, worker.IterableDatasetStopIteration):
                    if self._persistent_workers:
                        self._workers_status[data.worker_id] = False
                    else:
                        self._mark_worker_as_unavailable(data.worker_id)

                    self._try_put_index()

                    continue

            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]
                return self._process_data(data)

    def _try_put_index(self):
        try:
            index = self._next_index()
        except StopIteration:
            return
        for _ in range(self._num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            logger.debug('no worker active...')
            return

        # print(f'worker_queue_idx={worker_queue_idx}, _send_idx={self._send_idx}, index={index}')
        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_index()

        if isinstance(data, ExceptionWrapper):
            data.re_raise()

        # data = deserialize(data)

        return data

    def _mark_worker_as_unavailable(self, worker_id):
        q = self._index_queues[worker_id]
        q.put(None)  # indicate that no more data will be put on this queue by the current process

        self._workers_status[worker_id] = False

    def _shutdown_workers(self):
        # Python is shutting down
        python_exit_status = util.python_exit_status
        if python_exit_status is True or python_exit_status is None:
            return

        # shutdown all workers
        if not self._shutdown:
            self._shutdown = True

            try:
                self._workers_done_event.set()

                for worker_id in range(len(self._workers)):
                    if self._persistent_workers or self._workers_status[worker_id]:
                        self._mark_worker_as_unavailable(worker_id)

                for w in self._workers:
                    w.join(timeout=worker.MP_STATUS_CHECK_INTERVAL)

                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                if self._worker_pids_set:
                    self._worker_pids.clear()
                    self._worker_pids_set = False

                for w in self._workers:
                    if w.is_alive():
                        w.terminate()

    def __del__(self):
        self._shutdown_workers()


class DataLoader(_BaseDataLoader):
    def __init__(self, dataset: Dataset, num_workers: int = 0, background_generator=False, **kwargs):
        """
        Args:
            dataset:
            num_workers:
            background_generator: with BackgroundGenerator (default no)
            kwargs:
                - fn_to_tensor:
                - fn_worker_init:
                - transform:
                - repeat_in_batch:


        Returns:
        """
        if num_workers == 0:
            seed = tf.random.uniform((1, 1), 0, np.iinfo(np.int32).max, tf.int32).numpy()[0][0]
            set_rnd(dataset, int(seed))

        if 'fn_to_tensor' not in kwargs:
            kwargs.update({"fn_to_tensor": to_tensor.to_tf_tensor})

        if 'fn_worker_init' not in kwargs:
            kwargs.update({'fn_worker_init': self.worker_init_fn})

        if not isinstance(kwargs.get('repeat_in_batch', None), RepeatInBatch):
            logger.debug(
                f'repeat_in_batch should be instance of DatasetRepeat, but got: {kwargs.get("repeat_in_batch", None)}. '
                f'Default: RepeatInBatch("no") will be used'
            )
            kwargs.update({'repeat_in_batch': RepeatInBatch('no')})

        self.background_generator = background_generator

        super().__init__(dataset=dataset, num_workers=num_workers, **kwargs)

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """set different random seed for the transforms in different workers

        Args:
            worker_id:

        Returns:
        """
        worker_info = worker.get_worker_info()

        set_rnd(worker_info.dataset, seed=worker_info.seed)

    def __iter__(self):
        if self.background_generator:
            return BackgroundGenerator(super().__iter__())

        return super().__iter__()
