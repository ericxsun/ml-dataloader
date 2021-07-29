#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from abc import abstractmethod
from copy import copy
import multiprocessing as mp
import threading

from six.moves import queue
import zmq

from dataloader.pipeline.datapipe import DataPipeReEntrantGuard
from dataloader.pipeline.datapipe import ProxyDataPipe
from dataloader.pipeline.runner import _bind_guard
from dataloader.pipeline.runner import _get_pipe_name
from dataloader.pipeline.runner import _MultiProcessZMQDataPipe
from dataloader.pipeline.runner import _zmq_catch_error
from dataloader.util.concurrency import enable_death_signal
from dataloader.util.concurrency import StoppableThread
from dataloader.util.serialize import deserialize
from dataloader.util.serialize import serialize

__all__ = ['MultiThreadMapData', 'MultiProcessMapDataZMQ', 'MapData', 'MapDataProcessKind']


class _ParallelMapData(ProxyDataPipe):
    def __init__(self, datapipe, buffer_size):
        super().__init__(datapipe)

        if buffer_size <= 0:
            raise ValueError(f'buffer_size should be greater than 0, act: {buffer_size}')

        self._buffer_size = buffer_size
        self._buffer_occupancy = 0

        self._iter = None

    def reset(self):
        super().reset()
        dp = self.datapipe
        self._iter = dp.__iter__()

    @abstractmethod
    def _recv(self):
        pass

    @abstractmethod
    def _send(self, dp):
        pass

    def _recv_filter_none(self):
        ret = self._recv()

        if ret is None:
            raise RuntimeError(f'[{type(self)}] Map function cannot return None')

        return ret

    def _fill_buffer(self, cnt=None):
        if cnt is None:
            cnt = self._buffer_size - self._buffer_occupancy

        true_cnt = 0
        try:
            for _ in range(cnt):
                dp = next(self._iter)
                self._send(dp)
                true_cnt += 1
        except StopIteration:
            pass

        self._buffer_occupancy += true_cnt

    def get_data(self):
        self._fill_buffer()

        for dp in self._iter:
            self._send(dp)
            yield self._recv_filter_none()

        self._iter = self.datapipe.__iter__()  # refresh

        # clear buffer and re-fill
        for k in range(self._buffer_size):
            dp = self._recv_filter_none()
            self._buffer_occupancy -= 1
            if k == self._buffer_size - 1:
                self._fill_buffer()
            yield dp

    def __iter__(self):
        yield from self.get_data()


class MultiThreadMapData(_ParallelMapData):
    """
    Note:
        1. the produced data will have different ordering
        2. one should avoid starting many threads in main process to reduce GIL contention.
           The threads will only start in the process which calls :meth:`reset()`.
           Therefore you can use ``MultiProcessRunnerZMQ(MultiThreadMapData(...), 1)`` to reduce GIL contention.
    """
    class Worker(StoppableThread):
        def __init__(self, in_queue, out_queue, evt, map_func, timeout=5):
            super().__init__(evt, timeout)

            self._in_queue = in_queue
            self._out_queue = out_queue
            self.func = map_func

            self.daemon = True

        def run(self):
            try:
                while True:
                    dp = self.queue_get_stoppable(self._in_queue)
                    if self.stopped():
                        return

                    # cannot ignore None here. will lead to un-synced send/recv
                    obj = self.func(dp)
                    self.queue_put_stoppable(self._out_queue, obj)
            except Exception:
                if self.stopped():
                    pass
                else:
                    raise
            finally:
                self.stop()

    def __init__(self, datapipe, num_threads=None, map_func=None, *, buffer_size=200):
        try:
            buffer_size = min(buffer_size, len(datapipe))
        except Exception:  # datapipe may not have a len
            pass

        if num_threads is None or num_threads <= 0:
            raise ValueError('num_threads should be greater than 0')

        super().__init__(datapipe, buffer_size)

        self.num_threads = num_threads
        self.map_func = map_func
        self._threads = []
        self._evt = None
        self._in_queue = None
        self._out_queue = None
        self._guard = None

    def reset(self):
        super().reset()

        if self._threads:
            self._threads[0].stop()
            for t in self._threads:
                t.join()

        self._in_queue = queue.Queue()
        self._out_queue = queue.Queue()
        self._evt = threading.Event()

        self._threads = [
            MultiThreadMapData.Worker(self._in_queue, self._out_queue, self._evt, self.map_func)
            for _ in range(self.num_threads)
        ]

        for t in self._threads:
            t.start()

        self._guard = DataPipeReEntrantGuard()

        # Call once at the beginning, to ensure in-queue + out-queue has a total of buffer_size elements
        self._fill_buffer()

    def _recv(self):
        return self._out_queue.get()

    def _send(self, dp):
        self._in_queue.put(dp)

    def __iter__(self):
        with self._guard:
            yield from super().__iter__()

    def __del__(self):
        if self._evt is not None:
            self._evt.set()

        for p in self._threads:
            p.stop()
            p.join(timeout=5.0)


class MultiProcessMapDataZMQ(_ParallelMapData, _MultiProcessZMQDataPipe):
    """
    Note:
        1. the produced data will have different ordering
    """
    class Worker(mp.Process):
        def __init__(self, identity, map_func, pipe_name, hwm):
            super().__init__()
            self.identity = identity
            self.pipe_name = pipe_name
            self.map_func = map_func
            self.hwm = hwm

        def run(self):
            enable_death_signal(_warn=self.identity == b'0')
            ctx = zmq.Context()
            socket = ctx.socket(zmq.REP)
            socket.setsockopt(zmq.IDENTITY, self.identity)
            socket.set_hwm(self.hwm)
            socket.connect(self.pipe_name)

            while True:
                dp = deserialize(socket.recv(copy=False))
                dp = self.map_func(dp)
                socket.send(serialize(dp), copy=False)

    def __init__(self, datapipe, num_procs=None, map_func=None, *, buffer_size=200):
        try:
            buffer_size = min(buffer_size, len(datapipe))
        except Exception:  # ds may not have a length
            pass

        if num_procs is None or num_procs <= 0:
            raise ValueError('num_proc should be greater than 0')

        _ParallelMapData.__init__(self, datapipe, buffer_size)
        _MultiProcessZMQDataPipe.__init__(self)

        self.num_procs = num_procs
        self.map_func = map_func
        self._procs = []

        self._guard = None

    def reset(self):
        _MultiProcessZMQDataPipe.reset(self)
        _ParallelMapData.reset(self)

        self._guard = DataPipeReEntrantGuard()

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.set_hwm(self._buffer_size * 2)

        pipe_name = _get_pipe_name('datapipe-map')
        _bind_guard(self.socket, pipe_name)

        worker_hwm = int(self._buffer_size * 2 // self.num_procs)
        self._procs = [
            MultiProcessMapDataZMQ.Worker(f'{i}'.encode('utf-8'), self.map_func, pipe_name, worker_hwm)
            for i in range(self.num_procs)
        ]

        self._start_processes()
        self._fill_buffer()

    def _send(self, dp):
        msg = [b'', serialize(dp)]
        self.socket.send_multipart(msg, copy=False)

    def _recv(self):
        msg = self.socket.recv_multipart(copy=False)
        dp = deserialize(msg[1])

        return dp

    def __iter__(self):
        with self._guard, _zmq_catch_error(type(self).__name__):
            yield from super().__iter__()


class MapData(ProxyDataPipe):
    """
        Note:
            1. the produced data will keep the order,
               but could be slow compare with MultiThreadMapData and MultiProcessMapDataZMQ
        """
    def __init__(self, datapipe, map_func):
        super().__init__(datapipe)
        self.map_func = map_func

    def __iter__(self):
        for dp in self.datapipe:
            ret = self.map_func(copy(dp))

            if ret is not None:
                yield ret


class MapDataProcessKind:
    NORMAL = 'normal'
    MULTI_THREAD = 'multi-thread'
    MULTI_PROCESS = 'multi-process'

    kinds = {NORMAL, MULTI_THREAD, MULTI_PROCESS}
    kinds_str = ', '.join(kinds)
