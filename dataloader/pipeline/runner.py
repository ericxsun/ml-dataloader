#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from abc import ABC
import atexit
from contextlib import contextmanager
import errno
import itertools
import multiprocessing as mp
import os
import pickle
import sys
import traceback
import uuid
import weakref

import zmq

from dataloader import logger
from dataloader.pipeline.datapipe import DataPipe
from dataloader.pipeline.datapipe import DataPipeReEntrantGuard
from dataloader.pipeline.datapipe import DataPipeTerminated
from dataloader.util.concurrency import enable_death_signal
from dataloader.util.concurrency import start_proc_mask_signal
from dataloader.util.serialize import deserialize
from dataloader.util.serialize import serialize


class _ExceptionWrapper:
    MAGIC = b"EXC_MAGIC"
    """Wraps an exception plus traceback to communicate across threads"""
    def __init__(self, exc_info):
        # It is important that we don't store exc_info, see
        # NOTE [ Python Traceback Reference Cycle Problem ]
        self.exc_type = exc_info[0]
        self.exc_msg = ''.join(traceback.format_exception(*exc_info))

    def pack(self):
        return self.MAGIC + pickle.dumps(self)

    @staticmethod
    def unpack(dp):
        if isinstance(dp, bytes) and dp.startswith(_ExceptionWrapper.MAGIC):
            return pickle.loads(dp[len(_ExceptionWrapper.MAGIC):])


def _repeat_iter(get_iter):
    while True:
        yield from get_iter()


def _bind_guard(sock, name):
    try:
        sock.bind(name)
    except zmq.ZMQError:
        logger.error(f'ZMQError in socket.bind({name})')
        raise


def _get_pipe_name(name):
    if sys.platform.startswith('linux'):
        # linux supports abstract sockets: http://api.zeromq.org/4-1:zmq-ipc
        pipe_name = "ipc://@{}-pipe-{}".format(name, str(uuid.uuid1())[:8])
        pipe_dir = os.environ.get('DL_PIPE_DIR', None)
        if pipe_dir is not None:
            logger.warning("DL_PIPE_DIR is not used on Linux any more! Abstract sockets will be used.")
    else:
        pipe_dir = os.environ.get('DL_PIPE_DIR', None)
        if pipe_dir is not None:
            logger.info("ZMQ uses DL_PIPEDIR={}".format(pipe_dir))
        else:
            pipe_dir = '.'
        assert os.path.isdir(pipe_dir), pipe_dir

        filename = '{}/{}-pipe-{}'.format(pipe_dir.rstrip('/'), name, str(uuid.uuid1())[:6])
        assert not os.path.exists(filename), "Pipe {} exists! You may be unlucky.".format(filename)

        pipe_name = "ipc://{}".format(filename)

    return pipe_name


def del_weakref(x):
    o = x()
    if o is not None:
        o.__del__()


@contextmanager
def _zmq_catch_error(name):
    try:
        yield
    except zmq.ContextTerminated:
        logger.info(f'[{name}] Context terminated')
        raise DataPipeTerminated()
    except zmq.ZMQError as e:
        if e.errno == errno.ENOTSOCK:
            logger.info(f'[{name}] Socket closed')
            raise DataPipeTerminated()

        raise
    except Exception:
        raise


class _MultiProcessZMQDataPipe(DataPipe, ABC):
    def __init__(self):
        self._reset_done = False
        self._procs = []
        self.context = None
        self.socket = None

    def reset(self):
        assert not self._reset_done, 'reset() was called twice! This violates the api'
        self._reset_done = True

        # __del__ not guaranteed to get called at exit
        atexit.register(del_weakref, weakref.ref(self))

    def _start_processes(self):
        start_proc_mask_signal(self._procs)

    def __del__(self):
        try:
            if not self._reset_done:
                return

            if not self.context.closed:
                self.socket.close(0)
                self.context.destroy(0)

            for proc in self._procs:
                proc.terminate()
                proc.join(5)
        except Exception:
            pass


class MultiProcessRunnerZMQ(_MultiProcessZMQDataPipe):
    """run a datapipe in more than 1 processes, with ZeroMQ for communication.

    It'll fork the calling process of `reset`, can collect data from the give datapipe in each process by ZMQ ipc pipe.
    """
    class Worker(mp.Process):
        def __init__(self, datapipe, conn_name, hwm, idx):
            super().__init__()

            self.datapipe = datapipe
            self.conn_name = conn_name
            self.hwm = hwm
            self.idx = idx

        def run(self):
            enable_death_signal(_warn=self.idx == 0)
            self.datapipe.reset()
            _iter = _repeat_iter(lambda: self.datapipe)

            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.set_hwm(self.hwm)
            socket.connect(self.conn_name)

            try:
                while True:
                    try:
                        dp = next(_iter)
                        socket.send(serialize(dp), copy=False)
                    except Exception:
                        dp = _ExceptionWrapper(sys.exc_info()).pack()
                        socket.send(serialize(dp), copy=False)
                        raise
            except KeyboardInterrupt:
                pass
            finally:
                socket.close(0)
                context.destroy(0)

    def __init__(self, datapipe, num_procs=1, hwm=50):
        """

        Args:
            datapipe:
            num_procs: number of processes to use
            hwm: the zmq "high-water mark" (queue size) for both sender and receiver
        """
        super().__init__()

        self.datapipe = datapipe
        self.num_procs = num_procs
        self._hwm = hwm

        if num_procs > 1:
            logger.info(
                'MultiProcessRunnerZMQ will fork a datapipe more than one time, this assumes the data are i.i.d.'
            )

        try:
            self._sz = datapipe.__len__()
        except NotImplementedError:
            self._sz = -1

        self._guard = None
        self.context = None
        self.socket = None

    def reset(self):
        super().reset()
        self._guard = DataPipeReEntrantGuard()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.set_hwm(self._hwm)

        pipe_name = _get_pipe_name('datapipe')
        _bind_guard(self.socket, pipe_name)

        self._procs = [
            MultiProcessRunnerZMQ.Worker(self.datapipe, pipe_name, self._hwm, idx) for idx in range(self.num_procs)
        ]
        self._start_processes()

    def _recv(self):
        ret = deserialize(self.socket.recv(copy=False))

        exc = _ExceptionWrapper.unpack(ret)

        if exc is not None:
            logger.error(f'Exception "{str(exc.exc_type)}" in worker')
            raise exc.exc_type(exc.exc_msg)

        return ret

    def __len__(self):
        return self.datapipe.__len__()

    def __iter__(self):
        with self._guard, _zmq_catch_error('MultiProcessRunnerZMQ'):
            for k in itertools.count():
                if 0 < self._sz <= k:
                    break

                yield self._recv()
