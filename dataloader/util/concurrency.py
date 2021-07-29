#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from contextlib import contextmanager
import multiprocessing as mp
import platform
import signal
import threading

from six.moves import queue

from dataloader import logger


class StoppableThread(threading.Thread):
    """A thread that has a 'stop' event"""

    def __init__(self, evt=None, timeout=5):
        """

        Args:
            evt: threading.Event, if None, will create one
            timeout: put/get timeout
        """
        super().__init__()

        if evt is None:
            evt = threading.Event()

        self._stop_evt = evt
        self._timeout = timeout

    def stop(self):
        self._stop_evt.set()

    def stopped(self):
        """
        Returns:
            whether the thread is stopped or not
        """
        return self._stop_evt.isSet()

    def queue_put_stoppable(self, q, obj):
        """put obj to queue, but will give up when the thread is stopped"""
        while not self.stopped():
            try:
                q.put(obj, timeout=self._timeout)
                break
            except queue.Full:
                pass

    def queue_get_stoppable(self, q):
        """get obj from queue, but will give up when the thread is stopped"""
        while not self.stopped():
            try:
                return q.get(timeout=self._timeout)
            except queue.Empty:
                pass


def enable_death_signal(_warn=True):
    """set death signal of the current process, so the current process will be cleaned with guarantee in case the
    parent dies accidentally
    """

    if platform.system() != 'Linux':
        return

    try:
        import prctl  # pip install python-prctl
    except ImportError:
        if _warn:
            logger.warning(
                '"import prctl" failed! Install python-prctl so that processes can be cleaned with guarantee'
            )
        return
    else:
        if not hasattr(prctl, 'set_pdeathsig'):
            raise RuntimeError(
                'prctl.set_pdeathsig does not exist! Note that you need to install "python-prctl" instead of "prctl"'
            )
        # is SIGHUP a good choice?
        prctl.set_pdeathsig(signal.SIGHUP)


@contextmanager
def mask_sigint():
    """
    Returns:
        If called in main thread, returns a context where ``SIGINT`` is ignored, and yield True.
        Otherwise yield False.
    """
    if threading.current_thread() == threading.main_thread():
        sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        yield True

        signal.signal(signal.SIGINT, sigint_handler)
    else:
        yield False


def start_proc_mask_signal(proc):
    """start processes with SIGINT ignored

    Args:
        proc: multiprocessing process or processes

    Returns:

    Note:
        The signal mask is only applied when called from main thread.
    """
    if not isinstance(proc, list):
        proc = [proc]

    with mask_sigint():
        for p in proc:
            if isinstance(p, mp.Process):
                if mp.get_start_method() == 'fork':
                    msg = (
                        'Starting a process with fork method is efficient but not safe and may cause deadlock or crash,'
                        'Use forkserver or spawn method instead if you run into such issues'
                    )
                    logger.warning(msg)

            p.start()

