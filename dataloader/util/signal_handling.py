#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import signal
import threading

# Whether SIGCHLD handler is set for DataLoader worker failures
# Only one handler needs to be set for all DataLoaders in a process
_SIGCHLD_handler_set = False


def set_sigchld_handler():
    if not isinstance(threading.current_thread(), threading._MainThread):
        return

    global _SIGCHLD_handler_set
    if _SIGCHLD_handler_set:
        return

    previous_handler = signal.getsignal(signal.SIGCHLD)
    if not callable(previous_handler):
        previous_handler = None

    def handler(signum, frame):
        if previous_handler is not None:
            previous_handler(signum, frame)

    signal.signal(signal.SIGCHLD, handler)
    _SIGCHLD_handler_set = True
