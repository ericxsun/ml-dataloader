#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import atexit
from datetime import datetime
import os

_RNG_SEED = None


def fix_rng_seed(seed):
    """call at the beginning of program to fix rng seed

    Args:
        seed:

    Returns:

    """
    global _RNG_SEED
    _RNG_SEED = int(seed)


def get_rng(obj=None):
    """generate a good RNG with time, pid and the object

    Args:
        obj:

    Returns:
        np.random.RandomState: the RNG
    """
    import numpy as np

    seed = (id(obj) + os.getpid() + int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295

    return np.random.RandomState(seed)


# interval (in seconds) to check status of processes to avoid hanging in multiprocessing
MP_STATUS_CHECK_INTERVAL = 5.0

"""Whether python is shutting down, this flag is guaranteed to be set before the python core library resources are
freed, but python may already be exiting fro some time when this is set. 

Hook to set this flag is `set_python_exit_flag`, and is inspired by a similar
hook in Python 3.7 multiprocessing library:
https://github.com/python/cpython/blob/d4d60134b29290049e28df54f23493de4f1824b6/Lib/multiprocessing/util.py#L277-L327
"""
python_exit_status = False


def set_python_exit_flag():
    global python_exit_status
    python_exit_status = True


atexit.register(set_python_exit_flag)
