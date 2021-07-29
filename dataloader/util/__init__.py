#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from datetime import datetime
import os

import numpy as np

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
    seed = (id(obj) + os.getpid() + int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295

    if _RNG_SEED is not None:
        seed = _RNG_SEED

    return np.random.RandomState(seed)
