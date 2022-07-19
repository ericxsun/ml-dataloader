#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import collections
import sys
import traceback
from typing import Any
from typing import Tuple

import numpy as np

from dataloader import logger

MAX_SEED = np.iinfo(np.uint32).max + 1


def _is_sequence_iterable(obj) -> bool:
    return isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str)


def to_tuple(values: Any) -> Tuple[Any, ...]:
    if not _is_sequence_iterable(values):
        values = (values, )

    return tuple(values)


def bytes_to_str(text):
    if text is not None and isinstance(text, bytes):
        try:
            return text.decode('utf-8')
        except UnicodeDecodeError as e:
            logger.error(f'decode failed text={text}: {e}')
            raise e

    return text


class KeyErrorMessage(str):
    """str subclass that returns itself in repr"""
    def __repr__(self):
        return self


class ExceptionWrapper(object):
    """Wraps an exception plus traceback to communicate across threads"""
    def __init__(self, exc_info=None, where='in background'):
        if exc_info is None:
            exc_info = sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = ''.join(traceback.format_exception(*exc_info))
        self.where = where

    def re_raise(self):
        r"""Re-raises the wrapped exception in the current thread"""
        msg = f'Caught {self.exc_type.__name__} {self.where}.\n Original {self.exc_msg}'

        if self.exc_type == KeyError:
            msg = KeyErrorMessage(msg)
        elif getattr(self.exc_type, 'message', None):
            raise self.exc_type(message=msg)

        try:
            exception = self.exc_type(msg)
        except TypeError:
            raise RuntimeError(msg) from None

        raise exception


def set_rnd(obj, seed):
    """

    Args:
        obj:
        seed:

    Returns:

    """
    if not hasattr(obj, '__dict__'):
        return seed

    if hasattr(obj, 'set_random_state'):
        obj.set_random_state(seed=seed % MAX_SEED)
        return seed + 1

    for key in obj.__dict__:
        if key.startswith('__'):
            continue
        seed = set_rnd(obj.__dict__[key], seed=seed)

    return seed


def get_offset(filename):
    logger.debug(f'loading offset from {filename}')
    with open(filename, 'rb') as fd:
        offset = [0]
        while fd.readline():
            offset.append(fd.tell())

        offset = offset[:-1]

    n_data = len(offset)
    logger.debug(f'loading offset done: n_offset={n_data}')

    return offset, n_data


def get_from_registry(key: str, registry):
    if hasattr(key, 'lower'):
        key = key.lower()

    value = registry.get(key, None)
    if value is not None:
        return value

    raise ValueError(f'key "{key}" not supported, available options: {registry.keys()}')
