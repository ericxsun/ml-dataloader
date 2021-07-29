#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import collections
from typing import Any
from typing import Tuple

from dataloader import logger


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
