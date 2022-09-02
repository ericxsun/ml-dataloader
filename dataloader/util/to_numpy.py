#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import collections
import re

import numpy as np

_np_str_obj_array_pattern = re.compile(r'[SaUO]')


map_np_dtypes = {
    type(1): np.int,
    type(1.0): np.float32,
}


def to_np_array(batch):
    """puts each data field into a tensor with outer dimension batch size

        Examples:
            >>> to_np_array([10, 20])
            array([10, 20])
            >>> to_np_array([1.0, 2.0])
            array([1., 2.], dtype=float32)
        """
    elem = batch[0]
    elem_type = type(elem)

    dtype = map_np_dtypes.get(elem_type, None)

    if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            if _np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(f'batch must be numbers, dicts or lists; found {elem.dtype}')

            return batch

        if elem.shape == ():  # scalars
            return np.asarray(batch, dtype=dtype)

    if isinstance(elem, float):
        return np.asarray(batch, dtype=dtype)

    if isinstance(elem, int):
        return np.asarray(batch, dtype=dtype)

    if isinstance(elem, (str, bytes)):
        return batch

    if isinstance(elem, collections.abc.Mapping):
        return {key: to_np_array([d[key] for d in batch]) for key in elem}

    if isinstance(elem, tuple):
        return elem_type(to_np_array(samples) for samples in zip(*batch))

    if isinstance(elem, collections.abc.Sequence):
        return np.asarray(batch, dtype=map_np_dtypes.get(type(elem[0]), None))

    raise TypeError(f'batch must be numbers, dicts or lists; found {elem.dtype}')
