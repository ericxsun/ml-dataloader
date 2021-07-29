#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import collections
import re

import numpy as np
import tensorflow as tf

_np_str_obj_array_pattern = re.compile(r'[SaUO]')


def to_tf_tensor(batch):
    """puts each data field into a tensor with outer dimension batch size

    Examples:
        >>> to_tf_tensor([10, 20])
        <tf.Tensor: shape=(2,), dtype=int32, numpy=array([10, 20], dtype=int32)>
    """
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, tf.Tensor):
        # 使用 tf.stack 和 tf.concat 在多线程加载数据时 (num_workers > 0) 会 卡住, 原因未知
        # mac python3.8 + tf2.4.1 正常, python3.7 + tf2.4.1/2.3.2 卡住
        # 线上物理机 python3.8 + tf.2.4.1 卡住
        return tf.convert_to_tensor(np.stack([b.numpy() for b in batch], axis=0))

    if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            if _np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    f'default_collate: batch must contain tensors, numpy arrays, '
                    f'numbers, dicts or lists; found {elem.dtype}'
                )

            return to_tf_tensor([tf.convert_to_tensor(b) for b in batch])

        if elem.shape == ():  # scalars
            return tf.convert_to_tensor(batch)

    if isinstance(elem, float):
        return tf.convert_to_tensor(batch, dtype=tf.float64)

    if isinstance(elem, int):
        return tf.convert_to_tensor(batch)

    if isinstance(elem, (str, bytes)):
        return batch

    if isinstance(elem, collections.abc.Mapping):
        return {key: to_tf_tensor([d[key] for d in batch]) for key in elem}

    if isinstance(elem, tuple):
        return elem_type(to_tf_tensor(samples) for samples in zip(*batch))

    if isinstance(elem, collections.abc.Sequence):
        return tf.convert_to_tensor(batch)

    raise TypeError(
        f'default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {elem_type}'
    )
