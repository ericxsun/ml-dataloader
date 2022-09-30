#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import collections
import re

_np_str_obj_array_pattern = re.compile(r'[SaUO]')


def to_pt_tensor(batch):
    """puts each data field into a tensor with outer dimension batch size

    Examples:
        >>> to_pt_tensor([10, 20])
        tensor([10, 20])
    """
    import torch

    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)

    if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if _np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(f'batch must be tensors, numpy arrays, numbers, dicts or lists; found {elem.dtype}')

            return to_pt_tensor([torch.as_tensor(b) for b in batch])

        if elem.shape == ():  # scalars
            return torch.as_tensor(batch)

    if isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)

    if isinstance(elem, int):
        return torch.tensor(batch)

    if isinstance(elem, (str, bytes)):
        return batch

    if isinstance(elem, collections.abc.Mapping):
        return elem_type({key: to_pt_tensor([d[key] for d in batch]) for key in elem})

    if isinstance(elem, tuple):
        return elem_type(to_pt_tensor(samples) for samples in zip(*batch))

    if isinstance(elem, collections.abc.Sequence):
        return torch.as_tensor(batch)

    raise TypeError(f'batch must be tensors, numpy arrays, numbers, dicts or lists; found {elem.dtype}')
