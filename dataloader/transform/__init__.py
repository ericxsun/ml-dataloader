#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import abc
from typing import Any
from typing import Callable
from typing import Collection
from typing import Hashable
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

__all__ = [
    'apply_transform',
    'Transform', 'Randomize', 'ComposeTransform', 'DictTransform', 'LambdaTransform', 'LambdaDictTransform'
]

from dataloader import logger
from dataloader.util.misc import to_tuple


def apply_transform(transform: Callable, data):
    if transform is None:
        return data

    if isinstance(data, (list, tuple)):
        return [transform(item) for item in data]

    return transform(data)


class Transform:
    @abc.abstractmethod
    def __call__(self, data: Any):
        raise NotImplementedError(f'Subclass {self.__class__.__name__} must implement this method')


class Randomize:
    R: np.random.RandomState = np.random.RandomState()
    MAX_SEED = np.iinfo(np.uint32).max + 1

    def set_random_state(self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None):
        if seed is not None:
            _seed = id(seed) if not isinstance(seed, (int, np.integer)) else seed
            _seed = _seed % self.MAX_SEED
            self.R = np.random.RandomState(_seed)

            return self

        if state is not None:
            if not isinstance(state, np.random.RandomState):
                raise TypeError(f'state must be None or a np.random.RandomState but is {type(state).__name__}')
            self.R = state
            return self

        self.R = np.random.RandomState()
        return self

    @abc.abstractmethod
    def randomize(self, data: Any):
        raise NotImplementedError(f'Subclass {self.__class__.__name__} must implement this method')


class ComposeTransform(Randomize, Transform):
    """to chain a series of calls together in a sequence"""
    def __init__(self, transforms: Optional[Union[Sequence[Callable], Callable]] = None, seed: Optional[int] = None):
        if transforms is None:
            transforms = []

        self.transforms = to_tuple(transforms)
        self.set_random_state(seed)

    def set_random_state(self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None):
        super().set_random_state(seed=seed, state=state)

        for _transform in self.transforms:
            if not isinstance(_transform, Randomize):
                continue

            _transform.set_random_state(seed=self.R.randint(self.MAX_SEED, dtype='uint32'))

        return self

    def randomize(self, data: Optional[Any] = None):
        for _transform in self.transforms:
            if not isinstance(_transform, Randomize):
                continue

            try:
                _transform.randomize(data)
            except TypeError as type_error:
                name = type(_transform).__name__
                logger.warning(
                    f'Transform "{name}" in ComposeTransform not randomized\n{name}.{type_error}.', RuntimeWarning
                )

    def __call__(self, input_):
        for _transform in self.transforms:
            input_ = apply_transform(_transform, input_)

        return input_


class DictTransform(Transform):
    def __init__(self, keys: Union[Collection[Hashable], Hashable], *args, **kwargs):
        self.keys: Tuple[Hashable, ...] = to_tuple(keys)

        if not self.keys:
            raise ValueError('keys must be non empty')

        for key in self.keys:
            if not isinstance(key, Hashable):
                raise TypeError(f'keys must be one of (Hashable, Iterable[Hashable]) but is {type(keys).__name__}')

        self.args = args
        self.kwargs = kwargs

    @abc.abstractmethod
    def __call__(self, data):
        """

        Args:
            data: a python dict, data[key] is a numpy ndarray

        Returns:

        """
        raise NotImplementedError(f'Subclass {self.__class__.__name__} must implement this method')


class LambdaTransform(Transform):
    """apply user-defined lambda func as a transform

    Examples:
        >>> image = np.ones((10, 2, 2))
        >>> image.shape
        (10, 2, 2)
        >>> trans = LambdaTransform(func=lambda x: x[:4, :, :])
        >>> trans(image).shape
        (4, 2, 2)
    """
    def __init__(self, func: Optional[Callable] = None, **kwargs):
        if func is not None and not callable(func):
            raise TypeError(f'func must be None or callable but is {type(func).__name__}')

        self.func = func
        self.kwargs = kwargs

    def __call__(self, data, func: Optional[Callable] = None):
        if func is not None:
            if not callable(func):
                raise TypeError(f'func must be None or callable but is {type(func).__name__}')

            return func(data, **self.kwargs)

        if self.func is not None:
            return self.func(data, **self.kwargs)

        raise ValueError('incompatible values: func=None and self.func=None')


class LambdaDictTransform(DictTransform):
    """apply user-defined lambda func as a map transform(almost same as LambdaTransform but on dicted data)

    Examples:
        >>> data = {'image': np.random.random((10, 2, 2)), 'label': np.ones((10, 2, 2))}
        >>> trans = LambdaDictTransform(keys=['image', 'label'], func=lambda x: x[:4, :, :])
        >>> trans(data)['label'].shape
        (4, 2, 2)
    """
    def __init__(
        self,
        keys: Union[Collection[Hashable], Hashable],
        func: Union[Sequence[Callable], Callable],
        as_single=True,
        **kwargs
    ):
        """

        Args:
            keys: data fields to be transformed by `func`
            func: user-defined funcs
            as_single: treat each element in list/tuple individually, then func will be applied on each
            **kwargs: the params used in func
        """
        super().__init__(keys)

        self._as_single = as_single

        self.func = to_tuple(func)
        self.lambda_func = LambdaTransform(**kwargs)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            for func in self.func:
                if isinstance(d[key], (list, tuple)) and self._as_single:
                    d[key] = [self.lambda_func(x, func=func) for x in d[key]]
                elif isinstance(d[key], dict):
                    d[key] = {k: self.lambda_func(v, func=func) for k, v in d[key].items()}
                else:
                    d[key] = self.lambda_func(d[key], func=func)

        return d
