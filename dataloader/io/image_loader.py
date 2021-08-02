#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from typing import Collection
from typing import Hashable
from typing import Optional
from typing import Union

import numpy as np

from dataloader.io.loader import Loader
from dataloader.io.reader import Reader
from dataloader.transform import DictTransform


class LoadDictImage(DictTransform):
    def __init__(
        self,
        keys: Union[Collection[Hashable], Hashable],
        *args,
        reader: Optional[Union[Reader, str]] = None,
        dtype: Optional[np.dtype] = np.float32,
        **kwargs,
    ):
        """loading images with data format like {'image_1': image_path_1, 'image_2': image_path_2, ...}

        Args:
            keys:

            reader:
            dtype:

            *args: additional parameters for reader if providing a reader name
            **kwargs: additional parameters for reader if providing a reader name
        """
        super().__init__(keys)

        self._loader = Loader(reader, *args, dtype, **kwargs)

    def __call__(self, data, reader: Optional[Reader] = None):
        """

        Args:
            data:
            reader:

        Returns:
            the loaded image
        """
        res = dict(data)
        for key in self.keys:
            res[key] = self._loader(res[key], reader)

        return res
