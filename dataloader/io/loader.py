#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np

from dataloader.io.reader import Reader
from dataloader.io.registry import get_reader_class
from dataloader.transform import Transform


class Loader(Transform):
    """load file or files from provided path based on reader"""

    def __init__(self, reader: Union[Reader, str], *args, dtype: np.dtype = np.float32, **kwargs):
        """

        Args:
            reader: register reader to load file
            dtype: if not None convert the loaded to this data type

            *args: additional parameters for reader if providing a reader name
            **kwargs: additional parameters for reader if providing a reader name
        """
        self._reader = reader

        if isinstance(reader, str):
            reader = reader.lower()
            self._reader = get_reader_class(reader)(*args, **kwargs)

        self.dtype = dtype

    def __call__(self, filename: Union[Sequence[str], str], reader: Optional[Reader] = None):
        """

        Args:
            filename: path file or file-like object or a list of files
                      Will be saved in meta_data with key `filename_or_obj`
            reader:

        Returns:
            ndarray:
        """
        if reader is None:
            reader = self._reader

        if reader is None:
            raise RuntimeError(f'can not find suitable reader for this file: {filename}')

        ndarray = reader.read(filename)
        ndarray = ndarray.astype(self.dtype)

        return ndarray
