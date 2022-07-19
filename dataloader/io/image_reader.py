#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from typing import List
from typing import Sequence
from typing import Union

import numpy as np

from dataloader.io.reader import Reader
from dataloader.util.misc import to_tuple
from dataloader.io.registry import register_reader


@register_reader('reader_image_cv2')
class Cv2ImageReader(Reader):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.args = args
        self.kwargs = kwargs

    def read(self, data: Union[Sequence[str], str], **kwargs):
        """Read image data from specified file or files

        Args:
            data:
            **kwargs:

        Returns:
            ndarray: stacked image ndarray
        """
        import cv2

        images: List[np.ndarray] = list()

        filenames: Sequence[str] = to_tuple(data)

        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)

        for name in filenames:
            image = cv2.imread(name, **kwargs_)
            images.append(image)

        ndarray = np.stack(images, axis=0) if len(images) > 1 else images[0]

        return ndarray
