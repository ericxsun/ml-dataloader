#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from dataloader.io.image_reader import Cv2ImageReader

reader_registry = {
    'reader_image_cv2': Cv2ImageReader
}
