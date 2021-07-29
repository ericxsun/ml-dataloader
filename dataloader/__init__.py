#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import logging
import os
import sys

logging_level_registry = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    None: logging.INFO
}

LOG_FMT = '[%(levelname)1.1s %(asctime)s.%(msecs)03d %(module)s:%(lineno)d] %(message)s'

logging_level = logging_level_registry[os.getenv('DL_LOGGING_LEVEL')]

logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
logger.propagate = False

formatter = logging.Formatter(LOG_FMT)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)

logger.addHandler(handler)
