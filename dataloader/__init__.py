#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import logging
import os
import sys

LOG_FMT = '[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s'

logging.getLogger('tensorflow').setLevel(os.getenv('LOG_TF', 'ERROR'))

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv('LOG_DL', 'CRITICAL'))
logger.propagate = False

formatter = logging.Formatter(LOG_FMT, datefmt='%Y-%m-%d,%H:%M:%S')

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)

logger.addHandler(handler)
