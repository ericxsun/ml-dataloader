#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import multiprocess as multiprocessing

from dataloader.transform import apply_transform


def pool_transform(batch, transform, processes=20):
    def pool_apply_transform(elem):
        return apply_transform(transform, elem)

    pool = multiprocessing.Pool(processes=processes)

    processed = pool.map(pool_apply_transform, batch)
    processed = list(processed)

    pool.close()
    pool.join()

    return processed
