#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from dataloader.util.fetcher import IterableDatasetFetcher
from dataloader.util.fetcher import MapDatasetFetcher


class DatasetKind:
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, is_batch, drop_last, transform):
        """get data

        Args:
            kind: Map or Iterable
            dataset:
            is_batch: sample index is  mini-batch or a single index
            drop_last: drop last batch if batch_size does not match
            transform: how to transform raw data into feature

        Returns:
            Fetcher
        """
        if kind == DatasetKind.Map:
            return MapDatasetFetcher(dataset, is_batch, drop_last, transform)

        return IterableDatasetFetcher(dataset, is_batch, drop_last, transform)
