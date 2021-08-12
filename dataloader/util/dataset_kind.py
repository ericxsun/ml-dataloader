#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from dataloader.util.fetcher import IterableDatasetFetcher
from dataloader.util.fetcher import MapDatasetFetcher


class DatasetKind:
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, is_batch, drop_last, fn_to_tensor, transform, repeat_in_batch):
        """get data

        Args:
            kind: Map or Iterable
            dataset:
            is_batch: sample index is  mini-batch or a single index
            drop_last: drop last batch if batch_size does not match
            fn_to_tensor: how to transform data into tensor
            transform: how to transform raw data into feature
            repeat_in_batch: repeat data item in batch

        Returns:
            Fetcher
        """
        if kind == DatasetKind.Map:
            return MapDatasetFetcher(dataset, is_batch, drop_last, fn_to_tensor, transform, repeat_in_batch)

        return IterableDatasetFetcher(dataset, is_batch, drop_last, fn_to_tensor, transform, repeat_in_batch)
