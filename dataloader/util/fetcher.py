#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from dataloader.transform import apply_transform
from dataloader.util.batch import RepeatInBatch


class IterableDatasetFetcher:
    def __init__(self, dataset, is_batch, drop_last, fn_to_tensor, transform, repeat_in_batch: RepeatInBatch):
        """

        Args:
            dataset:
            is_batch: sample index is  mini-batch or a single index
            drop_last: drop last batch if batch_size does not match
            fn_to_tensor: how to transform data into tensor
            transform: how to transform raw data into feature
            repeat_in_batch: repeat data item in batch
        """
        self.dataset_iter = iter(dataset)
        self.is_batch = is_batch
        self.drop_last = drop_last
        self.fn_to_tensor = fn_to_tensor
        self.transform = transform
        self.repeat_in_batch = repeat_in_batch

    def fetch(self, possibly_batched_index):
        if self.is_batch:
            data = []

            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    break

            if len(data) == 0 or (self.drop_last and len(data) < len(possibly_batched_index)):
                raise StopIteration
        else:
            data = next(self.dataset_iter)

        data = apply_transform(self.transform, data)

        if self.is_batch:
            data = self.repeat_in_batch.repeat(data)

        data = self.fn_to_tensor(data)

        return data


class MapDatasetFetcher:
    def __init__(self, dataset, is_batch, drop_last, fn_to_tensor, transform, repeat_in_batch: RepeatInBatch):
        """

        Args:
            dataset:
            is_batch: sample index is  mini-batch or a single index
            drop_last: drop last batch if batch_size does not match
            fn_to_tensor: how to transform data into tensor
            transform: how to transform raw data into feature
            repeat_in_batch: repeat data item in batch
        """
        self.dataset = dataset
        self.is_batch = is_batch
        self.drop_last = drop_last
        self.fn_to_tensor = fn_to_tensor
        self.transform = transform
        self.repeat_in_batch = repeat_in_batch

    def fetch(self, possibly_batched_index):
        if self.is_batch:
            data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]

        data = apply_transform(self.transform, data)

        if self.is_batch:
            data = self.repeat_in_batch.repeat(data)

        data = self.fn_to_tensor(data)

        return data
