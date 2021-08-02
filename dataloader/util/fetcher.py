#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from dataloader.transform import apply_transform


class IterableDatasetFetcher:
    def __init__(self, dataset, is_batch, drop_last, transform):
        """

        Args:
            dataset:
            is_batch: sample index is  mini-batch or a single index
            drop_last: drop last batch if batch_size does not match
            transform: how to transform raw data into feature
        """
        self.dataset_iter = iter(dataset)
        self.is_batch = is_batch
        self.drop_last = drop_last
        self.transform = transform

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

        return data


class MapDatasetFetcher:
    def __init__(self, dataset, is_batch, drop_last, transform):
        """

        Args:
            dataset:
            is_batch: sample index is  mini-batch or a single index
            drop_last: drop last batch if batch_size does not match
            transform: how to transform raw data into feature
        """
        self.dataset = dataset
        self.is_batch = is_batch
        self.drop_last = drop_last
        self.transform = transform

    def fetch(self, possibly_batched_index):
        if self.is_batch:
            data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]

        data = apply_transform(self.transform, data)

        return data
