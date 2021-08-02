#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from typing import Callable
from typing import Optional

from dataloader.pipeline.batch import Batch
from dataloader.pipeline.batch import RepeatInBatch
from dataloader.pipeline.dataset import Dataset
from dataloader.pipeline.dataset import shuffle_dataset
from dataloader.pipeline.processor import MapData
from dataloader.pipeline.processor import MapDataProcessKind
from dataloader.pipeline.processor import MultiProcessMapDataZMQ
from dataloader.pipeline.processor import MultiThreadMapData
from dataloader.pipeline.runner import MultiProcessRunnerZMQ
from dataloader.transform import apply_transform

__all__ = ['DataLoader']


class DataLoader:
    def __init__(
        self, dataset: Dataset, batch_size, shuffle=False, drop_last=False,
        repeat_in_batch_kind='no', repeat_in_batch_time=0,
        transform: Optional[Callable] = None,
        to_tensor_func: Optional[Callable] = None,
        processor_kind: MapDataProcessKind = MapDataProcessKind.NORMAL,
        num_threads=20, num_procs=20, buffer_size=10,
        runner_num_procs=1
    ):
        """load data into batch

        Args:
            dataset: @see Dataset
            batch_size:
            shuffle: shuffle data
            drop_last: drop last batch or not if len(last_batch) < batch_size, default False
            repeat_in_batch_kind: repeat kind, @see RepeatInBatch
            repeat_in_batch_time: repeat times, @see RepeatInBatch
            processor_kind: what kind of method to process the data, @see MapDataProcessKind
            num_threads: num threads used in MultiThread process
            num_procs: num processes used in MultiProcess
            buffer_size: buffer size used in MultiThread/MultiProcess
            runner_num_procs: num processes to run the process in MultiProcessRunner
            transform: callable for transforming raw data to features
            to_tensor_func: callable for mapping features to tensor, default to_tf_tensor (@see to_tensor.to_tf_tensor)
        """
        dataset = shuffle_dataset(dataset, shuffle)

        if transform is None:
            transform = lambda e: e

        if processor_kind == MapDataProcessKind.NORMAL:
            dataset = MapData(dataset, map_func=lambda e: apply_transform(transform, e))
        elif processor_kind == MapDataProcessKind.MULTI_THREAD:
            dataset = MultiThreadMapData(
                dataset, num_threads=num_threads, buffer_size=buffer_size,
                map_func=lambda e: apply_transform(transform, e)
            )
        elif processor_kind == MapDataProcessKind.MULTI_PROCESS:
            dataset = MultiProcessMapDataZMQ(
                dataset, num_procs=num_procs, buffer_size=buffer_size,
                map_func=lambda e: apply_transform(transform, e)
            )
        else:
            raise ValueError(
                f'not supported processor_kind: {processor_kind}. Choose one from [{MapDataProcessKind.kinds_str}]'
            )

        dataset = MultiProcessRunnerZMQ(dataset, num_procs=runner_num_procs)

        repeat_in_batch = RepeatInBatch(repeat_in_batch_kind, repeat_in_batch_time)
        dataset = Batch(dataset, batch_size, drop_last, repeat_in_batch, to_tensor_func)

        self.dataset = dataset
        self.dataset.reset()

    def __iter__(self):
        yield from self.dataset.__iter__()

    def __len__(self):
        return len(self.dataset)
