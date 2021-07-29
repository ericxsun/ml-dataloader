# ml-dataloader

**ml-dataloader** is an **efficient** and **flexible** data loading pipeline for deep learning, written in pure Python.


## Install

```
pip install --upgrade git+https://github.com/ericxsun/ml-dataloader.git

# or pip install ml-dataloader
```

## Examples

1. suppose data store in python list

```python
from dataloader.dataset import Dataset, DataKind
from dataloader.dataloader import DataLoader
from dataloader.pipeline import MapDataProcessKind

data = list(range(10))
kind = DataKind.MEM_SEQ
dataset = Dataset(data, kind, shuffle=False)

dl = DataLoader(dataset, batch_size=2, processor_kind=MapDataProcessKind.NORMAL)
for batch in dl:
    print(batch)

# tf.Tensor([0 1], shape=(2,), dtype=int32)
# tf.Tensor([2 3], shape=(2,), dtype=int32)
# tf.Tensor([4 5], shape=(2,), dtype=int32)
# tf.Tensor([6 7], shape=(2,), dtype=int32)
# tf.Tensor([8 9], shape=(2,), dtype=int32)
```

2. suppose `train.tsv` storing the data

```python
from dataloader.dataset import Dataset, DataKind
from dataloader.dataloader import DataLoader
from dataloader.pipeline import MapDataProcessKind

filename = 'train.tsv'
kind = DataKind.FILE
dataset = Dataset(filename, kind, shuffle=False)

dl = DataLoader(dataset, batch_size=2, processor_kind=MapDataProcessKind.MULTI_PROCESS, num_procs=20)
for batch in dl:
    print(batch)
```

**NOTES**:

1. the fully supported parameters, pls ref to [DataLoader](./dataloader/dataloader.py) definition

## Refs:

- [pytorch-data](https://github.com/pytorch/pytorch/tree/master/torch/utils/data)
- [tensorpack-dataflow](https://github.com/tensorpack/dataflow)
- [performance-tuning](https://github.com/tensorpack/tensorpack/blob/master/docs/tutorial/performance-tuning.md)
- [tensorpack-benchmark](https://github.com/tensorpack/benchmarks/blob/master/ResNet-Horovod/imagenet-resnet-horovod.py)