# ml-dataloader

**ml-dataloader** is an **efficient** and **flexible** data loading pipeline for deep learning, written in pure Python.


## Install

`pip install ml-dataloader`


## Examples (similar to Pytorch-dataloader)

- suppose data store in python list

```python
from dataloader.dataset import Dataset
from dataloader.dataloader import DataLoader
from dataloader.util.data_kind import DataKind

data = list(range(10))
dataset = Dataset(data, kind=DataKind.MEM_SEQ)

dl = DataLoader(dataset, batch_size=2, shuffle=False)
for batch in dl:
    print(batch)

# tf.Tensor([0 1], shape=(2,), dtype=int32)
# tf.Tensor([2 3], shape=(2,), dtype=int32)
# tf.Tensor([4 5], shape=(2,), dtype=int32)
# tf.Tensor([6 7], shape=(2,), dtype=int32)
# tf.Tensor([8 9], shape=(2,), dtype=int32)
```

- data with index
```python
from dataloader.dataset import DatasetWithIndex
from dataloader.dataloader import DataLoader
from dataloader.util.data_kind import DataKind

data = list(range(10))
dataset = DatasetWithIndex(data, kind=DataKind.MEM_SEQ)

dl = DataLoader(dataset, batch_size=2, shuffle=False)
for batch in dl:
    print(batch)

# (tf.Tensor([0, 1], shape=(2,), dtype=int32), tf.Tensor([0, 1], shape=(2,), dtype=int32))
# (tf.Tensor([2, 3], shape=(2,), dtype=int32), tf.Tensor([2, 3], shape=(2,), dtype=int32))
# (tf.Tensor([4, 5], shape=(2,), dtype=int32), tf.Tensor([4, 5], shape=(2,), dtype=int32))
# (tf.Tensor([6, 7], shape=(2,), dtype=int32), tf.Tensor([6, 7], shape=(2,), dtype=int32))
# (tf.Tensor([8, 9], shape=(2,), dtype=int32), tf.Tensor([8, 9], shape=(2,), dtype=int32))
```

- suppose `train.tsv` storing the data

```python
from dataloader.dataset import Dataset
from dataloader.dataloader import DataLoader
from dataloader.util.data_kind import DataKind

filename = 'train.tsv'
dataset = Dataset(filename, kind=DataKind.FILE)

dl = DataLoader(dataset, batch_size=2, shuffle=True)
for batch in dl:
    print(batch)
```

- suppose `train.tsv` storing the data and using `mmap`

```python
from dataloader.dataset import Dataset
from dataloader.dataloader import DataLoader
from dataloader.util.data_kind import DataKind

filename = 'train.tsv'

dataset = Dataset(filename, kind=DataKind.MMAP_FILE)

dl = DataLoader(dataset, batch_size=2, shuffle=True)
for batch in dl:
    print(batch)
```

**NOTES**:

- if transform is slow, the dataloader will be stuck while num_workers > 0


## Examples with Pipeline (similar to Tensorpack-dataflow)

- suppose data store in python list

```python
from dataloader.pipeline.dataset import Dataset
from dataloader.pipeline.dataloader import DataLoader
from dataloader.pipeline.processor import MapDataProcessKind
from dataloader.util.data_kind import DataKind

data = list(range(10))
dataset = Dataset(data, kind=DataKind.MEM_SEQ)

dl = DataLoader(dataset, batch_size=2, shuffle=False, processor_kind=MapDataProcessKind.NORMAL)
for batch in dl:
    print(batch)

# tf.Tensor([0 1], shape=(2,), dtype=int32)
# tf.Tensor([2 3], shape=(2,), dtype=int32)
# tf.Tensor([4 5], shape=(2,), dtype=int32)
# tf.Tensor([6 7], shape=(2,), dtype=int32)
# tf.Tensor([8 9], shape=(2,), dtype=int32)
```

- suppose `train.tsv` storing the data

```python
from dataloader.pipeline.dataset import Dataset
from dataloader.pipeline.dataloader import DataLoader
from dataloader.pipeline.processor import MapDataProcessKind
from dataloader.util.data_kind import DataKind

filename = 'train.tsv'
dataset = Dataset(filename, kind=DataKind.FILE)

dl = DataLoader(dataset, batch_size=2, shuffle=True, processor_kind=MapDataProcessKind.MULTI_PROCESS, num_procs=20)
for batch in dl:
    print(batch)
```

- suppose `train.tsv` storing the data and using `mmap`

```python
from dataloader.pipeline.dataset import Dataset
from dataloader.pipeline.dataloader import DataLoader
from dataloader.pipeline.processor import MapDataProcessKind
from dataloader.util.data_kind import DataKind

filename = 'train.tsv'

dataset = Dataset(filename, kind=DataKind.MMAP_FILE)

dl = DataLoader(dataset, batch_size=2, shuffle=True, processor_kind=MapDataProcessKind.MULTI_PROCESS, num_procs=20)
for batch in dl:
    print(batch)
```

**NOTES**:

1. the fully supported parameters, pls ref to [DataLoader](https://github.com/ericxsun/ml-dataloader/blob/main/dataloader/dataloader.py) definition
2. with [MultiThreadMapData/MultiProcessMapDataZMQ](https://github.com/ericxsun/ml-dataloader/blob/main/dataloader/pipeline/processor.py), the order won't be kept as defined in dataset
3. in order to keep order as defined in `Dataset`, [MapData](https://github.com/ericxsun/ml-dataloader/blob/main/dataloader/pipeline/processor.py) can be used, but it could be slow compare with MultiThreadMapData and MultiProcessMapDataZMQ. Another way, process the data with [pool_transform](https://github.com/ericxsun/ml-dataloader/blob/main/dataloader/transform/misc.py), then pass the processed data as `DataKind.MEM_SEQ` kind into `Dataset`, i.e., `dataset = Dataset(processed, DataKind.MEM_SEQ)`, and avoid using `MultiThreadMapData/MultiProcessMapDataZMQ` 

## Refs:

- [pytorch-data](https://github.com/pytorch/pytorch/tree/master/torch/utils/data)
- [MONAI](https://github.com/Project-MONAI/MONAI)
- [tensorpack-dataflow](https://github.com/tensorpack/dataflow)
- [performance-tuning](https://github.com/tensorpack/tensorpack/blob/master/docs/tutorial/performance-tuning.md)
- [tensorpack-benchmark](https://github.com/tensorpack/benchmarks/blob/master/ResNet-Horovod/imagenet-resnet-horovod.py)

## FAQ

- `[__NSPlaceholderDate initialize] may have been in progress in another thread when fork()`

  if one meets such error, please set `OBJC_DISABLE_INITIALIZE_FORK_SAFETY` as yes: `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`

- publish to pypi (on mac)

  ```
  pip install pypandoc twine
  brew install pandoc
  
  python setup.py sdist
  twine upload --verbose --repository-url "https://upload.pypi.org/legacy/" -u "pypi-username" -p "pypi-password" dist/*
  ```