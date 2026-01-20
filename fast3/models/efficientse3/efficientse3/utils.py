import torch
from torch_geometric.data import Data, Batch, Dataset
import lmdb
import contextlib
import gzip
import importlib
import json
import io
#import msgpack
import pickle as pkl
from pathlib import Path
import gc
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)

def split_dataset(D):
  n_train = int(len(D) * 0.9)
  n_val = len(D) - n_train
  n_test1 = 0
  n_test2 = 0
  return torch.utils.data.random_split(D, [n_train, n_val, n_test1, n_test2], generator=torch.Generator().manual_seed(42))

def to_torch(data, has_coords=False):
    node_features = torch.FloatTensor(data['x'])
    edge_index = torch.LongTensor(data['edge_index']).view(2, -1)
    y = torch.FloatTensor(data['affinity']).view(-1, 1)
    correct = torch.FloatTensor(data['correct']).view(-1, 1)
    name = data['name']
    d = Data(x=node_features, edge_index=edge_index, y=y, correct=correct, name=name)
    if has_coords:
        d.coords = torch.FloatTensor(data['coords'])
    else:
        d.edge_attr = torch.FloatTensor(data['edge_attr']).view(-1, 1)
    return d

class LMDBDataset(Dataset):
    """
    Creates a dataset from an lmdb file. Adapted from `TAPE <https://github.com/songlab-cal/tape/blob/master/tape/datasets.py>`_.

    :param data_file: path to LMDB file containing dataset
    :type data_file: Union[str, Path]
    :param transform: Transformation function to apply to each item.
    :type transform: Function, optional

    """

    def __init__(self, data_file, transform=None, use_cache=False, readahead=False):
        """constructor
        """
        if type(data_file) is list:
            if len(data_file) != 1:
                raise RuntimeError("Need exactly one filepath for lmdb")
            data_file = data_file[0]

        self.data_file = Path(data_file).absolute()
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)

        env = lmdb.open(str(self.data_file), max_readers=100, readonly=True,
                        lock=False, readahead=readahead, meminit=False)

        with env.begin(write=False) as txn:
            self._num_examples = int(txn.get(b'num_examples'))
            self._serialization_format = \
                txn.get(b'serialization_format').decode()

        self._env = env
        self._transform = transform
        self.cache = {}
        if use_cache:
            print('Using cache')
        self.use_cache = use_cache


    def __len__(self) -> int:
        return self._num_examples

    def get(self, i):
        return self[i]

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        if index in self.cache:
            return self.cache[index]
        with self._env.begin(write=False) as txn:
            compressed = txn.get(str(index).encode())
            buf = io.BytesIO(compressed)
            with gzip.GzipFile(fileobj=buf, mode="rb") as f:
                serialized = f.read()
            item = deserialize(serialized, self._serialization_format)
        if self._transform:
            item = self._transform(item)
        if self.use_cache:
            self.cache[index] = item
        return item

def deserialize(x, serialization_format):
    """
    Deserializes dataset `x` assuming format given by `serialization_format` (pkl, json, msgpack).
    """
    gc.disable()
    if serialization_format == 'pkl':
        return pkl.loads(x)
    elif serialization_format == 'json':
        serialized = json.loads(x)
    elif serialization_format == 'msgpack':
        serialized = msgpack.unpackb(x)
    else:
        raise RuntimeError('Invalid serialization format')
    gc.enable()
    return serialized

def serialize(x, serialization_format):
    """
    Serializes dataset `x` in format given by `serialization_format` (pkl, json, msgpack).
    """
    if serialization_format == 'pkl':
        # Pickle
        # Memory efficient but brittle across languages/python versions.
        return pkl.dumps(x)
    elif serialization_format == 'json':
        # JSON
        # Takes more memory, but widely supported.
        serialized = json.dumps(
            x, default=lambda df: json.loads(
                df.to_json(orient='split', double_precision=6))).encode()
    elif serialization_format == 'msgpack':
        # msgpack
        # A bit more memory efficient than json, a bit less supported.
        serialized = msgpack.packb(
            x, default=lambda df: df.to_dict(orient='split'))
    else:
        raise RuntimeError('Invalid serialization format')
    return serialized

def make_lmdb_dataset(dataset, output_lmdb, num_examples=None, filter_fn=None, serialization_format='json'):
    """
    Make an LMDB dataset from an input dataset.

    :param dataset: Input dataset to convert
    :type dataset: torch.utils.data.Dataset
    :param output_lmdb: Path to output LMDB.
    :type output_lmdb: Union[str, Path]
    :param filter_fn: Filter to decided if removing files.
    :type filter_fn: lambda x -> True/False
    :param serialization_format: How to serialize an entry.
    :type serialization_format: 'json', 'msgpack', 'pkl'
    :param include_bonds: Include bond information (only available for SDF yet).
    :type include_bonds: bool
    """
    if not num_examples:
    	num_examples = len(dataset)
    print(f'{num_examples} examples')
    env = lmdb.open(str(output_lmdb), map_size=int(1e11))

    with env.begin(write=True) as txn:
        try:
            i = 0
            for x in tqdm(dataset, total=num_examples):
                if filter_fn is not None and filter_fn(x):
                    continue
                buf = io.BytesIO()
                with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
                    f.write(serialize(x, serialization_format))
                compressed = buf.getvalue()
                result = txn.put(str(i).encode(), compressed, overwrite=True)
                if not result:
                    raise RuntimeError(f'LMDB entry {i} in {str(output_lmdb)} '
                                       'already exists')
                i += 1
        finally:
            txn.put(b'num_examples', str(i).encode())
            txn.put(b'serialization_format', serialization_format.encode())

def simpler(x):
    return torch.cat([x[:, 1:10], (x[:, 14] == 1).unsqueeze(-1), (x[:, 14] == -1).unsqueeze(-1)], dim=-1)

def non_interacting_edge_mask(edge_index, ligand_index):
    i = edge_index[0]
    j = edge_index[1]
    from_ligand = (i[..., None] == ligand_index).any(-1).squeeze()
    to_ligand = (j[..., None] == ligand_index).any(-1).squeeze()
    non_int_mask = from_ligand == to_ligand
    return non_int_mask
