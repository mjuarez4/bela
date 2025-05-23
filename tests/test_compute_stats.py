import importlib
import sys
import types
from pathlib import Path
import pytest


@pytest.fixture()
def stub_env(monkeypatch, tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    monkeypatch.setitem(sys.modules, 'bela.common.dataset', types.ModuleType('bela.common.dataset'))

    numpy_mod = types.ModuleType('numpy')
    def eye(n):
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    numpy_mod.eye = eye
    numpy_mod.load = lambda p: {'base': eye(4)}
    numpy_mod.savez = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'numpy', numpy_mod)

    jax_mod = types.ModuleType('jax')
    class _Path:
        def __init__(self, key):
            self.key = key
    def _map(fn, *trees):
        if isinstance(trees[0], dict):
            return {k: _map(fn, *[t[k] for t in trees]) for k in trees[0]}
        return fn(*trees)
    def _map_path(fn, tree, path=()):
        if isinstance(tree, dict):
            return {k: _map_path(fn, v, path + (_Path(k),)) for k, v in tree.items()}
        return fn(path, tree)
    jax_mod.tree = types.SimpleNamespace(map=_map, map_with_path=_map_path)
    monkeypatch.setitem(sys.modules, 'jax', jax_mod)

    torch_mod = types.ModuleType('torch')
    class Tensor:
        def __init__(self, data):
            self.data = data
        @property
        def shape(self):
            if isinstance(self.data[0], list):
                return (len(self.data), len(self.data[0]))
            return (len(self.data),)
        def mean(self, dim=0):
            cols = list(zip(*self.data))
            return Tensor([sum(c)/len(c) for c in cols])
        def std(self, dim=0):
            m = self.mean(dim).data
            cols = list(zip(*self.data))
            return Tensor([ (sum((x-m[i])**2 for x in col)/len(col))**0.5 for i, col in enumerate(cols)])
        def max(self, dim=0):
            cols = list(zip(*self.data))
            return Tensor([max(c) for c in cols]), None
        def min(self, dim=0):
            cols = list(zip(*self.data))
            return Tensor([min(c) for c in cols]), None
        def __getitem__(self, idx):
            val = self.data[idx]
            return Tensor(val) if isinstance(val, list) else val
    torch_mod.Tensor = Tensor
    def _concat(seq, dim=0):
        data = []
        for t in seq:
            data.extend(t.data)
        return Tensor(data)
    torch_mod.cat = _concat
    torch_mod.concatenate = _concat
    torch_mod.stack = lambda seq, dim=0: Tensor([t.data for t in seq])
    def _zeros(*shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return Tensor([[0] * shape[-1] for _ in range(shape[0])])
    torch_mod.zeros = _zeros
    class device:
        def __init__(self, typ):
            self.type = typ
    torch_mod.device = device
    utils_mod = types.ModuleType('torch.utils')
    data_mod = types.ModuleType('torch.utils.data')
    class DataLoader:
        def __init__(self, dataset, num_workers=0, prefetch_factor=4, batch_size=1, shuffle=False, sampler=None, pin_memory=False, drop_last=False):
            self.dataset = dataset
            self.batch_size = batch_size
            self.sampler = sampler
            self.shuffle = shuffle
        def __iter__(self):
            idxs = list(range(len(self.dataset))) if self.sampler is None else list(iter(self.sampler))
            if self.shuffle:
                import random
                random.shuffle(idxs)
            for i in idxs:
                yield self.dataset[i]
    data_mod.DataLoader = DataLoader
    distr_mod = types.ModuleType('torch.utils.data.distributed')
    class DistributedSampler:
        def __init__(self, data, num_replicas=None, rank=None, shuffle=True, seed=0):
            self.data = data
        def __iter__(self):
            return iter(range(len(self.data)))
        def __len__(self):
            return len(self.data)
    distr_mod.DistributedSampler = DistributedSampler
    data_mod.distributed = distr_mod
    utils_mod.data = data_mod
    torch_mod.utils = utils_mod
    for name, mod in [
        ('torch', torch_mod),
        ('torch.utils', utils_mod),
        ('torch.utils.data', data_mod),
        ('torch.utils.data.distributed', distr_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    trans_mod = types.ModuleType('pytorch3d.transforms')
    trans_mod.matrix_to_rotation_6d = lambda x: x
    monkeypatch.setitem(sys.modules, 'pytorch3d.transforms', trans_mod)

    xgym_mod = types.ModuleType('xgym')
    xgym_mod.BASE = tmp_path
    monkeypatch.setitem(sys.modules, 'xgym', xgym_mod)
    robot_mod = types.ModuleType('xgym.calibrate.urdf.robot')
    class RobotTree:
        def set_pose(self, joints):
            class Link:
                def get_matrix(self):
                    return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
            return {'link_eef': Link()}
    robot_mod.RobotTree = RobotTree
    monkeypatch.setitem(sys.modules, 'xgym.calibrate.urdf.robot', robot_mod)

    sampler_mod = types.ModuleType('lerobot.common.datasets.sampler')
    class EpisodeAwareSampler:
        def __init__(self, data_index, drop_n_last_frames=None, shuffle=True):
            self.data = data_index
        def __iter__(self):
            return iter(range(len(self.data)))
        def __len__(self):
            return len(self.data)
    sampler_mod.EpisodeAwareSampler = EpisodeAwareSampler

    utils2_mod = types.ModuleType('lerobot.common.datasets.utils')
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
    utils2_mod.cycle = cycle

    datasets_mod = types.ModuleType('lerobot.common.datasets')
    datasets_mod.sampler = sampler_mod
    datasets_mod.utils = utils2_mod
    common_mod = types.ModuleType('lerobot.common')
    common_mod.datasets = datasets_mod
    lerobot_mod = types.ModuleType('lerobot')
    lerobot_mod.common = common_mod
    for name, mod in [
        ('lerobot', lerobot_mod),
        ('lerobot.common', common_mod),
        ('lerobot.common.datasets', datasets_mod),
        ('lerobot.common.datasets.sampler', sampler_mod),
        ('lerobot.common.datasets.utils', utils2_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    tqdm_mod = types.ModuleType('tqdm')
    tqdm_mod.tqdm = lambda it, total=None, desc=None, leave=None: it
    monkeypatch.setitem(sys.modules, 'tqdm', tqdm_mod)

    lr_ds_mod = types.ModuleType('lerobot.common.datasets.lerobot_dataset')
    class LeRobotDataset(list):
        pass
    lr_ds_mod.LeRobotDataset = LeRobotDataset
    monkeypatch.setitem(sys.modules, 'lerobot.common.datasets.lerobot_dataset', lr_ds_mod)

    yield device, EpisodeAwareSampler, DistributedSampler


def _import_util():
    sys.modules.pop('bela.common.datasets.util', None)
    return importlib.import_module('bela.common.datasets.util')


def test_compute_stats(stub_env, monkeypatch):
    util = _import_util()

    def simple_postprocess(batch, batchspec, head, flat=True):
        batch['heads'] = [head, 'shared']
        batch['action_is_pad'] = util.torch.zeros( (len(batch['x'].data), 1) )
        return batch
    monkeypatch.setattr(util, 'postprocess', simple_postprocess)

    dataset = [
        {'x': util.torch.Tensor([[1, 2]])},
        {'x': util.torch.Tensor([[3, 4]])},
        {'x': util.torch.Tensor([[5, 6]])},
        {'x': util.torch.Tensor([[7, 8]])},
    ]
    stats = util.compute_stats(dataset, {}, head='human', batch_size=1, num_workers=0)
    assert stats['x']['mean'].data == [4.0, 5.0]
    assert stats['x']['count'] == 4
