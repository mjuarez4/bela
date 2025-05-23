import importlib
import sys
import types
from pathlib import Path

import pytest

@pytest.fixture()
def stub_env(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    monkeypatch.setitem(sys.modules, 'bela.common.dataset', types.ModuleType('bela.common.dataset'))

    torch_stub = types.ModuleType('torch')
    class device:
        def __init__(self, typ):
            self.type = typ
    torch_stub.device = device
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
    torch_stub.utils = utils_mod
    for name, mod in [
        ('torch', torch_stub),
        ('torch.utils', utils_mod),
        ('torch.utils.data', data_mod),
        ('torch.utils.data.distributed', distr_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

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

    yield device, EpisodeAwareSampler, DistributedSampler

def _import_dataloader():
    sys.modules.pop('bela.common.dataloader', None)
    return importlib.import_module('bela.common.dataloader')

def test_make_dataloaders_non_distributed(stub_env):
    device_cls, Epis, _ = stub_env
    dl = _import_dataloader()
    class Dataset(list):
        pass
    ds = Dataset([0, 1, 2, 3])
    ds.episode_data_index = ds
    cfg = types.SimpleNamespace(batch_size=2, num_workers=0, policy=types.SimpleNamespace(drop_n_last_frames=1), seed=0)
    iters, samplers = dl.make_dataloaders({'train': ds}, cfg, False, 0, 1, device_cls('cpu'))
    assert next(iter(iters['train'])) == 0
    assert isinstance(samplers['train'], Epis)

def test_make_dataloaders_distributed(stub_env):
    device_cls, _, Dist = stub_env
    dl = _import_dataloader()
    class Dataset(list):
        pass
    ds = Dataset([0, 1, 2, 3])
    ds.episode_data_index = ds
    cfg = types.SimpleNamespace(batch_size=2, num_workers=0, policy=types.SimpleNamespace(drop_n_last_frames=1), seed=0)
    iters, samplers = dl.make_dataloaders({'train': ds}, cfg, True, 0, 2, device_cls('cpu'))
    assert isinstance(samplers['train'], Dist)
