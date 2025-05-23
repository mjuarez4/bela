from typing import Any

from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
import torch
from torch.utils.data.distributed import DistributedSampler


def make_dataloaders(
    datasets: dict[str, Any],
    cfg,
    is_distributed: bool,
    rank: int,
    world_size: int,
    device: torch.device,
):
    """Return dataloader iterators for each dataset."""
    dl_iters, samplers = {}, {}
    bsz = cfg.batch_size // world_size if is_distributed else cfg.batch_size
    for head, dataset in datasets.items():
        if is_distributed:
            if hasattr(cfg.policy, "drop_n_last_frames"):
                episodic_sampler = EpisodeAwareSampler(
                    dataset.episode_data_index,
                    drop_n_last_frames=cfg.policy.drop_n_last_frames,
                    shuffle=True,
                )
                sampler = DistributedSampler(
                    episodic_sampler,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=cfg.seed or 0,
                )
            else:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=cfg.seed or 0,
                )
            shuffle = False
        else:
            if hasattr(cfg.policy, "drop_n_last_frames"):
                shuffle = False
                sampler = EpisodeAwareSampler(
                    dataset.episode_data_index,
                    drop_n_last_frames=cfg.policy.drop_n_last_frames,
                    shuffle=True,
                )
            else:
                shuffle = True
                sampler = None
        samplers[head] = sampler
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            prefetch_factor=4,
            batch_size=bsz,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=device.type != "cpu",
            drop_last=False,
        )
        dl_iters[head] = cycle(loader)
    return dl_iters, samplers
