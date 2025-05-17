import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain
from typing import Callable

import egomimic
import einops
import jax
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
import tyro
from flax.traverse_util import flatten_dict, unflatten_dict
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import (
    ACT, ACTDecoder, ACTEncoder, ACTPolicy, ACTSinusoidalPositionEmbedding2d,
    create_sinusoidal_pos_embedding)
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from rich.pretty import pprint
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms.v2 import has_all

from bela.typ import Head, HeadSpec, Morph


def typespec(d):
    return jax.tree.map(lambda x: type(x), d)


def spec(d):
    return jax.tree.map(lambda x: x.shape, d)


@dataclass
class HybridConfig:

    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(repo_id="none")
    )
    human_repos: list[str] = field(default_factory=list)
    robot_repos: list[str] = field(default_factory=list)

    policy: ACTConfig = field(default_factory=ACTConfig)

from bela.common.policies.bela import BELA, BELAPolicy

def make_policy():

    batchspec = {
        "observation": {
            "robot": {
                "joints": PolicyFeature(FeatureType.STATE, (7,)),
                "image.side": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
                "image.wrist": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
                # "pose": PolicyFeature(FeatureType.STATE, (6,)),
                # "gripper": PolicyFeature(FeatureType.STATE, (1,)),
            },
            "human": {
                # "gripper": PolicyFeature(FeatureType.STATE, (1,)),
                "mano.hand_pose": PolicyFeature(
                    FeatureType.STATE, (15, 3)
                ),  # (15, 3, 3)),
                "mano.global_orient": PolicyFeature(
                    FeatureType.STATE, (3,)
                ),  # (3, 3)),
                "kp3d": PolicyFeature(FeatureType.STATE, (21, 3)),
            },
            "shared": {
                "image.low": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
                "cam.pose": PolicyFeature(FeatureType.STATE, (6,)),
            },
        },
    }

    def flatten_state_feat(x):
        if x.type == FeatureType.STATE:
            arr = np.zeros(x.shape).reshape(-1)
            return PolicyFeature(FeatureType.STATE, arr.shape)
        return x

    batchspec = jax.tree.map(flatten_state_feat, batchspec)

    # pprint(batchspec)
    input_features = flatten_dict(batchspec, sep=".")
    output_features = dict(input_features)
    output_features = jax.tree.map(
        lambda x: (
            PolicyFeature(FeatureType.ACTION, x.shape)
            if x.type == FeatureType.STATE
            else None
        ),
        output_features,
    )
    output_features = {k: v for k, v in output_features.items() if v is not None}
    output_features = {
        k.replace("observation.", "action."): v for k, v in output_features.items()
    }

    batchspec = batchspec | unflatten_dict(output_features, sep=".")

    state_features = {
        k: v for k, v in input_features.items() if v.type == FeatureType.STATE
    }
    pprint(batchspec)

    def compute_head(feat, head):
        headfeat = {k: v for k, v in feat if head in k}
        headfeat = {k: Head(None, v.shape) for k, v in headfeat.items()}
        return sum(list(headfeat.values()), Head(None, (0,)))

    headspec = HeadSpec(
        robot=Head(Morph.ROBOT, compute_head(state_features.items(), "robot").shape),
        human=Head(Morph.HUMAN, compute_head(state_features.items(), "human").shape),
        share=Head(Morph.HR, compute_head(state_features.items(), "shared").shape),
    )
    pprint(headspec)

    bs = 4
    chunk = 50

    def generate_example(x):
        shape = list(x.shape)
        if x.type == FeatureType.STATE:
            return np.zeros(([bs] + shape))
        if x.type == FeatureType.VISUAL:
            return np.zeros(([bs] + shape))
        if x.type == FeatureType.ACTION:
            return np.zeros(([bs, chunk] + shape))

    def generate_stats(x):
        return {
            "count": np.array(55),
            "max": np.random.random(x.shape),
            "min": np.random.random(x.shape),
            "mean": np.random.random(x.shape),
            "std": np.random.random(x.shape),
        }

    example_batch = jax.tree.map(generate_example, batchspec)
    example_batch = jax.tree.map(torch.Tensor, example_batch)
    example_batch = flatten_dict(example_batch, sep=".")

    example_batch["action_is_pad"] = torch.zeros((bs, chunk)).bool()

    example_stats = jax.tree.map(generate_stats, batchspec)
    example_stats = jax.tree.map(torch.Tensor, example_stats)
    is_leaf = lambda d, k: "count" in k
    example_stats = flatten_dict(example_stats, sep=".", is_leaf=is_leaf)
    pprint(spec(example_stats))

    policycfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk,
        n_action_steps=chunk,
    )
    pprint(policycfg)
    policy = BELAPolicy(
        config=policycfg,
        headspec=headspec,
        dataset_stats=example_stats,
    )

    policy(example_batch, heads=["human", "shared"])
    policy(example_batch, heads=["robot", "shared"])
    return policy

def main(cfg: HybridConfig):

    hdatasets, rdatasets = [], []
    for h in cfg.human_repos:
        cfg.dataset.repo_id = h
        hdatasets.append(make_dataset(cfg))
    for r in cfg.robot_repos:
        cfg.dataset.repo_id = r
        rdatasets.append(make_dataset(cfg))

    print(hdatasets)
    print(rdatasets)


if __name__ == "__main__":
    main(tyro.cli(HybridConfig))
