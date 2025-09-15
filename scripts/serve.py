#!/usr/bin/env python
import json
import logging
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Callable

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import (add_envs_task,
                                       check_env_attributes_and_types,
                                       preprocess_observation)
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (get_safe_torch_device, init_logging,
                                        inside_slurm)
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.types import NormalizationMode

import inspect
from dataclasses import is_dataclass, fields
def _allowed_keys_for(cls):
    """Return the set of constructor / dataclass field names for cls."""
    if is_dataclass(cls):
        return {f.name for f in fields(cls)}
    try:
        return set(inspect.signature(cls).parameters.keys())
    except (TypeError, ValueError):
        return set()

def _filter_kwargs_for(cls, data: dict) -> dict:
    """Keep only keys that cls actually accepts."""
    allowed = _allowed_keys_for(cls)
    return {k: v for k, v in data.items() if k in allowed}

def _find_model_cfg_path(root: Path) -> Path:
    """Find config.json across common checkpoint layouts."""
    candidates = [
        root / "pretrained_model" / "config.json",
        root / "config.json",
        root / "pretrained_model" / "pretrained_model" / "config.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "config.json not found. Tried:\n- " + "\n- ".join(map(str, candidates))
    )

def _load_act_config(ckpt_root: Path):
    """Load model config, drop HF-only keys, and build ACTConfig safely."""
    cfg_path = _find_model_cfg_path(ckpt_root)
    with open(cfg_path, "r") as f:
        raw = json.load(f)

    # Drop known HF / hub / metadata keys ONLY (keep device / use_amp if present)
    DROP = {
        "push_to_hub", "repo_id", "hub_model_id", "hub_private_repo", "hub_strategy",
        "tags", "license", "_name_or_path", "transformers_version",
        "torch_dtype", "architectures", "id2label", "label2id", "type",
    }
    for k in list(raw.keys()):
        if k in DROP:
            raw.pop(k, None)

    filtered = _filter_kwargs_for(ACTConfig, raw)
    dropped = sorted(set(raw) - set(filtered))
    if dropped:
        logging.info("[serve] Dropping unsupported ACTConfig keys: %s", dropped)

    act_cfg = ACTConfig(**filtered)

    # Coerce enum values if present
    if getattr(act_cfg, "normalization_mapping", None):
        act_cfg.normalization_mapping = {
            k: NormalizationMode(v) for k, v in act_cfg.normalization_mapping.items()
        }

    # Folder that actually holds model.safetensors, etc.
    act_cfg.pretrained_path = cfg_path.parent
    return act_cfg, cfg_path



#@parser.wrap()
# if __name__ == "__main__":
# init_logging()
# main()


import dataclasses
import enum
import logging
import socket

import tyro
from webpolicy.deploy import base_policy as _base_policy
from webpolicy.deploy.server import WebsocketPolicyServer

# from openpi.policies import policy as _policy
# from openpi.policies import policy_config as _policy_config

# from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    IRL = "irl"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    dir: str  # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    config: str | None = None  # Training config name (e.g., "pi0_aloha_sim").


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


from lerobot.configs.policies import PreTrainedConfig


@dataclasses.dataclass
class ServeConfig:
    """Arguments for the serve_policy script."""

    ckpt: Checkpoint

    # Environment to serve the policy for. This is only used when serving default policies.
    # env: EnvMode = EnvMode.IRL

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    host: str = "0.0.0.0"
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    # ckpt: Checkpoint | Default = dataclasses.field(default_factory=Default)

    def __post_init__(self):
        pass
        # self._policy =
        # policy_path = parser.get_path_arg("policy")
        # if policy_path:
        # cli_overrides = parser.get_cli_overrides("policy")


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="s3://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="s3://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="s3://openpi-assets/checkpoints/pi0_fast_libero",
    ),
}


# def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
# """Create a default policy for the given environment."""
# if checkpoint := DEFAULT_CHECKPOINT.get(env):
# return _policy_config.create_trained_policy(
# _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
# )
# raise ValueError(f"Unsupported environment mode: {env}")


# def create_policy(args: ServeConfig) -> _policy.Policy:
# """Create a policy from the given arguments."""
# match args.policy:
# case Checkpoint():
# return _policy_config.create_trained_policy(
# _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
# )
# case Default():
# return create_default_policy(args.env, default_prompt=args.default_prompt)


import jax
from rich.pretty import pprint


class LeRobotPolicy(_base_policy.BasePolicy):

    def __init__(self, policy: PreTrainedPolicy, steps) -> None:
        policy.eval()
        self.policy = policy
        self.steps = steps

    def infer(self, obs: dict[str, Tensor]) -> dict:
        if "reset" in obs and obs["reset"]:
            self.reset()
            return {"reset": True}

        spec = lambda arr: jax.tree.map(lambda x: x.shape, arr)
        # pprint(spec(obs))
        obs = preprocess_observation(obs)

        # hack
        obs["observation.image.side"] = obs.pop("observation.images.side")
        obs["observation.image.low"] = obs.pop("observation.images.low")
        obs["observation.image.wrist"] = obs.pop("observation.images.wrist")

        # pprint(spec(obs))
        to = lambda x: x.to(self.device, non_blocking=self.device.type == "cuda")
        obs = {key: to(obs[key]) for key in obs}
        # obs = add_envs_task(env, obs)

        with torch.inference_mode():
            actions = []
            for _ in range(self.steps):
                action = self.policy.select_action(obs)
                action = action.to("cpu").numpy()
                assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"
                actions.append(action)
            actions = np.array(actions).reshape(self.steps, -1)
        return {"action": actions}

    def reset(self, *args, **kwargs):
        return self.policy.reset()


class PolicyWrapper(_base_policy.BasePolicy):
    """A wrapper for a policy."""

    wrapper = True

    def __init__(self, policy: _base_policy.BasePolicy) -> None:
        self.policy = policy

    def infer(self, obs: dict[str, Tensor]) -> dict:
        self.policy.infer(obs)

    def reset(self, *args, **kwargs):
        return self.policy.reset(*args, **kwargs)


class ShapeValidatorPolicy(PolicyWrapper):
    """A policy that validates the shape of the given policy."""

    def __init__(self, policy: _base_policy.BasePolicy, expected_shape: tuple) -> None:
        self.policy = policy
        self.expected_shape = expected_shape

    def infer(self, obs: dict[str, Tensor]) -> dict:
        action = self.policy.infer(obs)
        if action.shape != self.expected_shape:
            raise ValueError(
                f"Action shape {action.shape} does not match expected shape {self.expected_shape}"
            )
        return action


class TryPolicy(_base_policy.BasePolicy):
    """A policy that tries to run the given policy."""

    def __init__(self, policy: _base_policy.BasePolicy) -> None:
        self.policy = policy

    def infer(self, obs: dict[str, Tensor]) -> dict:
        try:
            action = self.policy.infer(obs)
            return action
        except Exception as e:
            logging.error("Error in policy: %s", e)
            import traceback

            traceback.print_exc()

        return None

    def reset(self, *args, **kwargs):
        return self.policy.reset()


from pathlib import Path

from rich.pretty import pprint

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.transforms import ImageTransformsConfig
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig

"""
def main(cfg: ServeConfig) -> None:

    ckpt_file = Path(cfg.ckpt.dir) / "pretrained_model" / "config.json"
    ptm_file = Path(cfg.ckpt.dir) / "pretrained_model"
    with open(ckpt_file, "r") as f:
        ckpt = json.load(f)
    del ckpt["type"]
    policycfg = ACTConfig(**ckpt)

    policycfg.normalization_mapping = {
        k: NormalizationMode(v) for k, v in policycfg.normalization_mapping.items()
    }

    pprint(ckpt)
    policycfg.pretrained_path = ptm_file
    pprint(policycfg)

    train_file = Path(cfg.ckpt.dir) / "pretrained_model" / "train_config.json"
    with open(train_file, "r") as f:
        ckpt = json.load(f)

    traincfg = TrainPipelineConfig(**ckpt)
    traincfg.dataset = DatasetConfig(**traincfg.dataset)
    traincfg.dataset.image_transforms = ImageTransformsConfig(
        **traincfg.dataset.image_transforms
    )
    traincfg.observation_delta_indices = None
    traincfg.policy = policycfg
    pprint(traincfg)
    dataset = make_dataset(traincfg)
    ds_meta = dataset.meta
    del dataset

    # turn off ensembling
    policycfg.temporal_ensemble_coeff = None
    policycfg.temporal_ensemble_coeff = 0.01
    policycfg.n_action_steps = 1
    policycfg.n_action_steps = 25
    policycfg.n_action_steps = 50

    policy = make_policy(
        cfg=policycfg,
        ds_meta=ds_meta,
        # env_cfg=cfg.env,
    )
    policy.eval()
    # policy_metadata = policy.metadata

    # Record the policy's behavior.
    # if cfg.record:
    # policy = _policy.PolicyRecorder(policy, "policy_records")

    policy = LeRobotPolicy(policy, steps=policycfg.n_action_steps)
    device = get_safe_torch_device(policycfg.device, log=True)
    policy.device = device
    policy = TryPolicy(policy)
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type) if policycfg.use_amp else nullcontext(),
    ):

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

        server = WebsocketPolicyServer(
            policy=policy,
            host=cfg.host,
            port=cfg.port,
            # metadata=policy_metadata,
        )
        server.serve_forever()

"""

def main(cfg: ServeConfig) -> None:
    ckpt_root = Path(cfg.ckpt.dir)

    # --- Load ACT config robustly (tolerant paths; drop HF-only keys) ---
    policycfg, model_cfg_path = _load_act_config(ckpt_root)
    pprint(policycfg)

    # --- Load training args (optional, for dataset wiring) ---
    train_candidates = [
        model_cfg_path.parent / "train_config.json",
        ckpt_root / "pretrained_model" / "train_config.json",
        ckpt_root / "train_config.json",
    ]
    traincfg_json = None
    for tpath in train_candidates:
        if tpath.exists():
            with open(tpath, "r") as f:
                traincfg_json = json.load(f)
            break
    if traincfg_json is None:
        raise FileNotFoundError(
            "train_config.json not found. Tried:\n- " + "\n- ".join(map(str, train_candidates))
        )

    # Filter to what TrainPipelineConfig actually accepts
    traincfg = TrainPipelineConfig(**_filter_kwargs_for(TrainPipelineConfig, traincfg_json))

    # Conversions / enrichments you already had
    traincfg.dataset = DatasetConfig(**traincfg.dataset)
    traincfg.dataset.image_transforms = ImageTransformsConfig(**traincfg.dataset.image_transforms)
    traincfg.observation_delta_indices = None
    traincfg.policy = policycfg
    pprint(traincfg)

    dataset = make_dataset(traincfg)
    ds_meta = dataset.meta
    del dataset

    # --- Your existing policy build continues unchanged below ---
    policycfg.temporal_ensemble_coeff = None
    policycfg.temporal_ensemble_coeff = 0.01
    policycfg.n_action_steps = 1
    policycfg.n_action_steps = 25
    policycfg.n_action_steps = 50

    policy = make_policy(
        cfg=policycfg,
        ds_meta=ds_meta,
        # env_cfg=cfg.env,
    )
    policy.eval()

    policy = LeRobotPolicy(policy, steps=policycfg.n_action_steps)
    device = get_safe_torch_device(policycfg.device, log=True)
    policy.device = device
    policy = TryPolicy(policy)

    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type) if policycfg.use_amp else nullcontext(),
    ):
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

        server = WebsocketPolicyServer(
            policy=policy,
            host=cfg.host,
            port=cfg.port,
            # metadata=policy_metadata,
        )
        server.serve_forever()
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(ServeConfig))
