from __future__ import annotations

from contextlib import nullcontext
import logging
import os
import time
from typing import Any

from lerobot.common.envs.factory import make_env
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.train_utils import get_step_identifier
from lerobot.common.utils.utils import has_method
import torch
from torch.amp import GradScaler
import torch.distributed as dist
from torch.optim import Optimizer

# ---- distributed utils -----------------------------------------------------


def _read_env() -> tuple[bool, int, int, int]:
    """Fetch torchrun environment variables."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return True, rank, world_size, local_rank


def _init_device(local_rank: int) -> torch.device:
    """Return the device assigned to the current process."""
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def setup_distributed() -> tuple[bool, int, int, torch.device | None]:
    """Initialize distributed mode for torchrun."""
    is_dist, rank, world_size, local_rank = _read_env()
    if not is_dist:
        logging.info("RANK or WORLD_SIZE not found in environment variables")
        return False, 0, 1, None

    device = _init_device(local_rank)
    if not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
    logging.info(f"Process group initialized: rank={dist.get_rank()}, world_size={dist.get_world_size()}")
    return True, rank, world_size, device


def cleanup_distributed() -> None:
    """Destroy the process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ---- training --------------------------------------------------------------


def _compute_loss(policy: PreTrainedPolicy, batch: Any, use_amp: bool) -> tuple[torch.Tensor, dict]:
    """Forward pass through the policy."""
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        return policy.forward(batch)


def _step_optimizer(optimizer: Optimizer, scaler: GradScaler, lock=None) -> None:
    """Step the optimizer with optional locking."""
    ctx = lock if lock is not None else nullcontext()
    with ctx:
        scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """Backpropagate and update the policy."""
    start = time.perf_counter()
    loss, out_dict = _compute_loss(policy, batch, use_amp)
    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm, error_if_nonfinite=False)
    _step_optimizer(optimizer, grad_scaler, lock)
    if lr_scheduler is not None:
        lr_scheduler.step()
    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start
    return train_metrics, out_dict


def update_policy_multi(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batches: dict[str, Any],
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """Backpropagate using the mean loss from multiple batches."""
    start = time.perf_counter()

    # losses, out_dict = [], {}
    out_dict = {}
    loss, _infos = _compute_loss(policy, list(batches.values()), use_amp)

    for head, info in zip(batches.keys(), _infos):
        out_dict[head] = info

    # for head, batch in batches.items():
    # loss, out = _compute_loss(policy, batch, use_amp)
    # losses.append(loss)
    # out_dict[head] = out

    # loss = torch.stack(losses).mean()

    grad_scaler.scale(loss).backward()  # in favor of retain_graph for ddp
    # for i, l in enumerate(losses):
    # grad_scaler.scale(l).backward(retain_graph=(i < len(losses) - 1))

    grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm, error_if_nonfinite=False)
    _step_optimizer(optimizer, grad_scaler, lock)
    if lr_scheduler is not None:
        lr_scheduler.step()
    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start
    return train_metrics, out_dict


# ---- metrics ----------------------------------------------------------------


def all_gather_metrics(metrics_dict: dict, device: torch.device) -> dict:
    """Average numeric metrics across ranks."""
    result = {}
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            tensor = torch.tensor(value, device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            result[key] = tensor.item() / dist.get_world_size()
    return result


def gather_rewards(rewards: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Gather a variable number of reward tensors."""
    local_size = torch.tensor([rewards.shape[0]], device=device)
    size_list = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size)
    max_size = max(size.item() for size in size_list)
    if rewards.shape[0] < max_size:
        padding = torch.zeros(max_size - rewards.shape[0], *rewards.shape[1:], device=device)
        rewards = torch.cat([rewards, padding], dim=0)
    tensor_list = [torch.zeros_like(rewards) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, rewards)
    return torch.cat([t[: size_list[i].item()] for i, t in enumerate(tensor_list)], dim=0)


# ---- env helpers ------------------------------------------------------------


def create_distributed_envs(env_cfg, world_size: int, rank: int):
    """Build environment shards per rank."""
    env_seed = (getattr(env_cfg, "seed", 0) or 0) + rank * 1000
    return make_env(env_cfg, n_envs=getattr(env_cfg, "n_envs_per_process", 1), start_seed=env_seed)


def make_train_metrics() -> dict[str, AverageMeter]:
    """Meters for loss, grad norm, lr and timing."""
    return {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }


def prepare_log_dict(d: dict[str, float]) -> dict[str, float]:
    """Detach tensors and flatten keys for wandb."""
    from flax.traverse_util import flatten_dict, unflatten_dict
    import jax
    import torch
    from rich.pretty import pprint

    d = {k.replace(".", "/"): v for k, v in flatten_dict(d, sep="/").items()}
    pprint(unflatten_dict(d, sep="/"))

    def _det(x):
        return x.detach().cpu().item() if isinstance(x, torch.Tensor) else x

    return jax.tree.map(_det, d)


def eval_and_log(
    step: int,
    policy: PreTrainedPolicy,
    dataset_ref,
    cfg,
    eval_env,
    wandb_logger=None,
    is_distributed: bool = False,
):
    """Evaluate policy and optionally log to wandb."""

    from lerobot.scripts.eval import eval_policy

    device = get_device_from_parameters(policy)
    step_id = get_step_identifier(step, cfg.steps)
    logging.info(f"Eval policy at step {step}")
    start_t = time.time()
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
    ):
        p = policy.module if is_distributed else policy
        info = eval_policy(
            eval_env,
            p,
            cfg.eval.n_episodes,
            videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
            max_episodes_rendered=4,
            start_seed=cfg.seed,
        )
    eval_s = time.time() - start_t

    eval_metrics = {
        "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
        "pc_success": AverageMeter("success", ":.1f"),
        "eval_s": AverageMeter("eval_s", ":.3f"),
    }
    tracker = MetricsTracker(
        cfg.batch_size,
        dataset_ref.num_frames,
        dataset_ref.num_episodes,
        eval_metrics,
        initial_step=step,
    )
    tracker.eval_s = info["aggregated"].pop("eval_s")
    tracker.avg_sum_reward = info["aggregated"].pop("avg_sum_reward")
    tracker.pc_success = info["aggregated"].pop("pc_success")
    logging.info(f"[Eval time: {eval_s:.2f}s] {tracker}")
    if wandb_logger:
        log_dict = {**tracker.to_dict(), **info, "eval_time": eval_s}
        wandb_logger.log_dict(log_dict, step, mode="eval")
        wandb_logger.log_video(info["video_paths"][0], step, mode="eval")

