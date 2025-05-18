import logging
from tqdm import tqdm
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

import jax
import os
import time
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (get_step_checkpoint_dir,
                                              get_step_identifier,
                                              save_checkpoint,
                                              update_last_checkpoint)
from lerobot.common.utils.utils import (format_big_number,
                                        get_safe_torch_device, has_method,
                                        init_logging)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


from lerobot.common.utils.wandb_utils import WandBLogger
import train_tools as tt

from dataclasses import dataclass, field, asdict
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig

from pprint import pformat
#
from rich.pretty import pprint
import logging

import tyro

from lerobot.common.optim.optimizers import AdamWConfig, AdamConfig
from lerobot.common import envs

import bela
from bela.common.policies.bela import BELAPolicy

if True:
    from bela.common.policies.make import make_policy
else:
    from lerobot.common.policies.factory import make_policy


from flax.traverse_util import flatten_dict, unflatten_dict
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.configs.policies import PreTrainedConfig

@PreTrainedConfig.register_subclass("bela")
@dataclass
class BELAConfig(ACTConfig):

    @property
    def observation_delta_indices(self):
        # we do dynamic composition of action. 
        # since the prediction is future states
        return self.action_delta_indices

    @property
    def image_delta_indices(self):
        # only one image to save space 
        return [0]



@dataclass
class MyTrainConfig(TrainPipelineConfig):

    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(repo_id="none")
    )
    human_repos: list[str] = field(default_factory=list) 
    human_revisions: list[str] = field(default_factory=list)  # branch
    robot_repos: list[str] = field(default_factory=list)
    robot_revisions: list[str] = field(default_factory=list)  # branch
    env: envs.EnvConfig | None = None

    policy: BELAConfig = field(default_factory=BELAConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    optimizer: AdamWConfig | AdamConfig = field(default_factory=AdamWConfig)

    def validate(self):
        pass

    def to_dict(self):
        return asdict(self)


def spec(thing):
    return jax.tree.map(
        lambda x: x.shape if isinstance(x, torch.Tensor) else type(x), thing
    )

def main(cfg: MyTrainConfig):
    """
    Entry point for distributed training - compatible with torchrun
    """

    pprint(cfg)

    # Initialize distributed environment (required for torchrun)
    is_distributed, rank, world_size, device = tt.setup_distributed()
    if not is_distributed:
        logging.warning("Not running in distributed mode.")
        rank = 0
        world_size = 1
        device = get_safe_torch_device(cfg.policy.device, log=True)

    # Configure logging for each process
    if rank != 0:
        # Disable verbose logging for non-master processes
        logging.getLogger().setLevel(logging.WARNING)

    try:
        cfg.validate()
        if rank == 0:
            logging.info(pformat(cfg.to_dict()))

        # Initialize wandb only on the main process
        wandb_logger = None
        if rank == 0 and cfg.wandb.enable and cfg.wandb.project:
            wandb_logger = WandBLogger(cfg)
        elif rank == 0:
            logging.info("Logs will be saved locally.")

        # Set seed with rank-specific offset to ensure different batch sampling
        if cfg.seed is not None:
            set_seed(cfg.seed + rank)

        # Configure device settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        if rank == 0:
            logging.info("Creating dataset")
        dataset = bela.common.dataset.make_dataset(cfg) # dataset = make_dataset(cfg)


        # monkeypatch to build the action at runtime

        from functools import partial
        from typing import Callable

        def compose_action(x, key):
            joints = torch.stack(x["observation.state.joints"])
            gripper = torch.stack(x["observation.state.gripper"]).unsqueeze(-1)
            action = torch.cat([joints, gripper], dim=-1)
            return [a for a in action]  # return to list for compliance

        # dataset.qfns = { "action": compose_action }

        batchspec = {
            "observation": {
                "robot": {
                    "joints": PolicyFeature(FeatureType.STATE, (7,)),
                    "image.side": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
                    "image.wrist": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
                    # "pose": PolicyFeature(FeatureType.STATE, (6,)),
                    "gripper": PolicyFeature(FeatureType.STATE, (1,)),
                },
                "human": {
                    # "gripper": PolicyFeature(FeatureType.STATE, (1,)),
                    "mano.hand_pose": PolicyFeature( FeatureType.STATE, (15, 3)),  # (15, 3, 3)),
                    # "mano.global_orient": PolicyFeature( FeatureType.STATE, (3,)),  # (3, 3)),
                    "kp3d": PolicyFeature(FeatureType.STATE, (21, 3)),
                },
                "shared": {
                    "image.low": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
                    "cam.pose": PolicyFeature(FeatureType.STATE, (6,)),
                },
            },
        }
        batchspec = flatten_dict(batchspec, sep=".")
        sharespec = {k: v for k, v in batchspec.items() if k.startswith("observation.shared")}

        def postprocess(batch, h, flat=True):
            if 'task' in batch:
                batch.pop('task') # it is annoying to pprint
            batch = {k.replace("observation.", f"observation.{h}."): v for k, v in batch.items()}
            batch = {k.replace(".state.", f"."): v for k, v in batch.items()}

            # move shared features
            # if the key would be in sharespec, if named properly, then rename
            def canshare(k):
                sharekey = k.replace(f"observation.{h}.", "observation.shared.")
                return sharekey in sharespec, sharekey

            newbatch = {}
            for k, v in batch.items():
                can,key = canshare(k)
                # print(k, key)
                if can:
                    newbatch[key] = v
                else:
                    newbatch[k] = v
            batch = newbatch

            # create shared features that don't exist
            myfeat = 'observation.shared.cam.pose'
            if h == 'human':
                kp3d = batch['observation.human.kp3d']
                bs,t,n, *_ = kp3d.shape
                palm = kp3d[:,:,0] # b,t,n,3 -> b,t,3
                kp3d = kp3d[:,:,1:] # should be 20 now
                kp3d = kp3d.reshape(bs, t, -1) if flat else kp3d
                batch['observation.human.kp3d'] = kp3d

                bs, t, *_ = kp3d.shape
                rot = batch['observation.human.mano.global_orient']
                rot = matrix_to_rotation_6d(rot.reshape(-1,3,3))
                rot = rot.reshape(bs, t, -1)
                batch[myfeat] = torch.cat([palm, rot], dim=-1) 

                # convert mano.hand_pose to rotation_6d
                manopose = batch['observation.human.mano.hand_pose'] # b,t,m,3,3
                manopose = matrix_to_rotation_6d(manopose.reshape(-1,3,3))
                manopose = manopose.reshape(bs, t, -1, manopose.shape[-1])
                manopose = manopose.reshape(bs, t, -1) if flat else manopose
                batch['observation.human.mano.hand_pose'] = manopose

            if h == 'robot':
                pass

            # pprint(spec(batch))
            batch = {k: v for k, v in batch.items() if k in batchspec}
            # design action
            # everything is a state variable if image is not in the name
            isstate = lambda x: "image" not in x
            actions = {}
            for k, v in batch.items():
                if isstate(k):
                    a = batch[k]
                    # first one is qpos
                    q = batch[k][:,0] # b,0 not 0,t
                    actions[k.replace("observation.", f"action.")] = a
                    actions[k] = q
            batch = batch | actions

            batch["heads"] = [h,"shared"]
            return batch

        example = dataset[0]
        example.pop('task') 

        example = jax.tree.map(lambda *x:  torch.stack((x)), example,example )
        example = postprocess(example, h='human')
        pprint(spec(example))
        # quit()

        def compute_stats(dataset, h):
            # Compute statistics for normalization
            # TODO you will need separate stats for actions even though shared
            stats = {}
            data = []
            samples = list(range(len(dataset)))[:10]
            for i in tqdm(samples, total=len(samples), desc="Computing stats",  leave=False):
                d = dataset[i]
                d.pop('task')
                d = jax.tree.map(lambda x: x.unsqueeze(0), d)
                d = postprocess(d, h='human')
                d.pop('heads')
                data.append(d)
            data = jax.tree.map(lambda *x: torch.concatenate((x)), data[0], *data[1:])

            def make_stat(stack):
                return {
                    "mean": stack.mean(dim=0),
                    "std": stack.std(dim=0),
                    "max": stack.max(dim=0)[0],
                    "min": stack.min(dim=0)[0],
                    'count': stack.shape[0],
                }
            stats = jax.tree.map(lambda x: make_stat(x), data)

            def take_act(k,v):
                print(k[0].key)
                y = k[0].key.startswith("action") and isinstance(v, torch.Tensor)
                return v[0] if y else v
            stats = jax.tree.map_with_path(take_act, stats)
            return stats

        pprint(spec(compute_stats(dataset, h='human')))
        quit()

        """
        # Create environment used for evaluating checkpoints during training on simulation data.
        # For distributed setting, only the main process needs to handle evaluation
        eval_env = None
        if rank == 0 and cfg.eval_freq > 0 and cfg.env is not None:
            logging.info("Creating env")
            eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)
        """

        if rank == 0:
            logging.info("Creating policy")

        # policy = make_policy( cfg=cfg.policy, ds_meta=dataset.meta)
        policy = make_policy()
        policy = policy.to(device)

        # Important: Create optimizer BEFORE wrapping model in DDP
        # This ensures get_optim_params() works correctly
        if rank == 0:
            logging.info("Creating optimizer and scheduler")
        optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

        # Now wrap the policy with DDP and enable unused parameter detection
        if is_distributed:
            # Enable find_unused_parameters to handle parameters not used in forward pass
            policy = DDP(
                policy,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True,
            )
            if rank == 0:
                logging.info("DDP enabled with unused parameter detection")

        grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

        step = 0  # number of policy updates (forward + backward + optim)

        if cfg.resume:
            # Only the main process loads the checkpoint, then broadcast to all
            map_location = {"cuda:0": f"cuda:{rank}"}
            if rank == 0:
                state_dict = torch.load(cfg.checkpoint_path, map_location=map_location)
                step = state_dict["step"]
            else:
                state_dict = None

            if is_distributed:
                # Broadcast step from rank 0 to all other processes
                step_tensor = torch.tensor([step], device=device)
                dist.broadcast(step_tensor, src=0)
                step = step_tensor.item()

                # Broadcast model parameters
                if rank == 0:
                    model_state_dict = state_dict["model"]
                else:
                    model_state_dict = policy.module.state_dict()

                for param_name, param in model_state_dict.items():
                    dist.broadcast(param, src=0)

                policy.module.load_state_dict(model_state_dict)
            else:
                # Non-distributed mode
                policy.load_state_dict(state_dict["model"])

            # Optimizer and scheduler are initialized per process
            if rank == 0:
                optimizer.load_state_dict(state_dict["optimizer"])
                if lr_scheduler is not None and "lr_scheduler" in state_dict:
                    lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

        if rank == 0:
            num_learnable_params = sum(
                p.numel() for p in policy.parameters() if p.requires_grad
            )
            num_total_params = sum(p.numel() for p in policy.parameters())

            logging.info(f"Output dir: {cfg.output_dir}")
            if cfg.env is not None:
                logging.info(f"{cfg.env.task=}")
            logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
            logging.info(
                f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})"
            )
            logging.info(f"{dataset.num_episodes=}")
            logging.info(
                f"{num_learnable_params=} ({format_big_number(num_learnable_params)})"
            )
            logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
            logging.info(f"Distributed training on {world_size} GPUs")

        # Create distributed sampler to ensure different data on each GPU
        if is_distributed:
            if hasattr(cfg.policy, "drop_n_last_frames"):
                episodic_sampler = EpisodeAwareSampler(
                    dataset.episode_data_index,
                    drop_n_last_frames=cfg.policy.drop_n_last_frames,
                    shuffle=True,
                )
                # Wrap the episodic sampler with DistributedSampler
                sampler = DistributedSampler(
                    episodic_sampler,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=cfg.seed if cfg.seed is not None else 0,
                )
            else:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=cfg.seed if cfg.seed is not None else 0,
                )
            shuffle = False
        else:
            # For non-distributed case, use the original sampler logic
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

        # Calculate per-GPU batch size
        batch_size = cfg.batch_size // world_size if is_distributed else cfg.batch_size

        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            prefetch_factor=2,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=device.type != "cpu",
            drop_last=False,
        )
        dl_iter = cycle(dataloader)

        policy.train()

        train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }

        train_tracker = MetricsTracker(
            batch_size,
            dataset.num_frames,
            dataset.num_episodes,
            train_metrics,
            initial_step=step,
        )

        # Create a lock for multi-process optimizer updates if needed
        optimizer_lock = None

        if rank == 0:
            logging.info("Start distributed offline training on a fixed dataset")

        # Track the start time for interval timing
        interval_start_time = time.time()
        last_log_step = step

        logging.info(f"Start training on {world_size} GPUs")
        for _ in range(step, cfg.steps):
            # Add barrier to synchronize processes before each step
            if is_distributed:
                dist.barrier()

            # Set epoch for the sampler to ensure proper shuffling
            if is_distributed:
                sampler.set_epoch(step)

            start_time = time.perf_counter()
            batch = next(dl_iter)
            batch = postprocess(batch, h='human')

            is_leaf = lambda x: isinstance(x, torch.Tensor) or isinstance(x, list)
            # pprint(spec(batch))

            train_tracker.dataloading_s = time.perf_counter() - start_time

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            train_tracker, output_dict = tt.update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=cfg.policy.use_amp,
                lock=optimizer_lock,
            )

            # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
            # increment `step` here.
            step += 1
            train_tracker.step()
            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
            is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
            is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

            # Synchronize metrics across processes for logging
            if is_log_step:
                # Calculate interval time
                interval_time = time.time() - interval_start_time
                local_steps = step - last_log_step

                # Calculate global steps across all GPUs - for DDP, each GPU does the same steps
                # but processes different data, so total steps = local steps
                global_steps = local_steps

                # Calculate total batches processed - each GPU processes its own batches
                # so total batches = local steps * world_size
                total_batches = local_steps * world_size

                # Calculate total samples - each batch has batch_size samples per GPU
                total_samples = total_batches * batch_size

                # Calculate rates
                samples_per_second = total_samples / interval_time
                batches_per_second = total_batches / interval_time
                steps_per_second = global_steps / interval_time

                # Gather metrics from all processes
                metrics_dict = train_tracker.to_dict()
                if is_distributed:
                    global_metrics = tt.all_gather_metrics(metrics_dict, device)
                else:
                    global_metrics = metrics_dict

                if rank == 0:
                    # Manually update the metrics for reporting
                    # Create a new tracker to hold the aggregated metrics
                    agg_metrics = {}
                    for key in train_metrics:
                        agg_metrics[key] = AverageMeter(
                            train_metrics[key].name, train_metrics[key].fmt
                        )
                        if key in global_metrics:
                            agg_metrics[key].avg = global_metrics[key]

                    agg_tracker = MetricsTracker(
                        batch_size,
                        dataset.num_frames,
                        dataset.num_episodes,
                        agg_metrics,
                        initial_step=step,
                    )

                    # Log comprehensive throughput information
                    gpu_info = f"{world_size} GPU{'s' if world_size > 1 else ''}"
                    throughput = f"Throughput: {samples_per_second:.1f} samples/sec ({batches_per_second:.1f} batches/sec)"
                    time_info = f"Time: {interval_time:.2f}s for {global_steps} steps"
                    batch_info = (
                        f"Batch size: {batch_size}/GPU, {batch_size * world_size} total"
                    )

                    logging.info(
                        f"[{gpu_info} | {throughput} | {time_info} | {batch_info}] {agg_tracker}"
                    )

                    if wandb_logger:
                        wandb_log_dict = global_metrics
                        # Add timing information to wandb log
                        wandb_log_dict["interval_time"] = interval_time
                        wandb_log_dict["samples_per_second"] = samples_per_second
                        wandb_log_dict["batches_per_second"] = batches_per_second
                        wandb_log_dict["steps_per_second"] = steps_per_second
                        wandb_log_dict["total_gpus"] = world_size
                        wandb_log_dict["total_batch_size"] = batch_size * world_size
                        if output_dict:
                            wandb_log_dict.update(output_dict)
                        wandb_logger.log_dict(wandb_log_dict, step)

                train_tracker.reset_averages()

                # Reset interval timer and counters
                interval_start_time = time.time()
                last_log_step = step

            # Only the main process handles checkpoint saving
            if rank == 0 and cfg.save_checkpoint and is_saving_step:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(
                    cfg.output_dir, cfg.steps, step
                )
                # Unwrap the policy if using DDP
                save_policy = policy.module if is_distributed else policy
                save_checkpoint(
                    checkpoint_dir, step, cfg, save_policy, optimizer, lr_scheduler
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            # Only the main process handles evaluation
            if rank == 0 and cfg.env and is_eval_step:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                eval_start_time = time.time()
                with (
                    torch.no_grad(),
                    (
                        torch.autocast(device_type=device.type)
                        if cfg.policy.use_amp
                        else nullcontext()
                    ),
                ):
                    # Unwrap the policy if using DDP
                    eval_policy_model = policy.module if is_distributed else policy
                    eval_info = eval_policy(
                        eval_env,
                        eval_policy_model,
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                    )
                eval_time = time.time() - eval_start_time

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop(
                    "avg_sum_reward"
                )
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(f"[Eval time: {eval_time:.2f}s] {eval_tracker}")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_log_dict["eval_time"] = eval_time
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(
                        eval_info["video_paths"][0], step, mode="eval"
                    )

        # Cleanup
        if eval_env:
            eval_env.close()

        if is_distributed:
            tt.cleanup_distributed()

        if rank == 0:
            logging.info("End of distributed training")

    except Exception as e:
        logging.error(f"Error during training: {e}")
        if is_distributed:
            tt.cleanup_distributed()
        raise


if __name__ == "__main__":
    init_logging()
    main(tyro.cli(MyTrainConfig))
