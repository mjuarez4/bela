from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import logging
from pprint import pformat
import time

from flax.traverse_util import flatten_dict
import jax
from lerobot.common import envs
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.optim.optimizers import AdamConfig, AdamWConfig
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType
from lerobot.scripts.eval import eval_policy

#
from rich.pretty import pprint
import torch
from torch.amp import GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import tyro

import bela
from bela.common.datasets.util import DataStats, postprocess
import bela.train_tools as tt
from bela.typ import PolicyFeature
from bela.util import spec

if True:
    from bela.common.policies.make import make_policy
else:
    from lerobot.common.policies.factory import make_policy


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
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(repo_id="none"))
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

        datasets = {}
        for head in (heads := ["human", "robot"]):
            _cfg = deepcopy(cfg)
            _cfg.dataset.repo_id = cfg.human_repos[0] if head == "human" else cfg.robot_repos[0]
            _cfg.dataset.revision = cfg.human_revisions[0] if head == "human" else cfg.robot_revisions[0]

            dataset = bela.common.dataset.make_dataset(_cfg)
            datasets[head] = dataset

        # dataset = bela.common.dataset.make_dataset(cfg)  # dataset = make_dataset(cfg)

        _batchspec = {
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
                    "mano.hand_pose": PolicyFeature(FeatureType.STATE, (15, 3)),  # (15, 3, 3)),
                    # "mano.global_orient": PolicyFeature( FeatureType.STATE, (3,)),  # (3, 3)),
                    "kp3d": PolicyFeature(FeatureType.STATE, (21, 3)),
                },
                "shared": {
                    "image.low": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
                    "cam.pose": PolicyFeature(FeatureType.STATE, (6,)),
                },
            },
        }
        _batchspec = flatten_dict(_batchspec, sep=".")
        sharespec = {k: v for k, v in _batchspec.items() if k.startswith("observation.shared")}

        examples = {}
        for head in heads:
            example = datasets[head][0]
            example.pop("task")

            example = jax.tree.map(lambda *x: torch.stack((x)), example, example)
            example = postprocess(example, _batchspec, head=head)
            examples[head] = example
            pprint(spec(example))

        stats = {head: DataStats(head=head) for head in heads}
        for head, stat in stats.items():
            stat.maybe_compute(datasets[head], _batchspec)
            # pprint(find_torch_unstable(stat.stats))
            pprint(spec(stat.stats))

        assert "action_is_pad" in example, f"missing key=action_is_pad in {_head:=heads[-1]}"

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

        #
        # update batchspec to reflect the postprocessed dataset
        #
        newbatchspec = flatten_dict(stats["human"].stats | stats["robot"].stats, sep=".")
        newbatchspec = {_k.replace(".mean", ""): v for _k, v in newbatchspec.items() if "mean" in _k}

        def shape_to_policyfeat(path, t: torch.Tensor):
            """for jax.tree.map_with_path"""
            shape = t.shape
            path = ".".join([str(x.key) for x in path])
            if "action" in path:
                return PolicyFeature(FeatureType.ACTION, shape)
            if "image" in path:
                return PolicyFeature(FeatureType.VISUAL, shape)
            return PolicyFeature(FeatureType.STATE, shape)

        batchspec = jax.tree.map_with_path(shape_to_policyfeat, newbatchspec)

        # policy = make_policy( cfg=cfg.policy, ds_meta=dataset.meta)
        policy = make_policy(batchspec, stats, examples)
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
            num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
            num_total_params = sum(p.numel() for p in policy.parameters())

            dataset_ref = datasets[heads[0]]
            logging.info(f"Output dir: {cfg.output_dir}")
            if cfg.env is not None:
                logging.info(f"{cfg.env.task=}")
            logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
            logging.info(f"{dataset_ref.num_frames=} ({format_big_number(dataset_ref.num_frames)})")
            logging.info(f"{dataset_ref.num_episodes=}")
            logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
            logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
            logging.info(f"Distributed training on {world_size} GPUs")

        # Build a dataloader per dataset
        dl_iters, samplers = {}, {}
        batch_size = cfg.batch_size // world_size if is_distributed else cfg.batch_size
        for head in heads:
            dataset = datasets[head]
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
            dataloader = torch.utils.data.DataLoader(
                dataset,
                num_workers=cfg.num_workers,
                prefetch_factor=4,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                pin_memory=device.type != "cpu",
                drop_last=False,
            )
            dl_iters[head] = cycle(dataloader)

        dataset_ref = datasets[heads[0]]

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
            dataset_ref.num_frames,
            dataset_ref.num_episodes,
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

            if is_distributed:
                for sampler in samplers.values():
                    sampler.set_epoch(step)

            start_time = time.perf_counter()
            batches = {}
            for head in heads:
                batch = next(dl_iters[head])
                batch = postprocess(batch, batchspec, head=head)
                batches[head] = batch

            train_tracker.dataloading_s = time.perf_counter() - start_time

            for batch in batches.values():
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device, non_blocking=True)

            train_tracker, output_dict = tt.update_policy_multi(
                train_tracker,
                policy,
                batches,
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
                        agg_metrics[key] = AverageMeter(train_metrics[key].name, train_metrics[key].fmt)
                        if key in global_metrics:
                            agg_metrics[key].avg = global_metrics[key]

                    agg_tracker = MetricsTracker(
                        batch_size,
                        dataset_ref.num_frames,
                        dataset_ref.num_episodes,
                        agg_metrics,
                        initial_step=step,
                    )

                    # Log comprehensive throughput information
                    gpu_info = f"{world_size} GPU{'s' if world_size > 1 else ''}"
                    throughput = (
                        f"Throughput: {samples_per_second:.1f} samples/sec ({batches_per_second:.1f} batches/sec)"
                    )
                    time_info = f"Time: {interval_time:.2f}s for {global_steps} steps"
                    batch_info = f"Batch size: {batch_size}/GPU, {batch_size * world_size} total"

                    logging.info(f"[{gpu_info} | {throughput} | {time_info} | {batch_info}] {agg_tracker}")

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
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                # Unwrap the policy if using DDP
                save_policy = policy.module if is_distributed else policy
                save_checkpoint(checkpoint_dir, step, cfg, save_policy, optimizer, lr_scheduler)
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
                    torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
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
                    dataset_ref.num_frames,
                    dataset_ref.num_episodes,
                    eval_metrics,
                    initial_step=step,
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(f"[Eval time: {eval_time:.2f}s] {eval_tracker}")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_log_dict["eval_time"] = eval_time
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

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
